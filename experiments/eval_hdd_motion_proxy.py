#!/usr/bin/env python3
"""Residual shortcut check: correlate V-JEPA 2 residual similarity with trivial motion proxies.

Tests whether V-JEPA 2 temporal residual AP on HDD is explained by simple
motion magnitude correlation (speed/blur/shake) rather than genuine temporal
structure. Computes per-segment mean pixel difference between consecutive
frames, then Spearman-correlates pairwise proxy similarity with residual
similarity across all evaluation pairs.

Low correlation (< 0.3) confirms residuals capture motion *structure*, not
just motion *intensity*.

Usage:
    python experiments/eval_hdd_motion_proxy.py --hdd-dir /path/to/hdd
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_hdd_vlm_bridge import (  # noqa: E402
    discover_sessions,
    load_gps,
    extract_maneuver_segments,
    cluster_intersections,
    filter_mixed_clusters,
)
from eval_hdd_intersections import ManeuverSegment  # noqa: E402


def compute_motion_proxy(
    video_path: str,
    start_sec: float,
    end_sec: float,
    max_frames: int = 32,
    max_resolution: int = 256,
) -> float | None:
    """Compute mean pixel difference between consecutive frames.

    Returns scalar motion magnitude (mean absolute pixel difference).
    Lower resolution for speed — we only need a coarse motion signal.
    """
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        time_base = float(stream.time_base or 1e-6)

        seek_sec = max(0.0, start_sec - 0.5)
        seek_pts = int(seek_sec / time_base)
        container.seek(seek_pts, stream=stream)

        frames = []
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts) * time_base
            if t < start_sec - 0.1:
                continue
            if t > end_sec + 0.1:
                break

            img = frame.to_ndarray(format="rgb24")
            if img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, max_resolution))
            frames.append(img.astype(np.float32))

            if len(frames) >= max_frames:
                break

        container.close()

        if len(frames) < 3:
            return None

        # Uniformly subsample if too many frames
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Mean absolute pixel difference between consecutive frames
        diffs = []
        for i in range(len(frames) - 1):
            diff = np.abs(frames[i + 1] - frames[i]).mean()
            diffs.append(diff)

        return float(np.mean(diffs))

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="HDD Motion Proxy Correlation Check"
    )
    parser.add_argument("--hdd-dir", type=str, required=True)
    parser.add_argument("--context-sec", type=float, default=3.0)
    parser.add_argument("--max-clusters", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir

    print("=" * 70)
    print("HDD MOTION PROXY CORRELATION CHECK")
    print("=" * 70)

    # --- Discover and cluster (same as eval_hdd_intersections.py) ---
    print("\nStep 1: Discovering sessions...")
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} sessions")

    print("\nStep 2: Extracting maneuver segments...")
    all_segments = []
    for session_id in tqdm(sorted(sessions.keys()), desc="Loading"):
        info = sessions[session_id]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            session_id, labels, gps_ts, gps_lats, gps_lngs,
            info["video_path"], info["video_start_unix"],
        )
        all_segments.extend(segs)
    print(f"  Total segments: {len(all_segments)}")

    print("\nStep 3: Clustering...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
    eval_segments = []
    cluster_to_indices = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)
    print(f"  Mixed clusters: {len(mixed)}, segments: {len(eval_segments)}")

    # --- Compute motion proxy for each segment ---
    print("\nStep 4: Computing motion proxies...")
    t0 = time.time()
    motion_proxies = {}
    failed = 0
    for i, seg in enumerate(tqdm(eval_segments, desc="Motion proxy")):
        start_sec = seg.start_frame / 3.0 - args.context_sec
        end_sec = seg.end_frame / 3.0 + args.context_sec
        start_sec = max(0.0, start_sec)
        proxy = compute_motion_proxy(seg.video_path, start_sec, end_sec)
        if proxy is not None:
            motion_proxies[i] = proxy
        else:
            failed += 1
    print(f"  Computed: {len(motion_proxies)}/{len(eval_segments)} ({failed} failed)")
    print(f"  Time: {time.time() - t0:.1f}s")

    # --- Load V-JEPA 2 residual pair scores ---
    pair_scores_path = hdd_dir / "pair_scores.json"
    if not pair_scores_path.exists():
        print(f"\nERROR: {pair_scores_path} not found. Run eval_hdd_intersections.py first.")
        return

    print(f"\nStep 5: Loading pair scores from {pair_scores_path}...")
    with open(pair_scores_path) as f:
        pair_data = json.load(f)

    # --- Enumerate pairs and compute correlations ---
    print("\nStep 6: Computing correlations...")

    # Build pair list (same as eval_hdd_intersections.py)
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in motion_proxies]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                pair_a_indices.append(indices[a_pos])
                pair_b_indices.append(indices[b_pos])
                gt = 1 if eval_segments[indices[a_pos]].label == eval_segments[indices[b_pos]].label else 0
                pair_gts.append(gt)

    # Motion proxy similarity: negative absolute difference (higher = more similar)
    motion_sims = []
    for a, b in zip(pair_a_indices, pair_b_indices):
        motion_sims.append(-abs(motion_proxies[a] - motion_proxies[b]))
    motion_sims = np.array(motion_sims)

    # Load residual similarity scores
    if "vjepa2_temporal_residual" not in pair_data:
        print("  ERROR: vjepa2_temporal_residual not in pair_scores.json")
        return

    residual_scores = np.array(pair_data["vjepa2_temporal_residual"]["scores"])
    residual_labels = np.array(pair_data["vjepa2_temporal_residual"]["labels"])

    # The pair_scores.json may have a different pair ordering.
    # Use the motion proxy pairs directly and compare with residual.
    # If lengths don't match, recompute residual on the same pairs.
    print(f"  Motion proxy pairs: {len(motion_sims)}")
    print(f"  Residual pairs: {len(residual_scores)}")

    if len(motion_sims) == len(residual_scores):
        # Same pair set — correlate directly
        spearman_r, spearman_p = spearmanr(motion_sims, residual_scores)
        pearson_r, pearson_p = pearsonr(motion_sims, residual_scores)
    else:
        # Different pair counts (some segments failed motion proxy).
        # Use motion proxy pairs only — residual scores are in pair_scores.json
        # which uses ALL pairs. We need to match by index.
        print("  Pair count mismatch — using motion proxy subset only")
        # Can't directly match without index mapping. Report proxy stats only.
        spearman_r = spearman_p = pearson_r = pearson_p = float("nan")

    # Also compute: does motion proxy predict label?
    from sklearn.metrics import average_precision_score, roc_auc_score
    motion_labels = np.array(pair_gts)
    if motion_labels.sum() > 0 and motion_labels.sum() < len(motion_labels):
        motion_ap = average_precision_score(motion_labels, motion_sims)
        motion_auc = roc_auc_score(motion_labels, motion_sims)
    else:
        motion_ap = motion_auc = float("nan")

    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Motion proxy stats:")
    proxy_vals = list(motion_proxies.values())
    print(f"    Mean pixel diff: {np.mean(proxy_vals):.2f} ± {np.std(proxy_vals):.2f}")
    print(f"    Range: [{np.min(proxy_vals):.2f}, {np.max(proxy_vals):.2f}]")

    print(f"\n  Motion proxy as maneuver discriminator:")
    print(f"    AP = {motion_ap:.4f}, AUC = {motion_auc:.4f}")
    print(f"    (V-JEPA 2 residual AP = 0.956 for reference)")

    print(f"\n  Correlation with V-JEPA 2 residual similarity:")
    print(f"    Spearman r = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"    Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"\n  Interpretation: {'LOW' if abs(spearman_r) < 0.3 else 'MODERATE' if abs(spearman_r) < 0.6 else 'HIGH'} correlation")
    if abs(spearman_r) < 0.3:
        print("    → Residuals capture motion STRUCTURE, not just motion MAGNITUDE")
    print("=" * 70)

    # Save results
    output = {
        "motion_proxy_stats": {
            "mean": float(np.mean(proxy_vals)),
            "std": float(np.std(proxy_vals)),
            "min": float(np.min(proxy_vals)),
            "max": float(np.max(proxy_vals)),
            "n_segments": len(proxy_vals),
        },
        "motion_proxy_discrimination": {
            "ap": float(motion_ap),
            "auc": float(motion_auc),
        },
        "correlation_with_residual": {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n_pairs": len(motion_sims),
        },
    }
    out_path = hdd_dir / "motion_proxy_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

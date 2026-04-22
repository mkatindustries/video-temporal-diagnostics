#!/usr/bin/env python3
"""α-sweep for DTW similarity transform: exp(-α * d).

Tests whether the paper's fixed α values (α=1 for derivatives/encoder-seq,
α=5 for attention trajectories) are robust. Sweeps α ∈ {0.5, 1, 2, 5, 10}
on three narrative-critical cells:

  1. DINOv3 HDD temporal-derivative DTW (paper: AP=0.498, α=1)
  2. V-JEPA 2 HDD encoder-seq DTW (paper: AP=0.942, α=1)
  3. DINOv3 VCDB temporal-derivative DTW (paper: AP=0.775, α=1)

Usage:
    python experiments/alpha_sweep.py \
        --hdd-dir /path/to/hdd \
        --vcdb-dir /path/to/vcdb/core_dataset
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_hdd_intersections import (
    ManeuverSegment,
    bootstrap_ap,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_gps,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch

ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Temporal derivatives from CLS embeddings
# ---------------------------------------------------------------------------


def compute_derivatives(embeddings: torch.Tensor) -> torch.Tensor:
    """First-order temporal derivatives, L2-normalized. (T,D) -> (T-1,D)."""
    if embeddings.shape[0] < 2:
        return embeddings[:0]  # empty
    derivs = embeddings[1:] - embeddings[:-1]
    return F.normalize(derivs, dim=-1)


# ---------------------------------------------------------------------------
# VCDB helpers
# ---------------------------------------------------------------------------


def load_vcdb_annotations(ann_dir: str) -> set[tuple[str, str]]:
    copy_pairs = set()
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".txt"):
            continue
        cat = fname.replace(".txt", "")
        with open(os.path.join(ann_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                vid_a = os.path.join(cat, parts[0].strip())
                vid_b = os.path.join(cat, parts[1].strip())
                if vid_a != vid_b:
                    pair = tuple(sorted([vid_a, vid_b]))
                    copy_pairs.add(pair)
    return copy_pairs


def sample_vcdb_pairs(keys, copy_pairs):
    key_set = set(keys)
    pos = [(a, b) for a, b in copy_pairs if a in key_set and b in key_set]
    n_pos = len(pos)
    neg = []
    sampled = set(pos)
    rng = np.random.RandomState(42)
    attempts = 0
    while len(neg) < n_pos and attempts < n_pos * 20:
        i, j = rng.randint(0, len(keys)), rng.randint(0, len(keys))
        if i == j:
            attempts += 1
            continue
        p = tuple(sorted([keys[i], keys[j]]))
        if p not in copy_pairs and p not in sampled:
            sampled.add(p)
            neg.append(p)
        attempts += 1
    return pos + neg, np.array([1]*len(pos) + [0]*len(neg))


# ---------------------------------------------------------------------------
# Core: compute DTW distances then sweep α
# ---------------------------------------------------------------------------


def compute_pairwise_dtw(
    seqs: dict,
    pairs: list[tuple],
    device: torch.device,
    chunk_size: int = 512,
    normalize: bool = True,
) -> np.ndarray:
    """Compute raw DTW distances for all pairs. Returns (N,) array.

    Args:
        normalize: Normalize sequences to [0, 1] before DTW. Must match
            the setting used in the paper's eval script for each fingerprint
            (False for temporal derivatives, True for encoder-seq).
    """
    dists = np.zeros(len(pairs))
    # Process in chunks for GPU memory
    for start in tqdm(range(0, len(pairs), chunk_size), desc="  DTW", leave=False):
        end = min(start + chunk_size, len(pairs))
        chunk_pairs = pairs[start:end]
        seqs_a = [seqs[a].to(device) for a, b in chunk_pairs if a in seqs and b in seqs]
        seqs_b = [seqs[b].to(device) for a, b in chunk_pairs if a in seqs and b in seqs]
        if not seqs_a:
            continue
        chunk_dists = dtw_distance_batch(seqs_a, seqs_b, normalize=normalize)
        # Map back
        idx = start
        for a, b in chunk_pairs:
            if a in seqs and b in seqs:
                dists[idx] = chunk_dists[idx - start].item()
            idx += 1
    return dists


def sweep_alpha(dists: np.ndarray, labels: np.ndarray, alphas: list[float]) -> dict:
    """Sweep α on pre-computed distances. Returns {α: {ap, ci_low, ci_high}}."""
    results = {}
    for alpha in alphas:
        sims = np.exp(-alpha * dists)
        ap, ci_lo, ci_hi = bootstrap_ap(sims, labels, n_resamples=2000)
        results[alpha] = {"ap": round(ap, 4), "ci_low": round(ci_lo, 4), "ci_high": round(ci_hi, 4)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="α-sweep for DTW similarity transform")
    parser.add_argument("--hdd-dir", type=str,
                        default=None)
    parser.add_argument("--vcdb-dir", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clusters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    hdd_dir = Path(args.hdd_dir)
    vcdb_dir = Path(args.vcdb_dir)
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("α-Sweep: DTW Similarity Transform Robustness")
    print(f"  α values: {ALPHA_VALUES}")
    print("=" * 70)

    all_results = {}

    # ==================================================================
    # HDD setup (shared across DINOv3 and V-JEPA 2 cells)
    # ==================================================================
    print("\n--- HDD setup ---")
    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="Sessions"):
        info = sessions[sid]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            sid, labels, gps_ts, gps_lats, gps_lngs,
            info["video_path"], info["video_start_unix"],
        )
        all_segments.extend(segs)

    clusters_raw = cluster_intersections(all_segments)
    clusters = filter_mixed_clusters(clusters_raw, max_clusters=args.max_clusters)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, members in clusters.items():
        for seg in members:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    # Enumerate HDD pairs
    hdd_pairs: list[tuple[int, int]] = []
    hdd_labels_list: list[int] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cid]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                hdd_pairs.append((indices[a], indices[b]))
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                hdd_labels_list.append(gt)
    hdd_labels = np.array(hdd_labels_list)
    print(f"  HDD: {len(hdd_pairs)} pairs ({int(hdd_labels.sum())} pos)")

    # ==================================================================
    # Cell 1: DINOv3 HDD temporal-derivative DTW
    # ==================================================================
    print("\n" + "=" * 70)
    print("Cell 1: DINOv3 HDD temporal-derivative DTW (paper α=1, AP=0.498)")
    print("=" * 70)

    dinov3_cache = project_root / "datasets" / "dinov3_hdd_frame_features.pt"
    print(f"  Loading features from {dinov3_cache}...")
    dinov3_feats = torch.load(dinov3_cache, map_location="cpu", weights_only=True)

    # Compute derivatives
    dinov3_derivs = {}
    for idx, emb in dinov3_feats["features"].items():
        d = compute_derivatives(emb)
        if d.shape[0] >= 2:
            dinov3_derivs[idx] = d

    # Filter pairs to those with derivatives
    hdd_d_pairs = [(a, b) for a, b in hdd_pairs if a in dinov3_derivs and b in dinov3_derivs]
    hdd_d_labels = np.array([
        1 if eval_segments[a].label == eval_segments[b].label else 0
        for a, b in hdd_d_pairs
    ])
    print(f"  Pairs with derivatives: {len(hdd_d_pairs)}")

    t0 = time.time()
    dists_dinov3_hdd = compute_pairwise_dtw(dinov3_derivs, hdd_d_pairs, device, normalize=False)
    print(f"  DTW computed ({time.time()-t0:.1f}s)")

    cell1 = sweep_alpha(dists_dinov3_hdd, hdd_d_labels, ALPHA_VALUES)
    all_results["dinov3_hdd_deriv"] = cell1

    print(f"\n  {'α':>6s}  {'AP':>6s}  {'95% CI':>14s}")
    print("  " + "-" * 28)
    for alpha, r in cell1.items():
        marker = " <-- paper" if alpha == 1.0 else ""
        print(f"  {alpha:>6.1f}  {r['ap']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{marker}")

    # ==================================================================
    # Cell 2: V-JEPA 2 HDD encoder-seq DTW
    # ==================================================================
    print("\n" + "=" * 70)
    print("Cell 2: V-JEPA 2 HDD encoder-seq DTW (paper α=1, AP=0.942)")
    print("=" * 70)

    vjepa2_cache = project_root / "datasets" / "vjepa2_hdd_encoder_features.pt"
    print(f"  Loading features from {vjepa2_cache}...")
    vjepa2_feats = torch.load(vjepa2_cache, map_location="cpu", weights_only=False)

    vjepa2_seqs = {}
    for idx, feat_dict in vjepa2_feats["features"].items():
        vjepa2_seqs[idx] = feat_dict["encoder_seq"]

    hdd_v_pairs = [(a, b) for a, b in hdd_pairs if a in vjepa2_seqs and b in vjepa2_seqs]
    hdd_v_labels = np.array([
        1 if eval_segments[a].label == eval_segments[b].label else 0
        for a, b in hdd_v_pairs
    ])
    print(f"  Pairs with encoder-seq: {len(hdd_v_pairs)}")

    t0 = time.time()
    dists_vjepa2_hdd = compute_pairwise_dtw(vjepa2_seqs, hdd_v_pairs, device, normalize=True)
    print(f"  DTW computed ({time.time()-t0:.1f}s)")

    cell2 = sweep_alpha(dists_vjepa2_hdd, hdd_v_labels, ALPHA_VALUES)
    all_results["vjepa2_hdd_encoder"] = cell2

    print(f"\n  {'α':>6s}  {'AP':>6s}  {'95% CI':>14s}")
    print("  " + "-" * 28)
    for alpha, r in cell2.items():
        marker = " <-- paper" if alpha == 1.0 else ""
        print(f"  {alpha:>6.1f}  {r['ap']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{marker}")

    # ==================================================================
    # Cell 3: DINOv3 VCDB temporal-derivative DTW
    # ==================================================================
    print("\n" + "=" * 70)
    print("Cell 3: DINOv3 VCDB temporal-derivative DTW (paper α=1, AP=0.775)")
    print("=" * 70)

    vcdb_cache = vcdb_dir / "feature_cache" / "dinov3_sr10_mf100.pt"
    print(f"  Loading features from {vcdb_cache}...")
    vcdb_feats = torch.load(vcdb_cache, map_location="cpu", weights_only=False)

    vcdb_derivs = {}
    for key, feat_dict in vcdb_feats.items():
        emb = feat_dict["embeddings"]
        d = compute_derivatives(emb)
        if d.shape[0] >= 2:
            vcdb_derivs[key] = d

    # Load annotations and sample pairs
    copy_pairs = load_vcdb_annotations(str(vcdb_dir / "annotation"))
    keys = sorted(vcdb_derivs.keys())
    vcdb_pairs, vcdb_labels = sample_vcdb_pairs(keys, copy_pairs)
    # Filter to pairs with derivatives
    vcdb_d_pairs = [(a, b) for a, b in vcdb_pairs if a in vcdb_derivs and b in vcdb_derivs]
    vcdb_d_labels = np.array([
        1 if (a, b) in copy_pairs or (b, a) in copy_pairs else 0
        for a, b in vcdb_d_pairs
    ])
    print(f"  Pairs with derivatives: {len(vcdb_d_pairs)} ({int(vcdb_d_labels.sum())} pos)")

    t0 = time.time()
    dists_dinov3_vcdb = compute_pairwise_dtw(vcdb_derivs, vcdb_d_pairs, device, normalize=False)
    print(f"  DTW computed ({time.time()-t0:.1f}s)")

    cell3 = sweep_alpha(dists_dinov3_vcdb, vcdb_d_labels, ALPHA_VALUES)
    all_results["dinov3_vcdb_deriv"] = cell3

    print(f"\n  {'α':>6s}  {'AP':>6s}  {'95% CI':>14s}")
    print("  " + "-" * 28)
    for alpha, r in cell3.items():
        marker = " <-- paper" if alpha == 1.0 else ""
        print(f"  {alpha:>6.1f}  {r['ap']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{marker}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: α-Sweep")
    print("=" * 70)
    print(f"  {'Cell':<30s}", end="")
    for alpha in ALPHA_VALUES:
        print(f"  {'α='+str(alpha):>8s}", end="")
    print()
    print("  " + "-" * 72)

    cell_names = {
        "dinov3_hdd_deriv": "DINOv3 HDD deriv (ref 0.498)",
        "vjepa2_hdd_encoder": "V-JEPA 2 HDD enc-seq (ref 0.942)",
        "dinov3_vcdb_deriv": "DINOv3 VCDB deriv (ref 0.775)",
    }
    for cell_key, cell_label in cell_names.items():
        print(f"  {cell_label:<30s}", end="")
        for alpha in ALPHA_VALUES:
            ap = all_results[cell_key][alpha]["ap"]
            print(f"  {ap:>8.3f}", end="")
        print()

    # Stability assessment
    print()
    for cell_key, cell_label in cell_names.items():
        aps = [all_results[cell_key][a]["ap"] for a in ALPHA_VALUES]
        spread = max(aps) - min(aps)
        best_alpha = ALPHA_VALUES[np.argmax(aps)]
        print(f"  {cell_label:<30s}  spread={spread:.3f}  best_α={best_alpha}")

    print("=" * 70)

    # Save
    out_path = project_root / "datasets" / "alpha_sweep_results.json"
    # Convert float keys to strings for JSON
    serializable = {}
    for cell_key, cell_results in all_results.items():
        serializable[cell_key] = {str(k): v for k, v in cell_results.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cluster-Level Bootstrap CIs for HDD and nuScenes.

Standard pair-level bootstrap underestimates CI width because pairs
within a GPS cluster share clips, violating independence. This script
resamples clusters (with replacement) and includes all pairs from each
sampled cluster, producing properly-widened CIs.

Usage:
    python experiments/eval_cluster_bootstrap.py \\
        --pairs-json datasets/hdd/pair_scores.json \\
        --segments-json datasets/hdd/cluster_segments.json

    # Or generate cluster info and run in one step:
    python experiments/eval_cluster_bootstrap.py --hdd-dir datasets/hdd
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from eval_hdd_intersections import (
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_gps,
    ManeuverSegment,
)


def cluster_bootstrap_ap(
    cluster_scores: dict[int, tuple[np.ndarray, np.ndarray]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Block-bootstrap AP by cluster.

    Resamples clusters with replacement, then pools all pairs from each
    sampled cluster before computing AP.

    Args:
        cluster_scores: {cluster_id: (scores_array, labels_array)}
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (point_ap, ci_low, ci_high)
    """
    # Pool all pairs for point estimate
    all_scores = np.concatenate([s for s, _ in cluster_scores.values()])
    all_labels = np.concatenate([l for _, l in cluster_scores.values()])

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return float("nan"), float("nan"), float("nan")

    point_ap = average_precision_score(all_labels, all_scores)

    cluster_ids = list(cluster_scores.keys())
    n_clusters = len(cluster_ids)
    rng = np.random.RandomState(seed)

    boot_aps = []
    for _ in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=n_clusters, replace=True)
        boot_scores = np.concatenate([cluster_scores[c][0] for c in sampled])
        boot_labels = np.concatenate([cluster_scores[c][1] for c in sampled])
        if boot_labels.sum() == 0 or boot_labels.sum() == len(boot_labels):
            boot_aps.append(point_ap)
        else:
            boot_aps.append(average_precision_score(boot_labels, boot_scores))

    boot_aps = np.array(boot_aps)
    ci_low = float(np.percentile(boot_aps, 2.5))
    ci_high = float(np.percentile(boot_aps, 97.5))

    return float(point_ap), ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-level bootstrap CIs for HDD"
    )
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--pairs-json", type=str, default=None,
                        help="Pre-computed pair_scores.json (skip re-computation)")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--max-clusters", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir

    print("=" * 70)
    print("CLUSTER-LEVEL BOOTSTRAP CIs (HDD)")
    print("=" * 70)

    # Load sessions and build cluster structure
    print("\nStep 1: Building cluster structure...")
    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="Loading"):
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

    clusters = cluster_intersections(all_segments)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    print(f"  {len(eval_segments)} segments, {len(mixed)} clusters")

    # Load pair scores
    pairs_path = Path(args.pairs_json) if args.pairs_json else hdd_dir / "pair_scores.json"
    if not pairs_path.exists():
        print(f"ERROR: {pairs_path} not found. Run eval_hdd_intersections.py first.")
        return

    with open(pairs_path) as f:
        pair_data = json.load(f)

    methods = list(pair_data.keys())
    print(f"  Methods: {methods}")

    # Build per-cluster pair mapping
    # We need to know which pairs belong to which cluster.
    # Pairs are enumerated in cluster order (same as eval_hdd_intersections.py).
    pair_to_cluster: list[int] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cid]
        n = len(indices)
        n_pairs = n * (n - 1) // 2
        pair_to_cluster.extend([cid] * n_pairs)

    # Compute cluster-level bootstrap for each method
    print(f"\nStep 2: Cluster-level bootstrap ({args.n_resamples} resamples)...")
    print(f"\n  {'Method':<30s}  {'Pair CI':>20s}  {'Cluster CI':>20s}  {'Width ratio':>12s}")
    print("  " + "-" * 85)

    results = {}
    for method in methods:
        scores = np.array(pair_data[method]["scores"])
        labels_arr = np.array(pair_data[method]["labels"])

        if len(scores) != len(pair_to_cluster):
            print(f"  {method}: pair count mismatch ({len(scores)} vs {len(pair_to_cluster)}), skipping")
            continue

        # Build per-cluster score/label arrays
        cluster_scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        pair_idx = 0
        for cid in sorted(cluster_to_indices.keys()):
            indices = cluster_to_indices[cid]
            n = len(indices)
            n_pairs = n * (n - 1) // 2
            c_scores = scores[pair_idx:pair_idx + n_pairs]
            c_labels = labels_arr[pair_idx:pair_idx + n_pairs]
            cluster_scores[cid] = (c_scores, c_labels)
            pair_idx += n_pairs

        # Pair-level bootstrap (standard)
        rng = np.random.RandomState(42)
        n = len(scores)
        point_ap = average_precision_score(labels_arr, scores)
        pair_boot = []
        for _ in range(args.n_resamples):
            idx = rng.randint(0, n, size=n)
            s, l = scores[idx], labels_arr[idx]
            if l.sum() == 0 or l.sum() == n:
                pair_boot.append(point_ap)
            else:
                pair_boot.append(average_precision_score(l, s))
        pair_boot = np.array(pair_boot)
        pair_ci_lo = float(np.percentile(pair_boot, 2.5))
        pair_ci_hi = float(np.percentile(pair_boot, 97.5))
        pair_width = pair_ci_hi - pair_ci_lo

        # Cluster-level bootstrap
        clust_ap, clust_ci_lo, clust_ci_hi = cluster_bootstrap_ap(
            cluster_scores, args.n_resamples
        )
        clust_width = clust_ci_hi - clust_ci_lo

        ratio = clust_width / max(pair_width, 1e-9)

        results[method] = {
            "ap": float(point_ap),
            "pair_ci": [pair_ci_lo, pair_ci_hi],
            "pair_width": pair_width,
            "cluster_ci": [clust_ci_lo, clust_ci_hi],
            "cluster_width": clust_width,
            "width_ratio": ratio,
        }

        print(
            f"  {method:<30s}  "
            f"[{pair_ci_lo:.4f}, {pair_ci_hi:.4f}]  "
            f"[{clust_ci_lo:.4f}, {clust_ci_hi:.4f}]  "
            f"{ratio:.1f}x wider"
        )

    # Save
    out_path = hdd_dir / "cluster_bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cluster-level bootstrap intervals and paired contrasts for HDD.

Standard pair-level bootstrap underestimates CI width because pairs
within a GPS cluster share clips, violating independence. This script
resamples clusters (with replacement) and includes all pairs from each
sampled cluster. Paired method differences use the same sampled clusters.

Usage:
    python experiments/eval_cluster_bootstrap.py \\
        --pairs-json datasets/hdd/pair_scores.json \\
        --segments-json datasets/hdd/cluster_segments.json

    # Or generate cluster info and run in one step:
    python experiments/eval_cluster_bootstrap.py --hdd-dir datasets/hdd
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import (
    ManeuverSegment,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_gps,
)
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from video_retrieval.diagnostics.statistics import (
    cluster_bootstrap_ap,
    paired_cluster_bootstrap_ap_difference,
)


def main():
    parser = argparse.ArgumentParser(description="Cluster-level bootstrap CIs for HDD")
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument(
        "--pairs-json",
        type=str,
        default=None,
        help="Pre-computed pair_scores.json (skip re-computation)",
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument(
        "--paired-methods",
        nargs=2,
        action="append",
        metavar=("METHOD_A", "METHOD_B"),
        help=(
            "Method pair for a cluster-bootstrap AP difference. May be repeated. "
            "Defaults to all method pairs with aligned scores."
        ),
    )
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
            sid,
            labels,
            gps_ts,
            gps_lats,
            gps_lngs,
            info["video_path"],
            info["video_start_unix"],
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
    cluster_scores_by_method: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for method in methods:
        scores = np.array(pair_data[method]["scores"])
        labels_arr = np.array(pair_data[method]["labels"])
        method_pair_clusters = np.asarray(pair_data[method].get("cluster_ids", pair_to_cluster))

        if len(scores) != len(method_pair_clusters):
            print(
                f"  {method}: score/cluster count mismatch "
                f"({len(scores)} vs {len(method_pair_clusters)}), skipping"
            )
            continue

        # Build per-cluster score/label arrays
        cluster_scores = {
            int(cid): (scores[method_pair_clusters == cid], labels_arr[method_pair_clusters == cid])
            for cid in np.unique(method_pair_clusters)
        }
        cluster_scores_by_method[method] = cluster_scores

        # Pair-level bootstrap (standard)
        rng = np.random.RandomState(42)
        n = len(scores)
        point_ap = average_precision_score(labels_arr, scores)
        pair_boot = []
        for _ in range(args.n_resamples):
            idx = rng.randint(0, n, size=n)
            sampled_scores, sampled_labels = scores[idx], labels_arr[idx]
            if sampled_labels.sum() == 0 or sampled_labels.sum() == n:
                pair_boot.append(point_ap)
            else:
                pair_boot.append(average_precision_score(sampled_labels, sampled_scores))
        pair_boot = np.array(pair_boot)
        pair_ci_lo = float(np.percentile(pair_boot, 2.5))
        pair_ci_hi = float(np.percentile(pair_boot, 97.5))
        pair_width = pair_ci_hi - pair_ci_lo

        # Cluster-level bootstrap
        clust_ap, clust_ci_lo, clust_ci_hi = cluster_bootstrap_ap(cluster_scores, args.n_resamples)
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

    requested_pairs = args.paired_methods
    if requested_pairs is None:
        aligned_methods = sorted(cluster_scores_by_method)
        requested_pairs = [
            [aligned_methods[i], aligned_methods[j]]
            for i in range(len(aligned_methods))
            for j in range(i + 1, len(aligned_methods))
        ]

    paired_results = {}
    print("\nStep 3: Paired cluster-bootstrap AP differences...")
    for method_a, method_b in requested_pairs:
        if method_a not in cluster_scores_by_method or method_b not in cluster_scores_by_method:
            print(f"  {method_a} vs {method_b}: method not found, skipping")
            continue
        key = f"{method_a}_minus_{method_b}"
        try:
            comparison = paired_cluster_bootstrap_ap_difference(
                cluster_scores_by_method[method_a],
                cluster_scores_by_method[method_b],
                args.n_resamples,
            )
        except ValueError as error:
            print(f"  {method_a} vs {method_b}: {error}, skipping")
            continue
        paired_results[key] = comparison
        ci_low, ci_high = comparison["ci"]
        print(
            f"  {method_a} - {method_b}: "
            f"{comparison['difference_a_minus_b']:+.4f} "
            f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        )

    # Save
    out_path = hdd_dir / "cluster_bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"marginal_intervals": results, "paired_differences": paired_results},
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

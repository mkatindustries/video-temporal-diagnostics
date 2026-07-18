#!/usr/bin/env python3
"""Grouped bootstrap intervals and paired AP contrasts from saved pair scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from video_retrieval.diagnostics.statistics import (
    cluster_bootstrap_ap,
    paired_cluster_bootstrap_ap_difference,
)


def load_grouped_scores(
    path: Path,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    with open(path) as input_file:
        pair_data = json.load(input_file)

    grouped = {}
    for method, data in pair_data.items():
        scores = np.asarray(data["scores"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        group_ids = np.asarray(data.get("cluster_ids", []), dtype=np.int64)
        if not (len(scores) == len(labels) == len(group_ids)):
            raise ValueError(f"{method} requires equally sized scores, labels, and cluster_ids")
        grouped[method] = {
            int(group_id): (scores[group_ids == group_id], labels[group_ids == group_id])
            for group_id in np.unique(group_ids)
        }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paired-methods",
        nargs=2,
        action="append",
        metavar=("METHOD_A", "METHOD_B"),
        help="Method pair for AP(A)-AP(B); may be repeated.",
    )
    args = parser.parse_args()

    grouped = load_grouped_scores(args.pairs_json)
    marginal = {}
    for method, cluster_scores in grouped.items():
        point, low, high = cluster_bootstrap_ap(cluster_scores, args.n_resamples, args.seed)
        marginal[method] = {
            "ap": point,
            "cluster_ci": [low, high],
            "n_clusters": len(cluster_scores),
        }

    method_pairs = args.paired_methods
    if method_pairs is None:
        methods = sorted(grouped)
        method_pairs = [
            [methods[i], methods[j]]
            for i in range(len(methods))
            for j in range(i + 1, len(methods))
        ]

    paired = {}
    for method_a, method_b in method_pairs:
        if method_a not in grouped or method_b not in grouped:
            raise ValueError(f"Unknown method pair: {method_a}, {method_b}")
        paired[f"{method_a}_minus_{method_b}"] = paired_cluster_bootstrap_ap_difference(
            grouped[method_a],
            grouped[method_b],
            args.n_resamples,
            args.seed,
        )

    result = {"marginal_intervals": marginal, "paired_differences": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

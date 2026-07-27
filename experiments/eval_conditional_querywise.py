#!/usr/bin/env python3
"""Evaluate matched, directed within-intersection retrieval from a fusion cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from video_retrieval.diagnostics.conditional_retrieval import (
    evaluate_conditional_querywise,
)

SCORE_CACHE_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-cache",
        required=True,
        help="Validated fusion score cache from eval_hdd_fusion.py or eval_nuscenes_fusion.py",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["hdd", "nuscenes"],
        help="Dataset name recorded in the output",
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-atol", type=float, default=1e-12)
    args = parser.parse_args()

    cache_path = Path(args.score_cache).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_bytes = cache_path.read_bytes()
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    raw_cache = torch.load(io.BytesIO(cache_bytes), map_location="cpu", weights_only=True)
    if not isinstance(raw_cache, dict):
        raise ValueError("score cache must contain a dictionary")
    if raw_cache.get("version") != SCORE_CACHE_VERSION:
        raise ValueError(
            f"unsupported score cache version {raw_cache.get('version')!r}; "
            f"expected {SCORE_CACHE_VERSION}"
        )
    for metadata_key in ("feature_cache_size", "feature_cache_mtime_ns"):
        if not isinstance(raw_cache.get(metadata_key), int):
            raise ValueError(f"score cache is missing integer metadata {metadata_key!r}")

    evaluation = evaluate_conditional_querywise(
        raw_cache,
        n_resamples=args.n_resamples,
        seed=args.seed,
        tie_atol=args.tie_atol,
    )
    results: dict[str, Any] = {
        "dataset": args.dataset,
        "score_cache": {
            "filename": cache_path.name,
            "size_bytes": len(cache_bytes),
            "sha256": cache_sha256,
            "version": raw_cache["version"],
            "feature_cache_size": raw_cache["feature_cache_size"],
            "feature_cache_mtime_ns": raw_cache["feature_cache_mtime_ns"],
        },
        **evaluation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)

    print(f"{args.dataset}: {results['n_queries']} eligible queries")
    for method, summary in results["methods"].items():
        metric = summary["query_macro_ap"]
        low, high = metric["cluster_ci"]
        print(f"  {method:<24s} mAP={metric['mean']:.6f} [{low:.6f}, {high:.6f}]")
    for contrast, summary in results["paired_ap_differences"].items():
        low, high = summary["ci"]
        print(
            f"  {contrast:<50s} {summary['difference_a_minus_b']:+.6f} "
            f"[{low:+.6f}, {high:+.6f}]"
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

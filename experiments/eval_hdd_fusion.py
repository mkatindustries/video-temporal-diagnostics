#!/usr/bin/env python3
"""Held-out score-level fusion for directed Honda HDD retrieval.

BoT cosine similarity retains global appearance/location information, while
encoder-sequence DTW supplies temporal discrimination. This script combines
their per-query z-scores with one weight selected by leave-one-intersection-out
cross-validation. The held-out cluster is excluded from both tuning queries and
tuning galleries, then its queries are evaluated against the full corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from eval_hdd_bof_dtw_rerank import load_evaluation_segments, summarize
from tqdm import tqdm

from video_retrieval.diagnostics.fusion import (
    evaluate_queries,
    leave_one_cluster_out_alpha,
    paired_cluster_bootstrap_mean_difference,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch

CACHE_VERSION = 1


class ScoreCache(TypedDict):
    version: int
    feature_cache_size: int
    feature_cache_mtime_ns: int
    dense_to_segment: torch.Tensor
    query_indices: torch.Tensor
    query_clusters: torch.Tensor
    label_ids: torch.Tensor
    bot_similarity: torch.Tensor
    dtw_distance: torch.Tensor


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def label_ids(labels: list[int]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def cache_metadata_matches(
    cache: ScoreCache,
    dense_to_segment: list[int],
    query_clusters: np.ndarray,
    query_labels: np.ndarray,
    feature_cache_path: Path,
) -> bool:
    stat = feature_cache_path.stat()
    if cache.get("version") != CACHE_VERSION:
        return False
    if cache.get("feature_cache_size") != stat.st_size:
        return False
    if cache.get("feature_cache_mtime_ns") != stat.st_mtime_ns:
        return False
    return (
        torch.equal(
            cache["dense_to_segment"],
            torch.as_tensor(dense_to_segment, dtype=torch.int64),
        )
        and torch.equal(
            cache["query_clusters"],
            torch.as_tensor(query_clusters, dtype=torch.int64),
        )
        and torch.equal(
            cache["label_ids"],
            torch.as_tensor(query_labels, dtype=torch.int64),
        )
    )


def build_score_cache(
    features: dict[int, dict],
    dense_to_segment: list[int],
    query_indices: np.ndarray,
    query_clusters: np.ndarray,
    query_labels: np.ndarray,
    device: torch.device,
    dtw_batch_size: int,
    feature_cache_path: Path,
) -> ScoreCache:
    n_segments = len(dense_to_segment)
    mean_embeddings = F.normalize(
        torch.stack([features[idx]["mean_emb"] for idx in dense_to_segment]).to(device),
        dim=-1,
    )
    bot_similarity = (mean_embeddings @ mean_embeddings.T).cpu()
    dtw_distance = torch.full((n_segments, n_segments), float("inf"), dtype=torch.float32)
    all_dense = np.arange(n_segments, dtype=np.int64)

    for query_dense in tqdm(query_indices, desc="Full-gallery DTW"):
        candidate_dense = all_dense[all_dense != query_dense]
        query_segment = dense_to_segment[int(query_dense)]
        query_sequence = features[query_segment]["encoder_seq"].to(device)
        for start in range(0, len(candidate_dense), dtw_batch_size):
            batch_dense = candidate_dense[start : start + dtw_batch_size]
            batch_sequences = [
                features[dense_to_segment[int(idx)]]["encoder_seq"].to(device)
                for idx in batch_dense
            ]
            distances = dtw_distance_batch(
                [query_sequence] * len(batch_sequences),
                batch_sequences,
                normalize=True,
            ).cpu()
            dtw_distance[int(query_dense), torch.as_tensor(batch_dense)] = distances

    stat = feature_cache_path.stat()
    return {
        "version": CACHE_VERSION,
        "feature_cache_size": stat.st_size,
        "feature_cache_mtime_ns": stat.st_mtime_ns,
        "dense_to_segment": torch.as_tensor(dense_to_segment, dtype=torch.int64),
        "query_indices": torch.as_tensor(query_indices, dtype=torch.int64),
        "query_clusters": torch.as_tensor(query_clusters, dtype=torch.int64),
        "label_ids": torch.as_tensor(query_labels, dtype=torch.int64),
        "bot_similarity": bot_similarity,
        "dtw_distance": dtw_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdd-dir", default="datasets/hdd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument(
        "--feature-cache",
        default="datasets/hdd/vjepa2_encoder_features.pt",
        help="Cache containing per-segment mean_emb and encoder_seq tensors.",
    )
    parser.add_argument(
        "--dist-cache",
        default="datasets/hdd/fusion_score_cache.pt",
        help="Reusable BoT/DTW score-matrix cache.",
    )
    parser.add_argument("--rebuild-dist-cache", action="store_true")
    parser.add_argument("--dtw-batch-size", type=int, default=256)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dtw_batch_size <= 0:
        parser.error("--dtw-batch-size must be positive")

    project_root = Path(__file__).parent.parent
    hdd_dir = resolve_path(project_root, args.hdd_dir)
    feature_cache_path = resolve_path(project_root, args.feature_cache)
    dist_cache_path = resolve_path(project_root, args.dist_cache)
    device = torch.device(args.device)

    eval_segments, _, segment_to_cluster = load_evaluation_segments(
        hdd_dir, args.max_clusters
    )
    if not feature_cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {feature_cache_path}; run eval_hdd_encoder_seq.py first"
        )
    feature_cache = torch.load(feature_cache_path, map_location="cpu", weights_only=True)
    features: dict[int, dict] = feature_cache["features"]
    dense_to_segment = [idx for idx in range(len(eval_segments)) if idx in features]
    if len(dense_to_segment) < 2:
        raise ValueError("feature cache contains fewer than two evaluation segments")

    clusters = np.asarray(
        [segment_to_cluster[segment] for segment in dense_to_segment], dtype=np.int64
    )
    labels = label_ids([eval_segments[segment].label for segment in dense_to_segment])
    relevance = (
        (clusters[:, None] == clusters[None, :])
        & (labels[:, None] == labels[None, :])
        & ~np.eye(len(dense_to_segment), dtype=bool)
    )
    query_indices = np.flatnonzero(relevance.any(axis=1)).astype(np.int64)
    if len(np.unique(clusters[query_indices])) < 2:
        raise ValueError("fusion evaluation requires at least two eligible intersection clusters")

    score_cache: ScoreCache | None = None
    if dist_cache_path.exists() and not args.rebuild_dist_cache:
        raw_cache = torch.load(
            dist_cache_path, map_location="cpu", weights_only=True
        )
        required_keys = set(ScoreCache.__required_keys__)
        if not isinstance(raw_cache, dict) or not required_keys.issubset(raw_cache):
            print(f"Ignoring stale or incompatible score cache: {dist_cache_path}")
        else:
            candidate_cache = cast(ScoreCache, raw_cache)
            if cache_metadata_matches(
                candidate_cache,
                dense_to_segment,
                clusters,
                labels,
                feature_cache_path,
            ):
                score_cache = candidate_cache
                print(f"Loaded validated score cache: {dist_cache_path}")
            else:
                print(f"Ignoring stale or incompatible score cache: {dist_cache_path}")

    if score_cache is None:
        score_cache = build_score_cache(
            features,
            dense_to_segment,
            query_indices,
            clusters,
            labels,
            device,
            args.dtw_batch_size,
            feature_cache_path,
        )
        dist_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(score_cache, dist_cache_path)
        print(f"Saved score cache: {dist_cache_path}")

    cached_query_indices = score_cache["query_indices"].numpy()
    if not np.array_equal(cached_query_indices, query_indices):
        raise ValueError("score cache has an incompatible eligible-query set")
    bot_similarity = score_cache["bot_similarity"].numpy().astype(np.float64)
    dtw_distance = score_cache["dtw_distance"].numpy().astype(np.float64)
    for query_idx in query_indices:
        valid = np.arange(len(dense_to_segment)) != query_idx
        if not np.all(np.isfinite(dtw_distance[query_idx, valid])):
            raise ValueError(f"score cache has incomplete DTW row for query {query_idx}")

    alpha_grid = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    loco = leave_one_cluster_out_alpha(
        bot_similarity,
        dtw_distance,
        relevance,
        clusters,
        query_indices,
        alpha_grid,
    )
    fused_ap = np.asarray(loco["ap"])
    fused_mrr = np.asarray(loco["mrr"])
    query_clusters = np.asarray(loco["query_clusters"])

    bot_ap, bot_mrr = evaluate_queries(
        bot_similarity,
        dtw_distance,
        relevance,
        clusters,
        query_indices,
        alpha=1.0,
    )
    dtw_ap, dtw_mrr = evaluate_queries(
        bot_similarity,
        dtw_distance,
        relevance,
        clusters,
        query_indices,
        alpha=0.0,
    )

    folds = loco["folds"]
    selected_alphas = np.asarray([fold["alpha"] for fold in folds], dtype=np.float64)
    results: dict[str, object] = {
        "protocol": {
            "directional": True,
            "gallery": "all cached HDD evaluation segments except the query",
            "relevance": "same GPS cluster and same maneuver label",
            "fusion": "alpha*z(BoT cosine) + (1-alpha)*z(-DTW distance), per query",
            "selection": (
                "leave-one-intersection-out; held-out cluster excluded from tuning "
                "queries and tuning galleries"
            ),
            "selection_tie_break": (
                "retain a sole maximizing endpoint; otherwise choose the middle "
                "maximizing grid value"
            ),
            "evaluation": "held-out-cluster queries against the full gallery",
            "uncertainty": (
                "intersection-cluster bootstrap over fixed out-of-fold query metrics; "
                "does not refit alpha inside bootstrap resamples"
            ),
        },
        "n_segments": len(dense_to_segment),
        "n_queries": len(query_indices),
        "n_clusters": len(np.unique(query_clusters)),
        "alpha_grid": alpha_grid.tolist(),
        "fused_loco": {
            "map": summarize(
                fused_ap.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
            "mrr": summarize(
                fused_mrr.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
        },
        "bot_full_gallery": {
            "map": summarize(
                bot_ap.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
            "mrr": summarize(
                bot_mrr.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
        },
        "encoder_seq_dtw_full_gallery": {
            "map": summarize(
                dtw_ap.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
            "mrr": summarize(
                dtw_mrr.tolist(), query_clusters.tolist(), args.n_bootstrap, args.seed
            ),
        },
        "paired_map_differences": {
            "fused_minus_bot": paired_cluster_bootstrap_mean_difference(
                fused_ap,
                bot_ap,
                query_clusters,
                args.n_bootstrap,
                args.seed,
            ),
            "fused_minus_encoder_seq_dtw": paired_cluster_bootstrap_mean_difference(
                fused_ap,
                dtw_ap,
                query_clusters,
                args.n_bootstrap,
                args.seed,
            ),
        },
        "alpha_selected": {
            "folds": folds,
            "median": float(np.median(selected_alphas)),
            "min": float(np.min(selected_alphas)),
            "max": float(np.max(selected_alphas)),
            "counts": {
                f"{alpha:.2f}": int(np.sum(np.isclose(selected_alphas, alpha)))
                for alpha in alpha_grid
                if np.any(np.isclose(selected_alphas, alpha))
            },
        },
    }

    output_path = hdd_dir / "fusion_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

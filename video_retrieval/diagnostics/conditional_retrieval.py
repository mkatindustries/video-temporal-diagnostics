"""Matched query-wise evaluation for conditional maneuver retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import torch

from video_retrieval.diagnostics.fusion import (
    paired_cluster_bootstrap_mean_difference,
    rank_metrics,
)

METHOD_SPECS = {
    "bot_cosine": ("bot_similarity", 1.0),
    "encoder_seq_dtw": ("dtw_distance", -1.0),
    "temporal_residual_dtw": ("temporal_residual_dtw_distance", -1.0),
}


def _as_numpy(cache: Mapping[str, object], key: str) -> np.ndarray:
    if key not in cache:
        raise ValueError(f"score cache is missing {key!r}")
    value = cache[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    query_clusters: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Summarize a query-macro mean with intersection-cluster resampling."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if values.ndim != 1 or query_clusters.shape != values.shape:
        raise ValueError("values and query_clusters must be equally sized 1D arrays")

    unique_clusters = np.unique(query_clusters)
    by_cluster = {
        cluster: values[query_clusters == cluster] for cluster in unique_clusters
    }
    rng = np.random.RandomState(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        selected = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        samples[sample_index] = np.mean(
            np.concatenate([by_cluster[cluster] for cluster in selected])
        )

    return {
        "mean": float(np.mean(values)),
        "cluster_ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "n_queries": int(len(values)),
        "n_clusters": int(len(unique_clusters)),
    }


def _heterogeneity(
    values_a: np.ndarray,
    values_b: np.ndarray,
    tie_atol: float,
) -> dict[str, float | int]:
    differences = values_a - values_b
    ties = np.abs(differences) <= tie_atol
    wins = differences > tie_atol
    losses = differences < -tie_atol
    n_queries = len(differences)
    return {
        "n_queries": int(n_queries),
        "win_count": int(np.sum(wins)),
        "tie_count": int(np.sum(ties)),
        "loss_count": int(np.sum(losses)),
        "win_rate": float(np.mean(wins)),
        "tie_rate": float(np.mean(ties)),
        "loss_rate": float(np.mean(losses)),
        "mean_ap_difference": float(np.mean(differences)),
        "median_ap_difference": float(np.median(differences)),
        "tie_atol": float(tie_atol),
    }


def evaluate_conditional_querywise(
    cache: Mapping[str, object],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    tie_atol: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate directed retrieval within each query's intersection cluster.

    The input is the validated square score cache produced by the HDD or nuScenes
    fusion evaluator. A candidate is relevant when it shares both the query's
    cluster and maneuver label. The evaluation fails closed if a query eligible for
    global retrieval lacks both a positive and a negative in its conditional gallery.
    """
    if tie_atol < 0.0:
        raise ValueError("tie_atol must be non-negative")

    clusters = _as_numpy(cache, "query_clusters").astype(np.int64, copy=False)
    labels = _as_numpy(cache, "label_ids").astype(np.int64, copy=False)
    dense_to_segment = _as_numpy(cache, "dense_to_segment").astype(
        np.int64, copy=False
    )
    if clusters.ndim != 1 or labels.shape != clusters.shape:
        raise ValueError("query_clusters and label_ids must be equally sized 1D arrays")
    if dense_to_segment.shape != clusters.shape:
        raise ValueError("dense_to_segment must align with query_clusters")
    if len(np.unique(dense_to_segment)) != len(dense_to_segment):
        raise ValueError("dense_to_segment must contain unique identifiers")

    n_segments = len(clusters)
    identity = np.eye(n_segments, dtype=bool)
    same_cluster = clusters[:, None] == clusters[None, :]
    relevance = same_cluster & (labels[:, None] == labels[None, :]) & ~identity
    conditional_gallery = same_cluster & ~identity
    positive_counts = relevance.sum(axis=1)
    gallery_counts = conditional_gallery.sum(axis=1)
    global_query_indices = np.flatnonzero(positive_counts > 0).astype(np.int64)
    conditional_query_indices = np.flatnonzero(
        (positive_counts > 0) & (positive_counts < gallery_counts)
    ).astype(np.int64)
    if len(conditional_query_indices) == 0:
        raise ValueError("score cache has no query with both a positive and a negative")
    if not np.array_equal(global_query_indices, conditional_query_indices):
        raise ValueError(
            "global and conditional eligible-query sets differ; matched evaluation "
            "requires every global query to have a conditional negative"
        )
    query_indices = global_query_indices

    scored_query_indices = _as_numpy(cache, "query_indices").astype(
        np.int64, copy=False
    )
    if scored_query_indices.ndim != 1:
        raise ValueError("query_indices must be a 1D array")
    if np.any((scored_query_indices < 0) | (scored_query_indices >= n_segments)):
        raise ValueError("query_indices contains an out-of-range row")
    if len(np.unique(scored_query_indices)) != len(scored_query_indices):
        raise ValueError("query_indices contains duplicate rows")
    missing_rows = np.setdiff1d(query_indices, scored_query_indices)
    if len(missing_rows):
        raise ValueError(
            f"score cache is missing {len(missing_rows)} matched eligible query row(s)"
        )

    method_scores: dict[str, np.ndarray] = {}
    for method, (cache_key, direction) in METHOD_SPECS.items():
        if cache_key not in cache:
            if method == "temporal_residual_dtw":
                continue
            raise ValueError(f"score cache is missing required method {cache_key!r}")
        matrix = _as_numpy(cache, cache_key).astype(np.float64, copy=False)
        if matrix.shape != (n_segments, n_segments):
            raise ValueError(
                f"{cache_key} must have shape {(n_segments, n_segments)}, "
                f"got {matrix.shape}"
            )
        global_gallery = np.ones((len(query_indices), n_segments), dtype=bool)
        global_gallery[np.arange(len(query_indices)), query_indices] = False
        if not np.all(np.isfinite(matrix[query_indices][global_gallery])):
            raise ValueError(
                f"{method} has a non-finite score in a matched full-gallery row"
            )
        method_scores[method] = direction * matrix

    per_method_ap = {
        method: np.empty(len(query_indices), dtype=np.float64)
        for method in method_scores
    }
    for output_index, query_index in enumerate(query_indices):
        valid_gallery = conditional_gallery[query_index]
        for method, scores in method_scores.items():
            if not np.all(np.isfinite(scores[query_index, valid_gallery])):
                raise ValueError(
                    f"{method} has a non-finite score in query {int(query_index)}'s gallery"
                )
            ap, _ = rank_metrics(
                scores[query_index], relevance[query_index], valid_gallery
            )
            per_method_ap[method][output_index] = ap

    query_clusters = clusters[query_indices]
    methods = {
        method: {
            "query_macro_ap": _cluster_bootstrap_mean(
                values, query_clusters, n_resamples, seed
            )
        }
        for method, values in per_method_ap.items()
    }

    paired_differences: dict[str, object] = {}
    heterogeneity: dict[str, object] = {}
    method_order = list(per_method_ap)
    for method_b, method_a in combinations(method_order, 2):
        key = f"{method_a}_minus_{method_b}"
        paired_differences[key] = paired_cluster_bootstrap_mean_difference(
            per_method_ap[method_a],
            per_method_ap[method_b],
            query_clusters,
            n_resamples,
            seed,
        )
        heterogeneity[key] = _heterogeneity(
            per_method_ap[method_a], per_method_ap[method_b], tie_atol
        )

    per_query = []
    for output_index, query_index in enumerate(query_indices):
        per_query.append(
            {
                "query_index": int(query_index),
                "segment_index": int(dense_to_segment[query_index]),
                "cluster_id": int(clusters[query_index]),
                "label_id": int(labels[query_index]),
                "n_gallery": int(gallery_counts[query_index]),
                "n_relevant": int(positive_counts[query_index]),
                "average_precision": {
                    method: float(values[output_index])
                    for method, values in per_method_ap.items()
                },
            }
        )

    return {
        "protocol": {
            "directional": True,
            "gallery": "all other cached segments in the query's intersection cluster",
            "relevance": "same intersection cluster and same maneuver label",
            "metric": "full conditional-gallery AP, macro-averaged over eligible queries",
            "eligibility": "at least one relevant and one non-relevant gallery candidate",
            "query_set": "exactly the global retrieval eligible-query set (fail-closed)",
            "query_identity": (
                "query_index is the score-cache row; segment_index is that row's "
                "dense_to_segment value"
            ),
            "uncertainty": "intersection-cluster bootstrap over fixed per-query AP",
            "pooled_pair_protocol": False,
        },
        "n_segments": int(n_segments),
        "n_queries": int(len(query_indices)),
        "n_global_eligible_queries": int(len(global_query_indices)),
        "n_conditional_eligible_queries": int(len(conditional_query_indices)),
        "n_clusters": int(len(np.unique(query_clusters))),
        "n_bootstrap": int(n_resamples),
        "seed": int(seed),
        "methods": methods,
        "paired_ap_differences": paired_differences,
        "heterogeneity": heterogeneity,
        "per_query": per_query,
    }

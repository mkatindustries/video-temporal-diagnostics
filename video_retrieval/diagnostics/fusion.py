"""Leakage-safe score fusion and grouped evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from video_retrieval.diagnostics.statistics import truncated_average_precision


class LocoFold(TypedDict):
    cluster_id: int
    alpha: float
    training_map: float
    training_map_by_alpha: list[float]
    n_train_queries: int
    n_test_queries: int


class LocoResult(TypedDict):
    ap: np.ndarray
    mrr: np.ndarray
    folds: list[LocoFold]
    query_clusters: np.ndarray


def zscore_over_gallery(scores: np.ndarray, valid_gallery: np.ndarray) -> np.ndarray:
    """Z-score one query's scores over its valid gallery candidates."""
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_gallery, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("scores and valid_gallery must be equally sized 1D arrays")
    if not np.any(valid):
        raise ValueError("valid_gallery must contain at least one candidate")
    if not np.all(np.isfinite(values[valid])):
        raise ValueError("valid gallery scores must be finite")

    output = np.zeros_like(values)
    mean = float(np.mean(values[valid]))
    std = float(np.std(values[valid]))
    if std > 0.0:
        output[valid] = (values[valid] - mean) / std
    return output


def fuse_score_row(
    bot_similarity: np.ndarray,
    dtw_distance: np.ndarray,
    valid_gallery: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse BoT similarity and negative DTW distance for one query."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    valid = np.asarray(valid_gallery, dtype=bool)
    bot_z = zscore_over_gallery(bot_similarity, valid)
    dtw_z = zscore_over_gallery(-np.asarray(dtw_distance, dtype=np.float64), valid)
    fused = alpha * bot_z + (1.0 - alpha) * dtw_z
    return np.where(valid, fused, -np.inf)


def reciprocal_rank(ranked_relevance: np.ndarray) -> float:
    """Return reciprocal rank for a relevance vector already in rank order."""
    hits = np.flatnonzero(ranked_relevance)
    return 0.0 if len(hits) == 0 else float(1.0 / (hits[0] + 1))


def rank_metrics(
    scores: np.ndarray,
    relevance: np.ndarray,
    valid_gallery: np.ndarray,
) -> tuple[float, float]:
    """Return AP and reciprocal rank after ranking the valid gallery."""
    values = np.asarray(scores, dtype=np.float64)
    relevant = np.asarray(relevance, dtype=np.int64)
    valid = np.asarray(valid_gallery, dtype=bool)
    if not (values.shape == relevant.shape == valid.shape) or values.ndim != 1:
        raise ValueError("scores, relevance, and valid_gallery must be equally sized rows")

    total_relevant = int(relevant[valid].sum())
    if total_relevant == 0:
        raise ValueError("query has no relevant item in the valid gallery")
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(-values[valid], kind="stable")]
    ranked_relevance = relevant[order]
    return (
        truncated_average_precision(ranked_relevance, total_relevant),
        reciprocal_rank(ranked_relevance),
    )


def evaluate_queries(
    bot_similarity: np.ndarray,
    dtw_distance: np.ndarray,
    relevance: np.ndarray,
    candidate_clusters: np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
    alpha: float,
    excluded_gallery_cluster: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fused AP/MRR for the requested query rows."""
    query_ids = np.asarray(query_indices, dtype=np.int64)
    clusters = np.asarray(candidate_clusters, dtype=np.int64)
    n_segments = len(clusters)
    expected_shape = (n_segments, n_segments)
    if not (
        np.asarray(bot_similarity).shape
        == np.asarray(dtw_distance).shape
        == np.asarray(relevance).shape
        == expected_shape
    ):
        raise ValueError("score and relevance matrices must be square over candidate_clusters")

    ap_values = np.empty(len(query_ids), dtype=np.float64)
    mrr_values = np.empty(len(query_ids), dtype=np.float64)
    base_valid = np.ones(n_segments, dtype=bool)
    if excluded_gallery_cluster is not None:
        base_valid &= clusters != excluded_gallery_cluster

    for output_idx, query_idx in enumerate(query_ids):
        valid = base_valid.copy()
        valid[query_idx] = False
        fused = fuse_score_row(
            bot_similarity[query_idx], dtw_distance[query_idx], valid, alpha
        )
        ap_values[output_idx], mrr_values[output_idx] = rank_metrics(
            fused, relevance[query_idx], valid
        )
    return ap_values, mrr_values


def select_alpha_by_map(alphas: np.ndarray, map_values: np.ndarray) -> int:
    """Select a maximizing alpha with a conservative, deterministic tie rule.

    If exactly one single-method endpoint is tied for the maximum, retain that
    endpoint instead of introducing unnecessary fusion. Otherwise choose the
    middle maximizing grid point, with the lower point winning an even tie.
    """
    alpha_values = np.asarray(alphas, dtype=np.float64)
    metrics = np.asarray(map_values, dtype=np.float64)
    if alpha_values.ndim != 1 or metrics.shape != alpha_values.shape:
        raise ValueError("alphas and map_values must be equally sized 1D arrays")
    best = np.flatnonzero(np.isclose(metrics, np.max(metrics), rtol=0.0, atol=1e-12))
    zero_best = best[np.isclose(alpha_values[best], 0.0, rtol=0.0, atol=1e-12)]
    one_best = best[np.isclose(alpha_values[best], 1.0, rtol=0.0, atol=1e-12)]
    if len(zero_best) == 1 and len(one_best) == 0:
        return int(zero_best[0])
    if len(one_best) == 1 and len(zero_best) == 0:
        return int(one_best[0])
    return int(best[(len(best) - 1) // 2])


def _zscore_query_matrix(
    scores: np.ndarray,
    query_indices: np.ndarray,
    valid_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-query normalization for one LOCO training fold."""
    rows = np.asarray(scores, dtype=np.float64)[query_indices]
    valid = np.broadcast_to(valid_candidates, rows.shape).copy()
    valid[np.arange(len(query_indices)), query_indices] = False
    if not np.all(np.isfinite(rows[valid])):
        raise ValueError("valid gallery scores must be finite")

    counts = valid.sum(axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("every query must have a valid gallery candidate")
    means = np.where(valid, rows, 0.0).sum(axis=1, keepdims=True) / counts
    centered = rows - means
    variances = np.where(valid, centered * centered, 0.0).sum(
        axis=1, keepdims=True
    ) / counts
    stds = np.sqrt(variances)
    normalized = np.zeros_like(rows)
    np.divide(centered, stds, out=normalized, where=valid & (stds > 0.0))
    return normalized, valid


def _row_average_precision(
    scores: np.ndarray,
    relevance: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Vectorized full-gallery AP for rows sharing one validity matrix."""
    ranked_scores = np.where(valid, scores, -np.inf)
    order = np.argsort(-ranked_scores, axis=1, kind="stable")
    valid_relevance = relevance & valid
    ranked_relevance = np.take_along_axis(valid_relevance, order, axis=1).astype(
        np.float64
    )
    total_relevant = np.sum(valid_relevance, axis=1)
    if np.any(total_relevant == 0):
        raise ValueError("every tuning query must retain a relevant gallery item")
    precision = np.cumsum(ranked_relevance, axis=1) / np.arange(
        1, ranked_relevance.shape[1] + 1
    )
    return np.sum(precision * ranked_relevance, axis=1) / total_relevant


def leave_one_cluster_out_alpha(
    bot_similarity: np.ndarray,
    dtw_distance: np.ndarray,
    relevance: np.ndarray,
    candidate_clusters: np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
    alpha_grid: Sequence[float] | np.ndarray,
) -> LocoResult:
    """Tune alpha without using held-out clusters as queries or gallery items."""
    query_ids = np.asarray(query_indices, dtype=np.int64)
    clusters = np.asarray(candidate_clusters, dtype=np.int64)
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    if query_ids.ndim != 1 or len(query_ids) == 0:
        raise ValueError("query_indices must be a non-empty 1D sequence")
    if alphas.ndim != 1 or len(alphas) == 0 or np.any((alphas < 0) | (alphas > 1)):
        raise ValueError("alpha_grid must be a non-empty 1D sequence within [0, 1]")

    query_clusters = clusters[query_ids]
    held_out_clusters = np.unique(query_clusters)
    if len(held_out_clusters) < 2:
        raise ValueError("LOCO evaluation requires at least two query clusters")

    output_ap = np.empty(len(query_ids), dtype=np.float64)
    output_mrr = np.empty(len(query_ids), dtype=np.float64)
    fold_results: list[LocoFold] = []

    for held_out in held_out_clusters:
        train_queries = query_ids[query_clusters != held_out]
        test_mask = query_clusters == held_out
        test_queries = query_ids[test_mask]

        valid_candidates = clusters != held_out
        train_bot_z, train_valid = _zscore_query_matrix(
            bot_similarity, train_queries, valid_candidates
        )
        train_dtw_z, dtw_valid = _zscore_query_matrix(
            -np.asarray(dtw_distance), train_queries, valid_candidates
        )
        if not np.array_equal(train_valid, dtw_valid):
            raise ValueError("BoT and DTW validity masks differ")
        train_relevance = np.asarray(relevance, dtype=bool)[train_queries]
        train_map = np.empty(len(alphas), dtype=np.float64)
        for alpha_idx, alpha in enumerate(alphas):
            fused = float(alpha) * train_bot_z + (1.0 - float(alpha)) * train_dtw_z
            train_ap = _row_average_precision(
                fused,
                train_relevance,
                train_valid,
            )
            train_map[alpha_idx] = float(np.mean(train_ap))

        best_idx = select_alpha_by_map(alphas, train_map)
        selected_alpha = float(alphas[best_idx])
        test_ap, test_mrr = evaluate_queries(
            bot_similarity,
            dtw_distance,
            relevance,
            clusters,
            test_queries,
            selected_alpha,
        )
        output_ap[test_mask] = test_ap
        output_mrr[test_mask] = test_mrr
        fold_results.append(
            {
                "cluster_id": int(held_out),
                "alpha": selected_alpha,
                "training_map": float(train_map[best_idx]),
                "training_map_by_alpha": train_map.tolist(),
                "n_train_queries": int(len(train_queries)),
                "n_test_queries": int(len(test_queries)),
            }
        )

    return {
        "ap": output_ap,
        "mrr": output_mrr,
        "folds": fold_results,
        "query_clusters": query_clusters,
    }


def bot_dtw_cascade(
    bot_similarity: np.ndarray,
    dtw_distance: np.ndarray,
    relevance: np.ndarray,
    query_indices: Sequence[int] | np.ndarray,
    k_values: Sequence[int] | np.ndarray,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """BoT top-k retrieval then DTW rerank of that candidate set.

    For each requested ``k`` and query, the gallery is every candidate except the
    query. ``bot`` keeps the top-k BoT ranking; ``dtw_rerank`` reorders that same
    top-k by ascending DTW distance. AP@k is truncated by the query's total number
    of relevant items (so AP@k never exceeds recall@k), matching
    ``eval_hdd_bof_dtw_rerank``. Returns
    ``{k: {"bot"|"dtw_rerank": {"ap"|"recall"|"mrr": np.ndarray}}}`` with per-query
    arrays aligned to ``query_indices``.
    """
    bot = np.asarray(bot_similarity, dtype=np.float64)
    dtw = np.asarray(dtw_distance, dtype=np.float64)
    relevant = np.asarray(relevance, dtype=bool)
    query_ids = np.asarray(query_indices, dtype=np.int64)
    ks = sorted({int(k) for k in k_values})
    n_segments = bot.shape[0]
    if not (bot.shape == dtw.shape == relevant.shape == (n_segments, n_segments)):
        raise ValueError("score and relevance matrices must be square and identically shaped")
    if query_ids.ndim != 1 or len(query_ids) == 0:
        raise ValueError("query_indices must be a non-empty 1D sequence")
    if not ks or ks[0] < 1:
        raise ValueError("k_values must be positive")
    max_k = min(ks[-1], n_segments - 1)

    out: dict[int, dict[str, dict[str, list[float]]]] = {
        k: {"bot": {"ap": [], "recall": [], "mrr": []},
            "dtw_rerank": {"ap": [], "recall": [], "mrr": []}}
        for k in ks
    }
    for query_idx in query_ids:
        valid = np.ones(n_segments, dtype=bool)
        valid[query_idx] = False
        query_relevance = relevant[query_idx]
        total_relevant = int(query_relevance[valid].sum())
        if total_relevant == 0:
            raise ValueError(f"query {int(query_idx)} has no relevant gallery item")
        scores = np.where(valid, bot[query_idx], -np.inf)
        bot_order = np.argsort(-scores, kind="stable")  # query lands last via -inf
        for k in ks:
            k_eff = min(k, max_k)
            top = bot_order[:k_eff]
            bot_rel = query_relevance[top].astype(np.int64)
            rerank = top[np.argsort(dtw[query_idx, top], kind="stable")]
            dtw_rel = query_relevance[rerank].astype(np.int64)
            for name, ranked in (("bot", bot_rel), ("dtw_rerank", dtw_rel)):
                out[k][name]["ap"].append(
                    truncated_average_precision(ranked, total_relevant)
                )
                out[k][name]["recall"].append(float(ranked.sum() / total_relevant))
                out[k][name]["mrr"].append(reciprocal_rank(ranked))
    return {
        k: {name: {metric: np.asarray(values, dtype=np.float64)
                   for metric, values in method.items()}
            for name, method in methods.items()}
        for k, methods in out.items()
    }


def paired_cluster_bootstrap_mean_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    query_clusters: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Paired cluster bootstrap for a difference of query-macro means."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    clusters = np.asarray(query_clusters, dtype=np.int64)
    if not (a.shape == b.shape == clusters.shape) or a.ndim != 1:
        raise ValueError("values and query_clusters must be equally sized 1D arrays")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    unique_clusters = np.unique(clusters)
    differences = a - b
    by_cluster = {
        cluster: differences[clusters == cluster] for cluster in unique_clusters
    }
    rng = np.random.RandomState(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for sample_idx in range(n_resamples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        samples[sample_idx] = np.mean(
            np.concatenate([by_cluster[cluster] for cluster in selected])
        )

    point = float(np.mean(differences))
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "difference_a_minus_b": point,
        "ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "bootstrap_probability_a_gt_b": float(np.mean(samples > 0.0)),
        "n_clusters": int(len(unique_clusters)),
    }

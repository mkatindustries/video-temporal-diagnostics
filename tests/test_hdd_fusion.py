from __future__ import annotations

import numpy as np

from video_retrieval.diagnostics.fusion import (
    LocoResult,
    fuse_score_row,
    leave_one_cluster_out_alpha,
)


def synthetic_retrieval(
    bot_is_optimal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clusters = np.repeat(np.arange(3), 2)
    n_segments = len(clusters)
    relevance = (clusters[:, None] == clusters[None, :]) & ~np.eye(
        n_segments, dtype=bool
    )
    good = np.where(relevance, 10.0, 0.0)
    bad = np.where(relevance, 10.0, 0.0)
    np.fill_diagonal(good, 0.0)
    np.fill_diagonal(bad, 0.0)

    if bot_is_optimal:
        bot_similarity = good
        dtw_distance = bad
    else:
        bot_similarity = -good
        dtw_distance = -bad
    query_indices = np.arange(n_segments)
    return bot_similarity, dtw_distance, relevance, clusters, query_indices


def fold_alphas(result: LocoResult) -> dict[int, float]:
    return {int(fold["cluster_id"]): float(fold["alpha"]) for fold in result["folds"]}


def test_zscore_fusion_respects_endpoint_rankings() -> None:
    bot_similarity = np.array([0.0, 0.9, 0.2, 0.5])
    dtw_distance = np.array([0.0, 0.8, 0.1, 0.4])
    valid = np.array([False, True, True, True])

    bot_only = fuse_score_row(bot_similarity, dtw_distance, valid, alpha=1.0)
    dtw_only = fuse_score_row(bot_similarity, dtw_distance, valid, alpha=0.0)

    assert np.argsort(-bot_only).tolist()[:3] == [1, 3, 2]
    assert np.argsort(-dtw_only).tolist()[:3] == [2, 3, 1]
    assert np.isneginf(bot_only[0])


def test_loco_selects_bot_endpoint_when_bot_is_optimal() -> None:
    inputs = synthetic_retrieval(bot_is_optimal=True)
    result = leave_one_cluster_out_alpha(*inputs, alpha_grid=np.linspace(0, 1, 21))

    assert set(fold_alphas(result).values()) == {1.0}


def test_loco_selects_dtw_endpoint_when_dtw_is_optimal() -> None:
    inputs = synthetic_retrieval(bot_is_optimal=False)
    result = leave_one_cluster_out_alpha(*inputs, alpha_grid=np.linspace(0, 1, 21))

    assert set(fold_alphas(result).values()) == {0.0}


def test_loco_alpha_does_not_use_held_out_cluster() -> None:
    rng = np.random.RandomState(7)
    clusters = np.repeat(np.arange(3), 3)
    n_segments = len(clusters)
    relevance = (clusters[:, None] == clusters[None, :]) & ~np.eye(
        n_segments, dtype=bool
    )
    bot_similarity = rng.normal(size=(n_segments, n_segments))
    dtw_distance = rng.uniform(size=(n_segments, n_segments))
    query_indices = np.arange(n_segments)
    alpha_grid = np.linspace(0, 1, 5)

    original = leave_one_cluster_out_alpha(
        bot_similarity,
        dtw_distance,
        relevance,
        clusters,
        query_indices,
        alpha_grid,
    )
    changed_bot = bot_similarity.copy()
    changed_dtw = dtw_distance.copy()
    held_out = clusters == 0
    changed_bot[held_out, :] = rng.normal(loc=100.0, size=(held_out.sum(), n_segments))
    changed_bot[:, held_out] = rng.normal(loc=-100.0, size=(n_segments, held_out.sum()))
    changed_dtw[held_out, :] = rng.uniform(100.0, 200.0, size=(held_out.sum(), n_segments))
    changed_dtw[:, held_out] = rng.uniform(200.0, 300.0, size=(n_segments, held_out.sum()))
    changed = leave_one_cluster_out_alpha(
        changed_bot,
        changed_dtw,
        relevance,
        clusters,
        query_indices,
        alpha_grid,
    )

    assert fold_alphas(original)[0] == fold_alphas(changed)[0]

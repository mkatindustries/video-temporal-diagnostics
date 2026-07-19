"""Tests for the pure BoT->DTW cascade helper reused by the nuScenes evaluator."""

from __future__ import annotations

import numpy as np

from video_retrieval.diagnostics.fusion import bot_dtw_cascade


def synthetic_cascade_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two intersection clusters of three segments; two relevant items per query."""
    clusters = np.repeat(np.arange(2), 3)
    n = len(clusters)
    relevance = (clusters[:, None] == clusters[None, :]) & ~np.eye(n, dtype=bool)
    # BoT ranks same-cluster items above others; DTW distance is small for them too.
    bot_similarity = np.where(relevance, 1.0, 0.0)
    dtw_distance = np.where(relevance, 0.0, 1.0)
    query_indices = np.arange(n)
    return bot_similarity, dtw_distance, relevance, query_indices


def test_cascade_output_structure() -> None:
    bot, dtw, rel, queries = synthetic_cascade_inputs()
    ks = [1, 2, 5]
    out = bot_dtw_cascade(bot, dtw, rel, queries, ks)
    assert set(out) == set(ks)
    for k in ks:
        assert set(out[k]) == {"bot", "dtw_rerank"}
        for method in ("bot", "dtw_rerank"):
            for metric in ("ap", "recall", "mrr"):
                arr = out[k][method][metric]
                assert arr.shape == (len(queries),)


def test_ap_at_k_never_exceeds_recall_at_k() -> None:
    rng = np.random.RandomState(3)
    clusters = np.repeat(np.arange(4), 3)
    n = len(clusters)
    relevance = (clusters[:, None] == clusters[None, :]) & ~np.eye(n, dtype=bool)
    bot = rng.normal(size=(n, n))
    dtw = rng.uniform(size=(n, n))
    queries = np.arange(n)
    out = bot_dtw_cascade(bot, dtw, relevance, queries, [1, 2, 5, 11])
    for k, methods in out.items():
        for method, metrics in methods.items():
            assert np.all(metrics["ap"] <= metrics["recall"] + 1e-9), (k, method)
            assert np.all((metrics["recall"] >= 0.0) & (metrics["recall"] <= 1.0))
            assert np.all((metrics["mrr"] >= 0.0) & (metrics["mrr"] <= 1.0))


def test_rerank_preserves_candidate_set_recall() -> None:
    # Reranking reorders the top-k but keeps the same candidates, so recall@k is
    # identical between the BoT ranking and its DTW rerank.
    bot, dtw, rel, queries = synthetic_cascade_inputs()
    out = bot_dtw_cascade(bot, dtw, rel, queries, [1, 2, 5])
    for k in (1, 2, 5):
        assert np.allclose(out[k]["bot"]["recall"], out[k]["dtw_rerank"]["recall"])


def test_recall_grows_and_truncates_below_total_relevant() -> None:
    # With perfect BoT ranking, recall@1 = 0.5 (1 of 2 relevant) and recall@2 = 1.0.
    bot, dtw, rel, queries = synthetic_cascade_inputs()
    out = bot_dtw_cascade(bot, dtw, rel, queries, [1, 2])
    assert np.allclose(out[1]["bot"]["recall"], 0.5)
    assert np.allclose(out[2]["bot"]["recall"], 1.0)
    assert np.all(out[1]["bot"]["ap"] <= out[2]["bot"]["ap"] + 1e-9)

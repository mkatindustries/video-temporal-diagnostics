"""Tests for matched conditional query-wise retrieval evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from video_retrieval.diagnostics.conditional_retrieval import (
    evaluate_conditional_querywise,
)


def synthetic_cache() -> dict[str, torch.Tensor]:
    clusters = np.repeat(np.arange(2), 4)
    labels = np.tile(np.array([0, 0, 1, 1]), 2)
    same_cluster = clusters[:, None] == clusters[None, :]
    relevant = same_cluster & (labels[:, None] == labels[None, :])

    bot_similarity = np.where(relevant, 0.0, 1.0)
    encoder_distance = np.where(relevant, 0.0, 1.0)
    np.fill_diagonal(bot_similarity, 1.0)
    np.fill_diagonal(encoder_distance, 0.0)
    return {
        "dense_to_segment": torch.arange(len(clusters)),
        "query_clusters": torch.from_numpy(clusters),
        "label_ids": torch.from_numpy(labels),
        "query_indices": torch.arange(len(clusters)),
        "bot_similarity": torch.from_numpy(bot_similarity),
        "dtw_distance": torch.from_numpy(encoder_distance),
        "temporal_residual_dtw_distance": torch.from_numpy(encoder_distance.copy()),
    }


def test_matched_querywise_ap_and_heterogeneity() -> None:
    result = evaluate_conditional_querywise(
        synthetic_cache(), n_resamples=100, seed=7
    )

    assert result["protocol"]["directional"] is True
    assert result["protocol"]["pooled_pair_protocol"] is False
    assert result["n_queries"] == 8
    assert result["n_clusters"] == 2
    assert result["methods"]["bot_cosine"]["query_macro_ap"]["mean"] == pytest.approx(
        1 / 3
    )
    assert result["methods"]["encoder_seq_dtw"]["query_macro_ap"]["mean"] == 1.0

    contrast = result["paired_ap_differences"]["encoder_seq_dtw_minus_bot_cosine"]
    assert contrast["difference_a_minus_b"] == pytest.approx(2 / 3)
    heterogeneity = result["heterogeneity"]["encoder_seq_dtw_minus_bot_cosine"]
    assert heterogeneity["win_count"] == 8
    assert heterogeneity["tie_count"] == 0
    assert heterogeneity["loss_count"] == 0

    residual_contrast = result["heterogeneity"][
        "temporal_residual_dtw_minus_encoder_seq_dtw"
    ]
    assert residual_contrast["tie_count"] == 8


def test_per_query_records_preserve_cache_identity() -> None:
    cache = synthetic_cache()
    cache["dense_to_segment"] = torch.arange(100, 108)
    result = evaluate_conditional_querywise(cache, n_resamples=10)

    first = result["per_query"][0]
    assert first == {
        "query_index": 0,
        "segment_index": 100,
        "cluster_id": 0,
        "label_id": 0,
        "n_gallery": 3,
        "n_relevant": 1,
        "average_precision": {
            "bot_cosine": pytest.approx(1 / 3),
            "encoder_seq_dtw": 1.0,
            "temporal_residual_dtw": 1.0,
        },
    }


def test_global_and_conditional_query_sets_must_match() -> None:
    cache = synthetic_cache()
    cache["label_ids"] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="eligible-query sets differ"):
        evaluate_conditional_querywise(cache, n_resamples=10)


def test_requires_scores_for_every_matched_query() -> None:
    cache = synthetic_cache()
    cache["query_indices"] = torch.arange(1, 8)
    with pytest.raises(ValueError, match="missing 1 matched eligible query row"):
        evaluate_conditional_querywise(cache, n_resamples=10)


@pytest.mark.parametrize(
    "query_indices, message",
    [
        (torch.tensor([0, 1, 8]), "out-of-range"),
        (torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7]), "duplicate"),
    ],
)
def test_rejects_invalid_scored_query_rows(
    query_indices: torch.Tensor, message: str
) -> None:
    cache = synthetic_cache()
    cache["query_indices"] = query_indices
    with pytest.raises(ValueError, match=message):
        evaluate_conditional_querywise(cache, n_resamples=10)


def test_rejects_nonfinite_conditional_score() -> None:
    cache = synthetic_cache()
    cache["dtw_distance"][0, 1] = float("inf")
    with pytest.raises(ValueError, match="non-finite score"):
        evaluate_conditional_querywise(cache, n_resamples=10)


def test_rejects_nonfinite_score_outside_conditional_gallery() -> None:
    cache = synthetic_cache()
    cache["dtw_distance"][0, 4] = float("inf")
    with pytest.raises(ValueError, match="matched full-gallery row"):
        evaluate_conditional_querywise(cache, n_resamples=10)


def test_residual_method_is_optional() -> None:
    cache = synthetic_cache()
    del cache["temporal_residual_dtw_distance"]
    result = evaluate_conditional_querywise(cache, n_resamples=10)

    assert set(result["methods"]) == {"bot_cosine", "encoder_seq_dtw"}
    assert set(result["paired_ap_differences"]) == {
        "encoder_seq_dtw_minus_bot_cosine"
    }

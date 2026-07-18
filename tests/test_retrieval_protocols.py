from __future__ import annotations

import numpy as np

from video_retrieval.diagnostics.statistics import (
    paired_cluster_bootstrap_ap_difference,
    truncated_average_precision,
)


def test_truncated_ap_is_bounded_by_recall() -> None:
    relevance = np.array([1, 0, 1, 0])
    total_relevant = 5

    average_precision = truncated_average_precision(relevance, total_relevant)
    recall = relevance.sum() / total_relevant

    assert average_precision <= recall
    assert average_precision == (1.0 + 2 / 3) / 5


def test_paired_cluster_bootstrap_preserves_pairing() -> None:
    labels = np.array([1, 0, 1, 0])
    scores_a = {
        1: (np.array([0.9, 0.1, 0.8, 0.2]), labels),
        2: (np.array([0.8, 0.2, 0.7, 0.3]), labels),
    }
    scores_b = {
        1: (np.array([0.1, 0.9, 0.2, 0.8]), labels),
        2: (np.array([0.2, 0.8, 0.3, 0.7]), labels),
    }

    result = paired_cluster_bootstrap_ap_difference(scores_a, scores_b, n_resamples=100, seed=7)

    assert result["difference_a_minus_b"] > 0
    assert result["ci"][0] > 0
    assert result["bootstrap_probability_a_gt_b"] == 1.0

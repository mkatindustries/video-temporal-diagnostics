"""Tests for retrieval outcome decomposition used by the Video4Real analysis."""

from __future__ import annotations

import numpy as np
import pytest

from video_retrieval.diagnostics.fusion import (
    OUTCOME_CATEGORIES,
    ranked_outcome_composition,
)


def test_ranked_outcomes_separate_maneuver_and_location_errors() -> None:
    clusters = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([0, 0, 1, 0, 0, 1])
    scores = np.zeros((6, 6), dtype=np.float64)

    # Query 0: relevant, same-cluster wrong-label, then wrong-cluster.
    scores[0, [1, 2, 3, 4, 5]] = [9.0, 8.0, 7.0, 6.0, 5.0]
    # Query 3: wrong-cluster, same-cluster wrong-label, then relevant.
    scores[3, [0, 5, 4, 1, 2]] = [9.0, 8.0, 7.0, 6.0, 5.0]

    result = ranked_outcome_composition(scores, clusters, labels, [0, 3], [1, 3])

    assert set(result[1]) == set(OUTCOME_CATEGORIES)
    assert np.array_equal(result[1]["relevant"], [1.0, 0.0])
    assert np.array_equal(result[1]["same_cluster_wrong_label"], [0.0, 0.0])
    assert np.array_equal(result[1]["wrong_cluster"], [0.0, 1.0])
    for category in OUTCOME_CATEGORIES:
        assert np.allclose(result[3][category], 1.0 / 3.0)


def test_ranked_outcome_fractions_sum_to_one_and_clip_large_k() -> None:
    clusters = np.array([0, 0, 1])
    labels = np.array([0, 1, 0])
    scores = np.array(
        [
            [np.inf, 2.0, 1.0],
            [2.0, np.inf, 1.0],
            [2.0, 1.0, np.inf],
        ]
    )

    result = ranked_outcome_composition(scores, clusters, labels, [0, 1, 2], [10])
    total = sum(result[10][category] for category in OUTCOME_CATEGORIES)
    assert np.allclose(total, 1.0)


def test_ranked_outcomes_reject_nonfinite_gallery_scores() -> None:
    scores = np.array([[0.0, np.inf], [1.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite gallery score"):
        ranked_outcome_composition(
            scores,
            candidate_clusters=np.array([0, 1]),
            candidate_labels=np.array([0, 0]),
            query_indices=[0],
            k_values=[1],
        )

"""Tests for the feature-vs-comparator factorial."""

import pytest
import torch

from video_retrieval.diagnostics import feature_comparator_decomposition


def _mean_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.mean(0), b.mean(0)))


def _ordered_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum(dim=1).mean())


def test_decomposition_evaluates_shared_factorial() -> None:
    baseline = {
        "a": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "b": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "c": torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
    }
    alternative = {key: value.clone() for key, value in baseline.items()}
    result = feature_comparator_decomposition(
        {"baseline": baseline, "alternative": alternative},
        {"mean": _mean_dot, "ordered": _ordered_dot},
        [("a", "b", 1), ("a", "c", 0)],
    )

    assert result["n_pairs"] == 2
    assert result["n_positive"] == 1
    assert set(result["ap_matrix"]) == {"baseline", "alternative"}
    assert result["ap_matrix"]["baseline"]["mean"] == pytest.approx(1.0)


def test_decomposition_requires_shared_pairs() -> None:
    with pytest.raises(ValueError, match="no pairs"):
        feature_comparator_decomposition(
            {"features": {"a": torch.ones(2, 2)}},
            {"mean": _mean_dot},
            [("a", "missing", 1)],
        )

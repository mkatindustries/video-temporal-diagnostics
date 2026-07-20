from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from eval_hdd_fusion import (  # noqa: E402
    ScoreCache,
    build_dtw_distance_matrix,
    save_score_cache,
    score_rows_complete,
)


def test_residual_matrix_builds_only_requested_query_rows() -> None:
    features = {
        10: {"temporal_residual": torch.tensor([[0.0], [1.0], [2.0]])},
        20: {"temporal_residual": torch.tensor([[0.0], [1.0], [3.0]])},
        30: {"temporal_residual": torch.tensor([[2.0], [1.0], [0.0]])},
    }
    query_indices = np.asarray([0, 2], dtype=np.int64)

    distance = build_dtw_distance_matrix(
        features,
        dense_to_segment=[10, 20, 30],
        query_indices=query_indices,
        device=torch.device("cpu"),
        dtw_batch_size=2,
        feature_key="temporal_residual",
        description="test residual DTW",
    )

    assert distance.shape == (3, 3)
    assert torch.isfinite(distance[0, 1:]).all()
    assert torch.isfinite(distance[2, :2]).all()
    assert torch.isinf(distance[1]).all()
    assert score_rows_complete(distance, 3, query_indices)
    assert not score_rows_complete(distance, 3, np.asarray([1]))


def test_residual_score_field_is_optional_for_legacy_caches() -> None:
    assert "temporal_residual_dtw_distance" not in ScoreCache.__required_keys__
    assert "temporal_residual_dtw_distance" in ScoreCache.__optional_keys__
    assert {"bot_similarity", "dtw_distance"}.issubset(ScoreCache.__required_keys__)


def test_atomic_cache_write_preserves_existing_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "score_cache.pt"
    cache_path.write_bytes(b"validated legacy cache")

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted save"):
        save_score_cache(cast(ScoreCache, {}), cache_path)

    assert cache_path.read_bytes() == b"validated legacy cache"
    assert list(tmp_path.iterdir()) == [cache_path]

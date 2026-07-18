"""Tests for temporal diagnostics: s_rev, scramble embeddings, scramble gradient.

Verifies correctness of compute_s_rev, scramble_embeddings, scramble_gradient,
and temporal_report from video_retrieval.diagnostics.
"""

import math

import torch

from video_retrieval.diagnostics import (
    compute_s_rev,
    scramble_embeddings,
    scramble_gradient,
    temporal_report,
)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Order-invariant similarity: cosine of mean-pooled embeddings."""
    a_mean = a.mean(dim=0)
    b_mean = b.mean(dim=0)
    return torch.nn.functional.cosine_similarity(a_mean, b_mean, dim=0).item()


def _ordered_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Order-sensitive similarity: mean per-timestep cosine similarity."""
    min_t = min(a.shape[0], b.shape[0])
    return torch.nn.functional.cosine_similarity(a[:min_t], b[:min_t], dim=1).mean().item()


# ---------------------------------------------------------------------------
# TestComputeSRev
# ---------------------------------------------------------------------------


class TestComputeSRev:
    """Test compute_s_rev reversal sensitivity metric."""

    def test_identical_forward_reverse(self):
        """Constant embeddings (all rows identical) should give s_rev=1.0."""
        torch.manual_seed(42)
        row = torch.randn(16)
        emb = row.unsqueeze(0).expand(10, -1).clone()
        result = compute_s_rev({"v1": emb}, _cosine_sim)

        assert abs(result["mean"] - 1.0) < 1e-6, (
            f"Expected s_rev=1.0 for constant embeddings, got {result['mean']}"
        )

    def test_mean_pooling_is_exactly_order_invariant(self):
        """Mean-pooled cosine is invariant even for asymmetric sequences."""
        torch.manual_seed(7)
        T, D = 20, 8
        ramp = torch.linspace(0, 1, T).unsqueeze(1)
        noise = torch.randn(T, D)
        emb = ramp * noise  # magnitude grows along time axis

        result = compute_s_rev({"v1": emb}, _cosine_sim)

        assert abs(result["mean"] - 1.0) < 1e-6, (
            f"Expected s_rev=1 for mean pooling, got {result['mean']}"
        )

    def test_empty_embeddings(self):
        """Empty dict should return NaN mean and std."""
        result = compute_s_rev({}, _cosine_sim)

        assert math.isnan(result["mean"]), (
            f"Expected NaN mean for empty embeddings, got {result['mean']}"
        )
        assert math.isnan(result["std"]), (
            f"Expected NaN std for empty embeddings, got {result['std']}"
        )
        assert result["per_video"] == {}

    def test_single_video(self):
        """Single video should return valid mean and std=0."""
        torch.manual_seed(99)
        emb = torch.randn(10, 16)
        result = compute_s_rev({"v1": emb}, _cosine_sim)

        assert not math.isnan(result["mean"]), "Mean should not be NaN for single video"
        assert abs(result["std"]) < 1e-8, f"Expected std=0 for single video, got {result['std']}"
        assert "v1" in result["per_video"]


# ---------------------------------------------------------------------------
# TestScrambleEmbeddings
# ---------------------------------------------------------------------------


class TestScrambleEmbeddings:
    """Test scramble_embeddings chunk-shuffle utility."""

    def test_no_scramble(self):
        """K=1 should return identical tensor."""
        torch.manual_seed(42)
        emb = torch.randn(20, 8)
        result = scramble_embeddings(emb, n_chunks=1)
        assert torch.equal(emb, result), "K=1 should return the input unchanged"

    def test_preserves_shape(self):
        """Output shape must match input shape."""
        torch.manual_seed(42)
        emb = torch.randn(20, 8)
        result = scramble_embeddings(emb, n_chunks=4)
        assert result.shape == emb.shape, f"Expected shape {emb.shape}, got {result.shape}"

    def test_deterministic(self):
        """Same seed should produce identical results."""
        torch.manual_seed(42)
        emb = torch.randn(20, 8)
        r1 = scramble_embeddings(emb, n_chunks=4, seed=123)
        r2 = scramble_embeddings(emb, n_chunks=4, seed=123)
        assert torch.equal(r1, r2), "Same seed should produce identical scrambles"

    def test_different_seeds(self):
        """Different seeds should produce different results for K>1."""
        torch.manual_seed(42)
        emb = torch.randn(20, 8)
        r1 = scramble_embeddings(emb, n_chunks=4, seed=0)
        r2 = scramble_embeddings(emb, n_chunks=4, seed=99)
        assert not torch.equal(r1, r2), "Different seeds should produce different scrambles"

    def test_k_larger_than_T(self):
        """K > T should still work (clamps to T chunks of size 1)."""
        torch.manual_seed(42)
        emb = torch.randn(5, 4)
        result = scramble_embeddings(emb, n_chunks=100)
        assert result.shape == emb.shape, f"Expected shape {emb.shape} when K>T, got {result.shape}"

    def test_uneven_split_uses_balanced_chunks(self):
        """Remainder frames must be distributed instead of forming one large tail."""
        emb = torch.arange(30).unsqueeze(1)
        # Seed 1 does not place originally adjacent chunks next to one another.
        result = scramble_embeddings(emb, n_chunks=16, seed=1).squeeze(1)
        positions = {int(value): idx for idx, value in enumerate(result)}

        # With balanced chunks, original neighbors are grouped only in pairs.
        longest_preserved_run = 1
        current_run = 1
        for value in range(1, 30):
            if positions[value] == positions[value - 1] + 1:
                current_run += 1
                longest_preserved_run = max(longest_preserved_run, current_run)
            else:
                current_run = 1
        assert longest_preserved_run <= 2


# ---------------------------------------------------------------------------
# TestScrambleGradient
# ---------------------------------------------------------------------------


class TestScrambleGradient:
    """Test scramble_gradient retrieval diagnostic."""

    def _make_dataset(self, n_match: int = 5, n_nonmatch: int = 5, T: int = 20, D: int = 16):
        """Build a small retrieval dataset with matched and non-matched pairs.

        Matched pairs share the same embeddings; non-matched pairs are random.
        This gives perfect AP=1.0 with a reasonable similarity function.
        """
        torch.manual_seed(42)
        emb_a: dict[str, torch.Tensor] = {}
        emb_b: dict[str, torch.Tensor] = {}
        pairs: list[tuple[str, str, int]] = []

        for i in range(n_match):
            key_a = f"match_a_{i}"
            key_b = f"match_b_{i}"
            shared = torch.randn(T, D)
            emb_a[key_a] = shared
            emb_b[key_b] = shared.clone()
            pairs.append((key_a, key_b, 1))

        for i in range(n_nonmatch):
            key_a = f"nonmatch_a_{i}"
            key_b = f"nonmatch_b_{i}"
            emb_a[key_a] = torch.randn(T, D)
            emb_b[key_b] = torch.randn(T, D)
            pairs.append((key_a, key_b, 0))

        return emb_a, emb_b, pairs

    def test_order_invariant_method(self):
        """Mean-pooled cosine should produce 'order-invariant' verdict."""
        emb_a, emb_b, pairs = self._make_dataset()
        result = scramble_gradient(
            emb_a,
            emb_b,
            pairs,
            _cosine_sim,
            k_values=(1, 4, 16),
        )

        assert result["verdict"] == "no-detected-sensitivity", (
            f"Expected 'no-detected-sensitivity' for mean-pooled cosine, "
            f"got '{result['verdict']}' (AP scores: {result['ap_scores']})"
        )
        assert len(result["ap_scores"]) == 3
        assert result["k_values"] == [1, 4, 16]

    def test_order_sensitive_method(self):
        """Per-timestep cosine should produce 'order-sensitive' verdict."""
        emb_a, emb_b, pairs = self._make_dataset()
        result = scramble_gradient(
            emb_a,
            emb_b,
            pairs,
            _ordered_sim,
            k_values=(1, 4, 16),
        )

        assert result["verdict"] == "order-sensitive", (
            f"Expected 'order-sensitive' for per-timestep similarity, "
            f"got '{result['verdict']}' (AP scores: {result['ap_scores']})"
        )

    def test_no_valid_pairs(self):
        """All pairs reference missing IDs -> ap_scores should be NaN."""
        emb_a = {"a1": torch.randn(10, 4)}
        emb_b = {"b1": torch.randn(10, 4)}
        pairs = [("missing_a", "missing_b", 1)]

        result = scramble_gradient(
            emb_a,
            emb_b,
            pairs,
            _cosine_sim,
            k_values=(1, 4),
        )

        assert all(math.isnan(s) for s in result["ap_scores"]), (
            f"Expected all NaN AP scores for missing pairs, got {result['ap_scores']}"
        )
        assert result["verdict"] == "inconclusive", (
            f"Expected 'inconclusive' verdict, got '{result['verdict']}'"
        )


# ---------------------------------------------------------------------------
# TestTemporalReport
# ---------------------------------------------------------------------------


class TestTemporalReport:
    """Test temporal_report combined diagnostic."""

    def test_combined_report(self):
        """Report should contain both scramble_gradient and reversal_sensitivity."""
        torch.manual_seed(42)
        T, D = 15, 8

        emb_a = {"v1": torch.randn(T, D), "v2": torch.randn(T, D)}
        emb_b = {"v3": torch.randn(T, D), "v4": torch.randn(T, D)}
        pairs = [("v1", "v3", 1), ("v2", "v4", 0)]

        report = temporal_report(
            emb_a,
            emb_b,
            pairs,
            _cosine_sim,
            k_values=(1, 4),
            seed=0,
        )

        assert "scramble_gradient" in report, "Report must contain 'scramble_gradient'"
        assert "reversal_sensitivity" in report, "Report must contain 'reversal_sensitivity'"

        sg = report["scramble_gradient"]
        assert "k_values" in sg, "scramble_gradient must contain 'k_values'"
        assert "ap_scores" in sg, "scramble_gradient must contain 'ap_scores'"
        assert "verdict" in sg, "scramble_gradient must contain 'verdict'"
        assert len(sg["ap_scores"]) == len(sg["k_values"])

        rs = report["reversal_sensitivity"]
        assert "mean" in rs, "reversal_sensitivity must contain 'mean'"
        assert "std" in rs, "reversal_sensitivity must contain 'std'"
        assert "n_videos" in rs, "reversal_sensitivity must contain 'n_videos'"
        assert rs["n_videos"] == 4, f"Expected 4 videos (union of both dicts), got {rs['n_videos']}"

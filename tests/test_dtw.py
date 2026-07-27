"""Correctness tests for GPU-vectorized batched DTW.

Compares the wavefront implementation against a reference Python-loop DTW
to ensure numerical equivalence.
"""

import pytest
import torch

from video_retrieval.fingerprints.dtw import (
    dtw_distance,
    dtw_distance_batch,
    dtw_distance_shuffled,
    dtw_distance_batch_shuffled,
    _normalize_sequence,
)


# ---------------------------------------------------------------------------
# Reference implementation (original Python double-for-loop)
# ---------------------------------------------------------------------------

def _reference_dtw(seq1: torch.Tensor, seq2: torch.Tensor, normalize: bool) -> float:
    """Reference DTW using Python loops — known-correct, slow."""
    s1 = seq1.clone()
    s2 = seq2.clone()

    if normalize:
        s1 = _normalize_sequence(s1)
        s2 = _normalize_sequence(s2)

    n, m = s1.shape[0], s2.shape[0]
    cost = torch.cdist(s1, s2)

    dtw = torch.full((n + 1, m + 1), float("inf"))
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j].item(),
                dtw[i, j - 1].item(),
                dtw[i - 1, j - 1].item(),
            )

    return dtw[n, m].item() / (n + m)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDTWDistance:
    """Test single-pair dtw_distance against reference.

    Tolerance accounts for float32 (wavefront torch.minimum) vs float64
    (reference .item() + Python min) accumulated rounding.
    """

    TOL = 1e-4

    def test_small_4d(self):
        """10x10 sequences with D=4 (trajectory-like)."""
        torch.manual_seed(42)
        s1 = torch.randn(10, 4)
        s2 = torch.randn(10, 4)

        got = dtw_distance(s1, s2, normalize=True)
        ref = _reference_dtw(s1, s2, normalize=True)
        assert abs(got - ref) < self.TOL, f"got={got}, ref={ref}"

    def test_asymmetric_1024d(self):
        """50x30 sequences with D=1024 (embedding-like)."""
        torch.manual_seed(123)
        s1 = torch.randn(50, 1024)
        s2 = torch.randn(30, 1024)

        got = dtw_distance(s1, s2, normalize=False)
        ref = _reference_dtw(s1, s2, normalize=False)
        assert abs(got - ref) < self.TOL, f"got={got}, ref={ref}"

    def test_square_100x100(self):
        """100x100 sequences with D=4."""
        torch.manual_seed(7)
        s1 = torch.randn(100, 4)
        s2 = torch.randn(100, 4)

        got = dtw_distance(s1, s2, normalize=True)
        ref = _reference_dtw(s1, s2, normalize=True)
        assert abs(got - ref) < self.TOL, f"got={got}, ref={ref}"

    def test_no_normalize(self):
        """normalize=False should match reference."""
        torch.manual_seed(99)
        s1 = torch.randn(20, 8)
        s2 = torch.randn(15, 8)

        got = dtw_distance(s1, s2, normalize=False)
        ref = _reference_dtw(s1, s2, normalize=False)
        assert abs(got - ref) < self.TOL, f"got={got}, ref={ref}"

    def test_identical_sequences(self):
        """Identical sequences should give distance 0."""
        s = torch.randn(10, 4)
        dist = dtw_distance(s, s.clone(), normalize=False)
        assert dist < 1e-6, f"Expected ~0, got {dist}"

    def test_single_frame(self):
        """Single-frame sequences."""
        s1 = torch.randn(1, 4)
        s2 = torch.randn(1, 4)
        got = dtw_distance(s1, s2, normalize=False)
        ref = _reference_dtw(s1, s2, normalize=False)
        assert abs(got - ref) < 1e-5


class TestDTWDistanceBatch:
    """Test batched DTW matches individual single-pair calls."""

    def test_batch_matches_individual(self):
        """Batch of 5 pairs should match 5 individual calls."""
        torch.manual_seed(42)
        N = 5
        seqs_a = [torch.randn(10 + i, 4) for i in range(N)]
        seqs_b = [torch.randn(8 + i, 4) for i in range(N)]

        batch_dists = dtw_distance_batch(seqs_a, seqs_b, normalize=True)
        individual_dists = [
            dtw_distance(a, b, normalize=True)
            for a, b in zip(seqs_a, seqs_b)
        ]

        for i in range(N):
            assert abs(batch_dists[i].item() - individual_dists[i]) < 1e-4, (
                f"Pair {i}: batch={batch_dists[i].item()}, "
                f"individual={individual_dists[i]}"
            )

    def test_batch_no_normalize(self):
        """Batch with normalize=False."""
        torch.manual_seed(55)
        N = 3
        seqs_a = [torch.randn(15, 1024) for _ in range(N)]
        seqs_b = [torch.randn(20, 1024) for _ in range(N)]

        batch_dists = dtw_distance_batch(seqs_a, seqs_b, normalize=False)
        for i in range(N):
            ref = dtw_distance(seqs_a[i], seqs_b[i], normalize=False)
            assert abs(batch_dists[i].item() - ref) < 1e-4

    def test_batch_variable_lengths(self):
        """Batch with highly variable sequence lengths."""
        torch.manual_seed(77)
        lengths_a = [5, 50, 10, 30]
        lengths_b = [30, 5, 40, 10]
        D = 4

        seqs_a = [torch.randn(l, D) for l in lengths_a]
        seqs_b = [torch.randn(l, D) for l in lengths_b]

        batch_dists = dtw_distance_batch(seqs_a, seqs_b, normalize=True)
        for i in range(len(lengths_a)):
            ref = dtw_distance(seqs_a[i], seqs_b[i], normalize=True)
            assert abs(batch_dists[i].item() - ref) < 1e-4, (
                f"Pair {i} (len {lengths_a[i]}x{lengths_b[i]}): "
                f"batch={batch_dists[i].item()}, ref={ref}"
            )

    def test_chunk_size(self):
        """Chunked processing gives same results as single chunk."""
        torch.manual_seed(88)
        N = 10
        seqs_a = [torch.randn(8, 4) for _ in range(N)]
        seqs_b = [torch.randn(8, 4) for _ in range(N)]

        full = dtw_distance_batch(seqs_a, seqs_b, normalize=True, chunk_size=N)
        chunked = dtw_distance_batch(seqs_a, seqs_b, normalize=True, chunk_size=3)

        for i in range(N):
            assert abs(full[i].item() - chunked[i].item()) < 1e-5

    def test_empty_batch(self):
        """Empty batch returns empty tensor."""
        result = dtw_distance_batch([], [], normalize=True)
        assert result.shape == (0,)


class TestDTWDistanceShuffled:
    """Order-ablation control: identical machinery to dtw_distance, only seq2's time
    axis is permuted. Mirrors the design validated for SoccerNet's shuffled-DTW control
    (encoder_seq_dtw vs encoder_seq_dtw_shuffled), now reused for the driving datasets."""

    def test_self_comparison_not_zero(self):
        """Shuffling breaks the identical-sequence zero-distance property, confirming
        the permutation actually changes what's compared (unlike dtw_distance(s, s))."""
        s = torch.linspace(0, 1, 20).unsqueeze(1).repeat(1, 4)  # smooth, structured
        unshuffled = dtw_distance(s, s.clone(), normalize=True)
        shuffled = dtw_distance_shuffled(
            s, s.clone(), pair_id="a|b", n_perms=10, normalize=True
        )
        assert unshuffled < 1e-6
        assert shuffled > 1e-3

    def test_reproducible(self):
        """Same pair_id + same inputs -> identical result across calls."""
        torch.manual_seed(2)
        s1 = torch.randn(15, 4)
        s2 = torch.randn(12, 4)
        pid = ("sessA", 10, "sessB", 20)
        d1 = dtw_distance_shuffled(s1, s2, pair_id=pid, n_perms=5)
        d2 = dtw_distance_shuffled(s1, s2, pair_id=pid, n_perms=5)
        assert d1 == d2

    def test_injective_across_separator_collision(self):
        """Pair ids that would collide under naive '|'-joined string concatenation must
        not share a permutation draw -- the exact bug class the SoccerNet round-4
        review caught (event ids there contain literal '|' characters)."""
        torch.manual_seed(3)
        s1 = torch.randn(10, 4)
        s2 = torch.randn(10, 4)
        d1 = dtw_distance_shuffled(s1, s2, pair_id=("a|b", "c"), n_perms=1)
        d2 = dtw_distance_shuffled(s1, s2, pair_id=("a", "b|c"), n_perms=1)
        assert d1 != d2

    def test_normalize_flag_respected(self):
        """normalize=False should skip the min-max step, same as dtw_distance."""
        torch.manual_seed(4)
        s1 = torch.randn(10, 4)
        s2 = torch.randn(10, 4)
        d_norm = dtw_distance_shuffled(s1, s2, pair_id="x", n_perms=3, normalize=True)
        d_raw = dtw_distance_shuffled(s1, s2, pair_id="x", n_perms=3, normalize=False)
        assert d_norm != d_raw


class TestDTWDistanceBatchShuffled:
    """Batched order-ablation control must match per-pair dtw_distance_shuffled calls
    exactly, since it's the same seeding scheme applied per pair per permutation."""

    def test_batch_matches_individual(self):
        torch.manual_seed(10)
        N = 4
        seqs_a = [torch.randn(10 + i, 4) for i in range(N)]
        seqs_b = [torch.randn(8 + i, 4) for i in range(N)]
        pair_ids = [("sessA", i, "sessB", i * 2) for i in range(N)]

        batch = dtw_distance_batch_shuffled(
            seqs_a, seqs_b, pair_ids=pair_ids, n_perms=5, normalize=True
        )
        individual = [
            dtw_distance_shuffled(a, b, pair_id=pid, n_perms=5, normalize=True)
            for a, b, pid in zip(seqs_a, seqs_b, pair_ids)
        ]
        for i in range(N):
            assert abs(batch[i].item() - individual[i]) < 1e-4, (
                f"Pair {i}: batch={batch[i].item()}, individual={individual[i]}"
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            dtw_distance_batch_shuffled(
                [torch.randn(5, 4)], [torch.randn(5, 4), torch.randn(5, 4)],
                pair_ids=["a"],
            )

    def test_empty_batch(self):
        result = dtw_distance_batch_shuffled([], [], pair_ids=[])
        assert result.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Test CUDA gives same results as CPU."""

    def test_single_pair_cuda(self):
        torch.manual_seed(42)
        s1 = torch.randn(20, 4)
        s2 = torch.randn(15, 4)

        cpu_dist = dtw_distance(s1, s2, normalize=True)
        cuda_dist = dtw_distance(s1.cuda(), s2.cuda(), normalize=True)

        assert abs(cpu_dist - cuda_dist) < 1e-4

    def test_batch_cuda(self):
        torch.manual_seed(42)
        N = 5
        seqs_a = [torch.randn(10 + i, 4) for i in range(N)]
        seqs_b = [torch.randn(8 + i, 4) for i in range(N)]

        cpu_dists = dtw_distance_batch(seqs_a, seqs_b, normalize=True)
        cuda_dists = dtw_distance_batch(
            [s.cuda() for s in seqs_a],
            [s.cuda() for s in seqs_b],
            normalize=True,
        )

        for i in range(N):
            assert abs(cpu_dists[i].item() - cuda_dists[i].item()) < 1e-4

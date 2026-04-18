"""Correctness tests for GPU-vectorized batched DTW.

Compares the wavefront implementation against a reference Python-loop DTW
to ensure numerical equivalence.
"""

import pytest
import torch

from video_retrieval.fingerprints.dtw import (
    dtw_distance,
    dtw_distance_batch,
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

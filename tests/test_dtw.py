"""Correctness tests for GPU-vectorized batched DTW.

Compares the wavefront implementation against a reference Python-loop DTW
to ensure numerical equivalence.
"""

import pytest
import torch

from video_retrieval.fingerprints.dtw import (
    _normalize_sequence,
    assignment_distance,
    assignment_distance_batch,
    dtw_distance,
    dtw_distance_batch,
    dtw_distance_batch_shuffled,
    dtw_distance_shuffled,
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

        seqs_a = [torch.randn(length, D) for length in lengths_a]
        seqs_b = [torch.randn(length, D) for length in lengths_b]

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


class TestAssignmentDistance:
    """Rigid order-agnostic matching over DTW's normalized L2 costs."""

    def test_known_optimal_assignment(self):
        seq1 = torch.tensor([[0.0], [10.0], [20.0]])
        seq2 = torch.tensor([[11.0], [19.0], [1.0]])

        # Optimal costs are 1 + 1 + 1, normalized by T1 + T2 = 6.
        assert assignment_distance(seq1, seq2, normalize=False) == pytest.approx(0.5)

    def test_order_invariant(self):
        torch.manual_seed(90)
        seq = torch.randn(8, 6)
        permuted = seq[torch.randperm(len(seq))]

        assert assignment_distance(seq, permuted, normalize=True) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_symmetric_for_distinct_sequences(self):
        torch.manual_seed(94)
        seq1 = torch.randn(8, 6)
        seq2 = torch.randn(8, 6)
        forward = assignment_distance(seq1, seq2, normalize=True)
        reverse = assignment_distance(seq2, seq1, normalize=True)
        assert forward > 0.0
        assert reverse == pytest.approx(forward, abs=1e-6)

    def test_normalize_flag_respected(self):
        seq1 = torch.tensor([[0.0], [1.0], [2.0]])
        seq2 = torch.tensor([[0.0], [10.0], [20.0]])

        assert assignment_distance(seq1, seq2, normalize=True) == pytest.approx(
            0.0, abs=1e-6
        )
        assert assignment_distance(seq1, seq2, normalize=False) == pytest.approx(4.5)

    def test_equal_length_required(self):
        with pytest.raises(ValueError, match="equal sequence lengths"):
            assignment_distance(torch.randn(4, 3), torch.randn(5, 3))

    def test_nonempty_float_input_required(self):
        with pytest.raises(ValueError, match="at least one timestep"):
            assignment_distance(torch.empty(0, 3), torch.empty(0, 3))
        with pytest.raises(ValueError, match="float32 or float64"):
            assignment_distance(
                torch.ones(3, 2, dtype=torch.float16),
                torch.ones(3, 2, dtype=torch.float16),
            )

    def test_production_shape_self_distance_matches_dtw(self):
        torch.manual_seed(91)
        seq = torch.randn(32, 1024)
        assignment = assignment_distance(seq, seq, normalize=True)
        dtw = dtw_distance(seq, seq, normalize=True)
        assert assignment == pytest.approx(dtw, abs=1e-6)


class TestAssignmentDistanceBatch:
    def test_batch_matches_individual_with_chunking(self):
        torch.manual_seed(92)
        lengths = [5, 5, 7, 3]
        seqs_a = [torch.randn(length, 4) for length in lengths]
        seqs_b = [torch.randn(length, 4) for length in lengths]

        batch = assignment_distance_batch(
            seqs_a, seqs_b, normalize=True, chunk_size=2
        )
        individual = [
            assignment_distance(seq1, seq2, normalize=True)
            for seq1, seq2 in zip(seqs_a, seqs_b)
        ]
        torch.testing.assert_close(batch, torch.tensor(individual), rtol=1e-5, atol=1e-6)
        assert batch.device.type == "cpu"
        assert batch.dtype == seqs_a[0].dtype

    def test_empty_batch(self):
        assert assignment_distance_batch([], []).shape == (0,)

    def test_invalid_batch_inputs(self):
        with pytest.raises(ValueError, match="same number"):
            assignment_distance_batch([torch.randn(3, 2)], [])
        with pytest.raises(ValueError, match="equal sequence lengths"):
            assignment_distance_batch(
                [torch.randn(3, 2)], [torch.randn(4, 2)]
            )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            assignment_distance_batch([], [], chunk_size=0)


class TestDTWDistanceShuffled:
    """Symmetric order ablation using the same machinery as dtw_distance."""

    def test_self_comparison_not_zero(self):
        """Shuffling breaks the identical-sequence zero-distance property, confirming
        the permutation actually changes what's compared (unlike dtw_distance(s, s))."""
        s = torch.linspace(0, 1, 20).unsqueeze(1).repeat(1, 4)  # smooth, structured
        unshuffled = dtw_distance(s, s.clone(), normalize=True)
        shuffled = dtw_distance_shuffled(
            s, s.clone(), pair_id=("a", "b"), n_perms=10, normalize=True
        )
        assert unshuffled < 1e-6
        assert shuffled > 1e-3

    def test_reproducible(self):
        """Same pair_id + same inputs -> identical result across calls."""
        torch.manual_seed(2)
        s1 = torch.randn(15, 4)
        s2 = torch.randn(12, 4)
        pid = (("sessA", 10), ("sessB", 20))
        d1 = dtw_distance_shuffled(s1, s2, pair_id=pid, n_perms=5)
        d2 = dtw_distance_shuffled(s1, s2, pair_id=pid, n_perms=5)
        assert d1 == d2

    def test_symmetric_for_unordered_pair(self):
        torch.manual_seed(20)
        s1 = torch.randn(11, 4)
        s2 = torch.randn(8, 4)
        pid = (("sessA", 10), ("sessB", 20))
        d12 = dtw_distance_shuffled(s1, s2, pair_id=pid, n_perms=5)
        d21 = dtw_distance_shuffled(s2, s1, pair_id=(pid[1], pid[0]), n_perms=5)
        assert d12 == pytest.approx(d21, abs=1e-7)

    def test_base_seed_selects_different_draws(self):
        torch.manual_seed(21)
        s1 = torch.randn(10, 4)
        s2 = torch.randn(10, 4)
        d1 = dtw_distance_shuffled(s1, s2, pair_id=("x", "y"), n_perms=1, base_seed=1)
        d2 = dtw_distance_shuffled(s1, s2, pair_id=("x", "y"), n_perms=1, base_seed=2)
        assert d1 != d2

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
        d_norm = dtw_distance_shuffled(
            s1, s2, pair_id=("x", "y"), n_perms=3, normalize=True
        )
        d_raw = dtw_distance_shuffled(
            s1, s2, pair_id=("x", "y"), n_perms=3, normalize=False
        )
        assert d_norm != d_raw

    def test_nonpositive_permutations_raise(self):
        with pytest.raises(ValueError, match="n_perms must be positive"):
            dtw_distance_shuffled(
                torch.randn(5, 4),
                torch.randn(5, 4),
                pair_id=("x", "y"),
                n_perms=0,
            )


class TestDTWDistanceBatchShuffled:
    """Batched order-ablation control must match per-pair dtw_distance_shuffled calls
    exactly, since it's the same seeding scheme applied per pair per permutation."""

    def test_batch_matches_individual(self):
        torch.manual_seed(10)
        N = 4
        seqs_a = [torch.randn(10 + i, 4) for i in range(N)]
        seqs_b = [torch.randn(8 + i, 4) for i in range(N)]
        pair_ids = [(("sessA", i), ("sessB", i * 2)) for i in range(N)]

        batch = dtw_distance_batch_shuffled(
            seqs_a,
            seqs_b,
            pair_ids=pair_ids,
            n_perms=5,
            base_seed=7,
            normalize=True,
            chunk_size=2,
        )
        individual = [
            dtw_distance_shuffled(
                a, b, pair_id=pid, n_perms=5, base_seed=7, normalize=True
            )
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
                pair_ids=[("a", "b")],
            )

    def test_empty_batch(self):
        result = dtw_distance_batch_shuffled([], [], pair_ids=[])
        assert result.shape == (0,)

    def test_nonpositive_permutations_raise(self):
        with pytest.raises(ValueError, match="n_perms must be positive"):
            dtw_distance_batch_shuffled([], [], pair_ids=[], n_perms=0)

    def test_nonpositive_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            dtw_distance_batch_shuffled([], [], pair_ids=[], chunk_size=0)


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

    def test_shuffled_cuda_matches_cpu(self):
        torch.manual_seed(30)
        seqs_a = [torch.randn(9, 4), torch.randn(7, 4)]
        seqs_b = [torch.randn(8, 4), torch.randn(10, 4)]
        pair_ids = [("a", "b"), ("c", "d")]

        cpu_dists = dtw_distance_batch_shuffled(
            seqs_a, seqs_b, pair_ids=pair_ids, n_perms=3
        )
        cuda_dists = dtw_distance_batch_shuffled(
            [seq.cuda() for seq in seqs_a],
            [seq.cuda() for seq in seqs_b],
            pair_ids=pair_ids,
            n_perms=3,
        )
        torch.testing.assert_close(cuda_dists.cpu(), cpu_dists, rtol=1e-3, atol=1e-4)

    def test_assignment_cuda_matches_cpu(self):
        torch.manual_seed(93)
        seqs_a = [torch.randn(8, 4), torch.randn(6, 4)]
        seqs_b = [torch.randn(8, 4), torch.randn(6, 4)]

        cpu_dists = assignment_distance_batch(seqs_a, seqs_b, chunk_size=1)
        cuda_dists = assignment_distance_batch(
            [seq.cuda() for seq in seqs_a],
            [seq.cuda() for seq in seqs_b],
            chunk_size=1,
        )
        torch.testing.assert_close(cuda_dists.cpu(), cpu_dists, rtol=1e-4, atol=1e-5)

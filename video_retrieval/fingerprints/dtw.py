"""GPU-vectorized Dynamic Time Warping using anti-diagonal wavefront processing.

The DTW recurrence dtw[i,j] = cost[i-1,j-1] + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
has the property that all cells on the same anti-diagonal (i+j = d) are independent.
For a T1xT2 matrix this reduces T1*T2 serial Python iterations to T1+T2 vectorized
torch ops. Batching N pairs into (N, T1, T2) processes all pairs simultaneously.
"""

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import torch
from scipy.optimize import linear_sum_assignment


def _canonical_pair_id(pair_id: tuple[object, object]) -> tuple[object, object]:
    """Return an orientation-invariant representation of two endpoint IDs."""
    if not isinstance(pair_id, tuple) or len(pair_id) != 2:
        raise ValueError("pair_id must be a tuple of two endpoint identifiers")
    endpoint1, endpoint2 = pair_id
    if repr(endpoint1) <= repr(endpoint2):
        return endpoint1, endpoint2
    return endpoint2, endpoint1


def _shuffle_permutation(
    length: int,
    *,
    pair_id: tuple[object, object],
    permutation_index: int,
    base_seed: int,
) -> torch.Tensor:
    """Build one reproducible per-pair permutation on CPU."""
    seed = int.from_bytes(
        hashlib.sha256(
            repr((base_seed, pair_id, permutation_index)).encode()
        ).digest()[:8],
        "little",
    ) % (2**63 - 1)
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed))


def _normalize_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Independently min-max normalize each feature dimension over time."""
    if seq.shape[0] <= 1:
        return seq
    min_vals = seq.min(dim=0).values
    max_vals = seq.max(dim=0).values
    range_vals = (max_vals - min_vals).clamp(min=1e-8)
    return (seq - min_vals) / range_vals


def _wavefront_dtw_batch(
    cost: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
) -> torch.Tensor:
    """Compute DTW distances for a batch using anti-diagonal wavefront.

    Args:
        cost: (N, T1, T2) pairwise cost matrices.
        lengths1: (N,) actual lengths for dim 1 (int64).
        lengths2: (N,) actual lengths for dim 2 (int64).

    Returns:
        (N,) DTW distances (unnormalized).
    """
    N, T1, T2 = cost.shape
    inf = float("inf")

    # DTW accumulation matrix with border of inf
    dtw = torch.full((N, T1 + 1, T2 + 1), inf, device=cost.device, dtype=cost.dtype)
    dtw[:, 0, 0] = 0.0

    # Process anti-diagonals d = i + j, where 1 <= d <= T1 + T2
    for d in range(1, T1 + T2 + 1):
        # Valid i range: 1 <= i <= T1 and 1 <= j=d-i <= T2
        i_start = max(1, d - T2)
        i_end = min(T1, d - 1)  # j = d - i >= 1 requires i <= d - 1
        if i_start > i_end:
            continue

        i_idx = torch.arange(i_start, i_end + 1, device=cost.device)
        j_idx = d - i_idx  # j = d - i, guaranteed 1 <= j <= T2

        # Gather predecessors: (N, len(i_idx))
        prev = torch.minimum(
            torch.minimum(
                dtw[:, i_idx - 1, j_idx],  # from above
                dtw[:, i_idx, j_idx - 1],  # from left
            ),
            dtw[:, i_idx - 1, j_idx - 1],  # from diagonal
        )

        # Update: dtw[i,j] = cost[i-1,j-1] + min(predecessors)
        dtw[:, i_idx, j_idx] = cost[:, i_idx - 1, j_idx - 1] + prev

    # Gather result at each pair's actual lengths
    return dtw[torch.arange(N, device=cost.device), lengths1, lengths2]


def dtw_distance(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    normalize: bool = True,
) -> float:
    """Compute DTW distance between two sequences.

    Drop-in replacement for the original Python-loop DTW. Uses the vectorized
    wavefront kernel internally.

    Args:
        seq1: First sequence (T1, D).
        seq2: Second sequence (T2, D).
        normalize: Independently min-max normalize each sequence and feature
            dimension over time before comparing.

    Returns:
        DTW accumulated cost divided by ``T1 + T2`` (lower = more similar).
    """
    s1 = seq1.clone()
    s2 = seq2.clone()

    if normalize:
        s1 = _normalize_sequence(s1)
        s2 = _normalize_sequence(s2)

    n, m = s1.shape[0], s2.shape[0]

    # Pairwise cost matrix
    cost = torch.cdist(s1, s2).unsqueeze(0)  # (1, n, m)
    lengths1 = torch.tensor([n], device=seq1.device, dtype=torch.long)
    lengths2 = torch.tensor([m], device=seq1.device, dtype=torch.long)

    dist = _wavefront_dtw_batch(cost, lengths1, lengths2)
    return dist.item() / (n + m)


def dtw_distance_batch(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    normalize: bool = True,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute DTW distances for a batch of sequence pairs.

    Handles variable-length sequences by padding to max length within each chunk.
    Processes in chunks of chunk_size pairs to limit GPU memory.

    Args:
        seqs_a: List of N tensors, each (T_i, D).
        seqs_b: List of N tensors, each (T_j, D).
        normalize: Independently min-max normalize each sequence and feature
            dimension over time before comparing.
        chunk_size: Max pairs per GPU batch (controls memory usage).

    Returns:
        (N,) tensor of DTW accumulated costs divided by ``T1 + T2``.
    """
    assert len(seqs_a) == len(seqs_b), "Must have same number of sequences"
    N = len(seqs_a)
    if N == 0:
        return torch.tensor([])

    device = seqs_a[0].device
    all_dists = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_a = seqs_a[start:end]
        chunk_b = seqs_b[start:end]
        B = end - start

        # Normalize if requested
        if normalize:
            chunk_a = [_normalize_sequence(s.clone()) for s in chunk_a]
            chunk_b = [_normalize_sequence(s.clone()) for s in chunk_b]

        # Get actual lengths
        lens1 = torch.tensor([s.shape[0] for s in chunk_a], device=device, dtype=torch.long)
        lens2 = torch.tensor([s.shape[0] for s in chunk_b], device=device, dtype=torch.long)

        T1_max = int(lens1.max().item())
        T2_max = int(lens2.max().item())

        # Build cost matrices per-pair to match single-pair cdist numerics
        # (torch.cdist may select different algorithms based on matrix size,
        # so batched cdist on padded tensors can diverge from per-pair cdist).
        cost = torch.full((B, T1_max, T2_max), float("inf"), device=device)
        for i in range(B):
            t1, t2 = int(lens1[i].item()), int(lens2[i].item())
            cost[i, :t1, :t2] = torch.cdist(
                chunk_a[i].unsqueeze(0),
                chunk_b[i].unsqueeze(0),
            ).squeeze(0)

        dists = _wavefront_dtw_batch(cost, lens1, lens2)

        # Normalize by the sum of input lengths, not the realized warping path.
        path_lengths = (lens1 + lens2).float()
        dists = dists / path_lengths

        all_dists.append(dists)

    return torch.cat(all_dists)


def _validate_assignment_pair(seq1: torch.Tensor, seq2: torch.Tensor) -> None:
    """Validate one rigid one-to-one assignment pair."""
    if seq1.ndim != 2 or seq2.ndim != 2:
        raise ValueError("assignment inputs must be rank-2 (T, D) tensors")
    if seq1.shape[0] == 0:
        raise ValueError("assignment inputs must contain at least one timestep")
    if seq1.shape[0] != seq2.shape[0]:
        raise ValueError(
            "one-to-one assignment requires equal sequence lengths, "
            f"got {seq1.shape[0]} and {seq2.shape[0]}"
        )
    if seq1.shape[1] != seq2.shape[1]:
        raise ValueError(
            "assignment inputs must have equal feature dimensions, "
            f"got {seq1.shape[1]} and {seq2.shape[1]}"
        )
    if seq1.device != seq2.device:
        raise ValueError("assignment inputs must be on the same device")
    if seq1.dtype != seq2.dtype:
        raise ValueError("assignment inputs must have the same dtype")
    if seq1.dtype not in (torch.float32, torch.float64):
        raise ValueError("assignment inputs must use float32 or float64")


def assignment_distance(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    normalize: bool = True,
) -> float:
    """Compute rigid min-cost one-to-one assignment distance.

    This uses the same independently min-max-normalized sequences, Euclidean
    pairwise cost, and ``T1 + T2`` normalization as :func:`dtw_distance`, but
    replaces DTW's monotonic path with a bijective Hungarian assignment. It is
    therefore an order-agnostic structural control, not a pure ordering
    ablation: it also removes DTW's one-to-many warping and endpoint anchoring.

    The two sequences must have equal nonzero lengths so every timestep is
    matched exactly once. This prevents rectangular assignment from silently
    leaving timesteps unmatched while still dividing by ``T1 + T2``.
    """
    return float(
        assignment_distance_batch([seq1], [seq2], normalize=normalize)[0].item()
    )


def assignment_distance_batch(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    normalize: bool = True,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute rigid one-to-one assignment distances for sequence pairs.

    Pairwise Euclidean costs are computed in device batches, grouped by
    sequence length within each chunk. Each cost block is transferred to CPU
    once, where SciPy's exact Hungarian solver processes its matrices. The
    returned tensor is moved back to the input device to match
    :func:`dtw_distance_batch`.

    Every pair must contain equal-length, nonempty ``(T, D)`` tensors. Lengths
    may differ between pairs.
    """
    if len(seqs_a) != len(seqs_b):
        raise ValueError("seqs_a and seqs_b must contain the same number of sequences")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not seqs_a:
        return torch.tensor([])

    device = seqs_a[0].device
    dtype = seqs_a[0].dtype
    for seq1, seq2 in zip(seqs_a, seqs_b):
        _validate_assignment_pair(seq1, seq2)
        if seq1.device != device:
            raise ValueError("all assignment inputs must be on the same device")
        if seq1.dtype != dtype:
            raise ValueError("all assignment inputs must have the same dtype")

    distances = torch.empty(len(seqs_a), dtype=dtype)
    for start in range(0, len(seqs_a), chunk_size):
        end = min(start + chunk_size, len(seqs_a))
        indices_by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in range(start, end):
            shape = seqs_a[index].shape
            indices_by_shape[(int(shape[0]), int(shape[1]))].append(index)

        for (length, _), indices in indices_by_shape.items():
            chunk_a = [seqs_a[index] for index in indices]
            chunk_b = [seqs_b[index] for index in indices]
            if normalize:
                chunk_a = [_normalize_sequence(seq) for seq in chunk_a]
                chunk_b = [_normalize_sequence(seq) for seq in chunk_b]

            costs = torch.cdist(torch.stack(chunk_a), torch.stack(chunk_b))
            costs_np = costs.detach().cpu().numpy()
            for local_index, output_index in enumerate(indices):
                row_indices, col_indices = linear_sum_assignment(costs_np[local_index])
                matched_cost = costs_np[local_index, row_indices, col_indices].sum()
                distances[output_index] = float(matched_cost) / (2 * length)

    return distances.to(device=device)


def dtw_distance_shuffled(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    *,
    pair_id: tuple[object, object],
    n_perms: int = 10,
    base_seed: int = 42,
    normalize: bool = True,
) -> float:
    """Symmetric order-ablation control for one unordered pair.

    Runs the identical DTW machinery as :func:`dtw_distance` -- same normalization,
    cost matrix, warp tolerance, and endpoint anchoring -- but randomly permutes each
    sequence in turn and averages the two one-sided distances over ``n_perms``
    independently-seeded draws. Averaging both directions makes the control invariant to
    the arbitrary ordering of an unordered driving pair. Because
    ``_normalize_sequence`` is a per-feature statistic over the *set* of values across
    time, it commutes with the permutation, so the only thing that differs from
    ``dtw_distance(seq1, seq2, normalize)`` is the temporal arrangement the monotonic
    path sees -- not the cost function, scale, or alignment tolerance.

    Args:
        pair_id: two stable, content-based endpoint identifiers, e.g.
            ``((session_id, start_frame), (session_id, start_frame))``. The endpoints
            are canonicalized before hashing, so reversing the pair does not select new
            draws. The canonical tuple is combined with the permutation index via
            ``repr`` (not string concatenation), so separator-like characters in IDs
            remain unambiguous.
        n_perms: number of independent permutation draws to average over.
        base_seed: seed namespace used to select a reproducible set of draws.

    Returns:
        Mean DTW distance (lower = more similar) over the ``n_perms`` shuffled draws.
    """
    if n_perms < 1:
        raise ValueError("n_perms must be positive")

    canonical_pair_id = _canonical_pair_id(pair_id)
    dists = []
    for k in range(n_perms):
        perm1 = _shuffle_permutation(
            seq1.shape[0],
            pair_id=canonical_pair_id,
            permutation_index=k,
            base_seed=base_seed,
        )
        perm2 = _shuffle_permutation(
            seq2.shape[0],
            pair_id=canonical_pair_id,
            permutation_index=k,
            base_seed=base_seed,
        )
        shuffle_first = dtw_distance(seq1[perm1], seq2, normalize=normalize)
        shuffle_second = dtw_distance(seq1, seq2[perm2], normalize=normalize)
        dists.append((shuffle_first + shuffle_second) / 2)
    return sum(dists) / len(dists)


def dtw_distance_batch_shuffled(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    *,
    pair_ids: Sequence[tuple[object, object]],
    n_perms: int = 10,
    base_seed: int = 42,
    normalize: bool = True,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Batched order-ablation control (see :func:`dtw_distance_shuffled`).

    For GPU-scale pair counts (HDD/nuScenes have thousands of within-cluster pairs,
    unlike SoccerNet's smaller per-match galleries), permutation materialization and
    DTW are both bounded by ``chunk_size`` rather than retaining shuffled copies for
    the full pair set.

    Args:
        pair_ids: one injective, content-based id per pair (see
            :func:`dtw_distance_shuffled`) -- length must match ``seqs_a``/``seqs_b``.
        n_perms: number of paired permutation draws to average over.
        base_seed: seed namespace used to select a reproducible set of draws.

    Returns:
        (N,) tensor of mean DTW distance per pair, averaged over ``n_perms`` draws.
    """
    if n_perms < 1:
        raise ValueError("n_perms must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    assert len(seqs_a) == len(seqs_b) == len(pair_ids), (
        "seqs_a, seqs_b, and pair_ids must have matching length"
    )
    n = len(seqs_a)
    if n == 0:
        return torch.tensor([])
    canonical_pair_ids = [_canonical_pair_id(pair_id) for pair_id in pair_ids]
    acc = torch.zeros(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_a = seqs_a[start:end]
        chunk_b = seqs_b[start:end]
        chunk_pair_ids = canonical_pair_ids[start:end]
        chunk_acc = torch.zeros(end - start)

        for k in range(n_perms):
            shuffled_a = []
            shuffled_b = []
            for seq_a, seq_b, pair_id in zip(chunk_a, chunk_b, chunk_pair_ids):
                perm_a = _shuffle_permutation(
                    seq_a.shape[0],
                    pair_id=pair_id,
                    permutation_index=k,
                    base_seed=base_seed,
                )
                perm_b = _shuffle_permutation(
                    seq_b.shape[0],
                    pair_id=pair_id,
                    permutation_index=k,
                    base_seed=base_seed,
                )
                shuffled_a.append(seq_a[perm_a])
                shuffled_b.append(seq_b[perm_b])
            dists_first = dtw_distance_batch(
                shuffled_a, chunk_b, normalize=normalize, chunk_size=chunk_size
            )
            dists_second = dtw_distance_batch(
                chunk_a, shuffled_b, normalize=normalize, chunk_size=chunk_size
            )
            chunk_acc += ((dists_first + dists_second) / 2).cpu()
        acc[start:end] = chunk_acc
    return acc / n_perms

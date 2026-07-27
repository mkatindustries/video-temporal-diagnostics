"""GPU-vectorized Dynamic Time Warping using anti-diagonal wavefront processing.

The DTW recurrence dtw[i,j] = cost[i-1,j-1] + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
has the property that all cells on the same anti-diagonal (i+j = d) are independent.
For a T1xT2 matrix this reduces T1*T2 serial Python iterations to T1+T2 vectorized
torch ops. Batching N pairs into (N, T1, T2) processes all pairs simultaneously.
"""

import hashlib

import torch


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


def dtw_distance_shuffled(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    *,
    pair_id: object,
    n_perms: int = 10,
    normalize: bool = True,
) -> float:
    """Order-ablation control for one pair.

    Runs the identical DTW machinery as :func:`dtw_distance` -- same normalization,
    cost matrix, warp tolerance, and endpoint anchoring -- but with ``seq2``'s time axis
    randomly permuted, averaged over ``n_perms`` independently-seeded draws. Because
    ``_normalize_sequence`` is a per-feature statistic over the *set* of values across
    time, it commutes with the permutation, so the only thing that differs from
    ``dtw_distance(seq1, seq2, normalize)`` is the temporal arrangement the monotonic
    path sees -- not the cost function, scale, or alignment tolerance.

    Args:
        pair_id: must be injective across pairs -- a tuple of stable, content-based
            identifiers (e.g. ``(session_id, start_frame, session_id, start_frame)``),
            not raw list indices that can be reordered across runs. Combined with the
            permutation index via ``repr`` (not string concatenation) so ids containing
            arbitrary separator characters can't collide -- this is the exact bug class
            the SoccerNet shuffled-DTW round-4 review caught and fixed.
        n_perms: number of independent permutation draws to average over.

    Returns:
        Mean DTW distance (lower = more similar) over the ``n_perms`` shuffled draws.
    """
    dists = []
    for k in range(n_perms):
        seed = int.from_bytes(
            hashlib.sha256(repr((pair_id, k)).encode()).digest()[:8], "little"
        ) % (2**63 - 1)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(seq2.shape[0], generator=gen)
        dists.append(dtw_distance(seq1, seq2[perm], normalize=normalize))
    return sum(dists) / len(dists)


def dtw_distance_batch_shuffled(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    *,
    pair_ids: list,
    n_perms: int = 10,
    normalize: bool = True,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Batched order-ablation control (see :func:`dtw_distance_shuffled`).

    For GPU-scale pair counts (HDD/nuScenes have thousands of within-cluster pairs,
    unlike SoccerNet's smaller per-match galleries), this processes all pairs at once
    per permutation draw via :func:`dtw_distance_batch`, rather than looping per pair.

    Args:
        pair_ids: one injective, content-based id per pair (see
            :func:`dtw_distance_shuffled`) -- length must match ``seqs_a``/``seqs_b``.

    Returns:
        (N,) tensor of mean DTW distance per pair, averaged over ``n_perms`` draws.
    """
    assert len(seqs_a) == len(seqs_b) == len(pair_ids), (
        "seqs_a, seqs_b, and pair_ids must have matching length"
    )
    n = len(seqs_a)
    if n == 0:
        return torch.tensor([])
    acc = torch.zeros(n)
    for k in range(n_perms):
        shuffled_b = []
        for i, seq in enumerate(seqs_b):
            seed = int.from_bytes(
                hashlib.sha256(repr((pair_ids[i], k)).encode()).digest()[:8], "little"
            ) % (2**63 - 1)
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(seq.shape[0], generator=gen)
            shuffled_b.append(seq[perm])
        dists = dtw_distance_batch(
            seqs_a, shuffled_b, normalize=normalize, chunk_size=chunk_size
        )
        acc += dists.cpu()
    return acc / n_perms

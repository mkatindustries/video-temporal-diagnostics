"""GPU-vectorized Dynamic Time Warping using anti-diagonal wavefront processing.

The DTW recurrence dtw[i,j] = cost[i-1,j-1] + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
has the property that all cells on the same anti-diagonal (i+j = d) are independent.
For a T1xT2 matrix this reduces T1*T2 serial Python iterations to T1+T2 vectorized
torch ops. Batching N pairs into (N, T1, T2) processes all pairs simultaneously.
"""

import torch
import torch.nn.functional as F


def _normalize_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Normalize sequence to [0, 1] per dimension."""
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
                dtw[:, i_idx - 1, j_idx],      # from above
                dtw[:, i_idx, j_idx - 1],       # from left
            ),
            dtw[:, i_idx - 1, j_idx - 1],       # from diagonal
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
        normalize: Normalize sequences to [0, 1] before comparing.

    Returns:
        DTW distance normalized by path length (lower = more similar).
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
        normalize: Normalize sequences to [0, 1] before comparing.
        chunk_size: Max pairs per GPU batch (controls memory usage).

    Returns:
        (N,) tensor of DTW distances normalized by path length.
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
                chunk_a[i].unsqueeze(0), chunk_b[i].unsqueeze(0),
            ).squeeze(0)

        dists = _wavefront_dtw_batch(cost, lens1, lens2)

        # Normalize by path length
        path_lengths = (lens1 + lens2).float()
        dists = dists / path_lengths

        all_dists.append(dists)

    return torch.cat(all_dists)

"""Temporal scramble gradient diagnostic.

Chunk-shuffles embeddings at increasing granularity (K chunks) and measures
retrieval performance at each level.  Order-invariant methods stay flat;
sequence-aware methods degrade monotonically.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor


def scramble_embeddings(embeddings: Tensor, n_chunks: int, seed: int = 0) -> Tensor:
    """Chunk-shuffle embeddings along the time axis.

    Args:
        embeddings: ``(T, D)`` tensor of frame-level embeddings.
        n_chunks: Number of equal chunks to split into before shuffling.
            When ``n_chunks <= 1`` the input is returned unchanged.
        seed: Random seed for reproducible permutation.

    Returns:
        Tensor of the same shape with chunks reordered.
    """
    if n_chunks <= 1:
        return embeddings

    n_chunks = min(n_chunks, embeddings.shape[0])
    chunks = list(torch.tensor_split(embeddings, n_chunks, dim=0))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(chunks))
    shuffled = [chunks[p] for p in perm]

    return torch.cat(shuffled, dim=0)


def scramble_gradient(
    embeddings_a: dict[str, Tensor],
    embeddings_b: dict[str, Tensor],
    pairs: list[tuple[str, str, int]],
    similarity_fn: Callable[[Tensor, Tensor], float],
    k_values: list[int] | tuple[int, ...] = (1, 4, 16),
    seed: int = 0,
) -> dict:
    """Run the temporal scramble gradient and return AP at each K.

    For each value of *K*, the B-side embeddings are chunk-shuffled into *K*
    pieces while A-side embeddings stay intact.  Retrieval AP is computed at
    each level.

    Args:
        embeddings_a: ``{video_id: (T, D)}`` reference embeddings.
        embeddings_b: ``{video_id: (T, D)}`` query embeddings (shuffled).
        pairs: ``[(id_a, id_b, label), ...]`` where *label* is 1 (match)
            or 0 (non-match).
        similarity_fn: ``(Tensor, Tensor) -> float`` pairwise similarity.
        k_values: Chunk counts to sweep.  ``K=1`` is the unscrambled baseline.
        seed: Base seed; per-video seeds are derived deterministically.

    Returns:
        Dict with ``"k_values"``, ``"ap_scores"``, and ``"verdict"``
        (``"no-detected-sensitivity"`` or ``"order-sensitive"``).
    """
    k_values = sorted(k_values)
    ap_scores: list[float] = []

    for k in k_values:
        scores: list[float] = []
        labels: list[int] = []

        for id_a, id_b, label in pairs:
            if id_a not in embeddings_a or id_b not in embeddings_b:
                continue

            emb_a = embeddings_a[id_a]
            emb_b = embeddings_b[id_b]

            # Deterministic per-video seed
            vid_seed = int(hashlib.md5(f"{id_b}_{k}_{seed}".encode()).hexdigest(), 16) % (2**31)
            emb_b_scrambled = scramble_embeddings(emb_b, n_chunks=k, seed=vid_seed)

            scores.append(similarity_fn(emb_a, emb_b_scrambled))
            labels.append(label)

        if not scores:
            ap_scores.append(float("nan"))
            continue

        ap_scores.append(float(average_precision_score(labels, scores)))

    # A flat finite-sample curve does not prove architectural invariance.
    if len(ap_scores) >= 2 and not any(np.isnan(ap_scores)) and 1 in k_values:
        baseline_idx = k_values.index(1)
        drop = ap_scores[baseline_idx] - ap_scores[-1]
        verdict = "order-sensitive" if drop > 0.05 else "no-detected-sensitivity"
    else:
        drop = float("nan")
        verdict = "inconclusive"

    return {
        "k_values": k_values,
        "ap_scores": ap_scores,
        "absolute_ap_drop": drop,
        "verdict": verdict,
    }

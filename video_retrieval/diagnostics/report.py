"""Combined temporal diagnostic report.

Runs both the scramble gradient and reversal sensitivity tests and returns
a single JSON-serializable summary.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .reversal import compute_s_rev
from .scramble import scramble_gradient


def temporal_report(
    embeddings_a: dict[str, Tensor],
    embeddings_b: dict[str, Tensor],
    pairs: list[tuple[str, str, int]],
    similarity_fn: Callable[[Tensor, Tensor], float],
    k_values: list[int] | tuple[int, ...] = (1, 4, 16),
    seed: int = 0,
) -> dict:
    """Run scramble gradient + s_rev and return a combined report.

    Args:
        embeddings_a: ``{video_id: (T, D)}`` reference embeddings.
        embeddings_b: ``{video_id: (T, D)}`` query embeddings.
        pairs: ``[(id_a, id_b, label), ...]``.
        similarity_fn: ``(Tensor, Tensor) -> float``.
        k_values: Chunk counts for the scramble gradient.
        seed: Random seed.

    Returns:
        JSON-serializable dict with ``"scramble_gradient"`` and
        ``"reversal_sensitivity"`` sections.
    """
    scramble = scramble_gradient(
        embeddings_a,
        embeddings_b,
        pairs,
        similarity_fn,
        k_values=k_values,
        seed=seed,
    )

    # Run s_rev on the union of both embedding dicts
    all_embeddings = {**embeddings_a, **embeddings_b}
    s_rev = compute_s_rev(all_embeddings, similarity_fn)

    return {
        "scramble_gradient": scramble,
        "reversal_sensitivity": {
            "mean": s_rev["mean"],
            "std": s_rev["std"],
            "n_videos": len(s_rev["per_video"]),
        },
    }

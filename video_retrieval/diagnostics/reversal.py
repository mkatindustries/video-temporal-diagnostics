"""Reversal sensitivity (s_rev) diagnostic.

Measures how similar a video's embedding is to its time-reversed version.
s_rev close to 1.0 means the method is blind to temporal reversal;
s_rev well below 1.0 means it can distinguish forward from backward.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import Tensor


def compute_s_rev(
    embeddings: dict[str, Tensor],
    similarity_fn: Callable[[Tensor, Tensor], float],
) -> dict:
    """Compute reversal sensitivity per video.

    For each video, compares the original ``(T, D)`` embedding sequence with
    its time-reversed version using *similarity_fn*.

    Args:
        embeddings: ``{video_id: (T, D)}`` frame-level embeddings.
        similarity_fn: ``(Tensor, Tensor) -> float`` pairwise similarity
            (higher = more similar).

    Returns:
        Dict with ``"mean"``, ``"std"``, and ``"per_video"``
        (``{video_id: s_rev_value}``).
    """
    per_video: dict[str, float] = {}

    for vid, emb in embeddings.items():
        emb_rev = emb.flip(0)
        per_video[vid] = similarity_fn(emb, emb_rev)

    values = list(per_video.values())

    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "std": float(np.std(values)) if values else float("nan"),
        "per_video": per_video,
    }

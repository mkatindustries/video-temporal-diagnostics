"""Diagnostic toolkit for temporal sensitivity in video retrieval embeddings.

Provides two reusable diagnostics:
- **Scramble gradient**: chunk-shuffle embeddings at increasing K to separate
  order-invariant from sequence-aware methods.
- **Reversal sensitivity (s_rev)**: forward vs. reversed similarity per video.

Quick start::

    from video_retrieval.diagnostics import temporal_report

    report = temporal_report(emb_a, emb_b, pairs, similarity_fn)
"""

from .reversal import compute_s_rev
from .report import temporal_report
from .scramble import scramble_embeddings, scramble_gradient

__all__ = [
    "compute_s_rev",
    "scramble_embeddings",
    "scramble_gradient",
    "temporal_report",
]

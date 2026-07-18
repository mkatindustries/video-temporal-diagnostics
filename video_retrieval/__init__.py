"""Video retrieval using non-semantic fingerprints.

This package explores video deduplication and retrieval using signals
that capture HOW a video moves through content space, rather than
WHAT semantic content it contains.

Key components:
- models: Video encoders (DINOv3, v-JEPA)
- fingerprints: Non-semantic fingerprinting methods
- diagnostics: Temporal sensitivity diagnostic toolkit
- utils: Video loading and processing utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .diagnostics import (
    compute_s_rev,
    feature_comparator_decomposition,
    scramble_embeddings,
    scramble_gradient,
    temporal_report,
)

if TYPE_CHECKING:
    from .fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
    from .models.dinov3 import DINOv3Encoder
    from .utils import extract_frames, load_video

__all__ = [
    "DINOv3Encoder",
    "TemporalDerivativeFingerprint",
    "TrajectoryFingerprint",
    "compute_s_rev",
    "feature_comparator_decomposition",
    "scramble_embeddings",
    "scramble_gradient",
    "temporal_report",
    "load_video",
    "extract_frames",
]


def __getattr__(name: str) -> Any:
    """Load model and video dependencies only when their symbols are requested."""
    if name == "DINOv3Encoder":
        from .models.dinov3 import DINOv3Encoder

        return DINOv3Encoder
    if name in {"TemporalDerivativeFingerprint", "TrajectoryFingerprint"}:
        from .fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint

        return {
            "TemporalDerivativeFingerprint": TemporalDerivativeFingerprint,
            "TrajectoryFingerprint": TrajectoryFingerprint,
        }[name]
    if name in {"load_video", "extract_frames"}:
        from .utils import extract_frames, load_video

        return {"load_video": load_video, "extract_frames": extract_frames}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

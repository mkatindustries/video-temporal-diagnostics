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

from .models.dinov3 import DINOv3Encoder
from .fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
from .diagnostics import compute_s_rev, scramble_embeddings, scramble_gradient, temporal_report
from .utils import load_video, extract_frames

__all__ = [
    "DINOv3Encoder",
    "TemporalDerivativeFingerprint",
    "TrajectoryFingerprint",
    "compute_s_rev",
    "scramble_embeddings",
    "scramble_gradient",
    "temporal_report",
    "load_video",
    "extract_frames",
]

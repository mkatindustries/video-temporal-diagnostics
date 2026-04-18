"""Video retrieval using non-semantic fingerprints.

This package explores video deduplication and retrieval using signals
that capture HOW a video moves through content space, rather than
WHAT semantic content it contains.

Key components:
- models: Video encoders (DINOv3, v-JEPA)
- fingerprints: Non-semantic fingerprinting methods
- utils: Video loading and processing utilities
"""

from .models.dinov3 import DINOv3Encoder
from .fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
from .utils import load_video, extract_frames

__all__ = [
    "DINOv3Encoder",
    "TemporalDerivativeFingerprint",
    "TrajectoryFingerprint",
    "load_video",
    "extract_frames",
]

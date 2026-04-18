"""Temporal Derivative Fingerprinting.

The key insight: two videos may share semantic content (e.g., Central Park),
but the *sequence of transitions* between frames is unique. By computing
the temporal derivative of embeddings (how the embedding changes frame-to-frame),
we capture motion and trajectory information independent of static content.

Example:
- Cyclist A: Harlem → Central Park → Financial District
- Cyclist B: Chelsea → Central Park → Queens

Both have "Central Park" embeddings, but:
- A's derivative shows: urban→park transition, then park→downtown transition
- B's derivative shows: urban→park transition, then park→residential transition

These derivative patterns are distinct fingerprints.
"""

import torch
import torch.nn.functional as F

from video_retrieval.fingerprints.dtw import dtw_distance as _dtw_distance_impl


def _dtw_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """Compute DTW distance between two sequences (unnormalized coordinates)."""
    return _dtw_distance_impl(seq1, seq2, normalize=False)


class TemporalDerivativeFingerprint:
    """Generate fingerprints from temporal derivatives of video embeddings.

    Captures how a video *moves through* embedding space, not just where it is.
    Uses DTW for sequence comparison to preserve temporal structure.
    """

    def __init__(
        self,
        derivative_order: int = 1,
        window_size: int = 1,
        normalize_derivatives: bool = True,
    ):
        """Initialize fingerprinter.

        Args:
            derivative_order: Order of temporal derivative (1=velocity, 2=acceleration).
            window_size: Frames to skip when computing derivatives (larger = coarser motion).
            normalize_derivatives: L2-normalize derivative vectors.
        """
        self.derivative_order = derivative_order
        self.window_size = window_size
        self.normalize_derivatives = normalize_derivatives

    def compute_derivatives(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal derivatives of embeddings.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Derivatives tensor (T-window_size*order, D).
        """
        derivatives = embeddings

        for _ in range(self.derivative_order):
            if derivatives.shape[0] <= self.window_size:
                break
            # d[t] = embed[t + window] - embed[t]
            derivatives = derivatives[self.window_size:] - derivatives[:-self.window_size]

        if self.normalize_derivatives and derivatives.shape[0] > 0:
            derivatives = F.normalize(derivatives, p=2, dim=1)

        return derivatives

    def compute_fingerprint(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fingerprint from video embeddings.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Derivative sequence (T-k, D) - the fingerprint IS the sequence.
        """
        return self.compute_derivatives(embeddings)

    def compare(
        self,
        fp1: torch.Tensor,
        fp2: torch.Tensor,
    ) -> float:
        """Compare two fingerprints using DTW.

        Args:
            fp1, fp2: Derivative sequences from compute_fingerprint.

        Returns:
            Similarity score in [0, 1] (higher = more similar).
        """
        if fp1.shape[0] == 0 or fp2.shape[0] == 0:
            return 0.0

        distance = _dtw_distance(fp1, fp2)
        # Convert distance to similarity (0 = identical, higher = more different)
        # Use exponential decay for bounded similarity
        return float(torch.exp(torch.tensor(-distance)).item())


class MultiScaleDerivativeFingerprint:
    """Compute temporal derivatives at multiple time scales.

    Different motions operate at different temporal scales:
    - Fine-grained: camera shake, small movements (1-2 frames)
    - Medium: scene transitions (5-10 frames)
    - Coarse: narrative/journey progression (30+ frames)

    Combining multiple scales captures motion at all levels.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
    ):
        """Initialize multi-scale fingerprinter.

        Args:
            window_sizes: List of window sizes for derivative computation.
        """
        self.window_sizes = window_sizes or [1, 5, 15]
        self.fingerprinters = [
            TemporalDerivativeFingerprint(
                derivative_order=1,
                window_size=ws,
            )
            for ws in self.window_sizes
        ]

    def compute_fingerprint(
        self,
        embeddings: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Compute multi-scale fingerprint.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Dict mapping window_size -> derivative sequence.
        """
        result = {}

        for ws, fp in zip(self.window_sizes, self.fingerprinters):
            if embeddings.shape[0] > ws:
                result[ws] = fp.compute_fingerprint(embeddings)

        return result

    def compare(
        self,
        fp1: dict[int, torch.Tensor],
        fp2: dict[int, torch.Tensor],
    ) -> float:
        """Compare multi-scale fingerprints.

        Averages DTW similarity across all shared scales.

        Args:
            fp1, fp2: Multi-scale fingerprints.

        Returns:
            Similarity score in [0, 1].
        """
        similarities = []

        for ws in self.window_sizes:
            if ws in fp1 and ws in fp2:
                sim = self.fingerprinters[self.window_sizes.index(ws)].compare(
                    fp1[ws], fp2[ws]
                )
                similarities.append(sim)

        if not similarities:
            return 0.0

        return sum(similarities) / len(similarities)

"""Trajectory-based fingerprinting using attention centroids.

The key insight: DINOv3's attention heads track salient objects and regions.
The trajectory of attention centroids over time encodes motion patterns
that are independent of what specific content is being attended to.

Example:
- Cyclist A in NYC: attention tracks cyclist moving left-to-right across frame
- Cyclist B in Paris: attention tracks cyclist moving left-to-right across frame
- Both have similar attention trajectories despite different semantic content

This makes attention trajectories a non-semantic motion fingerprint.
"""

import torch
import torch.nn.functional as F

from video_retrieval.fingerprints.dtw import dtw_distance


class TrajectoryFingerprint:
    """Generate fingerprints from attention centroid trajectories.

    Uses DTW for sequence comparison to preserve temporal structure.
    The trajectory IS the fingerprint - no lossy aggregation to statistics.
    """

    def __init__(
        self,
        smoothing_window: int = 3,
        compute_velocity: bool = True,
    ):
        """Initialize trajectory fingerprinter.

        Args:
            smoothing_window: Window for smoothing trajectory (reduces noise).
            compute_velocity: Also include velocity in the fingerprint.
        """
        self.smoothing_window = smoothing_window
        self.compute_velocity = compute_velocity

    def smooth_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply moving average smoothing to trajectory.

        Args:
            trajectory: Attention centroids (T, 2) where 2 is (x, y).

        Returns:
            Smoothed trajectory.
        """
        if self.smoothing_window <= 1 or trajectory.shape[0] < self.smoothing_window:
            return trajectory

        # Simple moving average using unfold
        T, D = trajectory.shape
        smoothed = []

        for d in range(D):
            signal = trajectory[:, d]
            # Pad signal
            pad_size = self.smoothing_window // 2
            padded = F.pad(signal.unsqueeze(0), (pad_size, pad_size), mode="replicate").squeeze(0)
            # Moving average
            kernel = torch.ones(self.smoothing_window, device=trajectory.device) / self.smoothing_window
            smoothed_d = F.conv1d(
                padded.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
            ).squeeze()
            smoothed.append(smoothed_d[:T])

        return torch.stack(smoothed, dim=1)

    def compute_fingerprint(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute fingerprint from attention centroid trajectory.

        Args:
            trajectory: Attention centroids (T, 2) from DINOv3Encoder.get_attention_centroids().

        Returns:
            Fingerprint tensor (T, D) where D=2 (position) or D=4 (position + velocity).
        """
        smoothed = self.smooth_trajectory(trajectory)

        if self.compute_velocity and smoothed.shape[0] > 1:
            # Compute velocity (first derivative)
            velocity = smoothed[1:] - smoothed[:-1]
            # Align lengths by dropping first position
            positions = smoothed[1:]
            # Concatenate position and velocity
            return torch.cat([positions, velocity], dim=1)
        else:
            return smoothed

    def compare(
        self,
        fp1: torch.Tensor,
        fp2: torch.Tensor,
    ) -> float:
        """Compare two trajectory fingerprints using DTW.

        Args:
            fp1, fp2: Fingerprints from compute_fingerprint.

        Returns:
            Similarity score in [0, 1] (higher = more similar).
        """
        if fp1.shape[0] == 0 or fp2.shape[0] == 0:
            return 0.0

        distance = dtw_distance(fp1.clone(), fp2.clone(), normalize=True)
        # Convert distance to similarity
        return float(torch.exp(torch.tensor(-distance * 5)).item())


class DTWTrajectoryMatcher:
    """Direct DTW comparison of trajectories.

    Use this for simple, direct trajectory comparison without
    additional processing (smoothing, velocity, etc.).
    """

    def __init__(self, normalize: bool = True):
        """Initialize DTW matcher.

        Args:
            normalize: Normalize trajectories to [0, 1] before matching.
        """
        self.normalize = normalize

    def compare(self, traj1: torch.Tensor, traj2: torch.Tensor) -> float:
        """Compare two trajectories using DTW.

        Args:
            traj1, traj2: Trajectories (T, 2).

        Returns:
            Similarity score in [0, 1] (higher = more similar).
        """
        if traj1.shape[0] == 0 or traj2.shape[0] == 0:
            return 0.0

        distance = dtw_distance(traj1.clone(), traj2.clone(), normalize=self.normalize)
        return 1.0 / (1.0 + distance)

"""Synthetic fingerprint tests: verify fingerprints capture trajectory differences.

Creates synthetic "videos" (sequences of embeddings/centroids) with known
motion patterns and asserts that fingerprints correctly distinguish them.
"""

import numpy as np
import pytest
import torch

from video_retrieval.fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
from video_retrieval.fingerprints.trajectory import DTWTrajectoryMatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_synthetic_trajectory(
    pattern: str,
    num_frames: int = 100,
    noise_level: float = 0.02,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate synthetic attention centroid trajectory.

    Args:
        pattern: Type of motion pattern.
        num_frames: Number of frames.
        noise_level: Gaussian noise to add.
        device: Device for tensor.

    Returns:
        Trajectory tensor (T, 2).
    """
    t = torch.linspace(0, 1, num_frames).to(device)

    if pattern == "linear_lr":
        # Linear left-to-right motion
        x = t
        y = torch.ones_like(t) * 0.5

    elif pattern == "linear_rl":
        # Linear right-to-left motion
        x = 1 - t
        y = torch.ones_like(t) * 0.5

    elif pattern == "linear_tb":
        # Linear top-to-bottom motion
        x = torch.ones_like(t) * 0.5
        y = t

    elif pattern == "circular_cw":
        # Circular clockwise motion
        x = 0.5 + 0.3 * torch.cos(2 * np.pi * t)
        y = 0.5 + 0.3 * torch.sin(2 * np.pi * t)

    elif pattern == "circular_ccw":
        # Circular counter-clockwise motion
        x = 0.5 + 0.3 * torch.cos(-2 * np.pi * t)
        y = 0.5 + 0.3 * torch.sin(-2 * np.pi * t)

    elif pattern == "zigzag":
        # Zigzag pattern
        x = t
        y = 0.5 + 0.2 * torch.sin(8 * np.pi * t)

    elif pattern == "stationary":
        # Mostly stationary with slight drift
        x = torch.ones_like(t) * 0.5 + 0.05 * t
        y = torch.ones_like(t) * 0.5 + 0.02 * torch.sin(2 * np.pi * t)

    elif pattern == "random_walk":
        # Random walk
        steps = torch.randn(num_frames, 2).to(device) * 0.03
        trajectory = torch.cumsum(steps, dim=0)
        trajectory = trajectory - trajectory.mean(dim=0)  # Center
        trajectory = trajectory / trajectory.abs().max() * 0.4 + 0.5  # Normalize to [0.1, 0.9]
        return trajectory + torch.randn_like(trajectory) * noise_level

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    trajectory = torch.stack([x, y], dim=1)

    # Add noise
    if noise_level > 0:
        trajectory = trajectory + torch.randn_like(trajectory) * noise_level

    return trajectory


def generate_synthetic_embeddings(
    trajectory: torch.Tensor,
    embedding_dim: int = 1024,
    noise_level: float = 0.1,
) -> torch.Tensor:
    """Generate synthetic embeddings that follow a trajectory in embedding space.

    The embeddings smoothly transition along a path in high-dimensional space,
    with the trajectory controlling the direction of motion.

    Args:
        trajectory: 2D trajectory (T, 2).
        embedding_dim: Dimension of embeddings.
        noise_level: Noise to add.

    Returns:
        Embeddings tensor (T, embedding_dim).
    """
    T = trajectory.shape[0]
    device = trajectory.device

    # Create a few random "anchor" embeddings
    num_anchors = 4
    anchors = torch.randn(num_anchors, embedding_dim).to(device)
    anchors = torch.nn.functional.normalize(anchors, dim=1)

    # Interpolate between anchors based on trajectory position
    # Map trajectory x,y to weights over anchors
    embeddings = []
    for i in range(T):
        x, y = trajectory[i]
        # Simple mapping: 4 corners of unit square to 4 anchors
        w00 = (1 - x) * (1 - y)
        w10 = x * (1 - y)
        w01 = (1 - x) * y
        w11 = x * y
        weights = torch.tensor([w00, w10, w01, w11]).to(device)
        weights = weights / weights.sum()

        emb = (weights.unsqueeze(1) * anchors).sum(dim=0)
        emb = torch.nn.functional.normalize(emb, dim=0)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)

    # Add noise
    if noise_level > 0:
        embeddings = embeddings + torch.randn_like(embeddings) * noise_level
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    return embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATTERNS = [
    "linear_lr",
    "linear_rl",
    "linear_tb",
    "circular_cw",
    "circular_ccw",
    "zigzag",
    "stationary",
]


@pytest.fixture(scope="module")
def trajectories():
    """Generate all synthetic trajectories once per module."""
    torch.manual_seed(42)
    return {p: generate_synthetic_trajectory(p, num_frames=100) for p in PATTERNS}


@pytest.fixture(scope="module")
def embeddings(trajectories):
    """Generate all synthetic embeddings once per module."""
    torch.manual_seed(42)
    return {p: generate_synthetic_embeddings(trajectories[p]) for p in PATTERNS}


@pytest.fixture(scope="module")
def traj_fingerprints(trajectories):
    """Compute trajectory fingerprints for all patterns."""
    fp = TrajectoryFingerprint()
    return {p: fp.compute_fingerprint(t) for p, t in trajectories.items()}


@pytest.fixture(scope="module")
def deriv_fingerprints(embeddings):
    """Compute temporal derivative fingerprints for all patterns."""
    fp = TemporalDerivativeFingerprint()
    return {p: fp.compute_fingerprint(e) for p, e in embeddings.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyntheticFingerprints:
    """Validate that fingerprints capture trajectory differences."""

    # -- H1: Opposite directions should be dissimilar -----------------------

    def test_h1_trajectory_opposite_directions(self, traj_fingerprints):
        """linear_lr vs linear_rl should be less similar than self-similarity."""
        fp = TrajectoryFingerprint()
        self_sim = fp.compare(
            traj_fingerprints["linear_lr"], traj_fingerprints["linear_lr"]
        )
        opposite_sim = fp.compare(
            traj_fingerprints["linear_lr"], traj_fingerprints["linear_rl"]
        )
        assert opposite_sim < self_sim, (
            f"Opposite-direction similarity ({opposite_sim:.4f}) should be "
            f"lower than self-similarity ({self_sim:.4f})"
        )

    def test_h1_temporal_derivative_opposite_directions(self, deriv_fingerprints):
        """Temporal derivative should also distinguish opposite directions."""
        fp = TemporalDerivativeFingerprint()
        self_sim = fp.compare(
            deriv_fingerprints["linear_lr"], deriv_fingerprints["linear_lr"]
        )
        opposite_sim = fp.compare(
            deriv_fingerprints["linear_lr"], deriv_fingerprints["linear_rl"]
        )
        assert opposite_sim < self_sim, (
            f"Opposite-direction derivative similarity ({opposite_sim:.4f}) "
            f"should be lower than self-similarity ({self_sim:.4f})"
        )

    # -- H2: Similar motion types cluster -----------------------------------

    def test_h2_circular_vs_linear_trajectory(self, traj_fingerprints):
        """circular_cw vs circular_ccw similarity should differ from
        circular_cw vs linear_lr."""
        fp = TrajectoryFingerprint()
        cw_ccw_sim = fp.compare(
            traj_fingerprints["circular_cw"], traj_fingerprints["circular_ccw"]
        )
        cw_linear_sim = fp.compare(
            traj_fingerprints["circular_cw"], traj_fingerprints["linear_lr"]
        )
        assert cw_ccw_sim != pytest.approx(cw_linear_sim, abs=0.01), (
            f"circular_cw vs circular_ccw ({cw_ccw_sim:.4f}) and "
            f"circular_cw vs linear_lr ({cw_linear_sim:.4f}) should differ"
        )

    # -- H3: Stationary vs moving should be very dissimilar -----------------

    def test_h3_stationary_vs_moving(self, traj_fingerprints):
        """Stationary vs zigzag should have low trajectory similarity."""
        fp = TrajectoryFingerprint()
        sim = fp.compare(
            traj_fingerprints["stationary"], traj_fingerprints["zigzag"]
        )
        assert sim < 0.5, (
            f"Stationary vs zigzag similarity ({sim:.4f}) should be low (<0.5)"
        )

    # -- Self-similarity ----------------------------------------------------

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_self_similarity_trajectory(self, pattern, traj_fingerprints):
        """Each pattern's trajectory fingerprint should have high self-similarity."""
        fp = TrajectoryFingerprint()
        sim = fp.compare(traj_fingerprints[pattern], traj_fingerprints[pattern])
        assert sim > 0.8, (
            f"{pattern} trajectory self-similarity ({sim:.4f}) should be >0.8"
        )

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_self_similarity_temporal_derivative(self, pattern, deriv_fingerprints):
        """Each pattern's derivative fingerprint should have high self-similarity."""
        fp = TemporalDerivativeFingerprint()
        sim = fp.compare(deriv_fingerprints[pattern], deriv_fingerprints[pattern])
        assert sim > 0.8, (
            f"{pattern} derivative self-similarity ({sim:.4f}) should be >0.8"
        )

    # -- DTW matcher --------------------------------------------------------

    def test_dtw_distinguishes_opposite_motions(self, trajectories):
        """DTWTrajectoryMatcher should give lower similarity for opposite motions
        than for identical ones."""
        dtw = DTWTrajectoryMatcher()
        self_sim = dtw.compare(trajectories["linear_lr"], trajectories["linear_lr"])
        opposite_sim = dtw.compare(
            trajectories["linear_lr"], trajectories["linear_rl"]
        )
        assert opposite_sim < self_sim, (
            f"DTW opposite-direction similarity ({opposite_sim:.4f}) should be "
            f"lower than self-similarity ({self_sim:.4f})"
        )

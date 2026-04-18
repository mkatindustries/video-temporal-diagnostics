#!/usr/bin/env python3
"""Synthetic test: Validate fingerprints capture trajectory differences.

Creates synthetic "videos" (sequences of embeddings/centroids) with known
motion patterns to verify that fingerprints correctly distinguish them.

This allows testing without needing actual video files.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from video_retrieval.fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
from video_retrieval.fingerprints.trajectory import DTWTrajectoryMatcher


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


def run_synthetic_test():
    """Run tests with synthetic trajectories."""
    print("=" * 70)
    print("Synthetic Trajectory Fingerprint Test")
    print("=" * 70)

    patterns = [
        "linear_lr",
        "linear_rl",
        "linear_tb",
        "circular_cw",
        "circular_ccw",
        "zigzag",
        "stationary",
    ]

    # Generate trajectories
    trajectories = {}
    embeddings = {}
    for pattern in patterns:
        trajectories[pattern] = generate_synthetic_trajectory(pattern, num_frames=100)
        embeddings[pattern] = generate_synthetic_embeddings(trajectories[pattern])

    # Initialize fingerprinters
    traj_fp = TrajectoryFingerprint()
    deriv_fp = TemporalDerivativeFingerprint()
    dtw = DTWTrajectoryMatcher()

    # Compute fingerprints
    traj_fingerprints = {p: traj_fp.compute_fingerprint(t) for p, t in trajectories.items()}
    deriv_fingerprints = {p: deriv_fp.compute_fingerprint(e) for p, e in embeddings.items()}

    # Print similarity matrices
    print("\n1. TRAJECTORY FINGERPRINT SIMILARITY")
    print("-" * 50)
    print_similarity_matrix(patterns, traj_fingerprints, traj_fp.compare)

    print("\n2. TEMPORAL DERIVATIVE SIMILARITY")
    print("-" * 50)
    print_similarity_matrix(patterns, deriv_fingerprints, deriv_fp.compare)

    print("\n3. DTW TRAJECTORY SIMILARITY")
    print("-" * 50)
    print_dtw_matrix(patterns, trajectories, dtw)

    # Test key hypotheses
    print("\n" + "=" * 70)
    print("Hypothesis Tests")
    print("=" * 70)

    # H1: Opposite directions should be dissimilar
    lr_rl_sim = traj_fp.compare(traj_fingerprints["linear_lr"], traj_fingerprints["linear_rl"])
    lr_tb_sim = traj_fp.compare(traj_fingerprints["linear_lr"], traj_fingerprints["linear_tb"])
    print(f"\nH1: Opposite directions dissimilar")
    print(f"  linear_lr vs linear_rl: {lr_rl_sim:.4f}")
    print(f"  linear_lr vs linear_tb: {lr_tb_sim:.4f}")

    # H2: Same motion type (circular) should be more similar to each other
    cw_ccw_sim = traj_fp.compare(
        traj_fingerprints["circular_cw"], traj_fingerprints["circular_ccw"]
    )
    cw_linear_sim = traj_fp.compare(
        traj_fingerprints["circular_cw"], traj_fingerprints["linear_lr"]
    )
    print(f"\nH2: Similar motion types cluster")
    print(f"  circular_cw vs circular_ccw: {cw_ccw_sim:.4f}")
    print(f"  circular_cw vs linear_lr: {cw_linear_sim:.4f}")

    # H3: Stationary vs moving should be very dissimilar
    stat_zigzag_sim = traj_fp.compare(
        traj_fingerprints["stationary"], traj_fingerprints["zigzag"]
    )
    print(f"\nH3: Stationary vs moving dissimilar")
    print(f"  stationary vs zigzag: {stat_zigzag_sim:.4f}")

    # Visualize trajectories
    visualize_trajectories(trajectories)


def print_similarity_matrix(patterns, fingerprints, compare_fn):
    """Print a similarity matrix."""
    n = len(patterns)

    # Header
    print("          ", end="")
    for p in patterns:
        print(f"{p[:8]:>10}", end="")
    print()

    # Rows
    for i, p1 in enumerate(patterns):
        print(f"{p1[:10]:10}", end="")
        for j, p2 in enumerate(patterns):
            sim = compare_fn(fingerprints[p1], fingerprints[p2])
            print(f"{sim:10.3f}", end="")
        print()


def print_dtw_matrix(patterns, trajectories, dtw):
    """Print DTW similarity matrix."""
    n = len(patterns)

    print("          ", end="")
    for p in patterns:
        print(f"{p[:8]:>10}", end="")
    print()

    for i, p1 in enumerate(patterns):
        print(f"{p1[:10]:10}", end="")
        for j, p2 in enumerate(patterns):
            sim = dtw.compare(trajectories[p1], trajectories[p2])
            print(f"{sim:10.3f}", end="")
        print()


def visualize_trajectories(trajectories: dict, save_path: Path | None = None):
    """Plot all trajectories for visual inspection."""
    n = len(trajectories)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for ax, (name, traj) in zip(axes, trajectories.items()):
        traj_np = traj.cpu().numpy()
        ax.plot(traj_np[:, 0], traj_np[:, 1], "b-", alpha=0.7)
        ax.scatter(traj_np[0, 0], traj_np[0, 1], c="green", s=100, marker="o", label="start")
        ax.scatter(traj_np[-1, 0], traj_np[-1, 1], c="red", s=100, marker="x", label="end")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.legend(loc="upper right")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nTrajectory plot saved to: {save_path}")
    else:
        plt.savefig("trajectories.png", dpi=150)
        print("\nTrajectory plot saved to: trajectories.png")

    plt.close()


if __name__ == "__main__":
    run_synthetic_test()

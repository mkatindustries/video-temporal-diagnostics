#!/usr/bin/env python3
"""Visualize DINOv3 attention maps and centroid trajectories."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from video_retrieval.models import DINOv3Encoder
from video_retrieval.utils.video import load_video


def get_attention_maps(
    encoder: DINOv3Encoder,
    frames: list[np.ndarray],
    layer: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract attention maps and centroids from frames.

    Returns:
        Tuple of (attention_maps (N, H, W), centroids (N, 2))
    """
    all_maps = []
    all_centroids = []

    for frame in frames:
        inputs = encoder._preprocess([frame])

        with torch.no_grad():
            outputs = encoder.model(**inputs)

        attentions = outputs.attentions
        attn = attentions[layer]  # (1, heads, N, N)
        attn = attn.mean(dim=1)  # (1, N, N)

        # Get CLS attention over patches
        skip_tokens = 1 + encoder.num_register_tokens
        cls_attn = attn[0, 0, skip_tokens:]  # (num_patches,)

        # Reshape to spatial grid
        num_patches = cls_attn.shape[0]
        h = w = int(num_patches ** 0.5)
        attn_map = cls_attn.view(h, w)

        # Normalize
        attn_map = attn_map / (attn_map.sum() + 1e-8)

        # Compute centroid
        y_coords = torch.arange(h, device=attn_map.device).float()
        x_coords = torch.arange(w, device=attn_map.device).float()
        centroid_y = (attn_map.sum(dim=1) * y_coords).sum() / h
        centroid_x = (attn_map.sum(dim=0) * x_coords).sum() / w

        all_maps.append(attn_map.cpu())
        all_centroids.append(torch.tensor([centroid_x.cpu(), centroid_y.cpu()]))

    return torch.stack(all_maps), torch.stack(all_centroids)


def visualize_attention_frames(
    frames: list[np.ndarray],
    attention_maps: torch.Tensor,
    centroids: torch.Tensor,
    output_path: str = "figures/attention_visualization.png",
    num_frames: int = 6,
):
    """Create visualization showing attention heatmaps on frames."""
    n = min(num_frames, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i, idx in enumerate(indices):
        frame = frames[idx]
        attn_map = attention_maps[idx].numpy()

        # Resize attention map to frame size
        h, w = frame.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize for visualization
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        # Draw centroid
        cx = int(centroids[idx, 0].item() * w)
        cy = int(centroids[idx, 1].item() * h)
        cv2.circle(overlay, (cx, cy), 10, (255, 255, 255), 3)
        cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), 2)

        # Plot original frame
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame {idx}")
        axes[0, i].axis("off")

        # Plot attention overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Attention (centroid: {centroids[idx, 0]:.2f}, {centroids[idx, 1]:.2f})")
        axes[1, i].axis("off")

    plt.suptitle("DINOv3 Attention Maps (CLS token attention over patches)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention visualization to: {output_path}")
    plt.close()


def visualize_trajectory(
    centroids: torch.Tensor,
    output_path: str = "figures/attention_trajectory.png",
):
    """Plot the attention centroid trajectory over time."""
    centroids_np = centroids.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 2D trajectory plot
    ax = axes[0]
    colors = np.linspace(0, 1, len(centroids_np))
    scatter = ax.scatter(centroids_np[:, 0], centroids_np[:, 1], c=colors, cmap="viridis", s=50)
    ax.plot(centroids_np[:, 0], centroids_np[:, 1], "k-", alpha=0.3, linewidth=1)

    # Mark start and end
    ax.scatter(centroids_np[0, 0], centroids_np[0, 1], c="green", s=200, marker="o", label="Start", zorder=5)
    ax.scatter(centroids_np[-1, 0], centroids_np[-1, 1], c="red", s=200, marker="X", label="End", zorder=5)

    ax.set_xlabel("X (horizontal)")
    ax.set_ylabel("Y (vertical)")
    ax.set_title("Attention Centroid Trajectory")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Image coordinates
    ax.legend()
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Time (normalized)")

    # 2. X position over time
    ax = axes[1]
    ax.plot(centroids_np[:, 0], "b-", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("X position")
    ax.set_title("Horizontal Attention Position")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 3. Y position over time
    ax = axes[2]
    ax.plot(centroids_np[:, 1], "r-", linewidth=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y position")
    ax.set_title("Vertical Attention Position")
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Attention Trajectory Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory visualization to: {output_path}")
    plt.close()


def main(video_path: str, sample_rate: int = 5, max_frames: int = 80):
    """Run attention visualization."""
    print("=" * 60)
    print("DINOv3 Attention Visualization")
    print("=" * 60)

    # Load video
    print(f"\nLoading video: {video_path}")
    frames, fps = load_video(video_path, sample_rate=sample_rate, max_frames=max_frames, max_resolution=518)
    print(f"  Loaded {len(frames)} frames")

    # Load encoder
    print("\nLoading DINOv3...")
    encoder = DINOv3Encoder(device="cuda" if torch.cuda.is_available() else "cpu")

    # Extract attention
    print("\nExtracting attention maps...")
    attention_maps, centroids = get_attention_maps(encoder, frames)
    print(f"  Attention maps: {attention_maps.shape}")
    print(f"  Centroids: {centroids.shape}")

    # Create visualizations
    print("\nCreating visualizations...")
    Path("figures").mkdir(exist_ok=True)

    visualize_attention_frames(frames, attention_maps, centroids)
    visualize_trajectory(centroids)

    # Print trajectory stats
    print("\nTrajectory Statistics:")
    print(f"  X range: [{centroids[:, 0].min():.3f}, {centroids[:, 0].max():.3f}]")
    print(f"  Y range: [{centroids[:, 1].min():.3f}, {centroids[:, 1].max():.3f}]")
    print(f"  Net motion: Δx={centroids[-1, 0] - centroids[0, 0]:.3f}, Δy={centroids[-1, 1] - centroids[0, 1]:.3f}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_bicycle.mp4"
    main(video_path)

#!/usr/bin/env python3
"""Experiment: Test temporal derivative and trajectory fingerprints.

This script tests whether non-semantic fingerprints can distinguish
videos with similar semantic content but different trajectories.

Usage:
    python experiments/test_fingerprints.py --video1 path/to/video1.mp4 --video2 path/to/video2.mp4
    python experiments/test_fingerprints.py --video-dir path/to/videos/
"""

import argparse
from pathlib import Path

import torch

from video_retrieval.models.dinov3 import DINOv3Encoder
from video_retrieval.fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
from video_retrieval.fingerprints.temporal_derivative import MultiScaleDerivativeFingerprint
from video_retrieval.fingerprints.trajectory import DTWTrajectoryMatcher
from video_retrieval.utils import load_video


def process_video(
    video_path: Path,
    encoder: DINOv3Encoder,
    device: str = "cuda",
    max_frames: int = 300,
    sample_rate: int = 2,
) -> dict:
    """Process a single video and extract all fingerprints.

    Args:
        video_path: Path to video file.
        encoder: DINOv3 encoder.
        device: Device to use.
        max_frames: Maximum frames to process.
        sample_rate: Sample every Nth frame.

    Returns:
        Dict with embeddings and fingerprints.
    """
    print(f"Processing: {video_path.name}")

    # Load video
    frames, fps = load_video(video_path, max_frames=max_frames, sample_rate=sample_rate)
    print(f"  Loaded {len(frames)} frames at {fps:.1f} FPS")

    # Get embeddings
    embeddings = encoder.encode_video(frames, batch_size=16)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Get attention centroids
    centroids = encoder.get_attention_centroids(frames, batch_size=16)
    print(f"  Centroids shape: {centroids.shape}")

    # Compute temporal derivative fingerprints
    td_fp = TemporalDerivativeFingerprint(
        derivative_order=1,
        window_size=1,
    )
    td_fingerprint = td_fp.compute_fingerprint(embeddings)

    # Multi-scale derivative fingerprint
    ms_fp = MultiScaleDerivativeFingerprint(
        window_sizes=[1, 5, 15],
    )
    ms_fingerprint = ms_fp.compute_fingerprint(embeddings)

    # Trajectory fingerprint
    traj_fp = TrajectoryFingerprint(
        compute_velocity=True,
        smoothing_window=3,
    )
    traj_fingerprint = traj_fp.compute_fingerprint(centroids)

    return {
        "embeddings": embeddings,
        "centroids": centroids,
        "td_fingerprint": td_fingerprint,
        "ms_fingerprint": ms_fingerprint,
        "traj_fingerprint": traj_fingerprint,
        "fps": fps,
        "num_frames": len(frames),
    }


def compare_videos(result1: dict, result2: dict) -> dict:
    """Compare two processed videos using all fingerprint methods.

    Args:
        result1, result2: Outputs from process_video.

    Returns:
        Dict with similarity scores from each method.
    """
    scores = {}

    # Temporal derivative similarity
    td_fp = TemporalDerivativeFingerprint()
    scores["temporal_derivative"] = td_fp.compare(
        result1["td_fingerprint"],
        result2["td_fingerprint"],
    )

    # Multi-scale derivative similarity
    ms_fp = MultiScaleDerivativeFingerprint()
    scores["multiscale_derivative"] = ms_fp.compare(
        result1["ms_fingerprint"],
        result2["ms_fingerprint"],
    )

    # Trajectory fingerprint similarity
    traj_fp = TrajectoryFingerprint()
    scores["trajectory_stats"] = traj_fp.compare(
        result1["traj_fingerprint"],
        result2["traj_fingerprint"],
    )

    # DTW trajectory similarity
    dtw = DTWTrajectoryMatcher(normalize=True)
    scores["trajectory_dtw"] = dtw.compare(
        result1["centroids"],
        result2["centroids"],
    )

    # Baseline: raw embedding similarity (semantic)
    emb1 = result1["embeddings"].mean(dim=0)
    emb2 = result2["embeddings"].mean(dim=0)
    scores["semantic_baseline"] = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()

    return scores


def main():
    parser = argparse.ArgumentParser(description="Test video fingerprinting methods")
    parser.add_argument("--video1", type=Path, help="First video path")
    parser.add_argument("--video2", type=Path, help="Second video path")
    parser.add_argument("--video-dir", type=Path, help="Directory of videos to compare pairwise")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per video")
    parser.add_argument("--sample-rate", type=int, default=2, help="Sample every Nth frame")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to DINOv3 checkpoint",
    )
    args = parser.parse_args()

    # Initialize encoder
    print("Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(
        **({"model_name": str(args.model_path)} if args.model_path else {}),
        device=args.device,
    )
    print("Encoder loaded.")

    if args.video_dir:
        # Process all videos in directory
        video_files = list(args.video_dir.glob("*.mp4")) + list(args.video_dir.glob("*.avi"))
        print(f"Found {len(video_files)} videos")

        results = {}
        for vf in video_files:
            results[vf.name] = process_video(
                vf, encoder, args.device, args.max_frames, args.sample_rate
            )

        # Pairwise comparison
        print("\n" + "=" * 60)
        print("Pairwise Similarities")
        print("=" * 60)

        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                scores = compare_videos(results[name1], results[name2])

                print(f"\n{name1} vs {name2}:")
                for method, score in scores.items():
                    print(f"  {method:25s}: {score:.4f}")

    elif args.video1 and args.video2:
        # Compare two specific videos
        result1 = process_video(
            args.video1, encoder, args.device, args.max_frames, args.sample_rate
        )
        result2 = process_video(
            args.video2, encoder, args.device, args.max_frames, args.sample_rate
        )

        scores = compare_videos(result1, result2)

        print("\n" + "=" * 60)
        print(f"Comparison: {args.video1.name} vs {args.video2.name}")
        print("=" * 60)
        for method, score in scores.items():
            print(f"  {method:25s}: {score:.4f}")

    else:
        parser.print_help()
        print("\nError: Provide either --video1 and --video2, or --video-dir")


if __name__ == "__main__":
    main()

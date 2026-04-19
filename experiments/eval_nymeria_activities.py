#!/usr/bin/env python3
"""Nymeria Egocentric Activity Discrimination Experiment.

Evaluates whether DINOv3 attention trajectory fingerprints can distinguish
different activities performed by the same person in the same environment,
where bag-of-frames fails because the visual background is identical.

The Nymeria dataset (Meta Aria glasses) provides egocentric video from
multi-activity sessions: same person, same day, different activities (actN).
DINOv3 attention should lock onto salient foreground elements (hands, objects),
producing attention trajectories that encode the wearer's action.

Pipeline:
1. Discover multi-activity person-sessions in Nymeria dataset
2. Extract uniformly-spaced 10s segments from each activity clip
3. Compute DINOv3 features (embeddings + attention centroids)
4. Compute V-JEPA 2 features (encoder tokens + temporal prediction residuals)
5. Compare all pairs within each person-session using 6 methods
6. Evaluate discrimination (AP, AUC) between same-activity and
   different-activity pairs

Hypothesis: Bag-of-frames collapses (same room = same embeddings).
Attention trajectory separates (different actions = different gaze patterns).

Usage:
    python experiments/eval_nymeria_activities.py [--device cuda]
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance, dtw_distance_batch
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_DURATION = 10.0    # seconds per segment
SEGMENTS_PER_CLIP = 20     # uniformly spaced segments per activity clip
SKIP_MARGIN = 30.0         # skip first/last 30s (setup/teardown)
MAX_CLIP_DURATION = 3600.0 # skip clips > 1 hour

# V-JEPA 2 constants
DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32   # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256     # 16h × 16w


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ActivitySegment:
    """A 10-second segment from one activity clip."""

    person_session: str    # "20230628_s1_hayley_little" — grouping key
    activity_id: str       # "act1", "act2", etc.
    sequence_name: str     # full directory name
    video_path: str        # path to data.mp4
    start_sec: float       # segment start within video
    end_sec: float         # segment end within video


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base)
    container.close()
    return duration


def discover_sequences(nymeria_dir: Path) -> dict[str, list[tuple[str, str, str, float]]]:
    """Discover multi-activity person-sessions in the Nymeria dataset.

    Scans for directories matching the naming convention:
        {date}_{session}_{first}_{last}_{actN}_{hash}

    Handles double-nested structure: {name}/{name}/data.mp4

    Args:
        nymeria_dir: Path to the Nymeria dataset root.

    Returns:
        Dict mapping person_session -> list of (activity_id, sequence_name,
        video_path, duration) tuples. Only includes multi-activity sessions.
    """
    # Pattern: date_session_first_last_actN_hash
    pattern = re.compile(
        r"^(\d{8}_s\d+_[a-z]+_[a-z]+)_(act\d+)_([a-z0-9]+)$"
    )

    # Collect all sequences grouped by person_session
    all_sequences: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
    skipped_duration = 0
    skipped_missing = 0

    for entry in sorted(nymeria_dir.iterdir()):
        if not entry.is_dir():
            continue

        match = pattern.match(entry.name)
        if not match:
            continue

        person_session = match.group(1)
        activity_id = match.group(2)
        sequence_name = entry.name

        # Handle double-nested structure: {name}/{name}/data.mp4
        video_path = entry / entry.name / "data.mp4"
        if not video_path.exists():
            # Try single-level
            video_path = entry / "data.mp4"
            if not video_path.exists():
                skipped_missing += 1
                continue

        try:
            duration = get_video_duration(str(video_path))
        except Exception:
            skipped_missing += 1
            continue

        if duration > MAX_CLIP_DURATION:
            skipped_duration += 1
            continue

        all_sequences[person_session].append(
            (activity_id, sequence_name, str(video_path), duration)
        )

    if skipped_duration:
        print(f"  Skipped {skipped_duration} clips > {MAX_CLIP_DURATION}s")
    if skipped_missing:
        print(f"  Skipped {skipped_missing} sequences with missing video")

    # Filter: keep only multi-activity person-sessions
    multi_activity = {
        ps: seqs for ps, seqs in all_sequences.items() if len(seqs) >= 2
    }

    return multi_activity


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


def extract_segments(
    sequences: dict[str, list[tuple[str, str, str, float]]],
) -> tuple[list[ActivitySegment], dict[str, list[int]]]:
    """Extract uniformly-spaced segments from each activity clip.

    Args:
        sequences: Dict from discover_sequences().

    Returns:
        Tuple of (segments list, person_to_indices mapping).
    """
    segments: list[ActivitySegment] = []
    person_to_indices: dict[str, list[int]] = defaultdict(list)

    for person_session, seq_list in sorted(sequences.items()):
        for activity_id, sequence_name, video_path, duration in seq_list:
            usable_start = SKIP_MARGIN
            usable_end = duration - SKIP_MARGIN

            if usable_end - usable_start < SEGMENT_DURATION:
                # Clip too short after margins
                continue

            # Place segments uniformly within usable range
            available = usable_end - usable_start - SEGMENT_DURATION
            if SEGMENTS_PER_CLIP <= 1:
                starts = [usable_start]
            else:
                step = available / (SEGMENTS_PER_CLIP - 1)
                starts = [usable_start + i * step for i in range(SEGMENTS_PER_CLIP)]

            for start in starts:
                idx = len(segments)
                segments.append(
                    ActivitySegment(
                        person_session=person_session,
                        activity_id=activity_id,
                        sequence_name=sequence_name,
                        video_path=video_path,
                        start_sec=start,
                        end_sec=start + SEGMENT_DURATION,
                    )
                )
                person_to_indices[person_session].append(idx)

    return segments, dict(person_to_indices)


# ---------------------------------------------------------------------------
# Video clip extraction (PyAV time-based seeking)
# ---------------------------------------------------------------------------


def load_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> list[np.ndarray]:
    """Extract frames from a video clip using time-based seeking.

    Uses PyAV seek() to jump near the clip start, avoiding decoding
    from frame 0 for long egocentric videos.

    Args:
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        target_fps: Target frame rate for extraction.
        max_resolution: Maximum height (preserves aspect ratio).

    Returns:
        List of RGB frames as numpy arrays.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [bad-argument-type]
    time_base = float(stream.time_base)

    # Seek to slightly before start (keyframe-based seeking)
    # pyrefly: ignore [no-matching-overload]
    seek_sec = max(0, start_sec - 1.0)
    seek_pts = int(seek_sec / time_base)
    container.seek(seek_pts, stream=stream)

    # Sample interval in terms of video frames
    sample_interval = video_fps / target_fps

    frames = []
    frames_in_range = 0
    next_sample = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts) * time_base

        if frame_time < start_sec:
            continue
        if frame_time > end_sec:
            break

        if frames_in_range >= next_sample:
            img = frame.to_ndarray(format="rgb24")

            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)
            next_sample += sample_interval

        frames_in_range += 1

    container.close()
    return frames


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def save_feature_cache(features: dict, cache_path: Path) -> None:
    """Save extracted features to disk as a .pt file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_features: dict[str, Any] = {}
    for k, v in features.items():
        if isinstance(v, dict):
            cpu_features[k] = {
                fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv
                for fk, fv in v.items()
            }
        elif isinstance(v, torch.Tensor):
            cpu_features[k] = v.cpu()
        else:
            cpu_features[k] = v
    torch.save(cpu_features, cache_path)
    print(f"  Cache saved to {cache_path}")


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load cached features from disk, or return None if not found."""
    if not cache_path.exists():
        return None
    print(f"  Loading cache from {cache_path}")
    return torch.load(cache_path, weights_only=False)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_clip_features(
    encoder: DINOv3Encoder,
    segments: list[ActivitySegment],
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> dict[int, dict]:
    """Extract DINOv3 features for all activity segments.

    Args:
        encoder: DINOv3 encoder.
        segments: Activity segments to process.
        target_fps: Frame extraction rate.
        max_resolution: Max frame height.

    Returns:
        Dict mapping segment index -> {
            'embeddings': (T, 1024),
            'centroids': (T, 2),
            'mean_emb': (1024,),
        }
    """
    features = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Prefetch first clip
        future = pool.submit(
            load_clip, segments[0].video_path, segments[0].start_sec,
            segments[0].end_sec, target_fps, max_resolution,
        )

        for i, seg in enumerate(tqdm(segments, desc="Extracting clip features")):
            try:
                frames = future.result()

                # Prefetch next while GPU works
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    future = pool.submit(
                        load_clip, next_seg.video_path, next_seg.start_sec,
                        next_seg.end_sec, target_fps, max_resolution,
                    )

                if len(frames) < 3:
                    failed += 1
                    continue

                emb = encoder.encode_frames(frames)
                centroids = encoder.get_attention_centroids(frames)
                mean_emb = F.normalize(emb.mean(dim=0), dim=0)

                features[i] = {
                    "embeddings": emb,
                    "centroids": centroids,
                    "mean_emb": mean_emb,
                }
            except Exception:
                failed += 1
                # Ensure prefetch is still queued for next iteration
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    future = pool.submit(
                        load_clip, next_seg.video_path, next_seg.start_sec,
                        next_seg.end_sec, target_fps, max_resolution,
                    )
                continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


def load_clip_vjepa2(
    video_path: str,
    start_sec: float,
    end_sec: float,
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Dynamically computes fps to sample ~64 frames from the clip duration,
    then pads or truncates to exactly VJEPA2_NUM_FRAMES.

    Args:
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        max_resolution: Maximum height (V-JEPA 2 expects 256).

    Returns:
        List of exactly VJEPA2_NUM_FRAMES RGB frames as numpy arrays.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0

    # Target fps to get ~64 frames from this duration
    target_fps = VJEPA2_NUM_FRAMES / duration

    frames = load_clip(
        video_path,
        start_sec,
        end_sec,
        target_fps=target_fps,
        max_resolution=max_resolution,
    )

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    # Pad by repeating last frame if too few
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])

    # Truncate if too many
    return frames[:VJEPA2_NUM_FRAMES]


def build_temporal_masks(
    n_context_steps: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context/target masks that split along the temporal axis.

    Context = all spatial positions at time steps 0..n_context_steps-1
    Target  = all spatial positions at time steps n_context_steps..T_PATCHES-1

    Returns position-index tensors of shape (1, n_positions).
    """
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)

    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)

    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def extract_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    segments: list[ActivitySegment],
    device: torch.device,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for all activity segments.

    For each segment, computes:
    - mean_emb: L2-normalized mean-pooled encoder embedding (1024,)
    - temporal_residual: per-timestep residual vectors (n_target, 1024)

    Args:
        model: V-JEPA 2 model.
        processor: V-JEPA 2 video processor.
        segments: Activity segments to process.
        device: Torch device.

    Returns:
        Dict mapping segment index -> {'mean_emb': ..., 'temporal_residual': ...}
    """
    n_context_steps = VJEPA2_T_PATCHES // 2  # 16
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps  # 16
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Prefetch first clip
        future = pool.submit(
            load_clip_vjepa2, segments[0].video_path,
            segments[0].start_sec, segments[0].end_sec,
        )

        for i, seg in enumerate(tqdm(segments, desc="Extracting V-JEPA 2 features")):
            try:
                frames = future.result()

                # Prefetch next while GPU works
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    future = pool.submit(
                        load_clip_vjepa2, next_seg.video_path,
                        next_seg.start_sec, next_seg.end_sec,
                    )

                if len(frames) < VJEPA2_NUM_FRAMES:
                    failed += 1
                    continue

                # pyrefly: ignore [not-callable]
                inputs = processor(videos=frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Encoder only: bag-of-tokens embedding
                    enc_out = model(**inputs, skip_predictor=True)
                    encoder_tokens = enc_out.last_hidden_state[0]  # (8192, 1024)
                    mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)  # (1024,)

                    # With predictor: temporal residuals
                    pred_out = model(
                        **inputs,
                        context_mask=[context_mask],
                        target_mask=[target_mask],
                    )
                    predicted = pred_out.predictor_output.last_hidden_state[0]
                    ground_truth = pred_out.predictor_output.target_hidden_state[0]

                    # Reshape to (n_target_steps, SPATIAL, D), mean over spatial
                    predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                    ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                    residual = (predicted - ground_truth).mean(dim=1)  # (n_target_steps, 1024)

                features[i] = {
                    "mean_emb": mean_emb.cpu(),
                    "temporal_residual": residual.cpu(),
                }
            except Exception:
                failed += 1
                # Ensure prefetch is still queued for next iteration
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    future = pool.submit(
                        load_clip_vjepa2, next_seg.video_path,
                        next_seg.start_sec, next_seg.end_sec,
                    )
                continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_all_similarities(
    segments: list[ActivitySegment],
    features: dict[int, dict],
    person_to_indices: dict[str, list[int]],
    vjepa2_features: dict[int, dict] | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each person-session.

    For all pairs within each person-session:
    - Ground truth: same activity_id = positive (1), different = negative (0)
    - Compute 4 DINOv3 similarity methods + optional 2 V-JEPA 2 methods

    Uses batched DTW on GPU for temporal derivative, attention trajectory,
    and V-JEPA 2 temporal residual comparisons.

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute all fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    # Collect all pairs first
    pairs = []  # (a_idx, b_idx, gt_label)
    for ps in sorted(person_to_indices.keys()):
        indices = [i for i in person_to_indices[ps] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                a_idx = indices[a_pos]
                b_idx = indices[b_pos]
                gt = 1 if segments[a_idx].activity_id == segments[b_idx].activity_id else 0
                pairs.append((a_idx, b_idx, gt))

    total_pairs = len(pairs)
    print(f"  Total pairs to compute: {total_pairs}")

    # --- Non-DTW similarities (fast, vectorized) ---
    print("  Computing BoF and Chamfer similarities...")
    bof_scores = []
    chamfer_scores = []
    labels = []
    for a_idx, b_idx, gt in tqdm(pairs, desc="  BoF/Chamfer"):
        labels.append(gt)

        # Bag-of-frames
        bof_sim = float(
            torch.dot(features[a_idx]["mean_emb"], features[b_idx]["mean_emb"]).item()
        )
        bof_scores.append(bof_sim)

        # Chamfer
        ea = features[a_idx]["embeddings"]
        eb = features[b_idx]["embeddings"]
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_scores.append((max_ab + max_ba) / 2)

    # --- Batched DTW: temporal derivative ---
    print("  Batching temporal derivative DTW...")
    deriv_a = [deriv_fps[a] for a, b, _ in pairs]
    deriv_b = [deriv_fps[b] for a, b, _ in pairs]
    deriv_dists = dtw_distance_batch(deriv_a, deriv_b, normalize=False)
    deriv_scores = torch.exp(-deriv_dists).tolist()

    # --- Batched DTW: attention trajectory ---
    print("  Batching attention trajectory DTW...")
    traj_a = [traj_fps[a] for a, b, _ in pairs]
    traj_b = [traj_fps[b] for a, b, _ in pairs]
    traj_dists = dtw_distance_batch(traj_a, traj_b, normalize=True)
    traj_scores = torch.exp(-traj_dists * 5).tolist()

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": (bof_scores, labels),
        "chamfer": (chamfer_scores, list(labels)),
        "temporal_derivative": (deriv_scores, list(labels)),
        "attention_trajectory": (traj_scores, list(labels)),
    }

    # --- Batched DTW: V-JEPA 2 temporal residual ---
    if vjepa2_features:
        print("  Batching V-JEPA 2 DTW...")
        vjepa2_pairs = [
            (a, b, gt)
            for a, b, gt in pairs
            if a in vjepa2_features and b in vjepa2_features
        ]
        vjepa2_labels = [gt for _, _, gt in vjepa2_pairs]

        # Bag-of-tokens (no DTW, just cosine)
        bot_scores = []
        for a_idx, b_idx, _ in vjepa2_pairs:
            va = vjepa2_features[a_idx]
            vb = vjepa2_features[b_idx]
            bot_sim = float(torch.dot(va["mean_emb"], vb["mean_emb"]).item())
            bot_scores.append(bot_sim)

        # Temporal residual DTW (batched)
        res_a = [vjepa2_features[a]["temporal_residual"] for a, b, _ in vjepa2_pairs]
        res_b = [vjepa2_features[b]["temporal_residual"] for a, b, _ in vjepa2_pairs]
        res_dists = dtw_distance_batch(res_a, res_b, normalize=True)
        res_scores = torch.exp(-res_dists).tolist()

        all_scores["vjepa2_bag_of_tokens"] = (bot_scores, vjepa2_labels)
        all_scores["vjepa2_temporal_residual"] = (res_scores, vjepa2_labels)

    return all_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrimination(results: dict, fig_dir: Path):
    """Generate AP/AUC bar chart."""
    methods = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    labels = ["Bag of\nFrames", "Chamfer", "Temporal\nDerivative", "Attention\nTrajectory"]
    colors = {
        "bag_of_frames": "#e74c3c",
        "chamfer": "#1abc9c",
        "temporal_derivative": "#2ecc71",
        "attention_trajectory": "#3498db",
        "vjepa2_bag_of_tokens": "#9b59b6",
        "vjepa2_temporal_residual": "#f39c12",
    }

    # Add V-JEPA 2 methods if present
    if "vjepa2_bag_of_tokens" in results:
        methods.append("vjepa2_bag_of_tokens")
        labels.append("V-JEPA 2\nBag of Tokens")
    if "vjepa2_temporal_residual" in results:
        methods.append("vjepa2_temporal_residual")
        labels.append("V-JEPA 2\nTemporal Res.")

    aps = [results[m]["ap"] for m in methods]
    aucs = [results[m]["auc"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AP
    bars = ax1.bar(
        range(len(methods)),
        aps,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, aps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylabel("Average Precision", fontsize=12)
    ax1.set_title("Activity Discrimination: AP", fontsize=13)
    ax1.set_ylim(0, min(1.0, max(aps) * 1.15 + 0.05))
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax1.legend()

    # AUC
    bars = ax2.bar(
        range(len(methods)),
        aucs,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, aucs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("Activity Discrimination: AUC", fontsize=13)
    ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.legend()

    fig.suptitle(
        "Nymeria: Same-Activity vs Different-Activity Discrimination\n"
        "(Same Person, Same Environment)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "nymeria_activity_discrimination.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_similarity_distributions(
    all_scores: dict[str, tuple[list[float], list[int]]], fig_dir: Path
):
    """Generate same-vs-different activity similarity histograms."""
    methods = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    titles = ["Bag of Frames", "Chamfer", "Temporal Derivative", "Attention Trajectory"]

    # Add V-JEPA 2 methods if present
    if "vjepa2_bag_of_tokens" in all_scores and all_scores["vjepa2_bag_of_tokens"][0]:
        methods.append("vjepa2_bag_of_tokens")
        titles.append("V-JEPA 2 Bag of Tokens")
    if "vjepa2_temporal_residual" in all_scores and all_scores["vjepa2_temporal_residual"][0]:
        methods.append("vjepa2_temporal_residual")
        titles.append("V-JEPA 2 Temporal Residual")

    n_methods = len(methods)
    ncols = 3 if n_methods > 4 else 2
    nrows = (n_methods + ncols - 1) // ncols

    color_same = "#3498db"
    color_diff = "#e74c3c"

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, method, title in zip(axes, methods, titles):
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)

        same = scores[labels == 1]
        diff = scores[labels == 0]

        lo = min(scores.min(), 0)
        hi = max(scores.max(), 1)
        bins = np.linspace(lo, hi, 40)

        ax.hist(
            same,
            bins=bins,
            alpha=0.6,
            color=color_same,
            label=f"Same activity (n={len(same)})",
            density=True,
        )
        ax.hist(
            diff,
            bins=bins,
            alpha=0.6,
            color=color_diff,
            label=f"Different activity (n={len(diff)})",
            density=True,
        )
        ax.set_xlabel("Similarity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # Mean lines
        if len(same) > 0:
            ax.axvline(same.mean(), color=color_same, linestyle="--", alpha=0.8)
        if len(diff) > 0:
            ax.axvline(diff.mean(), color=color_diff, linestyle="--", alpha=0.8)

    # Hide unused subplot axes
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Nymeria: Similarity Distributions\n"
        "(Same vs Different Activity, Same Person & Environment)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "nymeria_similarity_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Nymeria Egocentric Activity Discrimination"
    )
    parser.add_argument(
        "--nymeria-dir",
        type=str,
        default="datasets/nymeria",
        help="Path to Nymeria dataset directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass cache load (still saves after extraction)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    nymeria_dir = project_root / args.nymeria_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NYMERIA: EGOCENTRIC ACTIVITY DISCRIMINATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover sequences
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering multi-activity sequences...")
    t0 = time.time()
    sequences = discover_sequences(nymeria_dir)

    total_clips = sum(len(seqs) for seqs in sequences.values())
    print(f"  Found {len(sequences)} multi-activity person-sessions")
    print(f"  Total activity clips: {total_clips}")
    for ps in sorted(sequences.keys()):
        acts = sorted(a[0] for a in sequences[ps])
        print(f"    {ps}: {', '.join(acts)}")
    print(f"  Discovery time: {time.time() - t0:.1f}s")

    if not sequences:
        print("\nERROR: No multi-activity sessions found.")
        return

    # ------------------------------------------------------------------
    # Step 2: Extract segments
    # ------------------------------------------------------------------
    print(f"\nStep 2: Extracting segments ({SEGMENTS_PER_CLIP} per clip, "
          f"{SEGMENT_DURATION}s each)...")
    segments, person_to_indices = extract_segments(sequences)

    # Count stats
    activity_counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        activity_counts[seg.activity_id] += 1

    print(f"  Total segments: {len(segments)}")
    print(f"  Person-sessions with segments: {len(person_to_indices)}")
    for act_id in sorted(activity_counts.keys()):
        print(f"    {act_id}: {activity_counts[act_id]} segments")

    # Count expected pairs
    total_pairs = 0
    total_pos = 0
    total_neg = 0
    for ps, indices in person_to_indices.items():
        ps_segments = [segments[i] for i in indices]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                gt = 1 if ps_segments[a_pos].activity_id == ps_segments[b_pos].activity_id else 0
                total_pairs += 1
                if gt:
                    total_pos += 1
                else:
                    total_neg += 1

    print(f"  Expected pairs: {total_pairs} (pos={total_pos}, neg={total_neg})")

    # ------------------------------------------------------------------
    # Step 3: Extract DINOv3 features (with caching)
    # ------------------------------------------------------------------
    dinov3_cache = nymeria_dir / "cache" / "dinov3_features.pt"
    features = None if args.no_cache else load_feature_cache(dinov3_cache)

    if features is None:
        print("\nStep 3: Loading DINOv3 encoder...")
        encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

        print("\nStep 4: Extracting video clips and DINOv3 features...")
        t_feat_start = time.time()
        features = extract_clip_features(
            encoder,
            segments,
            target_fps=3.0,
            max_resolution=518,
        )
        t_feat = time.time() - t_feat_start
        print(f"  Feature extraction time: {t_feat:.1f}s")

        save_feature_cache(features, dinov3_cache)

        # Free encoder memory
        del encoder
        torch.cuda.empty_cache()
    else:
        print("\nStep 3-4: DINOv3 features loaded from cache")

    # ------------------------------------------------------------------
    # Step 4b: Extract V-JEPA 2 features (with caching)
    # ------------------------------------------------------------------
    vjepa2_cache = nymeria_dir / "cache" / "vjepa2_features.pt"
    vjepa2_features = None if args.no_cache else load_feature_cache(vjepa2_cache)

    if vjepa2_features is None:
        print("\nStep 4b: Loading V-JEPA 2 model...")
        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )

        print("  Extracting V-JEPA 2 features...")
        t_vjepa_start = time.time()
        vjepa2_features = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            segments,
            device=torch.device(args.device),
        )
        t_vjepa = time.time() - t_vjepa_start
        print(f"  V-JEPA 2 feature extraction time: {t_vjepa:.1f}s")

        save_feature_cache(vjepa2_features, vjepa2_cache)

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()
    else:
        print("\nStep 4b: V-JEPA 2 features loaded from cache")

    # ------------------------------------------------------------------
    # Step 5: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 5: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_all_similarities(
        segments, features, person_to_indices, vjepa2_features=vjepa2_features
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # Dump pair scores for bootstrap CI computation
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }
    pair_path = nymeria_dir / "pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair scores saved to {pair_path}")

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: ACTIVITY DISCRIMINATION (SAME PERSON, SAME ENVIRONMENT)")
    print("=" * 70)

    results = {}
    method_order = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    if "vjepa2_bag_of_tokens" in all_scores:
        method_order.append("vjepa2_bag_of_tokens")
    if "vjepa2_temporal_residual" in all_scores:
        method_order.append("vjepa2_temporal_residual")

    for method in method_order:
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)

        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())
        gap = same_mean - diff_mean

        results[method] = {
            "ap": float(ap),
            "auc": float(auc),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": float(gap),
        }

        print(
            f"  {method:<25s}  AP={ap:.4f}  AUC={auc:.4f}  "
            f"gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 7: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir)
    plot_similarity_distributions(all_scores, fig_dir)

    # Summary
    n_total_pairs = len(all_scores["bag_of_frames"][0])
    print("\nSummary:")
    print(f"  Multi-activity person-sessions: {len(sequences)}")
    print(f"  Total activity clips: {total_clips}")
    print(f"  Total segments: {len(segments)}")
    print(f"  Total pairs evaluated: {n_total_pairs}")

    print("\nDone.")


if __name__ == "__main__":
    main()

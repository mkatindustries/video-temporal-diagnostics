#!/usr/bin/env python3
"""Honda HDD Intersection Clustering + Maneuver Discrimination Experiment.

Evaluates whether DINOv3 attention trajectory fingerprints can distinguish
different maneuvers (left turn vs right turn) at the same intersection,
where bag-of-frames fails because the visual background is identical.

Pipeline:
1. Load HDD session data (labels, GPS, video paths)
2. Extract contiguous maneuver segments with GPS midpoints
3. Cluster intersections using DBSCAN on GPS coordinates
4. Extract video clips for qualifying maneuver segments
5. Compute DINOv3 features (embeddings + attention centroids)
6. Compare all pairs within each cluster using 4 methods
7. Evaluate discrimination (AP, AUC) between same-maneuver and
   different-maneuver pairs

Hypothesis: Bag-of-frames and Chamfer will have low discrimination
(same background -> high similarity regardless of maneuver). Attention
trajectory will discriminate (gaze direction differs between left and
right turns).

Usage:
    python experiments/eval_hdd_intersections.py [--device cuda]
    python experiments/eval_hdd_intersections.py --fps-downsample 2 5 10 15
"""

import argparse
import json
import os
import re
import time
import zoneinfo
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.fingerprints.trajectory import dtw_distance
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Bootstrap CI for AP
# ---------------------------------------------------------------------------


def bootstrap_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for Average Precision.

    Returns:
        (ap, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    ap = average_precision_score(labels, scores)

    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot_aps[i] = ap  # degenerate sample
        else:
            boot_aps[i] = average_precision_score(l, s)

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_aps, 100 * alpha))
    ci_high = float(np.percentile(boot_aps, 100 * (1 - alpha)))

    return float(ap), ci_low, ci_high


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MANEUVER_NAMES = {
    1: "intersection_passing",
    2: "left_turn",
    3: "right_turn",
}

# V-JEPA 2 constants
DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32  # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256  # 16h × 16w


@dataclass
class ManeuverSegment:
    """A contiguous maneuver segment from one session."""

    session_id: str
    label: int  # 1=intersection_passing, 2=left_turn, 3=right_turn
    start_frame: int  # label frame index (at 3 fps)
    end_frame: int  # label frame index (at 3 fps), exclusive
    lat: float  # GPS midpoint latitude
    lng: float  # GPS midpoint longitude
    video_path: str
    video_start_unix: float  # unix timestamp of video start


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def parse_video_start_time(video_filename: str) -> float:
    """Parse video start unix timestamp from filename.

    Filename format: 2017-02-27-10-17-27_new_0.75.mp4
    Timestamps are in US/Pacific local time (PST or PDT depending on date).
    Using America/Los_Angeles handles DST transitions automatically —
    DST started March 12 2017, so Feb/early-Mar sessions are PST (UTC-8)
    and Apr+ sessions are PDT (UTC-7).
    """
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", video_filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {video_filename}")

    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return dt.timestamp()


def discover_sessions(hdd_dir: Path) -> dict[str, dict]:
    """Discover all valid HDD sessions with labels, GPS, and video.

    Returns:
        Dict mapping session_id -> {
            'label_path': Path to label .npy,
            'video_path': str path to .mp4,
            'gps_path': str path to rtk_pos.csv,
            'video_start_unix': float,
        }
    """
    label_dir = hdd_dir / "labels" / "target"
    release_dir = hdd_dir / "release_2019_07_08"

    # Find all label files
    label_files = {}
    for f in sorted(label_dir.glob("*.npy")):
        session_id = f.stem
        label_files[session_id] = f

    # Find all session directories in release_2019_07_08
    sessions = {}
    for date_dir in sorted(release_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name

            if session_id not in label_files:
                continue

            # Find video
            camera_dir = session_dir / "camera" / "center"
            if not camera_dir.exists():
                continue
            video_files = list(camera_dir.glob("*.mp4"))
            if not video_files:
                continue
            video_path = video_files[0]

            # Find GPS
            gps_path = session_dir / "general" / "csv" / "rtk_pos.csv"
            if not gps_path.exists():
                continue

            # Parse video start time
            try:
                video_start_unix = parse_video_start_time(video_path.name)
            except ValueError:
                continue

            sessions[session_id] = {
                "label_path": label_files[session_id],
                "video_path": str(video_path),
                "gps_path": str(gps_path),
                "video_start_unix": video_start_unix,
            }

    return sessions


def load_gps(gps_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GPS data from rtk_pos.csv.

    Note: CSV headers are swapped -- column labeled 'lng' contains latitude
    (~37.39), column labeled 'lat' contains longitude (~-122.05).

    Returns:
        (timestamps, latitudes, longitudes) as numpy arrays.
    """
    data = np.genfromtxt(
        gps_path, delimiter=",", skip_header=1, usecols=(0, 2, 3), dtype=np.float64
    )
    timestamps = data[:, 0]
    # Headers say lng,lat but values are swapped
    latitudes = data[:, 1]  # column labeled 'lng' is actually latitude
    longitudes = data[:, 2]  # column labeled 'lat' is actually longitude
    return timestamps, latitudes, longitudes


def _add_segment(
    segments: list[ManeuverSegment],
    session_id: str,
    label: int,
    start_frame: int,
    end_frame: int,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    video_path: str,
    video_start_unix: float,
):
    """Add a maneuver segment with GPS midpoint lookup."""
    mid_frame = (start_frame + end_frame) // 2
    mid_ts = video_start_unix + mid_frame / 3.0

    # Find nearest GPS entry
    gps_idx = np.searchsorted(gps_timestamps, mid_ts)
    # pyrefly: ignore [no-matching-overload]
    gps_idx = min(gps_idx, len(gps_timestamps) - 1)

    lat = gps_lats[gps_idx]
    lng = gps_lngs[gps_idx]

    # Sanity check GPS values
    if np.isnan(lat) or np.isnan(lng) or abs(lat) < 1 or abs(lng) < 1:
        return

    segments.append(
        ManeuverSegment(
            session_id=session_id,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
            # pyrefly: ignore [bad-argument-type]
            lat=lat,
            # pyrefly: ignore [bad-argument-type]
            lng=lng,
            video_path=video_path,
            video_start_unix=video_start_unix,
        )
    )


def extract_maneuver_segments(
    session_id: str,
    labels: np.ndarray,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    video_path: str,
    video_start_unix: float,
    target_labels: tuple[int, ...] = (1, 2, 3),
) -> list[ManeuverSegment]:
    """Extract contiguous maneuver segments and their GPS midpoints.

    Args:
        session_id: Session identifier.
        labels: Frame-level labels at 3 fps, shape (T,).
        gps_timestamps: GPS unix timestamps.
        gps_lats: GPS latitudes.
        gps_lngs: GPS longitudes.
        video_path: Path to video file.
        video_start_unix: Unix timestamp of video start.
        target_labels: Which label values to extract.

    Returns:
        List of ManeuverSegment objects.
    """
    segments: list[ManeuverSegment] = []

    in_segment = False
    seg_start = 0
    seg_label = 0

    for i in range(len(labels)):
        if labels[i] in target_labels:
            if not in_segment or labels[i] != seg_label:
                if in_segment:
                    _add_segment(
                        segments,
                        session_id,
                        seg_label,
                        seg_start,
                        i,
                        gps_timestamps,
                        gps_lats,
                        gps_lngs,
                        video_path,
                        video_start_unix,
                    )
                seg_start = i
                seg_label = int(labels[i])
                in_segment = True
        else:
            if in_segment:
                _add_segment(
                    segments,
                    session_id,
                    seg_label,
                    seg_start,
                    i,
                    gps_timestamps,
                    gps_lats,
                    gps_lngs,
                    video_path,
                    video_start_unix,
                )
                in_segment = False

    # Close final segment
    if in_segment:
        _add_segment(
            segments,
            session_id,
            seg_label,
            seg_start,
            len(labels),
            gps_timestamps,
            gps_lats,
            gps_lngs,
            video_path,
            video_start_unix,
        )

    return segments


# ---------------------------------------------------------------------------
# Intersection clustering
# ---------------------------------------------------------------------------


def cluster_intersections(
    segments: list[ManeuverSegment],
    eps: float = 0.0003,
    min_samples: int = 3,
) -> dict[int, list[ManeuverSegment]]:
    """Cluster maneuver segments by GPS location using DBSCAN.

    Args:
        segments: All maneuver segments.
        eps: DBSCAN radius in degrees (~30m at lat 37.4N).
        min_samples: Minimum cluster size.

    Returns:
        Dict mapping cluster_id -> list of segments in that cluster.
    """
    coords = np.array([[s.lat, s.lng] for s in segments])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(
        coords
    )

    clusters: dict[int, list[ManeuverSegment]] = defaultdict(list)
    for i, cid in enumerate(clustering.labels_):
        if cid >= 0:  # Ignore noise (-1)
            clusters[cid].append(segments[i])

    return dict(clusters)


def filter_mixed_clusters(
    clusters: dict[int, list[ManeuverSegment]],
    max_clusters: int = 50,
) -> dict[int, list[ManeuverSegment]]:
    """Filter for clusters containing both left turns (2) AND right turns (3).

    Returns top clusters by size.
    """
    mixed = {}
    for cid, segs in clusters.items():
        labels = {s.label for s in segs}
        if 2 in labels and 3 in labels:
            mixed[cid] = segs

    # Sort by cluster size, take top N
    sorted_clusters = sorted(mixed.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_clusters[:max_clusters])


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

    Uses PyAV seek() to jump near the clip start, avoiding decoding from
    frame 0 for long (53+ min) dashcam videos.

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


def load_clip_vjepa2(
    video_path: str,
    start_sec: float,
    end_sec: float,
    max_resolution: int = 256,
    fps_cap: float | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Dynamically computes fps to sample ~64 frames from the clip duration,
    then pads or truncates to exactly VJEPA2_NUM_FRAMES.

    If fps_cap is specified, the effective fps is capped to simulate a
    lower-framerate source. Fewer unique frames are extracted and the
    remainder is padded by repeating the last frame, preserving the
    64-frame input contract while reducing temporal resolution.

    Args:
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        max_resolution: Maximum height (V-JEPA 2 expects 256).
        fps_cap: If set, cap extraction fps to this value.

    Returns:
        Tuple of (frames, stats) where frames is a list of exactly
        VJEPA2_NUM_FRAMES RGB frames as numpy arrays, and stats is a dict
        with 'unique_frames', 'dup_ratio', 'fps_eff', 'duration'.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0

    # Target fps to get ~64 frames from this duration
    target_fps = VJEPA2_NUM_FRAMES / duration

    # Apply fps cap if specified
    if fps_cap is not None and target_fps > fps_cap:
        target_fps = fps_cap

    frames = load_clip(
        video_path,
        start_sec,
        end_sec,
        target_fps=target_fps,
        max_resolution=max_resolution,
    )

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    unique_frames = len(frames)

    # Pad by repeating last frame if too few
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])

    # Truncate if too many
    frames = frames[:VJEPA2_NUM_FRAMES]
    dup_ratio = 1.0 - unique_frames / VJEPA2_NUM_FRAMES
    fps_eff = unique_frames / duration if duration > 0 else 0.0

    stats = {
        "unique_frames": unique_frames,
        "dup_ratio": max(0.0, dup_ratio),
        "fps_eff": fps_eff,
        "duration": duration,
    }
    return frames, stats


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_clip_features(
    encoder: DINOv3Encoder,
    segments: list[ManeuverSegment],
    context_sec: float = 3.0,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> dict[int, dict]:
    """Extract DINOv3 features for all maneuver segments.

    Args:
        encoder: DINOv3 encoder.
        segments: Maneuver segments to process.
        context_sec: Seconds of context before/after maneuver.
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

    for i, seg in enumerate(tqdm(segments, desc="Extracting clip features")):
        # Convert label frame indices to video time
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            frames = load_clip(
                seg.video_path,
                start_sec,
                end_sec,
                target_fps=target_fps,
                max_resolution=max_resolution,
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
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


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
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
    fps_cap: float | None = None,
) -> tuple[dict[int, dict], list[dict]]:
    """Extract V-JEPA 2 features for all maneuver segments.

    For each segment, computes:
    - mean_emb: L2-normalized mean-pooled encoder embedding (1024,)
    - temporal_residual: per-timestep residual vectors (n_target, 1024)

    Args:
        model: V-JEPA 2 model.
        processor: V-JEPA 2 video processor.
        segments: Maneuver segments to process.
        device: Torch device.
        context_sec: Seconds of context before/after maneuver.

    Returns:
        Tuple of (features_dict, clip_stats) where features_dict maps
        segment index -> {'mean_emb': ..., 'temporal_residual': ...}
        and clip_stats is a list of per-clip stats dicts.
    """
    n_context_steps = VJEPA2_T_PATCHES // 2  # 16
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps  # 16
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    clip_stats = []
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting V-JEPA 2 features")):
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            frames, stats = load_clip_vjepa2(
                seg.video_path, start_sec, end_sec, fps_cap=fps_cap
            )
            clip_stats.append(stats)
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
                predicted = pred_out.predictor_output.last_hidden_state[
                    0
                ]  # (n_target*256, 1024)
                ground_truth = pred_out.predictor_output.target_hidden_state[0]

                # Reshape to (n_target_steps, SPATIAL, D), mean over spatial
                predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                residual = (predicted - ground_truth).mean(
                    dim=1
                )  # (n_target_steps, 1024)

            features[i] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    if clip_stats:
        fps_effs = [s["fps_eff"] for s in clip_stats]
        dup_ratios = [s["dup_ratio"] for s in clip_stats]
        durations = [s["duration"] for s in clip_stats]
        print(
            f"  Clip stats: fps_eff={np.median(fps_effs):.1f} (median), "
            f"dup_ratio={np.mean(dup_ratios):.3f} (mean), "
            f"duration={np.median(durations):.1f}s (median)"
        )
    return features, clip_stats


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_all_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    vjepa2_features: dict[int, dict] | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each cluster, aggregated.

    Uses batched GPU DTW for temporal derivative, attention trajectory,
    and V-JEPA 2 temporal residual comparisons.

    For all pairs within each cluster:
    - Ground truth: same-maneuver label = positive (1), different = negative (0)
    - Compute 4 DINOv3 similarity methods + optional 2 V-JEPA 2 methods

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute all fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    # Enumerate all pairs across clusters
    print("  Enumerating pairs...")
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                pair_a_indices.append(indices[a_pos])
                pair_b_indices.append(indices[b_pos])
                gt = (
                    1
                    if segments[indices[a_pos]].label == segments[indices[b_pos]].label
                    else 0
                )
                pair_gts.append(gt)

    total_pairs = len(pair_gts)
    print(f"  Total pairs to compute: {total_pairs}")

    # --- Vectorized bag-of-frames (batch dot product) ---
    print("  Computing bag-of-frames similarities...")
    mean_embs_a = torch.stack([features[i]["mean_emb"] for i in pair_a_indices]).to(
        device
    )
    mean_embs_b = torch.stack([features[i]["mean_emb"] for i in pair_b_indices]).to(
        device
    )
    bof_sims = (mean_embs_a * mean_embs_b).sum(dim=1).cpu().tolist()

    # --- Vectorized Chamfer (loop but on GPU) ---
    print("  Computing Chamfer similarities...")
    chamfer_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        ea = features[a_idx]["embeddings"].to(device)
        eb = features[b_idx]["embeddings"].to(device)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims.append((max_ab + max_ba) / 2)

    # --- Batched DTW: temporal derivatives ---
    print("  Computing temporal derivative DTW (batched GPU)...")
    deriv_seqs_a = [deriv_fps[i].to(device) for i in pair_a_indices]
    deriv_seqs_b = [deriv_fps[i].to(device) for i in pair_b_indices]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).cpu().tolist()

    # --- Batched DTW: attention trajectories ---
    print("  Computing attention trajectory DTW (batched GPU)...")
    traj_seqs_a = [traj_fps[i].to(device) for i in pair_a_indices]
    traj_seqs_b = [traj_fps[i].to(device) for i in pair_b_indices]
    traj_dists = dtw_distance_batch(traj_seqs_a, traj_seqs_b, normalize=True)
    traj_sims = torch.exp(-5.0 * traj_dists).cpu().tolist()

    # Assemble results
    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": (bof_sims, list(pair_gts)),
        "chamfer": (chamfer_sims, list(pair_gts)),
        "temporal_derivative": (deriv_sims, list(pair_gts)),
        "attention_trajectory": (traj_sims, list(pair_gts)),
    }

    # --- V-JEPA 2 methods ---
    if vjepa2_features:
        # Find pairs where both have V-JEPA 2 features
        vjepa2_mask = [
            (a_idx in vjepa2_features and b_idx in vjepa2_features)
            for a_idx, b_idx in zip(pair_a_indices, pair_b_indices)
        ]
        v_a_indices = [a for a, m in zip(pair_a_indices, vjepa2_mask) if m]
        v_b_indices = [b for b, m in zip(pair_b_indices, vjepa2_mask) if m]
        v_gts = [g for g, m in zip(pair_gts, vjepa2_mask) if m]

        if v_a_indices:
            # Bag-of-tokens (batch dot product)
            print("  Computing V-JEPA 2 bag-of-tokens similarities...")
            v_mean_a = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_a_indices]
            ).to(device)
            v_mean_b = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_b_indices]
            ).to(device)
            bot_sims = (v_mean_a * v_mean_b).sum(dim=1).cpu().tolist()

            # Temporal residual DTW (batched GPU)
            print("  Computing V-JEPA 2 temporal residual DTW (batched GPU)...")
            res_seqs_a = [
                vjepa2_features[i]["temporal_residual"].to(device) for i in v_a_indices
            ]
            res_seqs_b = [
                vjepa2_features[i]["temporal_residual"].to(device) for i in v_b_indices
            ]
            res_dists = dtw_distance_batch(res_seqs_a, res_seqs_b, normalize=True)
            res_sims = torch.exp(-res_dists).cpu().tolist()

            all_scores["vjepa2_bag_of_tokens"] = (bot_sims, list(v_gts))
            all_scores["vjepa2_temporal_residual"] = (res_sims, list(v_gts))
        else:
            all_scores["vjepa2_bag_of_tokens"] = ([], [])
            all_scores["vjepa2_temporal_residual"] = ([], [])

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
    labels = [
        "Bag of\nFrames",
        "Chamfer",
        "Temporal\nDerivative",
        "Attention\nTrajectory",
    ]
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
    ax1.set_title("Maneuver Discrimination: AP", fontsize=13)
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
    ax2.set_title("Maneuver Discrimination: AUC", fontsize=13)
    ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.legend()

    fig.suptitle(
        "Honda HDD: Left Turn vs Right Turn at Same Intersection",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "hdd_maneuver_discrimination.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_similarity_distributions(
    all_scores: dict[str, tuple[list[float], list[int]]], fig_dir: Path
):
    """Generate same-vs-different maneuver similarity histograms."""
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
    if (
        "vjepa2_temporal_residual" in all_scores
        and all_scores["vjepa2_temporal_residual"][0]
    ):
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
            label=f"Same maneuver (n={len(same)})",
            density=True,
        )
        ax.hist(
            diff,
            bins=bins,
            alpha=0.6,
            color=color_diff,
            label=f"Different maneuver (n={len(diff)})",
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
        "Honda HDD: Similarity Distributions\n"
        "(Same vs Different Maneuver at Same Intersection)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "hdd_similarity_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Honda HDD Intersection Clustering + Maneuver Discrimination"
    )
    parser.add_argument(
        "--hdd-dir",
        type=str,
        default="datasets/hdd",
        help="Path to HDD dataset directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of mixed clusters to evaluate",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Seconds of context before/after maneuver",
    )
    parser.add_argument(
        "--fps-downsample",
        type=float,
        nargs="+",
        default=None,
        help="V-JEPA 2 fps cap values for frame-rate downsample experiment "
        "(e.g. --fps-downsample 2 5 10 15). Runs V-JEPA 2 extraction "
        "at each fps cap and reports AP with bootstrap CIs.",
    )
    parser.add_argument(
        "--context-sec-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Context window durations (seconds before/after maneuver) for "
        "density sweep experiment. Always uses 64 frames; varying "
        "context_sec changes effective FPS. "
        "(e.g. --context-sec-sweep 1.0 2.0 3.0 4.0 6.5 8.0)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("HONDA HDD: INTERSECTION CLUSTERING + MANEUVER DISCRIMINATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover sessions
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    t0 = time.time()
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} valid sessions (with labels, GPS, video)")
    print(f"  Discovery time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Extract maneuver segments with GPS midpoints
    # ------------------------------------------------------------------
    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []
    sessions_with_segments = 0

    for session_id in tqdm(sorted(sessions.keys()), desc="Loading sessions"):
        info = sessions[session_id]

        labels = np.load(info["label_path"])

        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue

        segs = extract_maneuver_segments(
            session_id,
            labels,
            gps_ts,
            gps_lats,
            gps_lngs,
            info["video_path"],
            info["video_start_unix"],
            target_labels=(1, 2, 3),
        )

        if segs:
            sessions_with_segments += 1
        all_segments.extend(segs)

    # Count by type
    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1

    print(
        f"  Total segments: {len(all_segments)} "
        f"from {sessions_with_segments} sessions"
    )
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(label_val, 0)}")

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections (DBSCAN eps=0.0003)...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters
    # ------------------------------------------------------------------
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    total_segs_in_mixed = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print(
            "\nERROR: No mixed clusters found. "
            "Cannot evaluate maneuver discrimination."
        )
        return

    # Build flat list of segments in qualifying clusters, with cluster mapping
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    eval_label_counts: dict[int, int] = defaultdict(int)
    for seg in eval_segments:
        eval_label_counts[seg.label] += 1

    print(f"\n  Segments for evaluation: {len(eval_segments)}")
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {eval_label_counts.get(label_val, 0)}")

    # ------------------------------------------------------------------
    # Step 5: Extract video clips and DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 4: Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    print("\nStep 5: Extracting video clips and DINOv3 features...")
    t_feat_start = time.time()
    features = extract_clip_features(
        encoder,
        eval_segments,
        context_sec=args.context_sec,
        target_fps=3.0,
        max_resolution=518,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5b: Extract V-JEPA 2 features
    # ------------------------------------------------------------------
    print("\nStep 5b: Loading V-JEPA 2 model...")
    from transformers import AutoModel, AutoVideoProcessor

    vjepa2_model = AutoModel.from_pretrained(VJEPA2_MODEL_NAME, trust_remote_code=True)
    vjepa2_model = vjepa2_model.to(args.device).eval()
    vjepa2_processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    print("  Extracting V-JEPA 2 features...")
    t_vjepa_start = time.time()
    vjepa2_features, vjepa2_clip_stats = extract_vjepa2_features(
        vjepa2_model,
        vjepa2_processor,
        eval_segments,
        device=torch.device(args.device),
        context_sec=args.context_sec,
    )
    t_vjepa = time.time() - t_vjepa_start
    print(f"  V-JEPA 2 feature extraction time: {t_vjepa:.1f}s")

    del vjepa2_model, vjepa2_processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Compute similarities across all clusters
    # ------------------------------------------------------------------
    print("\nStep 6: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_all_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        vjepa2_features=vjepa2_features,
        device=torch.device(args.device),
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: MANEUVER DISCRIMINATION AT SAME INTERSECTION")
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
            "gap": gap,
        }

        print(
            f"  {method:<25s}  AP={ap:.4f}  AUC={auc:.4f}  "
            f"gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    # Dump pair-level scores for bootstrap CI computation
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }

    pair_path = hdd_dir / "pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir)
    plot_similarity_distributions(all_scores, fig_dir)

    # Summary
    n_total_pairs = len(all_scores["bag_of_frames"][0])
    print("\nSummary:")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Mixed intersection clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    print(f"  Total pairs evaluated: {n_total_pairs}")

    print("\nDone.")

    # ------------------------------------------------------------------
    # Optional: FPS downsample experiment
    # ------------------------------------------------------------------
    if args.fps_downsample:
        print("\n" + "=" * 70)
        print("FPS DOWNSAMPLE EXPERIMENT (V-JEPA 2 Temporal Residual)")
        print("=" * 70)

        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )

        downsample_results: dict[str, list] = {}
        for fps_val in args.fps_downsample:
            print(f"\n  --- fps_cap = {fps_val} ---")
            ds_features, _ = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                eval_segments,
                device=torch.device(args.device),
                context_sec=args.context_sec,
                fps_cap=fps_val,
            )

            # Compute similarities for V-JEPA 2 methods only
            ds_scores = compute_all_similarities(
                eval_segments,
                features,
                cluster_to_indices,
                vjepa2_features=ds_features,
                device=torch.device(args.device),
            )

            for method_key in ["vjepa2_temporal_residual", "vjepa2_bag_of_tokens"]:
                if method_key not in ds_scores:
                    continue
                s, l = ds_scores[method_key]
                s_arr, l_arr = np.array(s), np.array(l)
                if l_arr.sum() == 0 or l_arr.sum() == len(l_arr):
                    continue
                ap_val, ci_lo, ci_hi = bootstrap_ap(s_arr, l_arr)
                print(
                    f"    {method_key:<30s}  AP={ap_val:.4f}  "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                )
                downsample_results.setdefault(method_key, []).append(
                    {
                        "fps_cap": fps_val,
                        "ap": ap_val,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                    }
                )

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

        # Save downsample results
        ds_out = fig_dir.parent / "datasets" / "hdd" / "fps_downsample_results.json"
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_out, "w") as f:
            json.dump(downsample_results, f, indent=2)
        print(f"\n  Downsample results saved to {ds_out}")

    # ------------------------------------------------------------------
    # Optional: Context window sweep (density experiment)
    # ------------------------------------------------------------------
    if args.context_sec_sweep:
        print("\n" + "=" * 70)
        print("CONTEXT WINDOW SWEEP (V-JEPA 2 Temporal Residual)")
        print("=" * 70)
        print(f"  Windows: {args.context_sec_sweep}")
        print(f"  Fixed 64-frame input; varying context changes fps_eff")

        # Load V-JEPA 2 if not already loaded by fps_downsample
        if not args.fps_downsample:
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
        else:
            # Re-load since fps_downsample deleted it
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )

        sweep_out = (
            fig_dir.parent / "datasets" / "hdd" / "context_sec_sweep_results.json"
        )
        sweep_out.parent.mkdir(parents=True, exist_ok=True)

        # Resume from partial results if they exist
        sweep_results = {}
        completed_windows: set[float] = set()
        if sweep_out.exists():
            with open(sweep_out) as f:
                sweep_results = json.load(f)
            for method_entries in sweep_results.values():
                for entry in method_entries:
                    completed_windows.add(entry["context_sec"])
            if completed_windows:
                print(
                    f"  Resuming: skipping {len(completed_windows)} completed windows: "
                    f"{sorted(completed_windows)}"
                )

        for ctx_sec in args.context_sec_sweep:
            if ctx_sec in completed_windows:
                print(
                    f"\n  --- context_sec = {ctx_sec} --- SKIPPED (already completed)"
                )
                continue

            print(f"\n  --- context_sec = {ctx_sec} ---")
            ctx_features, ctx_stats = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                eval_segments,
                device=torch.device(args.device),
                context_sec=ctx_sec,
            )

            # Compute similarities for V-JEPA 2 methods only
            ctx_scores = compute_all_similarities(
                eval_segments,
                features,
                cluster_to_indices,
                vjepa2_features=ctx_features,
                device=torch.device(args.device),
            )

            # Aggregate clip stats
            fps_effs = [s["fps_eff"] for s in ctx_stats]
            dup_ratios = [s["dup_ratio"] for s in ctx_stats]
            durations = [s["duration"] for s in ctx_stats]
            stats_summary = {
                "fps_eff_median": float(np.median(fps_effs)),
                "fps_eff_mean": float(np.mean(fps_effs)),
                "dup_ratio_mean": float(np.mean(dup_ratios)),
                "duration_median": float(np.median(durations)),
                "unique_frames_mean": float(
                    np.mean([s["unique_frames"] for s in ctx_stats])
                ),
            }
            print(
                f"    fps_eff={stats_summary['fps_eff_median']:.1f} (median), "
                f"dup_ratio={stats_summary['dup_ratio_mean']:.3f}, "
                f"unique_frames={stats_summary['unique_frames_mean']:.1f}"
            )

            for method_key in ["vjepa2_temporal_residual", "vjepa2_bag_of_tokens"]:
                if method_key not in ctx_scores:
                    continue
                s, l = ctx_scores[method_key]
                s_arr, l_arr = np.array(s), np.array(l)
                if l_arr.sum() == 0 or l_arr.sum() == len(l_arr):
                    continue
                ap_val, ci_lo, ci_hi = bootstrap_ap(s_arr, l_arr)
                print(
                    f"    {method_key:<30s}  AP={ap_val:.4f}  "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                )
                sweep_results.setdefault(method_key, []).append(
                    {
                        "context_sec": ctx_sec,
                        "ap": ap_val,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                        **stats_summary,
                    }
                )

            # Save incrementally after each window
            with open(sweep_out, "w") as f:
                json.dump(sweep_results, f, indent=2)
            n_done = len(completed_windows) + 1
            completed_windows.add(ctx_sec)
            print(
                f"    [checkpoint saved: {n_done}/{len(args.context_sec_sweep)} windows]"
            )

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

        print(f"\n  Context sweep results saved to {sweep_out}")


if __name__ == "__main__":
    main()

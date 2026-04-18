#!/usr/bin/env python3
"""Honda HDD Cross-Session Maneuver Discrimination Experiment.

Evaluates HDD maneuver discrimination restricted to CROSS-SESSION pairs only.
A reviewer concern is that same-session pairs share lighting, camera position,
and weather — making the discrimination task easier and potentially inflating
AP values. This experiment filters to only cross-session pairs within each
intersection cluster.

Reports both "all pairs" AP (original protocol) and "cross-session only" AP
in a comparison table with bootstrap CIs. The key finding is whether the
RANK ORDER of methods changes — if it doesn't, the original protocol is
validated.

Pipeline:
1. Load HDD session data (labels, GPS, video paths)
2. Extract contiguous maneuver segments with GPS midpoints
3. Cluster intersections using DBSCAN on GPS coordinates
4. Extract video clips and features for all 6 methods
5. Compute similarities for ALL pairs (original) and CROSS-SESSION pairs
6. Report comparison table with bootstrap CIs

Usage:
    python experiments/eval_hdd_cross_session.py --hdd-dir datasets/hdd

    # Skip V-JEPA 2 (faster, DINOv3-only methods)
    python experiments/eval_hdd_cross_session.py --hdd-dir datasets/hdd --skip-vjepa2

    # Smoke test (2 clusters)
    python experiments/eval_hdd_cross_session.py --hdd-dir datasets/hdd --max-clusters 2
"""

import argparse
import json
import re
import time
import zoneinfo
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import cv2
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
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Constants
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
VJEPA2_SPATIAL = 256  # 16h x 16w

# Display names for the summary table
METHOD_DISPLAY_NAMES = {
    "bag_of_frames": "DINOv3 BoF",
    "chamfer": "Chamfer",
    "attention_trajectory": "Attn Traj",
    "temporal_derivative": "Temp Deriv",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "V-JEPA 2 Temp. Res.",
}


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
) -> tuple[list[np.ndarray], dict]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Dynamically computes fps to sample ~64 frames from the clip duration,
    then pads or truncates to exactly VJEPA2_NUM_FRAMES.

    Args:
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        max_resolution: Maximum height (V-JEPA 2 expects 256).

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
            frames, stats = load_clip_vjepa2(seg.video_path, start_sec, end_sec)
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
# Pair enumeration (all pairs + cross-session filtering)
# ---------------------------------------------------------------------------


def enumerate_pairs(
    segments: list[ManeuverSegment],
    features_available: set[int],
    cluster_to_indices: dict[int, list[int]],
    cross_session_only: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Enumerate segment pairs within each cluster.

    Args:
        segments: All evaluation segments.
        features_available: Set of segment indices that have features.
        cluster_to_indices: Mapping from cluster ID to segment indices.
        cross_session_only: If True, only return pairs where the two
            segments come from different session_ids.

    Returns:
        (pair_a_indices, pair_b_indices, pair_gts) — parallel lists of
        segment index A, segment index B, and ground-truth label (1 if
        same maneuver, 0 if different).
    """
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []

    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features_available]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                idx_a = indices[a_pos]
                idx_b = indices[b_pos]

                # Cross-session filter: skip same-session pairs
                if cross_session_only:
                    if segments[idx_a].session_id == segments[idx_b].session_id:
                        continue

                pair_a_indices.append(idx_a)
                pair_b_indices.append(idx_b)
                gt = 1 if segments[idx_a].label == segments[idx_b].label else 0
                pair_gts.append(gt)

    return pair_a_indices, pair_b_indices, pair_gts


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_similarities_for_pairs(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    pair_a_indices: list[int],
    pair_b_indices: list[int],
    pair_gts: list[int],
    vjepa2_features: dict[int, dict] | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities for the given pairs.

    Computes 4 DINOv3 methods + optional 2 V-JEPA 2 methods.

    Args:
        segments: All evaluation segments.
        features: DINOv3 features per segment index.
        pair_a_indices: Segment A indices for each pair.
        pair_b_indices: Segment B indices for each pair.
        pair_gts: Ground-truth labels (1=same maneuver, 0=different).
        vjepa2_features: Optional V-JEPA 2 features per segment index.
        device: Torch device.

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_pairs = len(pair_gts)
    if total_pairs == 0:
        return {}

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute all fingerprints for the indices we need
    needed_indices = set(pair_a_indices) | set(pair_b_indices)
    deriv_fps = {}
    traj_fps = {}
    for idx in needed_indices:
        if idx in features:
            deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
            traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    # --- Vectorized bag-of-frames (batch dot product) ---
    mean_embs_a = torch.stack([features[i]["mean_emb"] for i in pair_a_indices]).to(
        device
    )
    mean_embs_b = torch.stack([features[i]["mean_emb"] for i in pair_b_indices]).to(
        device
    )
    bof_sims = (mean_embs_a * mean_embs_b).sum(dim=1).cpu().tolist()

    # --- Vectorized Chamfer (loop but on GPU) ---
    chamfer_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        ea = features[a_idx]["embeddings"].to(device)
        eb = features[b_idx]["embeddings"].to(device)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims.append((max_ab + max_ba) / 2)

    # --- Batched DTW: temporal derivatives ---
    deriv_seqs_a = [deriv_fps[i].to(device) for i in pair_a_indices]
    deriv_seqs_b = [deriv_fps[i].to(device) for i in pair_b_indices]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).cpu().tolist()

    # --- Batched DTW: attention trajectories ---
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
        # Filter pairs where both have V-JEPA 2 features
        vjepa2_mask = [
            (a_idx in vjepa2_features and b_idx in vjepa2_features)
            for a_idx, b_idx in zip(pair_a_indices, pair_b_indices)
        ]
        v_a_indices = [a for a, m in zip(pair_a_indices, vjepa2_mask) if m]
        v_b_indices = [b for b, m in zip(pair_b_indices, vjepa2_mask) if m]
        v_gts = [g for g, m in zip(pair_gts, vjepa2_mask) if m]

        if v_a_indices:
            # Bag-of-tokens (batch dot product)
            v_mean_a = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_a_indices]
            ).to(device)
            v_mean_b = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_b_indices]
            ).to(device)
            bot_sims = (v_mean_a * v_mean_b).sum(dim=1).cpu().tolist()

            # Temporal residual DTW (batched GPU)
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
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_scores(
    all_scores: dict[str, tuple[list[float], list[int]]],
    method_order: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute AP, AUC, and bootstrap CIs for each method.

    Args:
        all_scores: Dict mapping method_name -> (scores_list, labels_list).
        method_order: Order to evaluate methods.
        n_resamples: Bootstrap resamples for CI.
        seed: Random seed for bootstrap.

    Returns:
        Dict mapping method_name -> {ap, ci_low, ci_high, auc, n_pos, n_neg}.
    """
    results = {}
    for method in method_order:
        if method not in all_scores:
            continue
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0 or len(scores) == 0:
            results[method] = {
                "ap": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap, ci_low, ci_high = bootstrap_ap(
            scores, labels, n_resamples=n_resamples, seed=seed
        )
        auc = roc_auc_score(labels, scores)

        results[method] = {
            "ap": float(ap),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "auc": float(auc),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    return results


def print_comparison_table(
    all_results: dict[str, dict],
    cross_results: dict[str, dict],
    method_order: list[str],
):
    """Print a side-by-side comparison table of all-pairs vs cross-session AP.

    Args:
        all_results: Results from all-pairs evaluation.
        cross_results: Results from cross-session-only evaluation.
        method_order: Order of methods for display.
    """
    print("\n" + "=" * 85)
    print("COMPARISON: ALL-PAIRS vs CROSS-SESSION ONLY")
    print("=" * 85)

    header = (
        f"{'Method':<25s} | {'All-Pairs AP':>16s} | "
        f"{'Cross-Session AP':>16s} | {'Delta AP':>10s}"
    )
    print(header)
    print("-" * 85)

    for method in method_order:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)

        all_r = all_results.get(method, {})
        cross_r = cross_results.get(method, {})

        all_ap = all_r.get("ap", float("nan"))
        all_ci_lo = all_r.get("ci_low", float("nan"))
        all_ci_hi = all_r.get("ci_high", float("nan"))

        cross_ap = cross_r.get("ap", float("nan"))
        cross_ci_lo = cross_r.get("ci_low", float("nan"))
        cross_ci_hi = cross_r.get("ci_high", float("nan"))

        if np.isnan(all_ap) or np.isnan(cross_ap):
            delta_str = "N/A"
        else:
            delta = cross_ap - all_ap
            delta_str = f"{delta:+.3f}"

        # Format with CI half-width
        if not np.isnan(all_ap):
            all_half = (all_ci_hi - all_ci_lo) / 2
            all_str = f"{all_ap:.3f} +/- {all_half:.2f}"
        else:
            all_str = "N/A"

        if not np.isnan(cross_ap):
            cross_half = (cross_ci_hi - cross_ci_lo) / 2
            cross_str = f"{cross_ap:.3f} +/- {cross_half:.2f}"
        else:
            cross_str = "N/A"

        print(
            f"{display_name:<25s} | {all_str:>16s} | "
            f"{cross_str:>16s} | {delta_str:>10s}"
        )

    print("=" * 85)

    # Check rank order preservation
    ranked_all = sorted(
        [
            (m, all_results[m]["ap"])
            for m in method_order
            if m in all_results and not np.isnan(all_results[m].get("ap", float("nan")))
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    ranked_cross = sorted(
        [
            (m, cross_results[m]["ap"])
            for m in method_order
            if m in cross_results
            and not np.isnan(cross_results[m].get("ap", float("nan")))
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    rank_all = [m for m, _ in ranked_all]
    rank_cross = [m for m, _ in ranked_cross]

    print(
        f"\nRank order (all-pairs):      {' > '.join(METHOD_DISPLAY_NAMES.get(m, m) for m in rank_all)}"
    )
    print(
        f"Rank order (cross-session):  {' > '.join(METHOD_DISPLAY_NAMES.get(m, m) for m in rank_cross)}"
    )

    if rank_all == rank_cross:
        print("\nRank order is PRESERVED -- original protocol is validated.")
    else:
        print("\nRank order CHANGED -- cross-session filtering affects method ranking.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Honda HDD Cross-Session Maneuver Discrimination"
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
        "--skip-vjepa2",
        action="store_true",
        help="Skip V-JEPA 2 feature extraction (faster, DINOv3-only)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("HONDA HDD: CROSS-SESSION MANEUVER DISCRIMINATION")
    print("=" * 70)
    print(
        "Evaluating whether same-session pairs inflate AP by sharing\n"
        "lighting, camera position, and weather conditions."
    )

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

    # Count unique sessions per cluster for cross-session statistics
    cross_session_cluster_count = 0
    for cid, indices in cluster_to_indices.items():
        session_ids = {eval_segments[i].session_id for i in indices}
        if len(session_ids) >= 2:
            cross_session_cluster_count += 1

    print(
        f"\n  Clusters with >= 2 sessions (eligible for cross-session pairs): "
        f"{cross_session_cluster_count}/{len(mixed)}"
    )

    # ------------------------------------------------------------------
    # Step 5: Extract DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 5: Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    print("  Extracting video clips and DINOv3 features...")
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
    # Step 5b: Extract V-JEPA 2 features (optional)
    # ------------------------------------------------------------------
    vjepa2_features: dict[int, dict] | None = None

    if not args.skip_vjepa2:
        print("\nStep 5b: Loading V-JEPA 2 model...")
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
        vjepa2_features, vjepa2_clip_stats = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            eval_segments,
            device=device,
            context_sec=args.context_sec,
        )
        t_vjepa = time.time() - t_vjepa_start
        print(f"  V-JEPA 2 feature extraction time: {t_vjepa:.1f}s")

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()
    else:
        print("\nStep 5b: Skipping V-JEPA 2 (--skip-vjepa2)")

    # ------------------------------------------------------------------
    # Step 6: Enumerate pairs (all pairs + cross-session only)
    # ------------------------------------------------------------------
    print("\nStep 6: Enumerating pairs...")

    features_available = set(features.keys())

    # All pairs (original protocol)
    all_a, all_b, all_gt = enumerate_pairs(
        eval_segments,
        features_available,
        cluster_to_indices,
        cross_session_only=False,
    )
    n_all = len(all_gt)
    n_all_pos = sum(all_gt)
    n_all_neg = n_all - n_all_pos

    # Cross-session only pairs
    cross_a, cross_b, cross_gt = enumerate_pairs(
        eval_segments,
        features_available,
        cluster_to_indices,
        cross_session_only=True,
    )
    n_cross = len(cross_gt)
    n_cross_pos = sum(cross_gt)
    n_cross_neg = n_cross - n_cross_pos

    # Same-session pairs (for reporting)
    n_same_session = n_all - n_cross

    print(f"  All pairs:           {n_all} (pos={n_all_pos}, neg={n_all_neg})")
    print(f"  Cross-session pairs: {n_cross} (pos={n_cross_pos}, neg={n_cross_neg})")
    print(f"  Same-session pairs:  {n_same_session} (filtered out)")
    if n_all > 0:
        print(f"  Cross-session fraction: {n_cross / n_all:.1%} of all pairs")

    if n_cross == 0:
        print(
            "\nERROR: No cross-session pairs found. Every cluster has "
            "segments from only one session."
        )
        return

    # ------------------------------------------------------------------
    # Step 7: Compute similarities for both protocols
    # ------------------------------------------------------------------
    print("\nStep 7a: Computing all-pairs similarities...")
    t_sim_start = time.time()
    all_scores = compute_similarities_for_pairs(
        eval_segments,
        features,
        all_a,
        all_b,
        all_gt,
        vjepa2_features=vjepa2_features,
        device=device,
    )
    t_all_sim = time.time() - t_sim_start
    print(f"  All-pairs similarity time: {t_all_sim:.1f}s")

    print("\nStep 7b: Computing cross-session similarities...")
    t_sim_start = time.time()
    cross_scores = compute_similarities_for_pairs(
        eval_segments,
        features,
        cross_a,
        cross_b,
        cross_gt,
        vjepa2_features=vjepa2_features,
        device=device,
    )
    t_cross_sim = time.time() - t_sim_start
    print(f"  Cross-session similarity time: {t_cross_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: Evaluate with bootstrap CIs
    # ------------------------------------------------------------------
    print("\nStep 8: Evaluating with bootstrap CIs (1000 resamples, seed=42)...")

    method_order = [
        "bag_of_frames",
        "chamfer",
        "attention_trajectory",
        "temporal_derivative",
    ]
    if not args.skip_vjepa2:
        method_order.append("vjepa2_bag_of_tokens")
        method_order.append("vjepa2_temporal_residual")

    all_results = evaluate_scores(all_scores, method_order)
    cross_results = evaluate_scores(cross_scores, method_order)

    # ------------------------------------------------------------------
    # Step 9: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: ALL-PAIRS PROTOCOL")
    print("=" * 70)
    for method in method_order:
        r = all_results.get(method, {})
        ap = r.get("ap", float("nan"))
        ci_lo = r.get("ci_low", float("nan"))
        ci_hi = r.get("ci_high", float("nan"))
        auc = r.get("auc", float("nan"))
        n_pos = r.get("n_pos", 0)
        n_neg = r.get("n_neg", 0)
        display = METHOD_DISPLAY_NAMES.get(method, method)
        print(
            f"  {display:<25s}  AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"AUC={auc:.4f}  (pos={n_pos}, neg={n_neg})"
        )

    print("\n" + "=" * 70)
    print("RESULTS: CROSS-SESSION ONLY PROTOCOL")
    print("=" * 70)
    for method in method_order:
        r = cross_results.get(method, {})
        ap = r.get("ap", float("nan"))
        ci_lo = r.get("ci_low", float("nan"))
        ci_hi = r.get("ci_high", float("nan"))
        auc = r.get("auc", float("nan"))
        n_pos = r.get("n_pos", 0)
        n_neg = r.get("n_neg", 0)
        display = METHOD_DISPLAY_NAMES.get(method, method)
        print(
            f"  {display:<25s}  AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"AUC={auc:.4f}  (pos={n_pos}, neg={n_neg})"
        )

    # Comparison table
    print_comparison_table(all_results, cross_results, method_order)

    # ------------------------------------------------------------------
    # Step 10: Save results
    # ------------------------------------------------------------------
    output_path = hdd_dir / "cross_session_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "max_clusters": args.max_clusters,
            "context_sec": args.context_sec,
            "skip_vjepa2": args.skip_vjepa2,
            "n_resamples": 1000,
            "seed": 42,
        },
        "pair_counts": {
            "all_pairs": n_all,
            "all_pairs_pos": n_all_pos,
            "all_pairs_neg": n_all_neg,
            "cross_session_pairs": n_cross,
            "cross_session_pos": n_cross_pos,
            "cross_session_neg": n_cross_neg,
            "same_session_pairs": n_same_session,
            "cross_session_fraction": n_cross / n_all if n_all > 0 else 0.0,
        },
        "all_pairs_results": {
            method: {
                "ap": r["ap"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "auc": r["auc"],
                "n_pos": r["n_pos"],
                "n_neg": r["n_neg"],
            }
            for method, r in all_results.items()
        },
        "cross_session_results": {
            method: {
                "ap": r["ap"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "auc": r["auc"],
                "n_pos": r["n_pos"],
                "n_neg": r["n_neg"],
            }
            for method, r in cross_results.items()
        },
        "delta_ap": {
            method: {
                "delta": cross_results[method]["ap"] - all_results[method]["ap"],
            }
            for method in method_order
            if method in all_results
            and method in cross_results
            and not np.isnan(all_results[method].get("ap", float("nan")))
            and not np.isnan(cross_results[method].get("ap", float("nan")))
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Summary
    print("\nSummary:")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Mixed intersection clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    print(f"  All pairs: {n_all}")
    print(f"  Cross-session pairs: {n_cross} ({n_cross / n_all:.1%})")

    print("\nDone.")


if __name__ == "__main__":
    main()

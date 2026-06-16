"""Shared utilities for Honda HDD intersection experiments.

Contains data loading, GPS clustering, video clip extraction, and feature
extraction functions used across multiple experiment scripts.
"""

import logging
import re
import zoneinfo
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from video_retrieval.models import DINOv3Encoder

logger = logging.getLogger(__name__)


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
    # Lazy import to avoid slow cv2 startup
    import cv2

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
        except Exception as e:
            logger.warning("Failed to extract clip features for segment %d: %s", i, e)
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
        except Exception as e:
            logger.warning("Failed to extract V-JEPA 2 features for segment %d: %s", i, e)
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


__all__ = [
    "DINOV3_MODEL_NAME",
    "MANEUVER_NAMES",
    "ManeuverSegment",
    "VJEPA2_MODEL_NAME",
    "VJEPA2_NUM_FRAMES",
    "VJEPA2_SPATIAL",
    "VJEPA2_T_PATCHES",
    "bootstrap_ap",
    "build_temporal_masks",
    "cluster_intersections",
    "discover_sessions",
    "extract_clip_features",
    "extract_maneuver_segments",
    "extract_vjepa2_features",
    "filter_mixed_clusters",
    "load_clip",
    "load_clip_vjepa2",
    "load_gps",
    "parse_video_start_time",
]

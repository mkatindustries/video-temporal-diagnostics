#!/usr/bin/env python3
"""nuScenes Intersection Clustering + Maneuver Discrimination Experiment.

Cross-dataset validation of the HDD maneuver discrimination result.
HDD showed V-JEPA 2 temporal residual dominates for distinguishing left vs
right turns at the same intersection (AP=0.968). This script tests whether
that finding generalizes to nuScenes — a different vehicle, different cameras,
different driving environment (Singapore/Boston vs Bay Area).

Pipeline:
1. Load nuScenes metadata (scene → sample → sample_data → ego_pose)
2. Load CAN bus pose + steering for each scene
3. Segment scenes by maneuver (steering angle + yaw rate → left/right/straight)
4. Cluster segments by ego_pose midpoint (DBSCAN, eps=30m)
5. Filter for mixed clusters (left + right turns at same location)
6. Extract DINOv3 + V-JEPA 2 features from CAM_FRONT keyframes
7. Compute pairwise similarities (6 methods)
8. Evaluate: AP/AUC for same vs different maneuver discrimination

Usage:
    python experiments/eval_nuscenes_intersections.py \
        --nuscenes-dir ./data/nuscenes --version v1.0-mini
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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
from video_retrieval.fingerprints.trajectory import dtw_distance
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANEUVER_NAMES = {
    1: "straight",
    2: "left_turn",
    3: "right_turn",
}

# V-JEPA 2 constants
DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32  # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256  # 16h × 16w


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ManeuverSegment:
    """A contiguous maneuver segment from one scene."""

    scene_name: str
    label: int  # 2=left_turn, 3=right_turn
    start_ts: float  # start timestamp (seconds, scene-relative utime)
    end_ts: float  # end timestamp (seconds)
    midpoint_x: float  # ego_pose midpoint x (meters, global frame)
    midpoint_y: float  # ego_pose midpoint y (meters, global frame)
    keyframe_paths: list[str] = field(default_factory=list)  # CAM_FRONT image paths


@dataclass
class NuScenesMetadata:
    """Pre-loaded nuScenes metadata lookup tables."""

    scenes: list[dict]
    token_to_sample: dict[str, dict]
    token_to_sample_data: dict[str, dict]
    token_to_ego_pose: dict[str, dict]
    cam_front_cal_tokens: set[str]  # calibrated_sensor tokens for CAM_FRONT


# ---------------------------------------------------------------------------
# nuScenes metadata loading (no devkit dependency)
# ---------------------------------------------------------------------------


def load_nuscenes_metadata(data_dir: Path, version: str) -> NuScenesMetadata:
    """Load nuScenes metadata JSONs and build lookup dicts.

    Reads scene.json, sample.json, sample_data.json, ego_pose.json,
    sensor.json, calibrated_sensor.json from {data_dir}/{version}/.
    """
    meta_dir = data_dir / version

    def _load(name: str) -> list[dict]:
        with open(meta_dir / name) as f:
            return json.load(f)

    scenes = _load("scene.json")
    samples = _load("sample.json")
    sample_data = _load("sample_data.json")
    ego_poses = _load("ego_pose.json")
    sensors = _load("sensor.json")
    cal_sensors = _load("calibrated_sensor.json")

    # Build lookup dicts
    token_to_sample = {s["token"]: s for s in samples}
    token_to_sample_data = {sd["token"]: sd for sd in sample_data}
    token_to_ego_pose = {ep["token"]: ep for ep in ego_poses}

    # Find CAM_FRONT sensor → calibrated_sensor tokens
    cam_front_sensor_token = None
    for s in sensors:
        if s["channel"] == "CAM_FRONT":
            cam_front_sensor_token = s["token"]
            break

    cam_front_cal_tokens = set()
    if cam_front_sensor_token:
        for cs in cal_sensors:
            if cs["sensor_token"] == cam_front_sensor_token:
                cam_front_cal_tokens.add(cs["token"])

    # Build sample_token → list of sample_data for fast lookup
    # (needed for finding CAM_FRONT data per sample)
    sample_to_data: dict[str, list[dict]] = defaultdict(list)
    for sd in sample_data:
        sample_to_data[sd["sample_token"]].append(sd)

    # Attach to metadata for use in get_scene_keyframes
    metadata = NuScenesMetadata(
        scenes=scenes,
        token_to_sample=token_to_sample,
        token_to_sample_data=token_to_sample_data,
        token_to_ego_pose=token_to_ego_pose,
        cam_front_cal_tokens=cam_front_cal_tokens,
    )
    # Store sample_to_data as extra attribute
    metadata._sample_to_data = sample_to_data  # type: ignore[attr-defined]

    print(
        f"  Loaded {version}: {len(scenes)} scenes, "
        f"{len(samples)} samples, {len(ego_poses)} ego_poses"
    )

    return metadata


def get_scene_keyframes(
    scene: dict, metadata: NuScenesMetadata, data_dir: Path
) -> list[tuple[float, str, float, float]]:
    """Walk sample chain for a scene and collect CAM_FRONT keyframes.

    Returns list of (timestamp_sec, image_path, ego_x, ego_y) tuples.
    """
    keyframes = []
    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = metadata.token_to_sample.get(sample_token)
        if sample is None:
            break

        # Find CAM_FRONT sample_data for this sample
        sample_datas = metadata._sample_to_data.get(sample_token, [])  # type: ignore[attr-defined]
        cam_front_sd = None
        for sd in sample_datas:
            if (
                sd["calibrated_sensor_token"] in metadata.cam_front_cal_tokens
                and sd["is_key_frame"]
            ):
                cam_front_sd = sd
                break

        if cam_front_sd is not None:
            # Get ego_pose for this sample_data
            ego_pose = metadata.token_to_ego_pose.get(cam_front_sd["ego_pose_token"])
            if ego_pose is not None:
                ts_sec = cam_front_sd["timestamp"] / 1e6  # microseconds → seconds
                img_path = str(data_dir / cam_front_sd["filename"])
                ego_x = ego_pose["translation"][0]
                ego_y = ego_pose["translation"][1]
                keyframes.append((ts_sec, img_path, ego_x, ego_y))

        # Next sample in chain
        sample_token = sample.get("next", "")
        if not sample_token:
            break

    return keyframes


# ---------------------------------------------------------------------------
# CAN bus loading + maneuver segmentation
# ---------------------------------------------------------------------------


def load_can_bus(
    can_dir: Path, scene_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load CAN bus steering + pose data for a scene.

    Returns:
        (timestamps_sec, steering_angles, yaw_rates, positions_xy, speeds)
        or None if files not found.
    """
    # Scene name format: "scene-0001" → file prefix "scene-0001"
    steering_path = can_dir / f"{scene_name}_steeranglefeedback.json"
    pose_path = can_dir / f"{scene_name}_pose.json"

    if not steering_path.exists() or not pose_path.exists():
        return None

    with open(steering_path) as f:
        steering_data = json.load(f)
    with open(pose_path) as f:
        pose_data = json.load(f)

    if not steering_data or not pose_data:
        return None

    # Build pose arrays (lower frequency ~50 Hz)
    pose_ts = np.array([p["utime"] / 1e6 for p in pose_data])  # → seconds
    pose_xy = np.array([[p["pos"][0], p["pos"][1]] for p in pose_data])
    yaw_rates = np.array([p["rotation_rate"][2] for p in pose_data])  # z-axis
    speeds = np.array([np.sqrt(p["vel"][0] ** 2 + p["vel"][1] ** 2) for p in pose_data])

    # Interpolate steering angles to pose timestamps
    steer_ts = np.array([s["utime"] / 1e6 for s in steering_data])
    steer_vals = np.array([s["value"] for s in steering_data])
    steering_interp = np.interp(pose_ts, steer_ts, steer_vals)

    return pose_ts, steering_interp, yaw_rates, pose_xy, speeds


def segment_maneuvers(
    timestamps: np.ndarray,
    steering_angles: np.ndarray,
    yaw_rates: np.ndarray,
    positions_xy: np.ndarray,
    speeds: np.ndarray,
    scene_name: str,
    min_duration: float = 2.0,
    smooth_window: int = 50,  # ~1 second at 50 Hz pose rate
) -> list[ManeuverSegment]:
    """Classify each timestep and extract contiguous maneuver segments.

    Uses AND logic: both yaw_rate AND steering must agree for a turn label.
    Applies sliding window smoothing before thresholding.

    Args:
        timestamps: Pose timestamps in seconds.
        steering_angles: Interpolated steering angle values.
        yaw_rates: Z-axis rotation rate (rad/s).
        positions_xy: Ego positions (N, 2) in meters.
        speeds: Scalar speed (m/s).
        scene_name: Scene identifier.
        min_duration: Minimum segment duration in seconds.
        smooth_window: Window size for mean smoothing (samples).

    Returns:
        List of ManeuverSegment objects (only left_turn=2 and right_turn=3).
    """
    n = len(timestamps)
    if n < smooth_window:
        return []

    # Smooth signals with sliding window mean
    kernel = np.ones(smooth_window) / smooth_window
    yaw_smooth = np.convolve(yaw_rates, kernel, mode="same")
    steer_smooth = np.convolve(steering_angles, kernel, mode="same")

    # Classify each timestep
    # AND logic: both signals must agree for a turn classification
    labels = np.ones(n, dtype=int)  # default: straight (1)

    for i in range(n):
        if speeds[i] < 0.5:
            labels[i] = 0  # stationary → skip
        elif yaw_smooth[i] > 0.02 and steer_smooth[i] > 0.5:
            labels[i] = 2  # left turn
        elif yaw_smooth[i] < -0.02 and steer_smooth[i] < -0.5:
            labels[i] = 3  # right turn

    # Extract contiguous segments of left/right turns
    segments: list[ManeuverSegment] = []
    in_segment = False
    seg_start = 0
    seg_label = 0

    for i in range(n):
        if labels[i] in (2, 3):
            if not in_segment or labels[i] != seg_label:
                if in_segment:
                    _finalize_segment(
                        segments,
                        scene_name,
                        seg_label,
                        seg_start,
                        i,
                        timestamps,
                        positions_xy,
                        min_duration,
                    )
                seg_start = i
                seg_label = int(labels[i])
                in_segment = True
        else:
            if in_segment:
                _finalize_segment(
                    segments,
                    scene_name,
                    seg_label,
                    seg_start,
                    i,
                    timestamps,
                    positions_xy,
                    min_duration,
                )
                in_segment = False

    # Close final segment
    if in_segment:
        _finalize_segment(
            segments,
            scene_name,
            seg_label,
            seg_start,
            n,
            timestamps,
            positions_xy,
            min_duration,
        )

    return segments


def _finalize_segment(
    segments: list[ManeuverSegment],
    scene_name: str,
    label: int,
    start_idx: int,
    end_idx: int,
    timestamps: np.ndarray,
    positions_xy: np.ndarray,
    min_duration: float,
):
    """Create a ManeuverSegment if duration >= min_duration."""
    duration = timestamps[end_idx - 1] - timestamps[start_idx]
    if duration < min_duration:
        return

    mid_idx = (start_idx + end_idx) // 2
    mid_idx = min(mid_idx, len(positions_xy) - 1)

    segments.append(
        ManeuverSegment(
            scene_name=scene_name,
            label=label,
            start_ts=timestamps[start_idx],
            end_ts=timestamps[end_idx - 1],
            midpoint_x=float(positions_xy[mid_idx, 0]),
            midpoint_y=float(positions_xy[mid_idx, 1]),
        )
    )


# ---------------------------------------------------------------------------
# Intersection clustering
# ---------------------------------------------------------------------------


def cluster_intersections(
    segments: list[ManeuverSegment],
    eps: float = 30.0,
    min_samples: int = 2,
) -> dict[int, list[ManeuverSegment]]:
    """Cluster maneuver segments by ego_pose location using DBSCAN.

    Args:
        segments: All maneuver segments.
        eps: DBSCAN radius in meters.
        min_samples: Minimum cluster size (relaxed for mini split).

    Returns:
        Dict mapping cluster_id → list of segments in that cluster.
    """
    coords = np.array([[s.midpoint_x, s.midpoint_y] for s in segments])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(
        coords
    )

    clusters: dict[int, list[ManeuverSegment]] = defaultdict(list)
    for i, cid in enumerate(clustering.labels_):
        if cid >= 0:
            clusters[cid].append(segments[i])

    return dict(clusters)


def filter_mixed_clusters(
    clusters: dict[int, list[ManeuverSegment]],
    max_clusters: int = 50,
) -> dict[int, list[ManeuverSegment]]:
    """Filter for clusters containing both left turns (2) AND right turns (3)."""
    mixed = {}
    for cid, segs in clusters.items():
        labels = {s.label for s in segs}
        if 2 in labels and 3 in labels:
            mixed[cid] = segs

    sorted_clusters = sorted(mixed.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_clusters[:max_clusters])


# ---------------------------------------------------------------------------
# Assign keyframes to segments
# ---------------------------------------------------------------------------


def assign_keyframes_to_segments(
    segments: list[ManeuverSegment],
    scene_keyframes: dict[str, list[tuple[float, str, float, float]]],
):
    """Attach CAM_FRONT keyframe paths to each segment based on timestamp overlap."""
    for seg in segments:
        keyframes = scene_keyframes.get(seg.scene_name, [])
        seg.keyframe_paths = [
            img_path
            for ts, img_path, _, _ in keyframes
            if seg.start_ts <= ts <= seg.end_ts
        ]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def load_keyframe_images(
    image_paths: list[str],
    max_resolution: int = 518,
) -> list[np.ndarray]:
    """Load JPEG keyframes and resize if needed."""
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max_resolution and img.shape[0] > max_resolution:
            scale = max_resolution / img.shape[0]
            new_h = max_resolution
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h))
        frames.append(img)
    return frames


def extract_segment_features_dinov3(
    encoder: DINOv3Encoder,
    segments: list[ManeuverSegment],
    max_resolution: int = 518,
) -> dict[int, dict]:
    """Extract DINOv3 features for maneuver segments from CAM_FRONT keyframes.

    Returns:
        Dict mapping segment index → {embeddings, centroids, mean_emb}.
    """
    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting DINOv3 features")):
        if len(seg.keyframe_paths) < 3:
            failed += 1
            continue

        try:
            frames = load_keyframe_images(seg.keyframe_paths, max_resolution)
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
    """Build context/target masks that split along the temporal axis."""
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)

    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)

    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def load_segment_frames_vjepa2(
    seg: ManeuverSegment,
    scene_keyframes: dict[str, list[tuple[float, str, float, float]]],
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Collect frames for a segment and pad/truncate to VJEPA2_NUM_FRAMES.

    Uses all keyframes (and duplicates if needed) to reach 64 frames.
    """
    # Get all keyframes for this scene within the segment timespan
    all_kf = scene_keyframes.get(seg.scene_name, [])
    paths = [p for ts, p, _, _ in all_kf if seg.start_ts <= ts <= seg.end_ts]

    frames = load_keyframe_images(paths, max_resolution)
    if not frames:
        raise ValueError(f"No frames for {seg.scene_name}")

    # Pad by repeating frames evenly if too few
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames = frames + frames  # double and truncate below

    return frames[:VJEPA2_NUM_FRAMES]


def extract_segment_features_vjepa2(
    model: torch.nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    scene_keyframes: dict[str, list[tuple[float, str, float, float]]],
    device: torch.device,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for maneuver segments.

    Returns:
        Dict mapping segment index → {mean_emb, encoder_seq, temporal_residual}.
    """
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting V-JEPA 2 features")):
        try:
            frames = load_segment_frames_vjepa2(seg, scene_keyframes)
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue

            # pyrefly: ignore [not-callable]
            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Encoder only: bag-of-tokens embedding
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]
                mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

                # Encoder sequence: spatially averaged per temporal position
                enc_reshaped = encoder_tokens.reshape(
                    VJEPA2_T_PATCHES, VJEPA2_SPATIAL, -1
                )
                encoder_seq = enc_reshaped.mean(dim=1)  # (32, 1024)

                # With predictor: temporal residuals
                pred_out = model(
                    **inputs,
                    context_mask=[context_mask],
                    target_mask=[target_mask],
                )
                predicted = pred_out.predictor_output.last_hidden_state[0]
                ground_truth = pred_out.predictor_output.target_hidden_state[0]

                predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                residual = (predicted - ground_truth).mean(dim=1)

            features[i] = {
                "mean_emb": mean_emb.cpu(),
                "encoder_seq": encoder_seq.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def save_feature_cache(features: dict, cache_path: Path) -> None:
    """Save extracted features to disk as a .pt file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_features = {}
    for k, v in features.items():
        cpu_features[k] = {
            fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv for fk, fv in v.items()
        }
    torch.save(cpu_features, cache_path)
    print(f"  Cache saved to {cache_path}")


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load cached features from disk, or return None if not found."""
    if not cache_path.exists():
        return None
    print(f"  Loading cache from {cache_path}")
    return torch.load(cache_path, weights_only=False)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_all_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    vjepa2_features: dict[int, dict] | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each cluster.

    Same 7 methods as HDD evaluation:
    - DINOv3: bag_of_frames, chamfer, temporal_derivative, attention_trajectory
    - V-JEPA 2: bag_of_tokens, encoder_seq_dtw, temporal_residual

    Returns:
        Dict mapping method_name → (scores_list, labels_list).
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": ([], []),
        "chamfer": ([], []),
        "temporal_derivative": ([], []),
        "attention_trajectory": ([], []),
    }
    if vjepa2_features:
        all_scores["vjepa2_bag_of_tokens"] = ([], [])
        all_scores["vjepa2_encoder_seq_dtw"] = ([], [])
        all_scores["vjepa2_temporal_residual"] = ([], [])

    total_pairs = 0
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                total_pairs += 1

    print(f"  Total pairs to compute: {total_pairs}")

    for cid in tqdm(sorted(cluster_to_indices.keys()), desc="  Cluster similarities"):
        indices = [i for i in cluster_to_indices[cid] if i in features]

        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                a_idx = indices[a_pos]
                b_idx = indices[b_pos]

                seg_a = segments[a_idx]
                seg_b = segments[b_idx]

                # Ground truth: same maneuver label = 1, different = 0
                gt = 1 if seg_a.label == seg_b.label else 0

                ea = features[a_idx]["embeddings"]
                eb = features[b_idx]["embeddings"]

                # Bag-of-frames (cosine of mean embeddings)
                bof_sim = float(
                    torch.dot(
                        features[a_idx]["mean_emb"], features[b_idx]["mean_emb"]
                    ).item()
                )
                all_scores["bag_of_frames"][0].append(bof_sim)
                all_scores["bag_of_frames"][1].append(gt)

                # Chamfer (max symmetric)
                sim_matrix = torch.mm(ea, eb.t())
                max_ab = sim_matrix.max(dim=1).values.mean().item()
                max_ba = sim_matrix.max(dim=0).values.mean().item()
                chamfer_sim = (max_ab + max_ba) / 2
                all_scores["chamfer"][0].append(chamfer_sim)
                all_scores["chamfer"][1].append(gt)

                # Temporal derivative DTW
                d_sim = deriv_fp.compare(deriv_fps[a_idx], deriv_fps[b_idx])
                all_scores["temporal_derivative"][0].append(d_sim)
                all_scores["temporal_derivative"][1].append(gt)

                # Attention trajectory DTW
                t_sim = traj_fp.compare(traj_fps[a_idx], traj_fps[b_idx])
                all_scores["attention_trajectory"][0].append(t_sim)
                all_scores["attention_trajectory"][1].append(gt)

                # V-JEPA 2 methods
                if (
                    vjepa2_features
                    and a_idx in vjepa2_features
                    and b_idx in vjepa2_features
                ):
                    va = vjepa2_features[a_idx]
                    vb = vjepa2_features[b_idx]

                    bot_sim = float(torch.dot(va["mean_emb"], vb["mean_emb"]).item())
                    all_scores["vjepa2_bag_of_tokens"][0].append(bot_sim)
                    all_scores["vjepa2_bag_of_tokens"][1].append(gt)

                    if "encoder_seq" in va and "encoder_seq" in vb:
                        enc_dist = dtw_distance(
                            va["encoder_seq"],
                            vb["encoder_seq"],
                            normalize=True,
                        )
                        enc_sim = float(torch.exp(torch.tensor(-enc_dist)).item())
                        all_scores["vjepa2_encoder_seq_dtw"][0].append(enc_sim)
                        all_scores["vjepa2_encoder_seq_dtw"][1].append(gt)

                    res_dist = dtw_distance(
                        va["temporal_residual"],
                        vb["temporal_residual"],
                        normalize=True,
                    )
                    res_sim = float(torch.exp(torch.tensor(-res_dist)).item())
                    all_scores["vjepa2_temporal_residual"][0].append(res_sim)
                    all_scores["vjepa2_temporal_residual"][1].append(gt)

    return all_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrimination(results: dict, fig_dir: Path):
    """Generate AP/AUC bar chart (same format as HDD)."""
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
        "vjepa2_encoder_seq_dtw": "#8e44ad",
        "vjepa2_temporal_residual": "#f39c12",
    }

    if "vjepa2_bag_of_tokens" in results:
        methods.append("vjepa2_bag_of_tokens")
        labels.append("V-JEPA 2\nBag of Tokens")
    if "vjepa2_encoder_seq_dtw" in results:
        methods.append("vjepa2_encoder_seq_dtw")
        labels.append("V-JEPA 2\nEncoder-Seq DTW")
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
        "nuScenes: Left Turn vs Right Turn at Same Intersection",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "nuscenes_maneuver_discrimination.png"
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

    if "vjepa2_bag_of_tokens" in all_scores and all_scores["vjepa2_bag_of_tokens"][0]:
        methods.append("vjepa2_bag_of_tokens")
        titles.append("V-JEPA 2 Bag of Tokens")
    if (
        "vjepa2_encoder_seq_dtw" in all_scores
        and all_scores["vjepa2_encoder_seq_dtw"][0]
    ):
        methods.append("vjepa2_encoder_seq_dtw")
        titles.append("V-JEPA 2 Encoder-Seq DTW")
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
        labels_arr = np.array(labels_list)

        same = scores[labels_arr == 1]
        diff = scores[labels_arr == 0]

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

        if len(same) > 0:
            ax.axvline(same.mean(), color=color_same, linestyle="--", alpha=0.8)
        if len(diff) > 0:
            ax.axvline(diff.mean(), color=color_diff, linestyle="--", alpha=0.8)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "nuScenes: Similarity Distributions\n"
        "(Same vs Different Maneuver at Same Intersection)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "nuscenes_similarity_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_cluster_map(
    clusters: dict[int, list[ManeuverSegment]],
    mixed_clusters: dict[int, list[ManeuverSegment]],
    fig_dir: Path,
):
    """Plot ego_pose centroids colored by cluster as a sanity check."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all clustered segments in gray
    for cid, segs in clusters.items():
        if cid in mixed_clusters:
            continue
        xs = [s.midpoint_x for s in segs]
        ys = [s.midpoint_y for s in segs]
        ax.scatter(xs, ys, c="lightgray", s=20, alpha=0.5, zorder=1)

    # Plot mixed clusters with colors
    cmap = plt.get_cmap("tab10")
    for i, (cid, segs) in enumerate(mixed_clusters.items()):
        color = cmap(i % 10)
        for seg in segs:
            marker = "^" if seg.label == 2 else "v"  # ▲ left, ▼ right
            ax.scatter(
                seg.midpoint_x,
                seg.midpoint_y,
                c=[color],
                s=80,
                marker=marker,
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )
        # Label cluster
        cx = np.mean([s.midpoint_x for s in segs])
        cy = np.mean([s.midpoint_y for s in segs])
        ax.annotate(
            f"C{cid}",
            (float(cx), float(cy)),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(
        "nuScenes Intersection Clusters\n"
        "(colored = mixed L/R, gray = single-type, ▲=left ▼=right)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    fig.tight_layout()

    path = fig_dir / "nuscenes_cluster_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="nuScenes Intersection Clustering + Maneuver Discrimination"
    )
    parser.add_argument(
        "--nuscenes-dir",
        type=str,
        default=None,
        required=True,
        help="Path to nuScenes data root",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes version to use",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-vjepa2",
        action="store_true",
        help="Skip V-JEPA 2 (DINOv3 only, faster iteration)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of mixed clusters to evaluate",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=2.0,
        help="Minimum maneuver duration in seconds",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable feature caching",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.nuscenes_dir)
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    cache_dir = data_dir / "feature_cache"

    print("=" * 70)
    print("NUSCENES: INTERSECTION CLUSTERING + MANEUVER DISCRIMINATION")
    print("=" * 70)
    print(f"  Version: {args.version}")
    print(f"  Data dir: {data_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load nuScenes metadata
    # ------------------------------------------------------------------
    print("\nStep 1: Loading nuScenes metadata...")
    t0 = time.time()
    metadata = load_nuscenes_metadata(data_dir, args.version)
    print(f"  Metadata load time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Load CAN bus + segment maneuvers
    # ------------------------------------------------------------------
    print("\nStep 2: Segmenting maneuvers from CAN bus data...")
    can_dir = data_dir / "can_bus" / "can_bus"

    all_segments: list[ManeuverSegment] = []
    scene_keyframes: dict[str, list[tuple[float, str, float, float]]] = {}
    scenes_with_segments = 0

    for scene in tqdm(metadata.scenes, desc="Processing scenes"):
        scene_name = scene["name"]

        # Load CAN bus
        can_data = load_can_bus(can_dir, scene_name)
        if can_data is None:
            continue

        timestamps, steering, yaw_rates, positions, speeds = can_data

        # Segment maneuvers
        segs = segment_maneuvers(
            timestamps,
            steering,
            yaw_rates,
            positions,
            speeds,
            scene_name,
            min_duration=args.min_segment_duration,
        )

        if segs:
            scenes_with_segments += 1
        all_segments.extend(segs)

        # Load keyframes for this scene
        kfs = get_scene_keyframes(scene, metadata, data_dir)
        if kfs:
            scene_keyframes[scene_name] = kfs

    # Count by type
    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1

    print(
        f"  Total segments: {len(all_segments)} "
        f"from {scenes_with_segments}/{len(metadata.scenes)} scenes"
    )
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        if label_val in (2, 3):
            print(f"    {name}: {label_counts.get(label_val, 0)}")

    if not all_segments:
        print("\n  No maneuver segments found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections
    # ------------------------------------------------------------------
    print(f"\nStep 3: Clustering intersections (DBSCAN eps=30m)...")
    clusters = cluster_intersections(
        all_segments,
        eps=30.0,
        min_samples=2,
    )
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters
    # ------------------------------------------------------------------
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    total_segs_in_mixed = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    # Plot cluster map regardless of whether we have mixed clusters
    if clusters:
        print("\n  Generating cluster map...")
        plot_cluster_map(clusters, mixed, fig_dir)

    if not mixed:
        print("\n  0 mixed clusters found — cannot evaluate maneuver discrimination.")
        if args.version == "v1.0-mini":
            print("  This is expected for the mini split (10 scenes).")
            print("  Use --version v1.0-trainval for real results.")
        return

    # Build flat list of segments in qualifying clusters
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    # Assign keyframes
    assign_keyframes_to_segments(eval_segments, scene_keyframes)

    eval_label_counts: dict[int, int] = defaultdict(int)
    for seg in eval_segments:
        eval_label_counts[seg.label] += 1

    print(f"\n  Segments for evaluation: {len(eval_segments)}")
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        if label_val in (2, 3):
            print(f"    {name}: {eval_label_counts.get(label_val, 0)}")

    kf_counts = [len(s.keyframe_paths) for s in eval_segments]
    print(
        f"  Keyframes per segment: min={min(kf_counts)}, "
        f"max={max(kf_counts)}, mean={np.mean(kf_counts):.1f}"
    )

    # ------------------------------------------------------------------
    # Step 5: Extract DINOv3 features
    # ------------------------------------------------------------------
    dinov3_cache_path = cache_dir / f"nuscenes_dinov3_{args.version}.pt"
    features = None
    if not args.no_cache:
        features = load_feature_cache(dinov3_cache_path)

    if features is None:
        print("\nStep 5: Loading DINOv3 encoder...")
        encoder = DINOv3Encoder(
            model_name=DINOV3_MODEL_NAME,
            device=args.device,
        )

        print("  Extracting DINOv3 features...")
        t_feat_start = time.time()
        features = extract_segment_features_dinov3(
            encoder,
            eval_segments,
            max_resolution=518,
        )
        t_feat = time.time() - t_feat_start
        print(f"  Feature extraction time: {t_feat:.1f}s")

        if not args.no_cache:
            save_feature_cache(features, dinov3_cache_path)

        del encoder
        torch.cuda.empty_cache()
    else:
        print("\nStep 5: DINOv3 features loaded from cache")

    # ------------------------------------------------------------------
    # Step 5b: Extract V-JEPA 2 features
    # ------------------------------------------------------------------
    vjepa2_features = None
    if not args.skip_vjepa2:
        vjepa2_cache_path = cache_dir / f"nuscenes_vjepa2_{args.version}.pt"
        if not args.no_cache:
            vjepa2_features = load_feature_cache(vjepa2_cache_path)

        if vjepa2_features is None:
            print("\nStep 5b: Loading V-JEPA 2 model...")
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME,
                trust_remote_code=True,
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME,
                trust_remote_code=True,
            )

            print("  Extracting V-JEPA 2 features...")
            t_vjepa_start = time.time()
            vjepa2_features = extract_segment_features_vjepa2(
                vjepa2_model,
                vjepa2_processor,
                eval_segments,
                scene_keyframes,
                device=torch.device(args.device),
            )
            t_vjepa = time.time() - t_vjepa_start
            print(f"  V-JEPA 2 feature extraction time: {t_vjepa:.1f}s")

            if not args.no_cache:
                save_feature_cache(vjepa2_features, vjepa2_cache_path)

            del vjepa2_model, vjepa2_processor
            torch.cuda.empty_cache()
        else:
            print("\nStep 5b: V-JEPA 2 features loaded from cache")

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
    if "vjepa2_encoder_seq_dtw" in all_scores and all_scores["vjepa2_encoder_seq_dtw"][0]:
        method_order.append("vjepa2_encoder_seq_dtw")
    if "vjepa2_temporal_residual" in all_scores:
        method_order.append("vjepa2_temporal_residual")

    for method in method_order:
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels_arr = np.array(labels_list)
        n_pos = int(labels_arr.sum())
        n_neg = len(labels_arr) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap = average_precision_score(labels_arr, scores)
        auc = roc_auc_score(labels_arr, scores)

        same_mean = float(scores[labels_arr == 1].mean())
        diff_mean = float(scores[labels_arr == 0].mean())
        gap = same_mean - diff_mean

        results[method] = {
            "ap": ap,
            "auc": auc,
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

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir)
    plot_similarity_distributions(all_scores, fig_dir)

    # Save results JSON
    results_path = data_dir / "intersection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Dump pair-level scores for bootstrap CI computation
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }

    pair_path = data_dir / "pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    # Summary
    n_total_pairs = len(all_scores["bag_of_frames"][0])
    print("\nSummary:")
    print(f"  Version: {args.version}")
    print(f"  Scenes: {len(metadata.scenes)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Intersection clusters: {len(clusters)}")
    print(f"  Mixed clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    print(f"  Total pairs evaluated: {n_total_pairs}")

    print("\nDone.")


if __name__ == "__main__":
    main()

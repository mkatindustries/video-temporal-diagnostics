#!/usr/bin/env python3
"""VLM Vision Tower Bridge Experiment on nuScenes.

Evaluates whether VLM vision tower embeddings (SigLIP from Gemma-4/Qwen3-VL,
CLIP from LLaVA-Video) can discriminate maneuvers at the same intersection
on nuScenes, extending the HDD VLM bridge to cross-dataset validation.

Two comparators per VLM:
  - Vision pooled (BoF): mean-pool per-frame vision embeddings -> cosine sim.
  - Vision seq DTW: per-frame pooled -> temporal derivative -> DTW similarity.

Protocol: nuScenes ego_pose DBSCAN clustering (eps=30m, min_samples=2),
CAN bus steering + yaw rate labels, 16 canonical PIL frames per segment,
1,000 bootstrap resamples for CIs, seed=42.

Hypothesis: Both comparators perform near chance (~0.50-0.55 AP) because
2D per-frame vision encoders cannot capture 3D ego-motion, matching the
HDD bridge findings and confirming the failure transfers cross-dataset.

Usage:
    # Gemma-4 SigLIP
    python experiments/eval_nuscenes_vlm_bridge.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family gemma4

    # LLaVA CLIP
    python experiments/eval_nuscenes_vlm_bridge.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family llava-video

    # With DINOv3/V-JEPA 2 baselines for comparison table
    python experiments/eval_nuscenes_vlm_bridge.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family gemma4 --include-baselines

    # Smoke test (2 clusters)
    python experiments/eval_nuscenes_vlm_bridge.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family llava-video --max-clusters 2
"""

import argparse
import json
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from video_retrieval.fingerprints import TemporalDerivativeFingerprint
from video_retrieval.fingerprints.dtw import dtw_distance_batch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLM_DEFAULT_PATHS = {
    "gemma4": "google/gemma-4-31B-it",
    "llava-video": "llava-hf/LLaVA-Video-7B-Qwen2-hf",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

VLM_DISPLAY_NAMES = {
    "gemma4": "Gemma-4 SigLIP",
    "llava-video": "LLaVA CLIP",
    "qwen3": "Qwen3 SigLIP",
}

MANEUVER_NAMES = {
    1: "straight",
    2: "left_turn",
    3: "right_turn",
}

DESCRIBE_PROMPT = "Describe the video."


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
# nuScenes Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ManeuverSegment:
    """A contiguous maneuver segment from one nuScenes scene."""

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
# nuScenes Metadata Loading (no devkit dependency)
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

    # Find CAM_FRONT sensor -> calibrated_sensor tokens
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

    # Build sample_token -> list of sample_data for fast lookup
    sample_to_data: dict[str, list[dict]] = defaultdict(list)
    for sd in sample_data:
        sample_to_data[sd["sample_token"]].append(sd)

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
            ego_pose = metadata.token_to_ego_pose.get(cam_front_sd["ego_pose_token"])
            if ego_pose is not None:
                ts_sec = cam_front_sd["timestamp"] / 1e6  # microseconds -> seconds
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
# CAN Bus Loading + Maneuver Segmentation
# ---------------------------------------------------------------------------


def load_can_bus(
    can_dir: Path, scene_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load CAN bus steering + pose data for a scene.

    Returns:
        (timestamps_sec, steering_angles, yaw_rates, positions_xy, speeds)
        or None if files not found.
    """
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
    pose_ts = np.array([p["utime"] / 1e6 for p in pose_data])  # -> seconds
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
            labels[i] = 0  # stationary -> skip
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
# Intersection Clustering
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
        Dict mapping cluster_id -> list of segments in that cluster.
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
# Assign Keyframes to Segments
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
# Discover nuScenes Segments (orchestration function)
# ---------------------------------------------------------------------------


def discover_nuscenes_segments(
    data_dir: Path,
    version: str,
    min_segment_duration: float = 2.0,
) -> tuple[
    list[ManeuverSegment],
    dict[str, list[tuple[float, str, float, float]]],
    NuScenesMetadata,
]:
    """Load nuScenes metadata and extract all maneuver segments.

    Returns:
        (all_segments, scene_keyframes, metadata)
    """
    # Step 1: Load metadata
    print("\n  Loading nuScenes metadata...")
    t0 = time.time()
    metadata = load_nuscenes_metadata(data_dir, version)
    print(f"  Metadata load time: {time.time() - t0:.1f}s")

    # Step 2: Load CAN bus + segment maneuvers
    print("\n  Segmenting maneuvers from CAN bus data...")
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
            min_duration=min_segment_duration,
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

    return all_segments, scene_keyframes, metadata


# ---------------------------------------------------------------------------
# Canonical Frame Sampling (from eval_hdd_vlm_bridge.py)
# ---------------------------------------------------------------------------


def compute_fps_eff(timestamps: list[float]) -> float:
    """Compute effective FPS from canonical frame timestamps.

    fps_eff = (N-1) / (t_last - t_first), representing the actual temporal
    spacing of sampled frames.
    """
    if len(timestamps) < 2:
        return 1.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 1.0
    return (len(timestamps) - 1) / duration


def sample_canonical_frames_from_keyframes(
    keyframe_paths: list[str],
    n_frames: int = 16,
    max_resolution: int = 518,
) -> tuple[list[Image.Image], list[float]]:
    """Sample uniformly spaced PIL frames from nuScenes keyframe images.

    nuScenes provides pre-extracted keyframes (JPEG images) rather than
    video files. This function loads keyframes and uniformly sub-samples
    or repeats to reach exactly n_frames.

    Returns:
        (pil_frames, synthetic_timestamps) where timestamps are evenly
        spaced [0, 1, ..., n_frames-1] / (n_frames-1) seconds.
    """
    if not keyframe_paths:
        return [], []

    # Load all available keyframes
    loaded_frames: list[Image.Image] = []
    for path in keyframe_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max_resolution and img.shape[0] > max_resolution:
            scale = max_resolution / img.shape[0]
            new_h = max_resolution
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h))
        loaded_frames.append(Image.fromarray(img))

    if not loaded_frames:
        return [], []

    # Uniformly sample n_frames from loaded frames (with repetition if needed)
    n_loaded = len(loaded_frames)
    selected_frames: list[Image.Image] = []
    for i in range(n_frames):
        idx = int(i * n_loaded / n_frames) if n_loaded > 1 else 0
        idx = min(idx, n_loaded - 1)
        selected_frames.append(loaded_frames[idx])

    # Synthetic timestamps (evenly spaced)
    if n_frames > 1:
        timestamps = [i / (n_frames - 1) * (n_loaded / 2.0) for i in range(n_frames)]
    else:
        timestamps = [0.0]

    return selected_frames, timestamps


# ---------------------------------------------------------------------------
# Feature Caching
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
# VLM Vision Adapters (copied from eval_hdd_vlm_bridge.py)
#
# Only vision-tower methods are needed: load, prepare_inputs,
# extract_vision_repr (pooled), extract_vision_seq (per-frame).
# No LLM inference, no generative methods, no layer ablation.
# ---------------------------------------------------------------------------


class VLMVisionAdapter(ABC):
    """Minimal adapter for VLM vision tower feature extraction."""

    @abstractmethod
    def load(
        self, model_path: str, device: torch.device, dtype=torch.bfloat16
    ) -> tuple[nn.Module, object]:
        """Load model + processor. Returns (model, processor)."""

    @abstractmethod
    def prepare_inputs(
        self,
        processor: object,
        frames: list[Image.Image],
        prompt: str,
        device: torch.device,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert PIL frames + prompt to model input tensors.

        The prompt is needed to produce correct pixel_values/position_ids
        even though we only use the vision tower.
        """

    @abstractmethod
    def extract_vision_repr(
        self, model: nn.Module, inputs: dict
    ) -> torch.Tensor | None:
        """Mean-pooled L2-normed vision tower embedding (D,)."""

    @abstractmethod
    def extract_vision_seq(
        self,
        model: nn.Module,
        inputs: dict,
        processor: object | None = None,
        frames: list[Image.Image] | None = None,
    ) -> torch.Tensor | None:
        """Per-frame vision tower embeddings as (T, D) sequence."""

    def extract_llm_repr(
        self, model: nn.Module, inputs: dict, layer_idx: int = -1
    ) -> torch.Tensor | None:
        """LLM-side representation at specified layer.

        Mean-pools all hidden states at the given layer, L2-normalizes.
        Returns (D,) vector or None on failure.
        """
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)
                emb = F.normalize(hidden.mean(dim=0), dim=0)
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"{type(self).__name__}.extract_llm_repr failed: {e}")
            return None


class GemmaVisionAdapter(VLMVisionAdapter):
    """Adapter for Gemma-4 SigLIP vision tower.

    Gemma-4 uses SigLIP-SO400M as its vision encoder. The vision tower
    requires pixel_position_ids from the processor to handle multi-image
    inputs. Output tokens are grouped by source image using position IDs.
    """

    def load(self, model_path, device, dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return model, processor

    def prepare_inputs(self, processor, frames, prompt, device, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f} for f in frames],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=frames,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def extract_vision_repr(self, model, inputs):
        try:
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return None
            with torch.no_grad():
                kwargs = {}
                pos_ids = inputs.get(
                    "image_position_ids", inputs.get("pixel_position_ids")
                )
                if pos_ids is not None:
                    kwargs["pixel_position_ids"] = pos_ids
                vision_out = model.model.vision_tower(pixel_values, **kwargs)
                tokens = vision_out.last_hidden_state
                emb = F.normalize(
                    tokens.reshape(-1, tokens.shape[-1]).mean(dim=0), dim=0
                )
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"GemmaVisionAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_seq(self, model, inputs, processor=None, frames=None):
        try:
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return None

            pos_ids = inputs.get("image_position_ids", inputs.get("pixel_position_ids"))
            if pos_ids is None:
                return None

            with torch.no_grad():
                vision_out = model.model.vision_tower(
                    pixel_values,
                    pixel_position_ids=pos_ids,
                )
                tokens = vision_out.last_hidden_state

                if tokens.ndim == 3:
                    # (T, N_tokens, D) — already per-frame
                    per_frame = F.normalize(tokens.mean(dim=1), dim=-1)
                elif tokens.ndim == 2:
                    # (total_tokens, D) — use position IDs to group by frame
                    if pos_ids.ndim == 2 and pos_ids.shape[1] >= 1:
                        img_indices = pos_ids[:, 0]  # image index per token
                        unique_imgs = img_indices.unique(sorted=True)
                        n_images = len(unique_imgs)

                        if n_images <= 1:
                            return None
                        frame_embs = []
                        for img_idx in unique_imgs:
                            mask = img_indices == img_idx
                            frame_tokens = tokens[mask]  # (N_i, D)
                            frame_embs.append(
                                F.normalize(frame_tokens.mean(dim=0), dim=0)
                            )
                        per_frame = torch.stack(frame_embs)  # (T, D)
                    else:
                        # Fallback: even split by number of input frames
                        T = len(frames) if frames is not None else 0
                        if T <= 1:
                            return None
                        toks_per_img = len(tokens) // T
                        if toks_per_img == 0:
                            return None
                        per_frame = torch.stack(
                            [
                                F.normalize(
                                    tokens[
                                        i * toks_per_img : (i + 1) * toks_per_img
                                    ].mean(dim=0),
                                    dim=0,
                                )
                                for i in range(T)
                            ]
                        )
                else:
                    return None

            return per_frame.cpu()
        except Exception as e:
            warnings.warn(f"GemmaVisionAdapter.extract_vision_seq failed: {e}")
            return None


class LlavaVisionAdapter(VLMVisionAdapter):
    """Adapter for LLaVA-Video CLIP-ViT-L vision tower.

    LLaVA-Video uses CLIP-ViT-L/14 as its vision encoder. Pixel values
    arrive as 5D tensors (B, T, C, H, W) for video input and are reshaped
    to 4D for the CLIP tower. Uses hidden_states[-2] (second-to-last layer)
    with CLS token skipped.
    """

    def load(self, model_path, device, dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return model, processor

    def prepare_inputs(self, processor, frames, prompt, device, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            videos=[frames],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def extract_vision_repr(self, model, inputs):
        try:
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                return None
            with torch.no_grad():
                # pixel_values may be 5D (B, T, C, H, W) for video — flatten
                # to 4D (B*T, C, H, W) for CLIP vision tower
                if pixel_values.ndim == 5:
                    B, T, C, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(B * T, C, H, W)
                vision_out = model.model.vision_tower(
                    pixel_values,
                    output_hidden_states=True,
                )
                # Use second-to-last hidden state, skip CLS token
                tokens = vision_out.hidden_states[-2][:, 1:]
                emb = F.normalize(
                    tokens.reshape(-1, tokens.shape[-1]).mean(dim=0),
                    dim=0,
                )
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"LlavaVisionAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_seq(self, model, inputs, processor=None, frames=None):
        try:
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                return None
            with torch.no_grad():
                if pixel_values.ndim == 5:
                    B, T, C, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(B * T, C, H, W)
                vision_out = model.model.vision_tower(
                    pixel_values,
                    output_hidden_states=True,
                )
                # (T, N_patches, D) — pool patches per frame, keep T
                tokens = vision_out.hidden_states[-2][:, 1:]  # skip CLS
                per_frame = F.normalize(tokens.mean(dim=1), dim=-1)  # (T, D)
            return per_frame.cpu()
        except Exception as e:
            warnings.warn(f"LlavaVisionAdapter.extract_vision_seq failed: {e}")
            return None


class QwenVisionAdapter(VLMVisionAdapter):
    """Adapter for Qwen3-VL vision tower.

    Qwen3-VL uses a ViT with 3D temporal-spatial merging. The vision tower
    is accessed via model.visual(pixel_values, grid_thw=grid_thw). Tokens
    are grouped by frame using grid_thw[0, 0] for the temporal dimension.
    """

    def load(self, model_path, device, dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return model, processor

    def prepare_inputs(self, processor, frames, prompt, device, **kwargs):
        from qwen_vl_utils import process_vision_info

        fps = kwargs.get("fps", 1.0)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        video_kwargs = {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
            for k, v in (video_kwargs or {}).items()
        }
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    @staticmethod
    def _get_visual(model):
        """Resolve the vision tower — Qwen2-VL uses model.visual, Qwen3-VL uses model.model.visual."""
        if hasattr(model, "visual"):
            return model.visual
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            return model.model.visual
        raise AttributeError("Cannot find .visual on Qwen model")

    @staticmethod
    def _unwrap_vision_output(output):
        """Qwen2-VL returns a raw tensor, Qwen3-VL returns BaseModelOutputWithDeepstackFeatures."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        raise TypeError(f"Unexpected vision output type: {type(output)}")

    def extract_vision_repr(self, model, inputs):
        try:
            pixel_values = inputs.get(
                "pixel_values_videos",
                inputs.get("pixel_values"),
            )
            grid_thw = inputs.get(
                "video_grid_thw",
                inputs.get("image_grid_thw"),
            )
            if pixel_values is None or grid_thw is None:
                return None

            visual = self._get_visual(model)
            with torch.no_grad():
                vision_tokens = self._unwrap_vision_output(
                    visual(pixel_values, grid_thw=grid_thw)
                )
                emb = F.normalize(vision_tokens.mean(dim=0), dim=0)
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"QwenVisionAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_seq(self, model, inputs, processor=None, frames=None):
        try:
            pixel_values = inputs.get(
                "pixel_values_videos",
                inputs.get("pixel_values"),
            )
            grid_thw = inputs.get(
                "video_grid_thw",
                inputs.get("image_grid_thw"),
            )
            if pixel_values is None or grid_thw is None:
                return None

            visual = self._get_visual(model)
            with torch.no_grad():
                vision_tokens = self._unwrap_vision_output(
                    visual(pixel_values, grid_thw=grid_thw)
                )
                n_tokens = vision_tokens.shape[0]
                n_frames = int(grid_thw[0, 0].item())
                tokens_per_frame = n_tokens // max(n_frames, 1)

                if tokens_per_frame < 1 or n_frames < 2:
                    return None

                frame_embs = []
                for i in range(n_frames):
                    start = i * tokens_per_frame
                    end = (i + 1) * tokens_per_frame if i < n_frames - 1 else n_tokens
                    frame_embs.append(
                        F.normalize(vision_tokens[start:end].mean(dim=0), dim=0)
                    )
                per_frame = torch.stack(frame_embs)  # (T, D)
            return per_frame.cpu()
        except Exception as e:
            warnings.warn(f"QwenVisionAdapter.extract_vision_seq failed: {e}")
            return None


VLM_ADAPTERS = {
    "gemma4": GemmaVisionAdapter,
    "llava-video": LlavaVisionAdapter,
    "qwen3": QwenVisionAdapter,
}


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_vlm_vision_features(
    adapter: VLMVisionAdapter,
    model: nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    n_frames: int = 16,
    device: torch.device | None = None,
    cache_path: Path | None = None,
    extract_llm: bool = False,
) -> dict[int, dict]:
    """Extract VLM vision tower features for all nuScenes maneuver segments.

    For each segment, extracts:
    - vision_repr: L2-normed mean-pooled vision embedding (D,)
    - vision_seq: L2-normed per-frame vision embeddings (T, D)
    - llm_repr: (optional) L2-normed mean-pooled LLM hidden state (D,)

    Args:
        adapter: VLM vision adapter instance.
        model: Loaded VLM model.
        processor: VLM processor/tokenizer.
        segments: NuScenes maneuver segments to process.
        n_frames: Canonical number of frames to sample per clip.
        device: Torch device.
        cache_path: Optional path for caching features to disk.
        extract_llm: If True, also extract LLM hidden state representations.

    Returns:
        Dict mapping segment index -> {
            'vision_repr': (D,) pooled embedding,
            'vision_seq': (T, D) per-frame sequence or None,
            'llm_repr': (D,) LLM hidden state or None (if extract_llm),
        }
    """
    if cache_path:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            return cached

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting VLM vision features")):
        try:
            pil_frames, timestamps = sample_canonical_frames_from_keyframes(
                seg.keyframe_paths,
                n_frames=n_frames,
                max_resolution=518,
            )
            if len(pil_frames) < 3:
                failed += 1
                continue

            fps_eff = compute_fps_eff(timestamps)
            inputs = adapter.prepare_inputs(
                processor,
                pil_frames,
                DESCRIBE_PROMPT,
                device,
                fps=fps_eff,
            )

            vision_repr = adapter.extract_vision_repr(model, inputs)
            vision_seq = adapter.extract_vision_seq(
                model,
                inputs,
                processor=processor,
                frames=pil_frames,
            )

            if vision_repr is None:
                failed += 1
                continue

            feat = {
                "vision_repr": vision_repr,
                "vision_seq": vision_seq,
            }
            if extract_llm:
                llm_repr = adapter.extract_llm_repr(model, inputs)
                feat["llm_repr"] = llm_repr

            features[i] = feat
        except Exception as e:
            warnings.warn(f"Segment {i} failed: {e}")
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")

    if cache_path:
        save_feature_cache(features, cache_path)

    return features


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------


def compute_vlm_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise VLM vision similarities within each cluster.

    Two comparators:
    - vision_pooled: cosine similarity of L2-normed pooled embeddings.
    - vision_seq_dtw: temporal derivative DTW on per-frame sequences.
      Uses the same TemporalDerivativeFingerprint and unnormalized DTW
      as the DINOv3 temporal derivative baseline (alpha=1).

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deriv_fp = TemporalDerivativeFingerprint()

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
    print(f"  Total pairs: {total_pairs}")

    if total_pairs == 0:
        return {}

    # --- Vision pooled: batch dot product ---
    print("  Computing vision pooled (cosine) similarities...")
    # Cast to float32 (VLM features are bf16)
    pooled_a = (
        torch.stack([features[i]["vision_repr"] for i in pair_a_indices])
        .float()
        .to(device)
    )
    pooled_b = (
        torch.stack([features[i]["vision_repr"] for i in pair_b_indices])
        .float()
        .to(device)
    )
    pooled_sims = (pooled_a * pooled_b).sum(dim=1).cpu().tolist()

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "vision_pooled": (pooled_sims, list(pair_gts)),
    }

    # --- Vision seq DTW: temporal derivative + batched GPU DTW ---
    # Only for segments where vision_seq is not None
    dtw_mask = [
        (
            features[a]["vision_seq"] is not None
            and features[b]["vision_seq"] is not None
        )
        for a, b in zip(pair_a_indices, pair_b_indices)
    ]
    dtw_a = [a for a, m in zip(pair_a_indices, dtw_mask) if m]
    dtw_b = [b for b, m in zip(pair_b_indices, dtw_mask) if m]
    dtw_gts = [g for g, m in zip(pair_gts, dtw_mask) if m]

    if dtw_a:
        print("  Computing vision seq DTW similarities...")
        # Pre-compute temporal derivative fingerprints (cast to float32)
        deriv_fps_cache: dict[int, torch.Tensor] = {}
        for idx in set(dtw_a + dtw_b):
            seq = features[idx]["vision_seq"].float()  # (T, D)
            deriv_fps_cache[idx] = deriv_fp.compute_fingerprint(seq)

        deriv_seqs_a = [deriv_fps_cache[i].to(device) for i in dtw_a]
        deriv_seqs_b = [deriv_fps_cache[i].to(device) for i in dtw_b]
        # normalize=False: same as DINOv3 temporal derivative baseline
        deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
        dtw_sims = torch.exp(-deriv_dists).cpu().tolist()

        all_scores["vision_seq_dtw"] = (dtw_sims, list(dtw_gts))
    else:
        print("  WARNING: No valid vision_seq pairs for DTW computation")

    # --- LLM hidden state: cosine similarity ---
    llm_mask = [
        (
            features[a].get("llm_repr") is not None
            and features[b].get("llm_repr") is not None
        )
        for a, b in zip(pair_a_indices, pair_b_indices)
    ]
    llm_a_idx = [a for a, m in zip(pair_a_indices, llm_mask) if m]
    llm_b_idx = [b for b, m in zip(pair_b_indices, llm_mask) if m]
    llm_gts = [g for g, m in zip(pair_gts, llm_mask) if m]

    if llm_a_idx:
        print("  Computing LLM hidden state (cosine) similarities...")
        llm_a = (
            torch.stack([features[i]["llm_repr"] for i in llm_a_idx])
            .float()
            .to(device)
        )
        llm_b = (
            torch.stack([features[i]["llm_repr"] for i in llm_b_idx])
            .float()
            .to(device)
        )
        llm_sims = (llm_a * llm_b).sum(dim=1).cpu().tolist()
        all_scores["llm_hidden"] = (llm_sims, list(llm_gts))

    return all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VLM Vision Tower Bridge Experiment on nuScenes"
    )
    parser.add_argument(
        "--nuscenes-dir",
        type=str,
        required=True,
        help="Path to nuScenes data root",
    )
    parser.add_argument(
        "--vlm-family",
        type=str,
        required=True,
        choices=list(VLM_ADAPTERS.keys()),
        help="VLM family to evaluate",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="Override model path (default: use VLM_DEFAULT_PATHS)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of mixed clusters to evaluate",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Number of canonical frames per clip (shared across VLMs)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes version to use",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Load pair_scores.json for DINOv3/V-JEPA 2 comparison table",
    )
    parser.add_argument(
        "--extract-llm",
        action="store_true",
        default=True,
        help="Also extract LLM hidden state representations (default: True)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.nuscenes_dir)
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    display_name = VLM_DISPLAY_NAMES[args.vlm_family]

    print("=" * 70)
    print(f"VLM VISION TOWER BRIDGE: {display_name} on nuScenes")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Version: {args.version}")
    print(f"  Frames per clip: {args.n_frames}")

    # ------------------------------------------------------------------
    # Step 1: Discover nuScenes segments
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering nuScenes segments...")
    t0 = time.time()
    all_segments, scene_keyframes, metadata = discover_nuscenes_segments(
        data_dir,
        args.version,
    )
    print(f"  Discovery time: {time.time() - t0:.1f}s")

    if not all_segments:
        print("\n  No maneuver segments found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2: Cluster intersections
    # ------------------------------------------------------------------
    print(f"\nStep 2: Clustering intersections (DBSCAN eps=30m)...")
    clusters = cluster_intersections(
        all_segments,
        eps=30.0,
        min_samples=2,
    )
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 3: Filter for mixed clusters
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
        if args.version == "v1.0-mini":
            print("  This is expected for the mini split (10 scenes).")
            print("  Use --version v1.0-trainval for real results.")
        return

    # Build flat list of segments in qualifying clusters, with cluster mapping
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    # Assign keyframes to eval segments
    assign_keyframes_to_segments(eval_segments, scene_keyframes)

    eval_label_counts: dict[int, int] = defaultdict(int)
    for seg in eval_segments:
        eval_label_counts[seg.label] += 1
    print(f"\n  Segments for evaluation: {len(eval_segments)}")
    for lv, name in sorted(MANEUVER_NAMES.items()):
        if lv in (2, 3):
            print(f"    {name}: {eval_label_counts.get(lv, 0)}")

    kf_counts = [len(s.keyframe_paths) for s in eval_segments]
    if kf_counts:
        print(
            f"  Keyframes per segment: min={min(kf_counts)}, "
            f"max={max(kf_counts)}, mean={np.mean(kf_counts):.1f}"
        )

    # ------------------------------------------------------------------
    # Step 4: Load VLM and extract vision features
    # ------------------------------------------------------------------
    print(f"\nStep 4: Loading {display_name} model...")
    adapter = VLM_ADAPTERS[args.vlm_family]()
    model, processor = adapter.load(model_path, torch.device(args.device))

    cache_dir = project_root / "datasets" / "nuscenes" / "cache"
    cache_suffix = "_llm" if args.extract_llm else ""
    cache_path = cache_dir / f"vlm_bridge_{args.vlm_family}{cache_suffix}_features.pt"

    print(f"\nStep 5: Extracting {display_name} vision features...")
    t_feat_start = time.time()
    features = extract_vlm_vision_features(
        adapter,
        model,
        processor,
        eval_segments,
        n_frames=args.n_frames,
        device=torch.device(args.device),
        cache_path=cache_path,
        extract_llm=args.extract_llm,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    # Free model memory
    del model, processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 6: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_vlm_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        device=torch.device(args.device),
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {display_name} — nuScenes MANEUVER DISCRIMINATION")
    print("=" * 70)

    results = {}
    for method_name in ["vision_pooled", "vision_seq_dtw", "llm_hidden"]:
        if method_name not in all_scores:
            continue
        scores_list, labels_list = all_scores[method_name]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method_name] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap_val, ci_lo, ci_hi = bootstrap_ap(scores, labels)
        auc = roc_auc_score(labels, scores)
        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())
        gap = same_mean - diff_mean

        results[method_name] = {
            "ap": float(ap_val),
            "ap_ci_low": ci_lo,
            "ap_ci_high": ci_hi,
            "auc": float(auc),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": gap,
        }

        method_display = f"{display_name} {method_name}"
        print(
            f"  {method_display:<35s}  AP={ap_val:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"AUC={auc:.4f}  gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    # ------------------------------------------------------------------
    # Comparison with baselines (optional)
    # ------------------------------------------------------------------
    baseline_results: dict[str, dict] = {}
    if args.include_baselines:
        print("\n  -- Running DINOv3 + V-JEPA 2 baselines --")

        # Import baseline models
        from video_retrieval.fingerprints.trajectory import dtw_distance
        from video_retrieval.models import DINOv3Encoder

        # --- DINOv3 baselines ---
        dinov3_cache_path = (
            project_root
            / "datasets"
            / "nuscenes"
            / "cache"
            / f"dinov3_nuscenes_{args.version}.pt"
        )
        dinov3_features = load_feature_cache(dinov3_cache_path)

        if dinov3_features is None:
            print("  Loading DINOv3 encoder...")
            encoder = DINOv3Encoder(
                model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
                device=args.device,
            )

            print("  Extracting DINOv3 features...")
            dinov3_features = {}
            failed_dino = 0
            for i, seg in enumerate(
                tqdm(eval_segments, desc="Extracting DINOv3 features")
            ):
                if len(seg.keyframe_paths) < 3:
                    failed_dino += 1
                    continue
                try:
                    frames = []
                    for path in seg.keyframe_paths:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img.shape[0] > 518:
                            scale = 518 / img.shape[0]
                            new_h = 518
                            new_w = int(img.shape[1] * scale)
                            img = cv2.resize(img, (new_w, new_h))
                        frames.append(img)
                    if len(frames) < 3:
                        failed_dino += 1
                        continue
                    emb = encoder.encode_frames(frames)
                    mean_emb = F.normalize(emb.mean(dim=0), dim=0)
                    dinov3_features[i] = {
                        "embeddings": emb,
                        "mean_emb": mean_emb,
                    }
                except Exception:
                    failed_dino += 1
                    continue
            print(
                f"  DINOv3 extracted: {len(dinov3_features)}/{len(eval_segments)} "
                f"({failed_dino} failed)"
            )
            save_feature_cache(dinov3_features, dinov3_cache_path)
            del encoder
            torch.cuda.empty_cache()
        else:
            print("  DINOv3 features loaded from cache")

        # Compute DINOv3 BoF and Chamfer similarities
        print("  Computing DINOv3 baseline similarities...")
        bof_scores_list: list[float] = []
        bof_labels_list: list[int] = []
        chamfer_scores_list: list[float] = []
        chamfer_labels_list: list[int] = []

        for cid in sorted(cluster_to_indices.keys()):
            indices = [i for i in cluster_to_indices[cid] if i in dinov3_features]
            for a_pos in range(len(indices)):
                for b_pos in range(a_pos + 1, len(indices)):
                    a_idx = indices[a_pos]
                    b_idx = indices[b_pos]
                    gt = (
                        1
                        if eval_segments[a_idx].label == eval_segments[b_idx].label
                        else 0
                    )

                    # Bag-of-frames (cosine of mean embeddings)
                    bof_sim = float(
                        torch.dot(
                            dinov3_features[a_idx]["mean_emb"],
                            dinov3_features[b_idx]["mean_emb"],
                        ).item()
                    )
                    bof_scores_list.append(bof_sim)
                    bof_labels_list.append(gt)

                    # Chamfer (max symmetric)
                    ea = dinov3_features[a_idx]["embeddings"]
                    eb = dinov3_features[b_idx]["embeddings"]
                    sim_matrix = torch.mm(ea, eb.t())
                    max_ab = sim_matrix.max(dim=1).values.mean().item()
                    max_ba = sim_matrix.max(dim=0).values.mean().item()
                    chamfer_sim = (max_ab + max_ba) / 2
                    chamfer_scores_list.append(chamfer_sim)
                    chamfer_labels_list.append(gt)

        if bof_scores_list:
            bl_scores_arr = np.array(bof_scores_list)
            bl_labels_arr = np.array(bof_labels_list)
            if bl_labels_arr.sum() > 0 and bl_labels_arr.sum() < len(bl_labels_arr):
                bl_ap, bl_ci_lo, bl_ci_hi = bootstrap_ap(bl_scores_arr, bl_labels_arr)
                bl_auc = roc_auc_score(bl_labels_arr, bl_scores_arr)
                baseline_results["DINOv3 BoF"] = {
                    "ap": bl_ap,
                    "ci_lo": bl_ci_lo,
                    "ci_hi": bl_ci_hi,
                    "auc": bl_auc,
                }

        if chamfer_scores_list:
            bl_scores_arr = np.array(chamfer_scores_list)
            bl_labels_arr = np.array(chamfer_labels_list)
            if bl_labels_arr.sum() > 0 and bl_labels_arr.sum() < len(bl_labels_arr):
                bl_ap, bl_ci_lo, bl_ci_hi = bootstrap_ap(bl_scores_arr, bl_labels_arr)
                bl_auc = roc_auc_score(bl_labels_arr, bl_scores_arr)
                baseline_results["DINOv3 Chamfer"] = {
                    "ap": bl_ap,
                    "ci_lo": bl_ci_lo,
                    "ci_hi": bl_ci_hi,
                    "auc": bl_auc,
                }

        # --- V-JEPA 2 baselines ---
        VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
        VJEPA2_NUM_FRAMES = 64
        VJEPA2_T_PATCHES = 32  # 64 frames / tubelet_size 2
        VJEPA2_SPATIAL = 256  # 16h x 16w

        vjepa2_cache_path = (
            project_root
            / "datasets"
            / "nuscenes"
            / "cache"
            / f"vjepa2_nuscenes_{args.version}.pt"
        )
        vjepa2_features = load_feature_cache(vjepa2_cache_path)

        if vjepa2_features is None:
            print("  Loading V-JEPA 2 model...")
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

            n_context_steps = VJEPA2_T_PATCHES // 2
            n_target_steps = VJEPA2_T_PATCHES - n_context_steps
            # Build temporal masks
            all_indices = torch.arange(
                VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=torch.device(args.device)
            )
            grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)
            context_mask = grid[:n_context_steps].reshape(-1).unsqueeze(0)
            target_mask = grid[n_context_steps:].reshape(-1).unsqueeze(0)

            print("  Extracting V-JEPA 2 features...")
            vjepa2_features = {}
            failed_vjepa = 0
            for i, seg in enumerate(
                tqdm(eval_segments, desc="Extracting V-JEPA 2 features")
            ):
                try:
                    # Load keyframe images, pad/truncate to VJEPA2_NUM_FRAMES
                    paths = seg.keyframe_paths
                    frames = []
                    for path in paths:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img.shape[0] > 256:
                            scale = 256 / img.shape[0]
                            new_h = 256
                            new_w = int(img.shape[1] * scale)
                            img = cv2.resize(img, (new_w, new_h))
                        frames.append(img)
                    if not frames:
                        failed_vjepa += 1
                        continue
                    while len(frames) < VJEPA2_NUM_FRAMES:
                        frames = frames + frames
                    frames = frames[:VJEPA2_NUM_FRAMES]

                    # pyrefly: ignore [not-callable]
                    inputs = vjepa2_processor(videos=frames, return_tensors="pt")
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        # Encoder only: bag-of-tokens embedding
                        enc_out = vjepa2_model(**inputs, skip_predictor=True)
                        encoder_tokens = enc_out.last_hidden_state[0]
                        mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

                        # With predictor: temporal residuals
                        pred_out = vjepa2_model(
                            **inputs,
                            context_mask=[context_mask],
                            target_mask=[target_mask],
                        )
                        predicted = pred_out.predictor_output.last_hidden_state[0]
                        ground_truth = pred_out.predictor_output.target_hidden_state[0]

                        predicted = predicted.reshape(
                            n_target_steps, VJEPA2_SPATIAL, -1
                        )
                        ground_truth = ground_truth.reshape(
                            n_target_steps, VJEPA2_SPATIAL, -1
                        )
                        residual = (predicted - ground_truth).mean(dim=1)

                    vjepa2_features[i] = {
                        "mean_emb": mean_emb.cpu(),
                        "temporal_residual": residual.cpu(),
                    }
                except Exception:
                    failed_vjepa += 1
                    continue
            print(
                f"  V-JEPA 2 extracted: {len(vjepa2_features)}/{len(eval_segments)} "
                f"({failed_vjepa} failed)"
            )
            save_feature_cache(vjepa2_features, vjepa2_cache_path)
            del vjepa2_model, vjepa2_processor
            torch.cuda.empty_cache()
        else:
            print("  V-JEPA 2 features loaded from cache")

        # Compute V-JEPA 2 BoT and TempRes similarities
        print("  Computing V-JEPA 2 baseline similarities...")
        bot_scores_list: list[float] = []
        bot_labels_list: list[int] = []
        tres_scores_list: list[float] = []
        tres_labels_list: list[int] = []

        for cid in sorted(cluster_to_indices.keys()):
            indices = [i for i in cluster_to_indices[cid] if i in vjepa2_features]
            for a_pos in range(len(indices)):
                for b_pos in range(a_pos + 1, len(indices)):
                    a_idx = indices[a_pos]
                    b_idx = indices[b_pos]
                    gt = (
                        1
                        if eval_segments[a_idx].label == eval_segments[b_idx].label
                        else 0
                    )

                    va = vjepa2_features[a_idx]
                    vb = vjepa2_features[b_idx]

                    bot_sim = float(torch.dot(va["mean_emb"], vb["mean_emb"]).item())
                    bot_scores_list.append(bot_sim)
                    bot_labels_list.append(gt)

                    res_dist = dtw_distance(
                        va["temporal_residual"],
                        vb["temporal_residual"],
                        normalize=True,
                    )
                    res_sim = float(torch.exp(torch.tensor(-res_dist)).item())
                    tres_scores_list.append(res_sim)
                    tres_labels_list.append(gt)

        if bot_scores_list:
            bl_scores_arr = np.array(bot_scores_list)
            bl_labels_arr = np.array(bot_labels_list)
            if bl_labels_arr.sum() > 0 and bl_labels_arr.sum() < len(bl_labels_arr):
                bl_ap, bl_ci_lo, bl_ci_hi = bootstrap_ap(bl_scores_arr, bl_labels_arr)
                bl_auc = roc_auc_score(bl_labels_arr, bl_scores_arr)
                baseline_results["V-JEPA 2 BoT"] = {
                    "ap": bl_ap,
                    "ci_lo": bl_ci_lo,
                    "ci_hi": bl_ci_hi,
                    "auc": bl_auc,
                }

        if tres_scores_list:
            bl_scores_arr = np.array(tres_scores_list)
            bl_labels_arr = np.array(tres_labels_list)
            if bl_labels_arr.sum() > 0 and bl_labels_arr.sum() < len(bl_labels_arr):
                bl_ap, bl_ci_lo, bl_ci_hi = bootstrap_ap(bl_scores_arr, bl_labels_arr)
                bl_auc = roc_auc_score(bl_labels_arr, bl_scores_arr)
                baseline_results["V-JEPA 2 Temp. Res."] = {
                    "ap": bl_ap,
                    "ci_lo": bl_ci_lo,
                    "ci_hi": bl_ci_hi,
                    "auc": bl_auc,
                }

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"  {'Method':<35s}  {'AP':>6s}  {'95% CI':>15s}  {'AUC':>6s}"
    print(header)
    print("  " + "-" * 66)

    # VLM results
    for method_name in ["vision_pooled", "vision_seq_dtw", "llm_hidden"]:
        if method_name not in results:
            continue
        r = results[method_name]
        method_display = f"{display_name} {method_name}"
        if "ap_ci_low" in r:
            ci_str = f"[{r['ap_ci_low']:.4f}, {r['ap_ci_high']:.4f}]"
            print(
                f"  {method_display:<35s}  {r['ap']:6.4f}  {ci_str:>15s}  "
                f"{r['auc']:6.4f}"
            )
        else:
            print(f"  {method_display:<35s}  {'N/A':>6s}  {'N/A':>15s}  {'N/A':>6s}")

    # Baseline results
    if baseline_results:
        print("  " + "-" * 66)
        for bl_name, bl_r in baseline_results.items():
            ci_str = f"[{bl_r['ci_lo']:.4f}, {bl_r['ci_hi']:.4f}]"
            print(
                f"  {bl_name:<35s}  {bl_r['ap']:6.4f}  {ci_str:>15s}  "
                f"{bl_r['auc']:6.4f}"
            )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = project_root / "datasets" / "nuscenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vlm_bridge_{args.vlm_family}_results.json"

    output = {
        "vlm_family": args.vlm_family,
        "model_path": model_path,
        "display_name": display_name,
        "dataset": "nuscenes",
        "version": args.version,
        "n_frames": args.n_frames,
        "n_segments_total": len(eval_segments),
        "n_segments_extracted": len(features),
        "n_clusters": len(mixed),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Dump pair-level scores for future analysis
    pair_path = out_dir / f"vlm_bridge_{args.vlm_family}_pair_scores.json"
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    # Summary
    print("\nSummary:")
    print(f"  Dataset: nuScenes ({args.version})")
    print(f"  VLM: {display_name}")
    print(f"  Scenes: {len(metadata.scenes)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Intersection clusters: {len(clusters)}")
    print(f"  Mixed clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    if all_scores:
        n_total_pairs = len(next(iter(all_scores.values()))[0])
        print(f"  Total pairs evaluated: {n_total_pairs}")

    print("\nDone.")


if __name__ == "__main__":
    main()

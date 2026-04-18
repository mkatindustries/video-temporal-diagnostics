#!/usr/bin/env python3
"""VLM Generative Probes on nuScenes — Forward/Reverse Direction Classification.

Runs VLM generative probes (forward/reverse direction classification) on
nuScenes maneuver segments. Fills 3 cells in Table 8 (VLM gen x nuScenes
for Gemma-4, LLaVA, Qwen3).

Protocol: nuScenes ego_pose DBSCAN clustering (eps=30m, min_samples=2),
CAN bus steering + yaw rate labels, 16 canonical PIL frames per segment,
3 direction prompts per segment (forward + reversed ordering), balanced
accuracy metric, seed=42.

Hypothesis: All three VLMs perform near chance (~0.50) because generative
VLMs cannot reliably detect temporal direction in driving video, consistent
with the EPIC-Kitchens and VCDB generative probe findings.

Usage:
    # Gemma-4
    python experiments/eval_nuscenes_vlm_generative.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family gemma4

    # LLaVA-Video
    python experiments/eval_nuscenes_vlm_generative.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family llava-video

    # Qwen3-VL
    python experiments/eval_nuscenes_vlm_generative.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family qwen3

    # Smoke test (2 clusters)
    python experiments/eval_nuscenes_vlm_generative.py \\
        --nuscenes-dir /path/to/nuscenes --vlm-family gemma4 --max-clusters 2
"""

import argparse
import json
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLM_DEFAULT_PATHS = {
    "gemma4": "google/gemma-4-31B-it",
    "llava-video": "llava-hf/LLaVA-Video-7B-Qwen2-hf",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

MANEUVER_NAMES = {
    1: "straight",
    2: "left_turn",
    3: "right_turn",
}

DIRECTION_PROMPTS = [
    "Is this video playing forward or in reverse? "
    "Answer with only FORWARD or REVERSE.",
    "Watch this video carefully. Is the temporal order normal (forward) "
    "or reversed? Reply FORWARD or REVERSE only.",
    "Determine if this video is playing in its original direction or has "
    "been reversed. Answer: FORWARD or REVERSE",
]


# ---------------------------------------------------------------------------
# nuScenes Data Structures (from eval_nuscenes_vlm_bridge.py)
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
# Canonical Frame Sampling
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
# Response Parsing
# ---------------------------------------------------------------------------


def parse_forward_reverse(response: str) -> str | None:
    """Parse FORWARD or REVERSE from model response."""
    upper = response.upper()
    if "FORWARD" in upper:
        return "FORWARD"
    elif "REVERSE" in upper:
        return "REVERSE"
    return None


# ---------------------------------------------------------------------------
# VLM Generative Adapters (from eval_vcdb_vlm_probes.py)
# ---------------------------------------------------------------------------


class VLMAdapter(ABC):
    """Uniform interface for VLM loading and generative inference."""

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
        """Convert PIL frames + prompt -> model input tensors."""

    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        inputs: dict,
        processor: object,
        max_new_tokens: int = 64,
    ) -> str:
        """Run generative inference, return decoded text."""

    def prepare_text_only_inputs(
        self,
        processor: object,
        prompt: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Prepare text-only inputs (no video/image) for bias testing."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement prepare_text_only_inputs"
        )


class GemmaAdapter(VLMAdapter):
    """Adapter for Gemma 4 models (multi-image video input)."""

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

    def prepare_text_only_inputs(self, processor, prompt, device):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def generate(self, model, inputs, processor, max_new_tokens=128):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            # Strip thinking block if present
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()


class LlavaVideoAdapter(VLMAdapter):
    """Adapter for LLaVA-Video (HF-converted llava-hf variant)."""

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

    def prepare_text_only_inputs(self, processor, prompt, device):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def generate(self, model, inputs, processor, max_new_tokens=128):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()


class QwenAdapter(VLMAdapter):
    """Adapter for Qwen3-VL models (generative probes only).

    Qwen3-VL uses a thinking mode (<think>...</think>) by default.
    This adapter parses past thinking tokens to extract the final answer,
    and uses a larger max_new_tokens budget to accommodate the thinking chain.
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

    def prepare_text_only_inputs(self, processor, prompt, device):
        """Prepare text-only inputs (no frames) for baseline measurement."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def generate(self, model, inputs, processor, max_new_tokens=256):
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()


VLM_ADAPTERS = {
    "gemma4": GemmaAdapter,
    "llava-video": LlavaVideoAdapter,
    "qwen3": QwenAdapter,
}


# ---------------------------------------------------------------------------
# Direction Classification Probe
# ---------------------------------------------------------------------------


def run_direction_probe(
    segments: list[ManeuverSegment],
    adapter: VLMAdapter,
    model: nn.Module,
    processor: object,
    device: torch.device,
    n_frames: int,
    prompts: list[str],
) -> dict:
    """Run forward/reverse direction classification with multiple prompts.

    For each segment: load keyframe images, run forward + reversed ordering
    with each prompt, compute balanced accuracy.

    Returns per-prompt and mean balanced accuracy, plus per-segment results.
    """
    per_prompt_results = {}
    all_balanced_accs = []
    per_segment_details = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  --- Direction Prompt {prompt_idx} ---")
        fwd_correct = 0
        rev_correct = 0
        fwd_total = 0
        rev_total = 0
        all_fwd_preds = []
        all_rev_preds = []

        for seg_idx, seg in enumerate(
            tqdm(segments, desc=f"Direction prompt_{prompt_idx}")
        ):
            try:
                pil_frames, timestamps = sample_canonical_frames_from_keyframes(
                    seg.keyframe_paths,
                    n_frames=n_frames,
                    max_resolution=518,
                )
                if len(pil_frames) < 3:
                    continue

                fps_eff = compute_fps_eff(timestamps)

                # Forward
                inputs = adapter.prepare_inputs(
                    processor,
                    pil_frames,
                    prompt,
                    device,
                    fps=fps_eff,
                )
                resp_fwd = adapter.generate(model, inputs, processor)
                pred_fwd = parse_forward_reverse(resp_fwd)

                if fwd_total + rev_total < 6 and prompt_idx == 0:
                    print(f"    [debug] fwd resp: {resp_fwd[:120]!r} -> {pred_fwd}")

                if pred_fwd is not None:
                    fwd_total += 1
                    all_fwd_preds.append(pred_fwd)
                    if pred_fwd == "FORWARD":
                        fwd_correct += 1

                # Reverse
                rev_frames = pil_frames[::-1]
                inputs = adapter.prepare_inputs(
                    processor,
                    rev_frames,
                    prompt,
                    device,
                    fps=fps_eff,
                )
                resp_rev = adapter.generate(model, inputs, processor)
                pred_rev = parse_forward_reverse(resp_rev)

                if fwd_total + rev_total < 6 and prompt_idx == 0:
                    print(f"    [debug] rev resp: {resp_rev[:120]!r} -> {pred_rev}")

                if pred_rev is not None:
                    rev_total += 1
                    all_rev_preds.append(pred_rev)
                    if pred_rev == "REVERSE":
                        rev_correct += 1

                if prompt_idx == 0:
                    per_segment_details.append(
                        {
                            "scene_name": seg.scene_name,
                            "label": MANEUVER_NAMES.get(seg.label, str(seg.label)),
                            "n_keyframes": len(seg.keyframe_paths),
                            "fwd_pred": pred_fwd,
                            "rev_pred": pred_rev,
                            "fwd_raw": resp_fwd[:200],
                            "rev_raw": resp_rev[:200],
                        }
                    )

            except Exception as e:
                if fwd_total + rev_total < 3:
                    print(
                        f"    [error] {seg.scene_name} seg {seg_idx}: {e}"
                    )
                continue

        fwd_acc = fwd_correct / max(fwd_total, 1)
        rev_acc = rev_correct / max(rev_total, 1)
        balanced_acc = (fwd_acc + rev_acc) / 2

        # Degeneracy detection: check if all predictions are the same
        all_preds = all_fwd_preds + all_rev_preds
        unique_preds = set(all_preds)
        is_degenerate = len(unique_preds) <= 1 and len(all_preds) > 0

        per_prompt_results[f"prompt_{prompt_idx}"] = {
            "balanced_acc": round(balanced_acc, 4),
            "forward_acc": round(fwd_acc, 4),
            "reverse_acc": round(rev_acc, 4),
            "fwd_total": fwd_total,
            "rev_total": rev_total,
            "is_degenerate": is_degenerate,
            "degenerate_label": list(unique_preds)[0] if is_degenerate else None,
        }
        all_balanced_accs.append(balanced_acc)

        print(
            f"  Prompt {prompt_idx}: fwd_acc={fwd_acc:.3f}, rev_acc={rev_acc:.3f}, "
            f"balanced={balanced_acc:.3f}"
            + (
                f" [DEGENERATE: always {list(unique_preds)[0]}]"
                if is_degenerate
                else ""
            )
        )

    mean_balanced_acc = float(np.mean(all_balanced_accs))
    std_balanced_acc = float(np.std(all_balanced_accs))
    print(f"\n  Mean balanced acc: {mean_balanced_acc:.3f} +/- {std_balanced_acc:.3f}")

    return {
        "per_prompt": per_prompt_results,
        "mean_balanced_acc": round(mean_balanced_acc, 4),
        "std_balanced_acc": round(std_balanced_acc, 4),
        "per_segment_details": per_segment_details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="nuScenes VLM Generative Probes: Forward/Reverse Direction Classification"
    )
    parser.add_argument(
        "--nuscenes-dir",
        type=str,
        required=True,
        help="Path to nuScenes dataset root directory",
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
        help="Override default model path for the VLM family",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of mixed intersection clusters to evaluate",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Number of frames to sample per segment",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes dataset version",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.nuscenes_dir)
    device = torch.device(args.device)

    # Resolve model path
    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    family = args.vlm_family

    print("=" * 70)
    print("nuScenes VLM GENERATIVE PROBES — Direction Classification")
    print("=" * 70)
    print(f"  VLM family:    {family}")
    print(f"  Model path:    {model_path}")
    print(f"  nuScenes dir:  {data_dir}")
    print(f"  Version:       {args.version}")
    print(f"  Max clusters:  {args.max_clusters}")
    print(f"  N frames:      {args.n_frames}")
    print(f"  Device:        {device}")

    # ------------------------------------------------------------------
    # Step 1: Discover nuScenes maneuver segments
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering nuScenes maneuver segments...")
    all_segments, scene_keyframes, metadata = discover_nuscenes_segments(
        data_dir, args.version
    )

    if len(all_segments) == 0:
        raise RuntimeError(
            "No maneuver segments found. Check nuScenes data directory "
            "and CAN bus data availability."
        )

    # ------------------------------------------------------------------
    # Step 2: Cluster intersections (DBSCAN eps=30m, min_samples=2)
    # ------------------------------------------------------------------
    print("\nStep 2: Clustering intersections...")
    clusters = cluster_intersections(all_segments, eps=30.0, min_samples=2)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 3: Filter for mixed clusters (both left+right turns)
    # ------------------------------------------------------------------
    print("\nStep 3: Filtering for mixed clusters...")
    mixed_clusters = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
    print(f"  Mixed clusters: {len(mixed_clusters)}")

    if len(mixed_clusters) == 0:
        raise RuntimeError(
            "No mixed clusters found (need clusters with both left and right turns). "
            "Try using v1.0-trainval for more data."
        )

    # Collect all segments from mixed clusters
    mixed_segments: list[ManeuverSegment] = []
    for cid, segs in mixed_clusters.items():
        mixed_segments.extend(segs)

    # ------------------------------------------------------------------
    # Step 4: Assign keyframes to segments
    # ------------------------------------------------------------------
    print("\nStep 4: Assigning keyframes to segments...")
    assign_keyframes_to_segments(mixed_segments, scene_keyframes)

    # Filter segments with enough keyframes
    segments_with_frames = [s for s in mixed_segments if len(s.keyframe_paths) >= 2]
    print(
        f"  Segments with keyframes: {len(segments_with_frames)} / {len(mixed_segments)}"
    )

    if len(segments_with_frames) == 0:
        raise RuntimeError(
            "No segments have sufficient keyframes. "
            "Check that nuScenes image data (samples/) is available."
        )

    # ------------------------------------------------------------------
    # Step 5: Load VLM
    # ------------------------------------------------------------------
    print(f"\nStep 5: Loading VLM ({family})...")
    adapter = VLM_ADAPTERS[family]()
    model, processor = adapter.load(model_path, device)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 6: Run direction probe (forward/reverse with 3 prompts)
    # ------------------------------------------------------------------
    print("\nStep 6: Direction classification probe...")
    direction_results = run_direction_probe(
        segments=segments_with_frames,
        adapter=adapter,
        model=model,
        processor=processor,
        device=device,
        n_frames=args.n_frames,
        prompts=DIRECTION_PROMPTS,
    )

    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Step 7: Cleanup model
    # ------------------------------------------------------------------
    del model, processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 8: Assemble and save results
    # ------------------------------------------------------------------
    print("\nStep 8: Saving results...")

    # Degeneracy summary
    direction_degenerate = any(
        v.get("is_degenerate", False) for v in direction_results["per_prompt"].values()
    )

    output = {
        "metadata": {
            "vlm_family": family,
            "model_path": model_path,
            "nuscenes_dir": str(data_dir),
            "version": args.version,
            "n_segments": len(segments_with_frames),
            "n_mixed_clusters": len(mixed_clusters),
            "n_frames": args.n_frames,
            "direction_prompts": DIRECTION_PROMPTS,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(t_elapsed, 1),
        },
        "direction": {
            "per_prompt": direction_results["per_prompt"],
            "mean_balanced_acc": direction_results["mean_balanced_acc"],
            "std_balanced_acc": direction_results["std_balanced_acc"],
            "is_degenerate": direction_degenerate,
        },
        "per_segment_details": direction_results.get("per_segment_details", []),
    }

    # Save
    results_dir = project_root / "datasets" / "nuscenes"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"vlm_generative_{family}_results.json"

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  VLM: {family} ({model_path})")
    print(f"  Segments evaluated: {len(segments_with_frames)}")
    print(f"  Mixed clusters: {len(mixed_clusters)}")
    print(f"  Elapsed: {t_elapsed:.0f}s")

    print(f"\n  Direction Classification:")
    for pname, pdata in direction_results["per_prompt"].items():
        deg = " [DEGENERATE]" if pdata.get("is_degenerate") else ""
        print(
            f"    {pname}: balanced_acc={pdata['balanced_acc']:.3f} "
            f"(fwd={pdata['forward_acc']:.3f}, rev={pdata['reverse_acc']:.3f}){deg}"
        )
    print(
        f"    Mean balanced acc: {direction_results['mean_balanced_acc']:.3f} "
        f"+/- {direction_results['std_balanced_acc']:.3f}"
    )

    if direction_degenerate:
        print("\n  WARNING: One or more prompts produced degenerate (constant) output.")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

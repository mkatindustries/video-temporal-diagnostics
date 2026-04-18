#!/usr/bin/env python3
"""VLM Vision Tower Bridge Experiment on Honda HDD.

Evaluates whether VLM vision tower embeddings (SigLIP from Gemma-4,
CLIP from LLaVA-Video) can discriminate maneuvers at the same intersection,
bridging the EPIC-Kitchens VLM probes with the HDD retrieval benchmark.

Two comparators per VLM:
  - Vision pooled (BoF): mean-pool per-frame vision embeddings → cosine sim.
  - Vision seq DTW: per-frame pooled → temporal derivative → DTW similarity.

Protocol: Same as Table 5 — 128 sessions, 1,687 maneuvers, 50 mixed-direction
clusters, DBSCAN eps=0.0003, min_samples=3, ±3 s context, 1,000 bootstrap
resamples for CIs, seed=42.

Hypothesis: Both comparators perform near chance (~0.50 AP) because 2D
per-frame vision encoders cannot capture 3D ego-motion. V-JEPA 2 temporal
residuals (AP 0.956) remain dominant.

Usage:
    # Gemma-4 SigLIP
    python experiments/eval_hdd_vlm_bridge.py \\
        --hdd-dir datasets/hdd --vlm-family gemma4

    # LLaVA CLIP
    python experiments/eval_hdd_vlm_bridge.py \\
        --hdd-dir datasets/hdd --vlm-family llava-video

    # With DINOv3/V-JEPA 2 baselines for comparison table
    python experiments/eval_hdd_vlm_bridge.py \\
        --hdd-dir datasets/hdd --vlm-family gemma4 --include-baselines

    # Smoke test (2 clusters)
    python experiments/eval_hdd_vlm_bridge.py \\
        --hdd-dir datasets/hdd --vlm-family llava-video --max-clusters 2
"""

import argparse
import json
import re
import sys
import time
import warnings
import zoneinfo
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import av
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

sys.path.insert(0, str(Path(__file__).parent))
from eval_hdd_intersections import ManeuverSegment  # noqa: E402


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
    1: "intersection_passing",
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
# HDD Data Loading (from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------


def parse_video_start_time(video_filename: str) -> float:
    """Parse video start unix timestamp from filename.

    Filename format: 2017-02-27-10-17-27_new_0.75.mp4
    Timestamps are in US/Pacific local time (PST or PDT depending on date).
    """
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", video_filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {video_filename}")
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return dt.timestamp()


def discover_sessions(hdd_dir: Path) -> dict[str, dict]:
    """Discover all valid HDD sessions with labels, GPS, and video."""
    label_dir = hdd_dir / "labels" / "target"
    release_dir = hdd_dir / "release_2019_07_08"

    label_files = {}
    for f in sorted(label_dir.glob("*.npy")):
        label_files[f.stem] = f

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

            camera_dir = session_dir / "camera" / "center"
            if not camera_dir.exists():
                continue
            video_files = list(camera_dir.glob("*.mp4"))
            if not video_files:
                continue
            video_path = video_files[0]

            gps_path = session_dir / "general" / "csv" / "rtk_pos.csv"
            if not gps_path.exists():
                continue

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
    """
    data = np.genfromtxt(
        gps_path, delimiter=",", skip_header=1, usecols=(0, 2, 3), dtype=np.float64
    )
    timestamps = data[:, 0]
    latitudes = data[:, 1]  # column labeled 'lng' is actually latitude
    longitudes = data[:, 2]  # column labeled 'lat' is actually longitude
    return timestamps, latitudes, longitudes


def _add_segment(
    segments,
    session_id,
    label,
    start_frame,
    end_frame,
    gps_timestamps,
    gps_lats,
    gps_lngs,
    video_path,
    video_start_unix,
):
    """Add a maneuver segment with GPS midpoint lookup."""
    mid_frame = (start_frame + end_frame) // 2
    mid_ts = video_start_unix + mid_frame / 3.0
    gps_idx = np.searchsorted(gps_timestamps, mid_ts)
    gps_idx = min(gps_idx, len(gps_timestamps) - 1)
    lat = gps_lats[gps_idx]
    lng = gps_lngs[gps_idx]
    if np.isnan(lat) or np.isnan(lng) or abs(lat) < 1 or abs(lng) < 1:
        return
    segments.append(
        ManeuverSegment(
            session_id=session_id,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
            lat=float(lat),
            lng=float(lng),
            video_path=video_path,
            video_start_unix=video_start_unix,
        )
    )


def extract_maneuver_segments(
    session_id,
    labels,
    gps_timestamps,
    gps_lats,
    gps_lngs,
    video_path,
    video_start_unix,
    target_labels=(1, 2, 3),
) -> list[ManeuverSegment]:
    """Extract contiguous maneuver segments and their GPS midpoints."""
    segments: list[ManeuverSegment] = []
    in_segment = False
    seg_start = seg_label = 0

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


def cluster_intersections(
    segments: list[ManeuverSegment],
    eps: float = 0.0003,
    min_samples: int = 3,
) -> dict[int, list[ManeuverSegment]]:
    """Cluster maneuver segments by GPS location using DBSCAN."""
    coords = np.array([[s.lat, s.lng] for s in segments])
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
# Canonical Frame Sampling (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def compute_fps_eff(timestamps: list[float]) -> float:
    """Compute effective FPS from canonical frame timestamps.

    fps_eff = (N-1) / (t_last - t_first), representing the actual temporal
    spacing of sampled frames. Used as a semantic hint for models like Qwen.
    """
    if len(timestamps) < 2:
        return 1.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 1.0
    return (len(timestamps) - 1) / duration


def sample_canonical_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = 16,
    max_resolution: int = 518,
) -> tuple[list[Image.Image], list[float]]:
    """Sample uniformly spaced PIL frames from a video clip.

    Uses PyAV seek() for efficient random access on long dashcam videos.
    Returns PIL Images directly (adapters receive PIL, not numpy).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    # pyrefly: ignore [bad-argument-type]
    time_base = float(stream.time_base or 0)

    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0

    # Compute target timestamps (uniform spacing)
    target_times = [
        start_sec + (i / max(n_frames - 1, 1)) * duration for i in range(n_frames)
    ]

    # Seek to before start
    # pyrefly: ignore [no-matching-overload]
    seek_sec = max(0.0, start_sec - 1.0)
    seek_pts = int(seek_sec / time_base)
    container.seek(seek_pts, stream=stream)

    # Collect all frames in range, then pick closest to targets
    all_frames = []
    all_times = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts) * time_base
        if frame_time < start_sec - 0.1:
            continue
        if frame_time > end_sec + 0.1:
            break

        img = frame.to_ndarray(format="rgb24")
        if max_resolution and img.shape[0] > max_resolution:
            scale = max_resolution / img.shape[0]
            new_h = max_resolution
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h))

        all_frames.append(img)
        all_times.append(frame_time)

    container.close()

    if len(all_frames) == 0:
        return [], []

    # Select frames closest to target timestamps
    all_times_arr = np.array(all_times)
    selected_frames = []
    selected_times = []
    for t in target_times:
        idx = int(np.argmin(np.abs(all_times_arr - t)))
        selected_frames.append(Image.fromarray(all_frames[idx]))
        selected_times.append(round(all_times[idx], 4))

    return selected_frames, selected_times


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
# VLM Vision Adapters (trimmed from eval_epic_temporal_order.py)
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
    context_sec: float = 3.0,
    n_frames: int = 16,
    device: torch.device | None = None,
    cache_path: Path | None = None,
    extract_llm: bool = False,
) -> dict[int, dict]:
    """Extract VLM vision tower features for all maneuver segments.

    For each segment, extracts:
    - vision_repr: L2-normed mean-pooled vision embedding (D,)
    - vision_seq: L2-normed per-frame vision embeddings (T, D)
    - llm_repr: (optional) L2-normed mean-pooled LLM hidden state (D,)

    Args:
        adapter: VLM vision adapter instance.
        model: Loaded VLM model.
        processor: VLM processor/tokenizer.
        segments: Maneuver segments to process.
        context_sec: Seconds of context before/after maneuver.
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
        # Convert label frame indices (at 3 fps) to video time
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            pil_frames, timestamps = sample_canonical_frames(
                seg.video_path,
                start_sec,
                end_sec,
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
      as the DINOv3 temporal derivative baseline (α=1).

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
    if any(llm_mask):
        print("  Computing LLM hidden state (cosine) similarities...")
        llm_a_idx = [a for a, m in zip(pair_a_indices, llm_mask) if m]
        llm_b_idx = [b for b, m in zip(pair_b_indices, llm_mask) if m]
        llm_gts = [g for g, m in zip(pair_gts, llm_mask) if m]

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
        description="VLM Vision Tower Bridge Experiment on Honda HDD"
    )
    parser.add_argument(
        "--hdd-dir",
        type=str,
        required=True,
        help="Path to HDD dataset directory",
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
        "--context-sec",
        type=float,
        default=3.0,
        help="Seconds of context before/after maneuver",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Number of canonical frames per clip (shared across VLMs)",
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
    hdd_dir = project_root / args.hdd_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    display_name = VLM_DISPLAY_NAMES[args.vlm_family]

    print("=" * 70)
    print(f"VLM VISION TOWER BRIDGE: {display_name} on HDD")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Frames per clip: {args.n_frames}")
    print(f"  Context: +/-{args.context_sec}s")

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
        )
        if segs:
            sessions_with_segments += 1
        all_segments.extend(segs)

    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1
    print(
        f"  Total segments: {len(all_segments)} "
        f"from {sessions_with_segments} sessions"
    )
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(lv, 0)}")

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
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {eval_label_counts.get(lv, 0)}")

    # ------------------------------------------------------------------
    # Step 5: Load VLM and extract vision features
    # ------------------------------------------------------------------
    print(f"\nStep 5: Loading {display_name} model...")
    adapter = VLM_ADAPTERS[args.vlm_family]()
    model, processor = adapter.load(model_path, torch.device(args.device))

    cache_dir = hdd_dir / "vlm_bridge_cache"
    cache_suffix = "_with_llm" if args.extract_llm else ""
    cache_path = cache_dir / f"{args.vlm_family}_vision_features{cache_suffix}.pt"

    print(f"\nStep 6: Extracting {display_name} vision features...")
    t_feat_start = time.time()
    features = extract_vlm_vision_features(
        adapter,
        model,
        processor,
        eval_segments,
        context_sec=args.context_sec,
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
    # Step 7: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 7: Computing pairwise similarities...")
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
    # Step 8: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {display_name} — HDD MANEUVER DISCRIMINATION")
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
    if args.include_baselines:
        pair_scores_path = hdd_dir / "pair_scores.json"
        if pair_scores_path.exists():
            print(f"\n  Loading baselines from {pair_scores_path}")
            with open(pair_scores_path) as f:
                baseline_data = json.load(f)

            baseline_names = {
                "bag_of_frames": "DINOv3 BoF",
                "chamfer": "DINOv3 Chamfer",
                "temporal_derivative": "DINOv3 Temp. Deriv.",
                "attention_trajectory": "DINOv3 Attn. Traj.",
                "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
                "vjepa2_temporal_residual": "V-JEPA 2 Temp. Res.",
            }

            print("\n  -- Baselines (from pair_scores.json) --")
            for method_key, bl_name in baseline_names.items():
                if method_key not in baseline_data:
                    continue
                bl_scores = np.array(baseline_data[method_key]["scores"])
                bl_labels = np.array(baseline_data[method_key]["labels"])
                if bl_labels.sum() == 0 or bl_labels.sum() == len(bl_labels):
                    continue
                bl_ap, bl_ci_lo, bl_ci_hi = bootstrap_ap(bl_scores, bl_labels)
                bl_auc = roc_auc_score(bl_labels, bl_scores)
                print(
                    f"  {bl_name:<35s}  AP={bl_ap:.4f}  [{bl_ci_lo:.4f}, {bl_ci_hi:.4f}]  "
                    f"AUC={bl_auc:.4f}"
                )
        else:
            print(f"\n  Baselines not found at {pair_scores_path}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = hdd_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vlm_bridge_{args.vlm_family}_results.json"

    output = {
        "vlm_family": args.vlm_family,
        "model_path": model_path,
        "display_name": display_name,
        "n_frames": args.n_frames,
        "context_sec": args.context_sec,
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

    print("\nDone.")


if __name__ == "__main__":
    main()

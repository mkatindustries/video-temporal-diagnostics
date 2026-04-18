#!/usr/bin/env python3
"""VCDB VLM Vision Tower Bridge Experiment.

Evaluates VLM vision tower embeddings on the VCDB copy detection benchmark,
filling 9 cells in Table 8: 3 vision_pooled, 3 vision_seq_dtw, 3 llm_hidden
across Gemma-4, LLaVA-Video, and Qwen3-VL.

Three comparators per VLM:
  - vision_pooled (BoF): mean-pool per-frame vision embeddings -> cosine sim.
  - vision_seq_dtw: per-frame pooled -> temporal derivative -> DTW similarity.
  - llm_hidden: mean-pool LLM hidden states -> cosine sim.

Protocol: Same balanced pair sampling as eval_vcdb.py -- all positive copy
pairs + equal number of sampled negatives (seed=42). AP and AUC via sklearn,
bootstrap CIs (1000 resamples, percentile method, seed=42).

Usage:
    python experiments/eval_vcdb_vlm_bridge.py \\
        --vcdb-dir datasets/vcdb/core_dataset --vlm-family gemma4

    python experiments/eval_vcdb_vlm_bridge.py \\
        --vcdb-dir datasets/vcdb/core_dataset --vlm-family llava-video

    python experiments/eval_vcdb_vlm_bridge.py \\
        --vcdb-dir datasets/vcdb/core_dataset --vlm-family qwen3
"""

import argparse
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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

DESCRIBE_PROMPT = "Describe the video."


# ---------------------------------------------------------------------------
# VCDB Loading (from eval_vcdb.py)
# ---------------------------------------------------------------------------


def discover_videos(vid_base_dir: str) -> list[str]:
    """Discover all video files under the VCDB core_dataset directory.

    Returns:
        List of relative paths like "baggio_penalty_1994/abc.mp4".
    """
    videos = []
    for cat in sorted(os.listdir(vid_base_dir)):
        cat_path = os.path.join(vid_base_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))
    return videos


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    """Load all VCDB annotations as global (videoA_path, videoB_path) pairs.

    Video paths are relative to vid_base_dir, e.g. "baggio_penalty_1994/abc.mp4".

    Returns:
        Set of (vidA_relpath, vidB_relpath) tuples (sorted order).
    """
    copy_pairs = set()
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".txt"):
            continue
        cat = fname.replace(".txt", "")
        with open(os.path.join(ann_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                vid_a = os.path.join(cat, parts[0].strip())
                vid_b = os.path.join(cat, parts[1].strip())
                if vid_a != vid_b:
                    pair = tuple(sorted([vid_a, vid_b]))
                    copy_pairs.add(pair)
    # pyrefly: ignore [bad-return]
    return copy_pairs


# ---------------------------------------------------------------------------
# Frame Sampling (from eval_vcdb_vlm_probes.py)
# ---------------------------------------------------------------------------


def sample_frames_from_video(
    video_path: str,
    n_frames: int = 16,
    max_resolution: int = 518,
) -> tuple[list[Image.Image], list[float]]:
    """Sample n_frames uniformly from the full video duration.

    Returns PIL images and their timestamps (seconds). Handles short
    videos by repeating the last frame.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base or 1e-6)

    # Estimate duration
    if stream.duration is not None:
        # pyrefly: ignore [unsupported-operation]
        duration = float(stream.duration * stream.time_base)
    else:
        duration = 60.0  # fallback

    # Compute target timestamps
    if duration <= 0:
        duration = 1.0
    target_times = [(i / max(n_frames - 1, 1)) * duration for i in range(n_frames)]

    # Decode all frames, then pick nearest to each target
    all_frames = []
    all_times = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts) * time_base

        img = frame.to_ndarray(format="rgb24")
        if max_resolution and img.shape[0] > max_resolution:
            scale = max_resolution / img.shape[0]
            new_h = max_resolution
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h))

        all_frames.append(img)
        all_times.append(frame_time)

        # Early exit: stop once well past last target
        if frame_time > target_times[-1] + 1.0 and len(all_frames) > n_frames * 2:
            break

    container.close()

    if len(all_frames) == 0:
        return [], []

    all_times_arr = np.array(all_times)
    selected_frames = []
    selected_times = []
    for t in target_times:
        idx = int(np.argmin(np.abs(all_times_arr - t)))
        selected_frames.append(Image.fromarray(all_frames[idx]))
        selected_times.append(round(all_times[idx], 4))

    return selected_frames, selected_times


# ---------------------------------------------------------------------------
# Effective FPS helper
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
# VLM Vision Adapters (from eval_hdd_vlm_bridge.py — self-contained)
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


def extract_vlm_features(
    adapter: VLMVisionAdapter,
    model: nn.Module,
    processor: object,
    vid_base_dir: str,
    video_relpaths: list[str],
    n_frames: int = 16,
    device: torch.device | None = None,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Extract VLM vision tower + LLM features for all VCDB videos.

    For each video, extracts:
    - vision_repr: L2-normed mean-pooled vision embedding (D,)
    - vision_seq: L2-normed per-frame vision embeddings (T, D)
    - llm_repr: L2-normed mean-pooled LLM hidden state (D,)

    Returns:
        Dict mapping video relpath -> {
            'vision_repr': (D,) pooled embedding,
            'vision_seq': (T, D) per-frame sequence or None,
            'llm_repr': (D,) LLM hidden state or None,
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

    for vp in tqdm(video_relpaths, desc="Extracting VLM features"):
        video_path = os.path.join(vid_base_dir, vp)
        try:
            pil_frames, timestamps = sample_frames_from_video(
                video_path,
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
            llm_repr = adapter.extract_llm_repr(model, inputs)

            if vision_repr is None:
                failed += 1
                continue

            features[vp] = {
                "vision_repr": vision_repr,
                "vision_seq": vision_seq,
                "llm_repr": llm_repr,
            }
        except Exception as e:
            failed += 1
            if failed <= 5:
                warnings.warn(f"Video {vp} failed: {e}")
            continue

    print(f"  Extracted: {len(features)}/{len(video_relpaths)} ({failed} failed)")

    if cache_path:
        save_feature_cache(features, cache_path)

    return features


# ---------------------------------------------------------------------------
# Pair Sampling
# ---------------------------------------------------------------------------


def build_pair_set(
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Build shared evaluation pair set: all positive + equal negatives.

    Args:
        keys: List of video relpaths with features.
        copy_pairs: Set of annotated copy pairs (sorted tuples).
        seed: Random seed for negative sampling.

    Returns:
        (pairs, labels) -- pairs as list of (vidA, vidB), labels as list of 0/1.
    """
    key_set = set(keys)
    n = len(keys)

    # All positive pairs that we have features for
    pos_pairs = []
    for a, b in copy_pairs:
        if a in key_set and b in key_set:
            pos_pairs.append((a, b))

    n_pos = len(pos_pairs)
    print(f"  Positive pairs with features: {n_pos}")

    # Sample equal number of negative pairs
    rng = np.random.RandomState(seed)
    all_pairs_set = set(pos_pairs)
    neg_pairs = []
    neg_attempts = 0
    n_neg_target = n_pos

    while len(neg_pairs) < n_neg_target and neg_attempts < n_neg_target * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        pair = (min(keys[i], keys[j]), max(keys[i], keys[j]))
        if pair not in copy_pairs and pair not in all_pairs_set:
            neg_pairs.append(pair)
            all_pairs_set.add(pair)  # avoid duplicates
        neg_attempts += 1

    print(
        f"  Total pairs to evaluate: {n_pos + len(neg_pairs)} "
        f"({n_pos} pos + {len(neg_pairs)} neg)"
    )

    # Combine
    pairs = pos_pairs + neg_pairs
    labels = [1] * n_pos + [0] * len(neg_pairs)

    return pairs, labels


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------


def compute_similarities(
    features: dict[str, dict],
    pairs: list[tuple[str, str]],
    labels: list[int],
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities for all three methods.

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deriv_fp = TemporalDerivativeFingerprint()

    # --- Vision pooled: cosine similarity ---
    print("  Computing vision pooled (cosine) similarities...")
    pooled_a = (
        torch.stack([features[a]["vision_repr"] for a, b in pairs])
        .float()
        .to(device)
    )
    pooled_b = (
        torch.stack([features[b]["vision_repr"] for a, b in pairs])
        .float()
        .to(device)
    )
    pooled_sims = (pooled_a * pooled_b).sum(dim=1).cpu().tolist()

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "vision_pooled": (pooled_sims, list(labels)),
    }

    # --- Vision seq DTW: temporal derivative + batched GPU DTW ---
    dtw_mask = [
        (
            features[a]["vision_seq"] is not None
            and features[b]["vision_seq"] is not None
        )
        for a, b in pairs
    ]
    dtw_pairs = [(a, b) for (a, b), m in zip(pairs, dtw_mask) if m]
    dtw_labels = [l for l, m in zip(labels, dtw_mask) if m]

    if dtw_pairs:
        print("  Computing vision seq DTW similarities...")
        # Pre-compute temporal derivative fingerprints (cast to float32)
        deriv_fps_cache: dict[str, torch.Tensor] = {}
        for a, b in dtw_pairs:
            for key in [a, b]:
                if key not in deriv_fps_cache:
                    seq = features[key]["vision_seq"].float()  # (T, D)
                    deriv_fps_cache[key] = deriv_fp.compute_fingerprint(seq)

        deriv_seqs_a = [deriv_fps_cache[a].to(device) for a, b in dtw_pairs]
        deriv_seqs_b = [deriv_fps_cache[b].to(device) for a, b in dtw_pairs]
        # normalize=False: same as DINOv3 temporal derivative baseline
        deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
        dtw_sims = torch.exp(-deriv_dists).cpu().tolist()

        all_scores["vision_seq_dtw"] = (dtw_sims, list(dtw_labels))
    else:
        print("  WARNING: No valid vision_seq pairs for DTW computation")

    # --- LLM hidden state: cosine similarity ---
    llm_mask = [
        (
            features[a].get("llm_repr") is not None
            and features[b].get("llm_repr") is not None
        )
        for a, b in pairs
    ]
    if any(llm_mask):
        print("  Computing LLM hidden state (cosine) similarities...")
        llm_pairs = [(a, b) for (a, b), m in zip(pairs, llm_mask) if m]
        llm_labels = [l for l, m in zip(labels, llm_mask) if m]

        llm_a = (
            torch.stack([features[a]["llm_repr"] for a, b in llm_pairs])
            .float()
            .to(device)
        )
        llm_b = (
            torch.stack([features[b]["llm_repr"] for a, b in llm_pairs])
            .float()
            .to(device)
        )
        llm_sims = (llm_a * llm_b).sum(dim=1).cpu().tolist()
        all_scores["llm_hidden"] = (llm_sims, list(llm_labels))

    return all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VCDB VLM Vision Tower Bridge Experiment"
    )
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb/core_dataset",
        help="Path to VCDB core_dataset directory",
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
        "--n-frames",
        type=int,
        default=16,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="(Not used -- reserved for compatibility)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    device = torch.device(args.device)

    # Resolve model path
    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    family = args.vlm_family

    print("=" * 70)
    print("VCDB VLM VISION TOWER BRIDGE EXPERIMENT")
    print("=" * 70)
    print(f"  VLM family: {family}")
    print(f"  Model path: {model_path}")
    print(f"  VCDB dir:   {vcdb_dir}")
    print(f"  N frames:   {args.n_frames}")
    print(f"  Device:     {device}")

    # ------------------------------------------------------------------
    # Step 1: Discover videos and load annotations
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering videos and loading annotations...")
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Annotated copy pairs: {len(copy_pairs)}")

    if len(videos) == 0:
        raise FileNotFoundError(
            f"No videos found in {vid_dir}. "
            "Ensure the VCDB core_dataset directory has category subdirectories "
            "containing video files."
        )

    # ------------------------------------------------------------------
    # Step 2: Load VLM
    # ------------------------------------------------------------------
    print(f"\nStep 2: Loading VLM ({family})...")
    adapter = VLM_ADAPTERS[family]()
    model, processor = adapter.load(model_path, device)

    # ------------------------------------------------------------------
    # Step 3: Extract features for all videos
    # ------------------------------------------------------------------
    cache_dir = vcdb_dir.parent / "vlm_bridge_cache"
    cache_path = cache_dir / f"{family}_vcdb_features.pt"

    print(f"\nStep 3: Extracting {family} features for {len(videos)} videos...")
    t0 = time.time()
    features = extract_vlm_features(
        adapter,
        model,
        processor,
        str(vid_dir),
        videos,
        n_frames=args.n_frames,
        device=device,
        cache_path=cache_path,
    )
    t_feat = time.time() - t0
    print(f"  Feature extraction time: {t_feat:.1f}s")

    # Free model memory
    del model, processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Build shared pair set
    # ------------------------------------------------------------------
    keys = sorted(features.keys())
    print(f"\nStep 4: Building shared pair set ({len(keys)} videos with features)...")
    pairs, labels = build_pair_set(keys, copy_pairs, seed=42)

    # ------------------------------------------------------------------
    # Step 5: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 5: Computing pairwise similarities...")
    t0 = time.time()
    all_scores = compute_similarities(features, pairs, labels, device=device)
    t_sim = time.time() - t0
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {family} -- VCDB COPY DETECTION")
    print("=" * 70)

    results = {}
    for method_name in ["vision_pooled", "vision_seq_dtw", "llm_hidden"]:
        if method_name not in all_scores:
            continue
        scores_list, labels_list = all_scores[method_name]
        scores_arr = np.array(scores_list)
        labels_arr = np.array(labels_list)
        n_pos = int(labels_arr.sum())
        n_neg = len(labels_arr) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method_name] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            print(
                f"  {family} {method_name:<20s}  AP=NaN  AUC=NaN  "
                f"(pos={n_pos}, neg={n_neg})"
            )
            continue

        ap_val, ci_lo, ci_hi = bootstrap_ap(scores_arr, labels_arr)
        auc = roc_auc_score(labels_arr, scores_arr)

        results[method_name] = {
            "ap": float(ap_val),
            "auc": float(auc),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        print(
            f"  {family} {method_name:<20s}  AP={ap_val:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"AUC={auc:.4f}  (pos={n_pos}, neg={n_neg})"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    out_dir = vcdb_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main results
    out_path = out_dir / f"vlm_bridge_{family}_results.json"
    output = {
        "vlm_family": family,
        "model_path": model_path,
        "n_frames": args.n_frames,
        "n_videos_total": len(videos),
        "n_videos_extracted": len(features),
        "n_copy_pairs": len(copy_pairs),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Pair-level scores for bootstrap
    pair_path = out_dir / f"vlm_bridge_{family}_pair_scores.json"
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

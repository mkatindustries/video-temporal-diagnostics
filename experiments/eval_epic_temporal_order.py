#!/usr/bin/env python3
"""EPIC-Kitchens Temporal Order Sensitivity + Multi-VLM Probes.

Evaluates temporal order sensitivity across DINOv3, V-JEPA 2, and optionally
three architecturally diverse VLMs (Qwen3-VL-8B, Gemma 4 31B, LLaVA-Video 7B)
via embedding extraction and generative probes on EPIC-Kitchens-100 cooking
sequences. Includes bootstrap CIs, multi-prompt variance, layer ablation,
text-only baselines, and temporal integrity consistency metrics.

Core task: Forward vs Reverse temporal order sensitivity (not action
classification). Uses ~10s sequences with 32 frames at ~3fps for
DINOv3, 64 for V-JEPA 2, and a configurable canonical frame count
(default 16) shared across all VLMs for input parity.

Usage:
    # DINOv3 only (local smoke test)
    python experiments/eval_epic_temporal_order.py \\
        --epic-dir datasets/epic_kitchens --max-sequences 5 --skip-vjepa2

    # Full DINOv3 + V-JEPA 2
    python experiments/eval_epic_temporal_order.py \\
        --epic-dir ./data/epic_kitchens

    # Qwen3-VL generative + embeddings with layer ablation
    python experiments/eval_epic_temporal_order.py \\
        --epic-dir ./data/epic_kitchens \\
        --vlm-family qwen3 --vlm-generative --vlm-integrity-probe \\
        --vlm-embeddings --vlm-layer-ablation --vlm-text-only-baseline \\
        --skip-vjepa2

    # Gemma 4 smoke test
    python experiments/eval_epic_temporal_order.py \\
        --epic-dir ./data/epic_kitchens \\
        --vlm-family gemma4 --vlm-generative --vlm-integrity-probe \\
        --vlm-embeddings --max-sequences 3 --skip-vjepa2

    # LLaVA-Video
    python experiments/eval_epic_temporal_order.py \\
        --epic-dir ./data/epic_kitchens \\
        --vlm-family llava-video --vlm-generative --vlm-embeddings \\
        --skip-vjepa2
"""

import argparse
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# V-JEPA 2
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_LOCAL_PATH = VJEPA2_MODEL_NAME
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32  # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256  # 16h × 16w

# DINOv3
DINOV3_LOCAL_PATH = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# VLM model paths
VLM_DEFAULT_PATHS = {
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma4": "google/gemma-4-31B-it",
    "llava-video": "llava-hf/LLaVA-Video-7B-Qwen2-hf",
}

# VLM prompts — 3 forward/reverse variants for prompt sensitivity
DEFAULT_CAUSALITY_PROMPTS = [
    "Analyze the physics, motion, and causality in this video. "
    "Is this video playing FORWARD normally, or is it playing in REVERSE (backwards)? "
    "Answer with a single word: either FORWARD or REVERSE.",
    "Is this video playing forward or in reverse? " "Answer FORWARD or REVERSE.",
    "Watch the actions in this video carefully. "
    "Are events happening in their natural chronological order, or is the video reversed? "
    "Answer FORWARD or REVERSE.",
]

DEFAULT_INTEGRITY_PROMPT = (
    "Is this video temporally intact (normal chronological order), or has it been "
    "temporally manipulated (reversed or scrambled)? "
    "Answer with one word: INTACT or TAMPERED."
)


def load_prompts_file(path: str) -> tuple[list[str], str]:
    """Load causality_prompts and integrity_prompt from a JSON file.

    Expected schema:
        {
            "causality_prompts": ["prompt0", "prompt1", ...],
            "integrity_prompt": "single prompt string"
        }

    Missing keys fall back to defaults.
    """
    with open(path) as f:
        data = json.load(f)
    causality = data.get("causality_prompts", DEFAULT_CAUSALITY_PROMPTS)
    integrity = data.get("integrity_prompt", DEFAULT_INTEGRITY_PROMPT)
    if not isinstance(causality, list) or not causality:
        raise ValueError("causality_prompts must be a non-empty list of strings")
    if not isinstance(integrity, str):
        raise ValueError("integrity_prompt must be a string")
    return causality, integrity


DESCRIBE_PROMPT = "Describe the video."


def compute_fps_eff(timestamps: list[float]) -> float:
    """Compute effective FPS from canonical frame timestamps.

    fps_eff = (N-1) / (t_last - t_first), representing the actual temporal
    spacing of sampled frames. Used as a semantic hint for models like Qwen
    that accept fps metadata.
    """
    if len(timestamps) < 2:
        return 1.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 1.0
    return (len(timestamps) - 1) / duration


# ---------------------------------------------------------------------------
# Feature caching (from eval_vcdb_scramble.py)
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
# Video clip extraction (from eval_hdd_intersections.py)
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
    frame 0 for long untrimmed kitchen recordings.

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
    time_base = float(stream.time_base or 0)

    # Seek to slightly before start (keyframe-based seeking)
    seek_sec = max(0.0, start_sec - 1.0)
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
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Dynamically computes fps to sample ~64 frames from the clip duration,
    then pads or truncates to exactly VJEPA2_NUM_FRAMES.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0

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

    return frames[:VJEPA2_NUM_FRAMES]


def sample_canonical_frames(
    video_path: str,
    start_sec: float,
    stop_sec: float,
    n_frames: int = 16,
    max_resolution: int = 518,
) -> tuple[list[Image.Image], list[float]]:
    """Sample a canonical set of frames shared by all VLMs for input parity.

    Uses uniform temporal sampling (n_frames evenly spaced within the clip
    window), not FPS-based. Returns PIL Images directly — adapters receive
    PIL, not numpy.

    Args:
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        stop_sec: Clip end time in seconds.
        n_frames: Number of frames to sample.
        max_resolution: Maximum height (preserves aspect ratio).

    Returns:
        (pil_frames, timestamps) — identical for all VLMs.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base or 0)

    duration = stop_sec - start_sec
    if duration <= 0:
        duration = 1.0

    # Compute target timestamps (uniform spacing)
    target_times = [
        start_sec + (i / max(n_frames - 1, 1)) * duration for i in range(n_frames)
    ]

    # Seek to before start
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
        if frame_time > stop_sec + 0.1:
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
# V-JEPA 2 temporal masks (from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Frame scrambling
# ---------------------------------------------------------------------------


def scramble_frames(frames: list, n_chunks: int, seed: int = 42) -> list:
    """Split frame list into n_chunks, shuffle with deterministic seed, reassemble.

    Works with both numpy arrays and PIL Images.

    Args:
        frames: List of frames (numpy or PIL).
        n_chunks: Number of chunks to split into.
        seed: Random seed for reproducible permutation.

    Returns:
        Scrambled frame list with same length as input.
    """
    if n_chunks <= 1 or len(frames) < n_chunks:
        return list(frames)

    T = len(frames)
    chunk_size = T // n_chunks

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else T
        chunks.append(frames[start:end])

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(chunks))
    shuffled = [chunks[p] for p in perm]

    result = []
    for chunk in shuffled:
        result.extend(chunk)
    return result


def scramble_tensor(
    tensor: torch.Tensor, n_chunks: int, seed: int = 42
) -> torch.Tensor:
    """Split tensor along dim 0 into n_chunks, shuffle chunks, reassemble."""
    if n_chunks <= 1:
        return tensor

    T = tensor.shape[0]
    chunk_size = T // n_chunks
    if chunk_size < 1:
        chunk_size = 1
        n_chunks = T

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else T
        chunks.append(tensor[start:end])

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(chunks))
    shuffled = [chunks[p] for p in perm]

    return torch.cat(shuffled, dim=0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: 1D array of per-sample values.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)

    point = float(values.mean())

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------


def load_sequences(epic_dir: Path, max_sequences: int = 200) -> list[dict]:
    """Load temporal order sequences from manifest.

    Args:
        epic_dir: EPIC-Kitchens data directory.
        max_sequences: Maximum number of sequences to use.

    Returns:
        List of sequence dicts with video_path resolved.
    """
    manifest_path = epic_dir / "annotations" / "temporal_order_sequences.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run: python scripts/download_epic_kitchens.py --epic-dir ... --setup"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]
    video_dir = epic_dir / "videos"

    # Resolve video paths and filter to available videos
    available = []
    for seq in sequences:
        vid = seq["video_id"]
        video_path = None
        for ext in [".MP4", ".mp4"]:
            candidate = video_dir / f"{vid}{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break

        if video_path is not None:
            seq["video_path"] = video_path
            available.append(seq)

    if not available:
        raise FileNotFoundError(
            f"No videos found in {video_dir}. "
            "Run: python scripts/download_epic_kitchens.py --download-videos"
        )

    # Limit
    if len(available) > max_sequences:
        # Deterministic subset: spread across participants
        rng = np.random.RandomState(42)
        indices = rng.choice(len(available), size=max_sequences, replace=False)
        indices.sort()
        available = [available[i] for i in indices]

    print(
        f"  Loaded {len(available)} sequences "
        f"({len(set(s['video_id'] for s in available))} videos)"
    )

    return available


# ---------------------------------------------------------------------------
# VLM Adapter Protocol + Implementations
# ---------------------------------------------------------------------------


class VLMAdapter(ABC):
    """Uniform interface for VLM loading, inference, and embedding extraction."""

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
        """Convert PIL frames + prompt -> model input tensors.

        Keyword args may include fps (float) for models that accept
        temporal spacing hints (e.g. Qwen's video_grid_thw).
        """

    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        inputs: dict,
        processor: object,
        max_new_tokens: int = 64,
    ) -> str:
        """Run generative inference, return decoded text."""

    def extract_vision_repr(
        self, model: nn.Module, inputs: dict
    ) -> torch.Tensor | None:
        """Best-effort vision-side representation (pre-projector).

        Returns mean-pooled L2-normed vector, or None if not exposed.
        """
        return None

    def extract_vision_seq(
        self,
        model: nn.Module,
        inputs: dict,
        processor=None,
        frames=None,
        debug_shapes: bool = False,
    ) -> torch.Tensor | None:
        """Per-frame vision-side embeddings as a temporal sequence.

        Returns (T, D) tensor of per-frame pooled vision tokens (one vector
        per input frame), preserving temporal order. Used for sequence-aware
        probes (DTW) to test whether the vision tower encodes any temporal
        signal, independent of mean-pooling.

        Returns None if not supported.
        """
        return None

    def extract_vision_token_repr(
        self, model: nn.Module, inputs: dict, layer_idx: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Extract LLM hidden states at vision-token positions only.

        Returns a tuple of three tensors:
            vision_only_pool: Mean-pooled LLM hidden states at vision-token
                positions, L2-normed. Shape (D,).
            vision_only_seq: Per-frame LLM hidden states — vision tokens are
                grouped by source frame, mean-pooled per frame. Shape (T, D).
            bos_token: First token's LLM hidden state. Shape (D,).

        Returns None if vision-token identification is not supported.
        """
        return None

    @abstractmethod
    def extract_llm_repr(
        self, model: nn.Module, inputs: dict, layer_idx: int = -1
    ) -> torch.Tensor:
        """LLM-side representation at specified layer.

        Always available via output_hidden_states=True.
        Returns mean-pooled L2-normed vector.
        """

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


class QwenAdapter(VLMAdapter):
    """Adapter for Qwen3-VL models.

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

        # Use fps_eff from canonical timestamps if provided, otherwise default 1.0
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
        # process_vision_info wraps scalar kwargs in lists (for batched videos);
        # unwrap single-element lists so the processor gets float, not [float]
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
        from qwen_vl_utils import process_vision_info

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
            # Decode with special tokens to handle thinking blocks
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            # Strip thinking block if present: <think>...</think>\n\nANSWER
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                # No thinking block — strip special tokens normally
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            # Remove any remaining special tokens
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()

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

            with torch.no_grad():
                vision_tokens = model.visual(pixel_values, grid_thw=grid_thw)
                emb = F.normalize(vision_tokens.mean(dim=0), dim=0)
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"QwenAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_token_repr(self, model, inputs, layer_idx=-1):
        try:
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                return None

            # Qwen3-VL uses a special token ID for video/image pad tokens.
            # The processor inserts these as placeholders where vision features
            # get injected. Typical token: <|image_pad|> or <|placeholder|>.
            # We detect them by checking for the image_token_id on the
            # processor (set during prepare_inputs), or by scanning for the
            # known Qwen vision placeholder token ID (151655 = <|image_pad|>).
            # Since we don't have the processor here, we identify vision tokens
            # as those whose input_id matches pixel_values token count.
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

            # Number of vision tokens = product of grid_thw entries
            # grid_thw shape: (n_videos, 3) — each row is (T, H, W)
            n_vision_tokens = int(grid_thw.prod(dim=-1).sum().item())

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)

                # Identify vision token positions: Qwen3-VL uses token ID
                # 151655 (<|image_pad|>) for vision placeholders in input_ids
                ids = input_ids[0]  # (seq_len,)

                # Try known Qwen vision placeholder token IDs
                # 151656 = <|video_pad|> (for video input)
                # 151655 = <|image_pad|> (for image input)
                vision_mask = None
                for candidate_id in [151656, 151655, 151654, 151653]:
                    mask = ids == candidate_id
                    if mask.sum().item() > 0:
                        vision_mask = mask
                        break

                if vision_mask is None or vision_mask.sum().item() == 0:
                    # Fallback: use the n_vision_tokens count from grid_thw
                    # to select the first n_vision_tokens non-special tokens
                    # This is a best-effort heuristic
                    return None

                n_found = int(vision_mask.sum().item())

                # vision_only_pool: mean of vision-token hidden states
                vision_hidden = hidden[vision_mask]  # (n_found, D)
                vision_only_pool = F.normalize(vision_hidden.mean(dim=0), dim=0)

                # vision_only_seq: group by frame
                # grid_thw gives us per-video (T, H, W); total tokens per
                # frame = H * W (spatial patches per temporal step)
                n_frames = int(grid_thw[0, 0].item())
                tokens_per_frame = n_found // max(n_frames, 1)
                if tokens_per_frame < 1 or n_frames < 2:
                    vision_only_seq = vision_only_pool.unsqueeze(0)
                else:
                    frame_embs = []
                    for i in range(n_frames):
                        start = i * tokens_per_frame
                        end = (
                            (i + 1) * tokens_per_frame if i < n_frames - 1 else n_found
                        )
                        frame_embs.append(
                            F.normalize(vision_hidden[start:end].mean(dim=0), dim=0)
                        )
                    vision_only_seq = torch.stack(frame_embs)  # (T, D)

                # bos_token: first token in sequence
                bos_token = F.normalize(hidden[0], dim=0)

            return (
                vision_only_pool.cpu(),
                vision_only_seq.cpu(),
                bos_token.cpu(),
            )
        except Exception as e:
            warnings.warn(f"QwenAdapter.extract_vision_token_repr failed: {e}")
            return None

    def extract_llm_repr(self, model, inputs, layer_idx=-1):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)
            emb = F.normalize(hidden.mean(dim=0), dim=0)
        return emb.cpu()


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
        # Build message with image placeholders
        image_tokens = "".join(["<image>"] * len(frames))
        full_prompt = f"{image_tokens}\n{prompt}"
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
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            # Decode with special tokens to handle thinking blocks
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            # Strip thinking block if present: <think>...</think>\n\nANSWER
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                # No thinking block — strip special tokens normally
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            # Remove any remaining special tokens
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()

    def extract_vision_repr(self, model, inputs):
        try:
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return None
            with torch.no_grad():
                # Gemma 4 vision tower requires pixel_position_ids;
                # processor outputs it as "image_position_ids"
                kwargs = {}
                pos_ids = inputs.get(
                    "image_position_ids", inputs.get("pixel_position_ids")
                )
                if pos_ids is not None:
                    kwargs["pixel_position_ids"] = pos_ids
                vision_out = model.model.vision_tower(
                    pixel_values,
                    **kwargs,
                )
                tokens = vision_out.last_hidden_state
                emb = F.normalize(
                    tokens.reshape(-1, tokens.shape[-1]).mean(dim=0), dim=0
                )
            return emb.cpu()
        except Exception as e:
            warnings.warn(f"GemmaAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_seq(
        self, model, inputs, processor=None, frames=None, debug_shapes=False
    ):
        """Extract per-frame vision features as a (T, D) sequence.

        Gemma 4's vision tower requires pixel_position_ids (cannot process
        frames independently). We use the full processor output, run the
        vision tower, then use pixel_position_ids to group output tokens
        by source image and pool per-frame.
        """
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
                tokens = vision_out.last_hidden_state  # (total_tokens, D)

                if debug_shapes:
                    print(f"    [debug] pixel_values: {pixel_values.shape}")
                    print(f"    [debug] pixel_position_ids: {pos_ids.shape}")
                    print(f"    [debug] vision tower output: {tokens.shape}")

                if tokens.ndim == 3:
                    # (T, N_tokens, D) — already per-frame
                    per_frame = F.normalize(tokens.mean(dim=1), dim=-1)
                elif tokens.ndim == 2:
                    # (total_tokens, D) — use position IDs to group by frame
                    # pixel_position_ids has shape (total_tokens, 3) or similar;
                    # first column is typically the image index
                    if pos_ids.ndim == 2 and pos_ids.shape[1] >= 1:
                        img_indices = pos_ids[:, 0]  # image index per token
                        unique_imgs = img_indices.unique(sorted=True)
                        n_images = len(unique_imgs)

                        if debug_shapes:
                            print(
                                f"    [debug] unique image indices: {n_images}, "
                                f"range [{unique_imgs.min()}, {unique_imgs.max()}]"
                            )

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
                        # Can't determine per-frame grouping; fall back to
                        # even split by number of input frames
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

            if debug_shapes:
                print(f"    [debug] per_frame output: {per_frame.shape}")

            return per_frame.cpu()
        except Exception as e:
            warnings.warn(f"GemmaAdapter.extract_vision_seq failed: {e}")
            return None

    def extract_vision_token_repr(self, model, inputs, layer_idx=-1):
        try:
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                return None

            pos_ids = inputs.get("image_position_ids", inputs.get("pixel_position_ids"))

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # (seq_len_out, D)

                ids = input_ids[0]  # (seq_len_in,)
                seq_len_in = len(ids)
                seq_len_out = hidden.shape[0]

                # Find vision token ID
                img_tok_id = getattr(model.config, "image_token_index", None)
                if img_tok_id is None:
                    for candidate_id in [258880, 262144, 256000]:
                        if (ids == candidate_id).sum().item() > 0:
                            img_tok_id = candidate_id
                            break

                if img_tok_id is None:
                    return None

                vision_positions = (ids == img_tok_id).nonzero(as_tuple=True)[0]
                n_placeholders = len(vision_positions)
                if n_placeholders == 0:
                    return None

                if seq_len_out > seq_len_in:
                    # Vision placeholders were expanded: each placeholder at
                    # position p in input_ids was replaced by
                    # tokens_per_placeholder vision tokens in hidden_states.
                    n_extra = seq_len_out - seq_len_in
                    tokens_per_placeholder = (
                        n_extra + n_placeholders
                    ) // n_placeholders

                    # Build list of vision hidden states grouped by frame
                    frame_hidden_list = []
                    for idx, orig_pos in enumerate(vision_positions):
                        # Each placeholder shifts subsequent positions by
                        # (tokens_per_placeholder - 1) for each prior expansion
                        expanded_pos = int(orig_pos.item()) + idx * (
                            tokens_per_placeholder - 1
                        )
                        frame_toks = hidden[
                            expanded_pos : expanded_pos + tokens_per_placeholder
                        ]
                        frame_hidden_list.append(frame_toks)
                    vision_hidden = torch.cat(frame_hidden_list, dim=0)
                else:
                    # No expansion: 1:1 mapping
                    vision_hidden = hidden[vision_positions]
                    frame_hidden_list = [hidden[p : p + 1] for p in vision_positions]

                n_found = vision_hidden.shape[0]
                if n_found == 0:
                    return None

                # vision_only_pool
                vision_only_pool = F.normalize(vision_hidden.mean(dim=0), dim=0)

                # vision_only_seq: group by frame (one frame per placeholder)
                n_images = n_placeholders
                if n_images >= 2:
                    frame_embs = []
                    for frame_toks in frame_hidden_list:
                        frame_embs.append(F.normalize(frame_toks.mean(dim=0), dim=0))
                    vision_only_seq = torch.stack(frame_embs)
                else:
                    vision_only_seq = vision_only_pool.unsqueeze(0)

                # bos_token
                bos_token = F.normalize(hidden[0], dim=0)

            return (
                vision_only_pool.cpu(),
                vision_only_seq.cpu(),
                bos_token.cpu(),
            )
        except Exception as e:
            warnings.warn(f"GemmaAdapter.extract_vision_token_repr failed: {e}")
            return None

    def extract_llm_repr(self, model, inputs, layer_idx=-1):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)
            emb = F.normalize(hidden.mean(dim=0), dim=0)
        return emb.cpu()


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
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            # Decode with special tokens to handle thinking blocks
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            # Strip thinking block if present: <think>...</think>\n\nANSWER
            if "</think>" in raw:
                response = raw.split("</think>", 1)[-1]
            else:
                # No thinking block — strip special tokens normally
                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
            # Remove any remaining special tokens
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                response = response.replace(tok, "")
        return response.strip()

    def extract_vision_repr(self, model, inputs):
        try:
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                return None
            with torch.no_grad():
                # pixel_values may be 5D (B, T, C, H, W) for video — flatten
                # to 4D (B*T, C, H, W) for CLIP/SigLIP vision tower
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
            warnings.warn(f"LlavaVideoAdapter.extract_vision_repr failed: {e}")
            return None

    def extract_vision_seq(
        self, model, inputs, processor=None, frames=None, debug_shapes=False
    ):
        try:
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                return None
            with torch.no_grad():
                if pixel_values.ndim == 5:
                    B, T, C, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(B * T, C, H, W)
                else:
                    T = pixel_values.shape[0]
                vision_out = model.model.vision_tower(
                    pixel_values,
                    output_hidden_states=True,
                )
                # (T, N_patches, D) — pool patches per frame, keep T
                tokens = vision_out.hidden_states[-2][:, 1:]  # skip CLS
                per_frame = F.normalize(tokens.mean(dim=1), dim=-1)  # (T, D)
            return per_frame.cpu()
        except Exception as e:
            warnings.warn(f"LlavaVideoAdapter.extract_vision_seq failed: {e}")
            return None

    def extract_vision_token_repr(self, model, inputs, layer_idx=-1):
        try:
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                return None

            # LLaVA-Video uses a special image token ID in input_ids where
            # vision features are injected. After the model's merge step,
            # these positions in the LLM hidden states correspond to vision
            # tokens.
            pixel_values = inputs.get(
                "pixel_values_videos",
                inputs.get("pixel_values"),
            )
            if pixel_values is None:
                return None

            # Determine number of frames and tokens per frame from pixel_values
            if pixel_values.ndim == 5:
                n_frames = pixel_values.shape[1]
            else:
                n_frames = pixel_values.shape[0]

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)

                # LLaVA uses image_token_index from config for vision placeholders
                ids = input_ids[0]  # (seq_len,)
                vision_mask = None

                img_tok_id = getattr(model.config, "image_token_index", None)
                if img_tok_id is not None:
                    mask = ids == img_tok_id
                    if mask.sum().item() > 0:
                        vision_mask = mask

                if vision_mask is None:
                    # Fallback: try known LLaVA image token IDs
                    for candidate_id in [32000, 32001, 64000]:
                        mask = ids == candidate_id
                        if mask.sum().item() > 0:
                            vision_mask = mask
                            break

                if vision_mask is None or vision_mask.sum().item() == 0:
                    # Last resort: after model forward, the hidden states
                    # have been expanded — vision tokens replace image_token_id.
                    # If we can't find them, we cannot proceed.
                    return None

                n_found = int(vision_mask.sum().item())

                # NOTE: After the LLM forward pass, the model's merge step
                # expands image tokens so hidden_states is longer than
                # input_ids. The vision_mask from input_ids identifies
                # *pre-merge* positions. After merging, each image token
                # position is replaced by (n_patches) vision tokens.
                # However, output_hidden_states returns states at the
                # *merged* sequence positions. We handle this by using the
                # full hidden state length — if the hidden state is longer
                # than input_ids, the vision tokens were expanded inline.
                if hidden.shape[0] > ids.shape[0]:
                    # Model expanded vision tokens — use all positions where
                    # vision features were inserted. The total number of
                    # vision tokens in the expanded sequence = hidden_len - text_len
                    text_len = ids.shape[0] - n_found
                    n_vision_expanded = hidden.shape[0] - text_len

                    # Find first vision token position in input_ids
                    vision_positions = torch.where(vision_mask)[0]
                    first_vis_pos = int(vision_positions[0].item())

                    # In the expanded hidden states, vision tokens start at
                    # first_vis_pos and span n_vision_expanded positions
                    vision_hidden = hidden[
                        first_vis_pos : first_vis_pos + n_vision_expanded
                    ]
                    n_found = n_vision_expanded
                else:
                    vision_hidden = hidden[vision_mask]

                # vision_only_pool
                vision_only_pool = F.normalize(vision_hidden.mean(dim=0), dim=0)

                # vision_only_seq: divide evenly by n_frames
                tokens_per_frame = n_found // max(n_frames, 1)
                if tokens_per_frame >= 1 and n_frames >= 2:
                    frame_embs = []
                    for i in range(n_frames):
                        start = i * tokens_per_frame
                        end = (
                            (i + 1) * tokens_per_frame if i < n_frames - 1 else n_found
                        )
                        frame_embs.append(
                            F.normalize(vision_hidden[start:end].mean(dim=0), dim=0)
                        )
                    vision_only_seq = torch.stack(frame_embs)
                else:
                    vision_only_seq = vision_only_pool.unsqueeze(0)

                # bos_token
                bos_token = F.normalize(hidden[0], dim=0)

            return (
                vision_only_pool.cpu(),
                vision_only_seq.cpu(),
                bos_token.cpu(),
            )
        except Exception as e:
            warnings.warn(f"LlavaVideoAdapter.extract_vision_token_repr failed: {e}")
            return None

    def extract_llm_repr(self, model, inputs, layer_idx=-1):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][0]  # (seq_len, D)
            emb = F.normalize(hidden.mean(dim=0), dim=0)
        return emb.cpu()


VLM_ADAPTERS = {
    "qwen3": QwenAdapter,
    "gemma4": GemmaAdapter,
    "llava-video": LlavaVideoAdapter,
}


# ---------------------------------------------------------------------------
# DINOv3 feature extraction
# ---------------------------------------------------------------------------


def extract_dinov3_features(
    sequences: list[dict],
    device: torch.device,
    target_fps: float = 3.0,
    max_resolution: int = 518,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Extract DINOv3 embeddings and attention centroids for all sequences.

    Args:
        sequences: List of sequence dicts with video_path.
        device: Torch device.
        target_fps: Frame extraction rate.
        max_resolution: Max frame height for DINOv3.
        cache_path: Optional path to cache features.

    Returns:
        Dict mapping sequence_id -> {
            'embeddings': (T, 1024),
            'centroids': (T, 2),
            'mean_emb': (1024,),
        }
    """
    if cache_path:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            return cached

    from video_retrieval.models import DINOv3Encoder

    print("  Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(model_name=DINOV3_LOCAL_PATH, device=device)

    features = {}
    failed = 0

    for seq in tqdm(sequences, desc="DINOv3 features"):
        try:
            frames = load_clip(
                seq["video_path"],
                seq["start_sec"],
                seq["stop_sec"],
                target_fps=target_fps,
                max_resolution=max_resolution,
            )
            if len(frames) < 3:
                failed += 1
                continue

            emb = encoder.encode_frames(frames)  # (T, 1024)
            centroids = encoder.get_attention_centroids(frames)  # (T, 2)
            mean_emb = F.normalize(emb.mean(dim=0), dim=0)

            features[seq["sequence_id"]] = {
                "embeddings": emb.cpu(),
                "centroids": centroids.cpu(),
                "mean_emb": mean_emb.cpu(),
            }
        except Exception as e:
            failed += 1
            continue

    print(f"  DINOv3: {len(features)}/{len(sequences)} ({failed} failed)")

    del encoder
    torch.cuda.empty_cache()

    if cache_path:
        save_feature_cache(features, cache_path)

    return features


# ---------------------------------------------------------------------------
# V-JEPA 2 feature extraction
# ---------------------------------------------------------------------------


def extract_vjepa2_features(
    sequences: list[dict],
    device: torch.device,
    reverse: bool = False,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for all sequences.

    Args:
        sequences: List of sequence dicts with video_path.
        device: Torch device.
        reverse: If True, reverse frame order before encoding.
        cache_path: Optional path to cache features.

    Returns:
        Dict mapping sequence_id -> {
            'mean_emb': (1024,),
            'temporal_residual': (n_target, 1024),
        }
    """
    if cache_path:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            return cached

    from transformers import AutoVideoProcessor, VJEPA2Model

    print(f"  Loading V-JEPA 2 model ({'reverse' if reverse else 'forward'})...")
    model = VJEPA2Model.from_pretrained(VJEPA2_LOCAL_PATH).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(VJEPA2_LOCAL_PATH)

    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for seq in tqdm(sequences, desc=f"V-JEPA 2 ({'rev' if reverse else 'fwd'})"):
        try:
            frames = load_clip_vjepa2(
                seq["video_path"],
                seq["start_sec"],
                seq["stop_sec"],
            )
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue

            if reverse:
                frames = frames[::-1]

            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]
                mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

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

            features[seq["sequence_id"]] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    direction = "reverse" if reverse else "forward"
    print(
        f"  V-JEPA 2 ({direction}): {len(features)}/{len(sequences)} "
        f"({failed} failed)"
    )

    del model, processor
    torch.cuda.empty_cache()

    if cache_path:
        save_feature_cache(features, cache_path)

    return features


# ---------------------------------------------------------------------------
# VLM embedding extraction (adapter-based)
# ---------------------------------------------------------------------------


def extract_vlm_features(
    sequences: list[dict],
    adapter: VLMAdapter,
    model_path: str,
    device: torch.device,
    n_frames: int = 16,
    max_resolution: int = 518,
    reverse: bool = False,
    layer_ablation: bool = False,
    vision_token_probe: bool = False,
    cache_path: Path | None = None,
    debug_shapes: bool = False,
) -> dict[str, dict]:
    """Extract VLM embeddings for all sequences using adapter protocol.

    Extracts vision repr (best-effort) and LLM repr at specified layers.
    Uses sample_canonical_frames() for input parity across all VLMs.

    Args:
        sequences: List of sequence dicts with video_path.
        adapter: VLMAdapter instance.
        model_path: Path to local model weights.
        device: Torch device.
        n_frames: Canonical frame count.
        max_resolution: Max frame height.
        reverse: If True, reverse frame order.
        layer_ablation: If True, extract at last/mid/penultimate layers.
        vision_token_probe: If True, extract vision-token-only LLM hidden
            state probes (vision_only_pool, vision_only_seq, bos_token).
        cache_path: Optional path to cache features.

    Returns:
        Dict mapping sequence_id -> {
            'vision_repr': (D,) or None,
            'llm_repr_last': (D,),
            'llm_repr_mid': (D,),          # only if layer_ablation
            'llm_repr_penultimate': (D,),   # only if layer_ablation
            'vision_only_pool': (D,),       # only if vision_token_probe
            'vision_only_seq': (T, D),      # only if vision_token_probe
            'bos_token': (D,),              # only if vision_token_probe
            'timestamps': list[float],
        }
    """
    if cache_path:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            return cached

    direction = "reverse" if reverse else "forward"
    print(f"  Loading VLM from {model_path} ({direction})...")
    model, processor = adapter.load(model_path, device)

    # Determine number of layers for mid-layer index
    n_layers = None
    if layer_ablation:
        try:
            if hasattr(model.config, "num_hidden_layers"):
                n_layers = model.config.num_hidden_layers
            elif hasattr(model.config, "text_config") and hasattr(
                model.config.text_config, "num_hidden_layers"
            ):
                n_layers = model.config.text_config.num_hidden_layers
        except Exception:
            pass
        if n_layers is None:
            print("  WARNING: Could not determine n_layers, using 32 as default")
            n_layers = 32

    features = {}
    failed = 0

    for seq in tqdm(sequences, desc=f"VLM emb ({direction})"):
        try:
            pil_frames, timestamps = sample_canonical_frames(
                seq["video_path"],
                seq["start_sec"],
                seq["stop_sec"],
                n_frames=n_frames,
                max_resolution=max_resolution,
            )
            if len(pil_frames) < 3:
                failed += 1
                continue

            if reverse:
                pil_frames = pil_frames[::-1]
                timestamps = timestamps[::-1]

            fps_eff = compute_fps_eff(timestamps)
            inputs = adapter.prepare_inputs(
                processor,
                pil_frames,
                DESCRIBE_PROMPT,
                device,
                fps=fps_eff,
            )

            # Vision repr (best-effort)
            vision_repr = adapter.extract_vision_repr(model, inputs)

            # Per-frame vision sequence for DTW probe
            debug_this = debug_shapes and (seq == sequences[0])
            vision_seq = adapter.extract_vision_seq(
                model,
                inputs,
                processor=processor,
                frames=pil_frames,
                debug_shapes=debug_this,
            )

            # LLM repr at last layer (always)
            llm_repr_last = adapter.extract_llm_repr(model, inputs, layer_idx=-1)

            feat_dict = {
                "vision_repr": vision_repr,
                "vision_seq": vision_seq,
                "llm_repr_last": llm_repr_last,
                "timestamps": timestamps,
            }

            # Layer ablation: mid and penultimate
            if layer_ablation and n_layers:
                mid_idx = n_layers // 2
                feat_dict["llm_repr_mid"] = adapter.extract_llm_repr(
                    model,
                    inputs,
                    layer_idx=mid_idx,
                )
                feat_dict["llm_repr_penultimate"] = adapter.extract_llm_repr(
                    model,
                    inputs,
                    layer_idx=-2,
                )

            # Vision-token-only LLM hidden state probes
            if vision_token_probe:
                vt_result = adapter.extract_vision_token_repr(
                    model,
                    inputs,
                    layer_idx=-1,
                )
                if vt_result is not None:
                    vt_pool, vt_seq, vt_bos = vt_result
                    feat_dict["vision_only_pool"] = vt_pool
                    feat_dict["vision_only_seq"] = vt_seq
                    feat_dict["bos_token"] = vt_bos

            features[seq["sequence_id"]] = feat_dict

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  WARNING: {seq['sequence_id']} failed: {e}")
            continue

    print(
        f"  VLM embedding ({direction}): {len(features)}/{len(sequences)} "
        f"({failed} failed)"
    )

    del model, processor
    torch.cuda.empty_cache()

    if cache_path:
        save_feature_cache(features, cache_path)

    return features


# ---------------------------------------------------------------------------
# VLM generative probes (adapter-based, multi-prompt)
# ---------------------------------------------------------------------------


def parse_forward_reverse(response: str) -> str | None:
    """Parse FORWARD or REVERSE from model response."""
    response_upper = response.upper()
    if "FORWARD" in response_upper:
        return "FORWARD"
    elif "REVERSE" in response_upper:
        return "REVERSE"
    return None


def parse_intact_tampered(response: str) -> str | None:
    """Parse INTACT or TAMPERED from model response."""
    response_upper = response.upper()
    if "INTACT" in response_upper:
        return "INTACT"
    elif "TAMPERED" in response_upper:
        return "TAMPERED"
    return None


def evaluate_vlm_generative(
    sequences: list[dict],
    adapter: VLMAdapter,
    model_path: str,
    device: torch.device,
    n_frames: int = 16,
    max_resolution: int = 518,
    causality_prompts: list[str] | None = None,
    integrity_prompt: str | None = None,
    run_integrity: bool = False,
    scramble_levels: list[int] | None = None,
    run_text_only: bool = True,
) -> dict:
    """Run VLM generative probes using adapter protocol.

    Runs causality prompt variants for forward/reverse classification,
    optional text-only baseline, and optional integrity probe with
    forward/reversed/scrambled transforms.

    Args:
        sequences: List of sequence dicts with video_path.
        adapter: VLMAdapter instance.
        model_path: Path to local model weights.
        device: Torch device.
        n_frames: Canonical frame count.
        max_resolution: Max frame height.
        causality_prompts: Forward/reverse prompt variants (defaults used if None).
        integrity_prompt: INTACT/TAMPERED prompt (defaults used if None).
        run_integrity: Whether to run the integrity probe.
        scramble_levels: Scramble chunk counts for integrity probe.
        run_text_only: Whether to run text-only baseline.

    Returns:
        Dict with per_prompt, mean/std balanced_acc, text_only_baseline, integrity.
    """
    if causality_prompts is None:
        causality_prompts = DEFAULT_CAUSALITY_PROMPTS
    if integrity_prompt is None:
        integrity_prompt = DEFAULT_INTEGRITY_PROMPT
    if scramble_levels is None:
        scramble_levels = [4, 16]

    print(f"  Loading VLM from {model_path}...")
    model, processor = adapter.load(model_path, device)

    # ---- Forward/Reverse probe with multiple prompts ----
    n_prompts = len(causality_prompts)
    print(f"\n  === Forward/Reverse Probe ({n_prompts} prompt variants) ===")

    per_prompt_results = {}
    all_balanced_accs = []

    for prompt_idx, prompt in enumerate(causality_prompts):
        print(f"\n  --- Prompt {prompt_idx} ---")
        fwd_correct = 0
        rev_correct = 0
        fwd_total = 0
        rev_total = 0

        for seq in tqdm(sequences, desc=f"Fwd/Rev prompt_{prompt_idx}"):
            try:
                pil_frames, timestamps = sample_canonical_frames(
                    seq["video_path"],
                    seq["start_sec"],
                    seq["stop_sec"],
                    n_frames=n_frames,
                    max_resolution=max_resolution,
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
                # Debug: log first 3 responses per prompt
                if fwd_total + rev_total < 6 and prompt_idx == 0:
                    print(f"    [debug] fwd resp: {resp_fwd[:120]!r} -> {pred_fwd}")
                if pred_fwd is not None:
                    fwd_total += 1
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
                    if pred_rev == "REVERSE":
                        rev_correct += 1

            except Exception as e:
                if fwd_total + rev_total < 3:
                    print(f"    [error] {seq['sequence_id']}: {e}")
                continue
        fwd_acc = fwd_correct / max(fwd_total, 1)
        rev_acc = rev_correct / max(rev_total, 1)
        balanced_acc = (fwd_acc + rev_acc) / 2

        per_prompt_results[f"prompt_{prompt_idx}"] = {
            "balanced_acc": round(balanced_acc, 4),
            "fwd_acc": round(fwd_acc, 4),
            "rev_acc": round(rev_acc, 4),
            "fwd_total": fwd_total,
            "rev_total": rev_total,
        }
        all_balanced_accs.append(balanced_acc)
        print(
            f"  Prompt {prompt_idx}: fwd={fwd_acc:.3f}, rev={rev_acc:.3f}, "
            f"balanced={balanced_acc:.3f}"
        )

    mean_balanced_acc = float(np.mean(all_balanced_accs))
    std_balanced_acc = float(np.std(all_balanced_accs))
    print(f"\n  Mean balanced acc: {mean_balanced_acc:.3f} +/- {std_balanced_acc:.3f}")

    # ---- Text-only baseline (per prompt) ----
    text_only_result = None
    if run_text_only:
        print("\n  === Text-Only Baseline ===")
        text_only_per_prompt = {}
        n_text_trials = min(len(sequences), 50)

        for prompt_idx, prompt in enumerate(causality_prompts):
            text_only_preds = []
            for _ in tqdm(range(n_text_trials), desc=f"Text-only prompt_{prompt_idx}"):
                try:
                    inputs = adapter.prepare_text_only_inputs(
                        processor,
                        prompt,
                        device,
                    )
                    resp = adapter.generate(model, inputs, processor)
                    pred = parse_forward_reverse(resp)
                    if pred is not None:
                        text_only_preds.append(pred)
                except Exception:
                    continue

            n_fwd = sum(1 for p in text_only_preds if p == "FORWARD")
            bias = n_fwd / max(len(text_only_preds), 1)
            text_only_per_prompt[f"prompt_{prompt_idx}"] = {
                "bias_rate_forward": round(bias, 4),
                "n_trials": len(text_only_preds),
                "n_forward": n_fwd,
            }
            print(
                f"  Prompt {prompt_idx} text-only bias: {bias:.3f} "
                f"({n_fwd}/{len(text_only_preds)})"
            )

        # Aggregate text-only bias
        all_bias = [v["bias_rate_forward"] for v in text_only_per_prompt.values()]
        text_only_result = {
            "per_prompt": text_only_per_prompt,
            "mean_bias_rate_forward": round(float(np.mean(all_bias)), 4),
        }

    # ---- Temporal integrity probe (optional) ----
    integrity_final = None
    if run_integrity:
        print("\n  === Temporal Integrity Probe ===")
        integrity_results = {
            "forward": {"correct": 0, "total": 0},
            "reverse": {"correct": 0, "total": 0},
        }
        for k in scramble_levels:
            integrity_results[f"scramble_k{k}"] = {"correct": 0, "total": 0}

        per_clip_all_correct = []

        for seq in tqdm(sequences, desc="Integrity probe"):
            try:
                pil_frames, timestamps = sample_canonical_frames(
                    seq["video_path"],
                    seq["start_sec"],
                    seq["stop_sec"],
                    n_frames=n_frames,
                    max_resolution=max_resolution,
                )
                if len(pil_frames) < 3:
                    continue

                fps_eff = compute_fps_eff(timestamps)
                clip_correct = True

                # Forward (should be INTACT)
                inputs = adapter.prepare_inputs(
                    processor,
                    pil_frames,
                    integrity_prompt,
                    device,
                    fps=fps_eff,
                )
                resp = adapter.generate(model, inputs, processor)
                pred = parse_intact_tampered(resp)
                if pred is not None:
                    integrity_results["forward"]["total"] += 1
                    if pred == "INTACT":
                        integrity_results["forward"]["correct"] += 1
                    else:
                        clip_correct = False

                # Reverse (should be TAMPERED)
                rev_frames = pil_frames[::-1]
                inputs = adapter.prepare_inputs(
                    processor,
                    rev_frames,
                    integrity_prompt,
                    device,
                    fps=fps_eff,
                )
                resp = adapter.generate(model, inputs, processor)
                pred = parse_intact_tampered(resp)
                if pred is not None:
                    integrity_results["reverse"]["total"] += 1
                    if pred == "TAMPERED":
                        integrity_results["reverse"]["correct"] += 1
                    else:
                        clip_correct = False

                # Scramble variants (should be TAMPERED)
                for k in scramble_levels:
                    scr_frames = scramble_frames(pil_frames, k, seed=42)
                    inputs = adapter.prepare_inputs(
                        processor,
                        scr_frames,
                        integrity_prompt,
                        device,
                        fps=fps_eff,
                    )
                    resp = adapter.generate(model, inputs, processor)
                    pred = parse_intact_tampered(resp)
                    key = f"scramble_k{k}"
                    if pred is not None:
                        integrity_results[key]["total"] += 1
                        if pred == "TAMPERED":
                            integrity_results[key]["correct"] += 1
                        else:
                            clip_correct = False

                per_clip_all_correct.append(1.0 if clip_correct else 0.0)

            except Exception:
                continue

        # Compute accuracy per condition
        integrity_accs = {}
        for condition, counts in integrity_results.items():
            total = counts["total"]
            correct = counts["correct"]
            acc = correct / max(total, 1)
            integrity_accs[f"{condition}_acc"] = round(acc, 4)
            print(f"  {condition}: {acc:.3f} ({correct}/{total})")

        consistency_arr = (
            np.array(per_clip_all_correct) if per_clip_all_correct else np.array([0.0])
        )
        cons_point, cons_ci_low, cons_ci_high = bootstrap_ci(consistency_arr)
        print(
            f"  Consistency (all correct): {cons_point:.3f} "
            f"[{cons_ci_low:.3f}, {cons_ci_high:.3f}]"
        )

        integrity_final = {
            **integrity_accs,
            "consistency_pct": round(cons_point, 4),
            "consistency_ci": [round(cons_ci_low, 4), round(cons_ci_high, 4)],
        }

    del model, processor
    torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "per_prompt": per_prompt_results,
        "mean_balanced_acc": round(mean_balanced_acc, 4),
        "std_balanced_acc": round(std_balanced_acc, 4),
        "prompts_used": causality_prompts,
    }
    if integrity_final:
        result["integrity"] = integrity_final
        result["integrity_prompt_used"] = integrity_prompt
    if text_only_result:
        result["text_only_baseline"] = text_only_result

    return result


# ---------------------------------------------------------------------------
# Order sensitivity evaluation
# ---------------------------------------------------------------------------


def evaluate_order_sensitivity(
    dinov3_features: dict[str, dict],
    vjepa2_fwd_features: dict[str, dict] | None,
    vjepa2_rev_features: dict[str, dict] | None,
    vlm_fwd_features: dict[str, dict] | None = None,
    vlm_rev_features: dict[str, dict] | None = None,
    vlm_family: str | None = None,
) -> dict:
    """Compute order sensitivity s_rev and balanced accuracy per method.

    s_rev = cosine_similarity(forward_embedding, reversed_embedding)
    Higher s_rev -> method is more order-invariant (blind to reversal)
    Lower s_rev -> method is more order-sensitive (detects reversal)

    For DINOv3, reversed = embeddings.flip(0) (zero cost, no positional encoding).
    For V-JEPA 2, reversed features must be separately extracted.

    Returns:
        Dict mapping method_name -> {s_rev_mean, s_rev_ci, balanced_acc, acc_ci}
    """
    from video_retrieval.fingerprints import (
        TemporalDerivativeFingerprint,
        TrajectoryFingerprint,
    )
    from video_retrieval.fingerprints.trajectory import dtw_distance

    results = {}

    # Sequence IDs common to all available feature sets
    seq_ids = sorted(dinov3_features.keys())

    # --- DINOv3 methods ---

    # 1. Bag-of-frames (mean embedding)
    print("  Computing Bag-of-frames s_rev...")
    s_rev_bof = []
    for sid in seq_ids:
        feat = dinov3_features[sid]
        emb = feat["embeddings"]
        fwd_mean = F.normalize(emb.mean(dim=0), dim=0)
        rev_mean = F.normalize(emb.flip(0).mean(dim=0), dim=0)  # Same as fwd_mean!
        sim = F.cosine_similarity(fwd_mean.unsqueeze(0), rev_mean.unsqueeze(0)).item()
        s_rev_bof.append(sim)

    s_rev_arr = np.array(s_rev_bof)
    point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
    # BoF is order-invariant (mean is permutation-invariant), so balanced acc is N/A
    results["bag_of_frames"] = {
        "s_rev_mean": round(point, 4),
        "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_acc": None,  # order-invariant, not meaningful
        "acc_ci": None,
    }

    # 2. Chamfer (pairwise min distances, order-invariant)
    print("  Computing Chamfer s_rev...")
    s_rev_chamfer = []
    for sid in seq_ids:
        feat = dinov3_features[sid]
        emb = feat["embeddings"]
        rev_emb = emb.flip(0)
        dists = torch.cdist(emb, rev_emb)
        chamfer = (dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()) / 2
        sim = float(torch.exp(-chamfer))
        s_rev_chamfer.append(sim)

    s_rev_arr = np.array(s_rev_chamfer)
    point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
    results["chamfer"] = {
        "s_rev_mean": round(point, 4),
        "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_acc": None,  # order-invariant, not meaningful
        "acc_ci": None,
    }

    # 3. Temporal derivative (DTW on embedding derivatives)
    print("  Computing Temporal derivative s_rev...")
    td_fp = TemporalDerivativeFingerprint()
    s_rev_td = []
    for sid in seq_ids:
        feat = dinov3_features[sid]
        emb = feat["embeddings"]
        d_fwd = td_fp.compute_fingerprint(emb)
        d_rev = td_fp.compute_fingerprint(emb.flip(0))
        sim = td_fp.compare(d_fwd, d_rev)
        s_rev_td.append(float(sim))

    s_rev_arr = np.array(s_rev_td)
    point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
    acc_arr = np.array([1.0 if s < 0.8 else 0.0 for s in s_rev_td])
    acc_point, acc_ci_low, acc_ci_high = bootstrap_ci(acc_arr)
    results["temporal_derivative"] = {
        "s_rev_mean": round(point, 4),
        "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_acc": round(acc_point, 4),
        "acc_ci": [round(acc_ci_low, 4), round(acc_ci_high, 4)],
    }

    # 4. Attention trajectory (DTW on centroid trajectories)
    print("  Computing Attention trajectory s_rev...")
    traj_fp = TrajectoryFingerprint()
    s_rev_traj = []
    for sid in seq_ids:
        feat = dinov3_features[sid]
        centroids = feat["centroids"]
        t_fwd = traj_fp.compute_fingerprint(centroids)
        t_rev = traj_fp.compute_fingerprint(centroids.flip(0))
        sim = traj_fp.compare(t_fwd, t_rev)
        s_rev_traj.append(float(sim))

    s_rev_arr = np.array(s_rev_traj)
    point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
    acc_arr = np.array([1.0 if s < 0.8 else 0.0 for s in s_rev_traj])
    acc_point, acc_ci_low, acc_ci_high = bootstrap_ci(acc_arr)
    results["attention_trajectory"] = {
        "s_rev_mean": round(point, 4),
        "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_acc": round(acc_point, 4),
        "acc_ci": [round(acc_ci_low, 4), round(acc_ci_high, 4)],
    }

    # --- V-JEPA 2 methods ---
    if vjepa2_fwd_features and vjepa2_rev_features:
        common_ids = sorted(
            set(vjepa2_fwd_features.keys()) & set(vjepa2_rev_features.keys())
        )

        # 5. V-JEPA 2 bag-of-tokens
        print("  Computing V-JEPA 2 BoT s_rev...")
        s_rev_vj_bot = []
        for sid in common_ids:
            fwd = vjepa2_fwd_features[sid]["mean_emb"]
            rev = vjepa2_rev_features[sid]["mean_emb"]
            sim = F.cosine_similarity(fwd.unsqueeze(0), rev.unsqueeze(0)).item()
            s_rev_vj_bot.append(sim)

        s_rev_arr = np.array(s_rev_vj_bot)
        point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
        results["vjepa2_bag_of_tokens"] = {
            "s_rev_mean": round(point, 4),
            "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
            "balanced_acc": 0.5,
            "acc_ci": [0.5, 0.5],
        }

        # 6. V-JEPA 2 temporal residual
        print("  Computing V-JEPA 2 temporal residual s_rev...")
        s_rev_vj_res = []
        for sid in common_ids:
            fwd_res = vjepa2_fwd_features[sid]["temporal_residual"]
            rev_res = vjepa2_rev_features[sid]["temporal_residual"]
            fwd_flat = fwd_res.reshape(-1)
            rev_flat = rev_res.reshape(-1)
            sim = F.cosine_similarity(
                fwd_flat.unsqueeze(0), rev_flat.unsqueeze(0)
            ).item()
            s_rev_vj_res.append(sim)

        s_rev_arr = np.array(s_rev_vj_res)
        point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
        acc_arr = np.array([1.0 if s < 0.8 else 0.0 for s in s_rev_vj_res])
        acc_point, acc_ci_low, acc_ci_high = bootstrap_ci(acc_arr)
        results["vjepa2_temporal_residual"] = {
            "s_rev_mean": round(point, 4),
            "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
            "balanced_acc": round(acc_point, 4),
            "acc_ci": [round(acc_ci_low, 4), round(acc_ci_high, 4)],
        }

    # --- VLM methods ---
    if vlm_fwd_features and vlm_rev_features:
        family_label = vlm_family or "vlm"
        common_ids = sorted(set(vlm_fwd_features.keys()) & set(vlm_rev_features.keys()))

        # Vision repr
        s_rev_vision = []
        has_vision = False
        for sid in common_ids:
            fwd_v = vlm_fwd_features[sid].get("vision_repr")
            rev_v = vlm_rev_features[sid].get("vision_repr")
            if fwd_v is not None and rev_v is not None:
                sim = F.cosine_similarity(fwd_v.unsqueeze(0), rev_v.unsqueeze(0)).item()
                s_rev_vision.append(sim)
                has_vision = True

        if has_vision and s_rev_vision:
            print(f"  Computing {family_label} vision_repr s_rev...")
            s_rev_arr = np.array(s_rev_vision)
            point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
            results[f"{family_label}_vision_repr"] = {
                "s_rev_mean": round(point, 4),
                "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
            }

        # Vision sequence DTW probe (per-frame, order-sensitive)
        td_fp = TemporalDerivativeFingerprint()
        s_rev_vis_seq = []
        has_vis_seq = False
        for sid in common_ids:
            fwd_seq = vlm_fwd_features[sid].get("vision_seq")
            rev_seq = vlm_rev_features[sid].get("vision_seq")
            if fwd_seq is not None and rev_seq is not None and len(fwd_seq) >= 3:
                # Cast to float32 — DTW uses torch.cdist which doesn't support bf16
                d_fwd = td_fp.compute_fingerprint(fwd_seq.float())
                d_rev = td_fp.compute_fingerprint(rev_seq.float())
                sim = td_fp.compare(d_fwd, d_rev)
                s_rev_vis_seq.append(float(sim))
                has_vis_seq = True

        if has_vis_seq and s_rev_vis_seq:
            print(f"  Computing {family_label} vision_seq_dtw s_rev...")
            s_rev_arr = np.array(s_rev_vis_seq)
            point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
            acc_arr = np.array([1.0 if s < 0.8 else 0.0 for s in s_rev_vis_seq])
            acc_point, acc_ci_low, acc_ci_high = bootstrap_ci(acc_arr)
            results[f"{family_label}_vision_seq_dtw"] = {
                "s_rev_mean": round(point, 4),
                "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
                "balanced_acc": round(acc_point, 4),
                "acc_ci": [round(acc_ci_low, 4), round(acc_ci_high, 4)],
            }

        # Derivative sign flip check (no DTW, direct diagnostic)
        # If reversal flips temporal direction, forward and reverse derivatives
        # should point in opposite directions: mean cos(Δz_fwd, Δz_rev) < 0
        deriv_cos_values = []
        for sid in common_ids:
            fwd_seq = vlm_fwd_features[sid].get("vision_seq")
            rev_seq = vlm_rev_features[sid].get("vision_seq")
            if fwd_seq is not None and rev_seq is not None and len(fwd_seq) >= 3:
                d_fwd = (fwd_seq[1:] - fwd_seq[:-1]).float()  # (T-1, D)
                d_rev = (rev_seq[1:] - rev_seq[:-1]).float()
                # Match lengths (should be equal, but be safe)
                min_len = min(len(d_fwd), len(d_rev))
                cos_per_t = F.cosine_similarity(
                    d_fwd[:min_len],
                    d_rev[:min_len],
                    dim=-1,
                )
                deriv_cos_values.append(float(cos_per_t.mean()))

        if deriv_cos_values:
            mean_deriv_cos = float(np.mean(deriv_cos_values))
            std_deriv_cos = float(np.std(deriv_cos_values))
            print(
                f"  {family_label} derivative sign flip: "
                f"mean cos(Δz_fwd, Δz_rev) = {mean_deriv_cos:.4f} "
                f"± {std_deriv_cos:.4f}"
            )
            results[f"{family_label}_deriv_sign_flip"] = {
                "mean_cos": round(mean_deriv_cos, 4),
                "std_cos": round(std_deriv_cos, 4),
            }

        # LLM repr at each available layer
        for layer_key in ["llm_repr_last", "llm_repr_mid", "llm_repr_penultimate"]:
            s_rev_llm = []
            for sid in common_ids:
                fwd_l = vlm_fwd_features[sid].get(layer_key)
                rev_l = vlm_rev_features[sid].get(layer_key)
                if fwd_l is not None and rev_l is not None:
                    sim = F.cosine_similarity(
                        fwd_l.unsqueeze(0), rev_l.unsqueeze(0)
                    ).item()
                    s_rev_llm.append(sim)

            if s_rev_llm:
                print(f"  Computing {family_label} {layer_key} s_rev...")
                s_rev_arr = np.array(s_rev_llm)
                point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
                results[f"{family_label}_{layer_key}"] = {
                    "s_rev_mean": round(point, 4),
                    "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
                }

        # Vision-token-only LLM hidden state probes
        # 1. vision_only_pool: cosine similarity between fwd and rev
        s_rev_vt_pool = []
        for sid in common_ids:
            fwd_vp = vlm_fwd_features[sid].get("vision_only_pool")
            rev_vp = vlm_rev_features[sid].get("vision_only_pool")
            if fwd_vp is not None and rev_vp is not None:
                sim = F.cosine_similarity(
                    fwd_vp.unsqueeze(0), rev_vp.unsqueeze(0)
                ).item()
                s_rev_vt_pool.append(sim)

        if s_rev_vt_pool:
            print(f"  Computing {family_label} vision_only_pool s_rev...")
            s_rev_arr = np.array(s_rev_vt_pool)
            point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
            results[f"{family_label}_vision_only_pool"] = {
                "s_rev_mean": round(point, 4),
                "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
            }

        # 2. vision_only_seq_dtw: temporal derivative DTW
        td_fp_vt = TemporalDerivativeFingerprint()
        s_rev_vt_seq = []
        for sid in common_ids:
            fwd_vs = vlm_fwd_features[sid].get("vision_only_seq")
            rev_vs = vlm_rev_features[sid].get("vision_only_seq")
            if fwd_vs is not None and rev_vs is not None and len(fwd_vs) >= 3:
                d_fwd = td_fp_vt.compute_fingerprint(fwd_vs.float())
                d_rev = td_fp_vt.compute_fingerprint(rev_vs.float())
                sim = td_fp_vt.compare(d_fwd, d_rev)
                s_rev_vt_seq.append(float(sim))

        if s_rev_vt_seq:
            print(f"  Computing {family_label} vision_only_seq_dtw s_rev...")
            s_rev_arr = np.array(s_rev_vt_seq)
            point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
            acc_arr = np.array([1.0 if s < 0.8 else 0.0 for s in s_rev_vt_seq])
            acc_point, acc_ci_low, acc_ci_high = bootstrap_ci(acc_arr)
            results[f"{family_label}_vision_only_seq_dtw"] = {
                "s_rev_mean": round(point, 4),
                "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
                "balanced_acc": round(acc_point, 4),
                "acc_ci": [round(acc_ci_low, 4), round(acc_ci_high, 4)],
            }

        # 3. bos_token: cosine similarity between fwd and rev
        s_rev_bos = []
        for sid in common_ids:
            fwd_b = vlm_fwd_features[sid].get("bos_token")
            rev_b = vlm_rev_features[sid].get("bos_token")
            if fwd_b is not None and rev_b is not None:
                sim = F.cosine_similarity(fwd_b.unsqueeze(0), rev_b.unsqueeze(0)).item()
                s_rev_bos.append(sim)

        if s_rev_bos:
            print(f"  Computing {family_label} bos_token s_rev...")
            s_rev_arr = np.array(s_rev_bos)
            point, ci_low, ci_high = bootstrap_ci(s_rev_arr)
            results[f"{family_label}_bos_token"] = {
                "s_rev_mean": round(point, 4),
                "s_rev_ci": [round(ci_low, 4), round(ci_high, 4)],
            }

    return results


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval_task(
    dinov3_features: dict[str, dict],
    sequences: list[dict],
    vjepa2_fwd_features: dict[str, dict] | None = None,
    vjepa2_rev_features: dict[str, dict] | None = None,
) -> dict:
    """Evaluate forward vs reversed retrieval (AP/AUC).

    For each sequence, the query is the forward video. The gallery contains:
    - Its own reversed version (positive if we want to detect reversal)
    - Other sequences' forward versions (negatives)

    We measure whether each method assigns higher similarity to same-video
    reversed vs different-video forward.

    Returns:
        Dict mapping method_name -> {ap, auc}.
    """
    from video_retrieval.fingerprints import (
        TemporalDerivativeFingerprint,
        TrajectoryFingerprint,
    )

    seq_ids = sorted(dinov3_features.keys())
    if len(seq_ids) < 3:
        print("  WARNING: Too few sequences for retrieval evaluation")
        return {}

    results = {}

    # --- Bag-of-frames retrieval ---
    print("  Computing BoF retrieval...")
    all_mean = torch.stack([dinov3_features[sid]["mean_emb"] for sid in seq_ids])
    all_rev_mean = torch.stack(
        [
            F.normalize(dinov3_features[sid]["embeddings"].flip(0).mean(dim=0), dim=0)
            for sid in seq_ids
        ]
    )

    scores_bof = []
    labels_bof = []
    for i in range(len(seq_ids)):
        query = all_mean[i]
        pos_sim = F.cosine_similarity(
            query.unsqueeze(0), all_rev_mean[i].unsqueeze(0)
        ).item()
        scores_bof.append(pos_sim)
        labels_bof.append(1)
        for j in range(len(seq_ids)):
            if j != i:
                neg_sim = F.cosine_similarity(
                    query.unsqueeze(0), all_mean[j].unsqueeze(0)
                ).item()
                scores_bof.append(neg_sim)
                labels_bof.append(0)

    ap = average_precision_score(labels_bof, scores_bof)
    auc = roc_auc_score(labels_bof, scores_bof)
    results["bag_of_frames"] = {"ap": round(ap, 4), "auc": round(auc, 4)}
    print(f"    BoF: AP={ap:.3f}, AUC={auc:.3f}")

    # --- Temporal derivative retrieval ---
    print("  Computing temporal derivative retrieval...")
    td_fp = TemporalDerivativeFingerprint()

    fwd_fps_td = {}
    rev_fps_td = {}
    for sid in seq_ids:
        emb = dinov3_features[sid]["embeddings"]
        fwd_fps_td[sid] = td_fp.compute_fingerprint(emb)
        rev_fps_td[sid] = td_fp.compute_fingerprint(emb.flip(0))

    scores_td = []
    labels_td = []
    for i, sid_q in enumerate(seq_ids):
        pos_sim = td_fp.compare(fwd_fps_td[sid_q], rev_fps_td[sid_q])
        scores_td.append(float(pos_sim))
        labels_td.append(1)
        for j, sid_g in enumerate(seq_ids):
            if j != i:
                neg_sim = td_fp.compare(fwd_fps_td[sid_q], fwd_fps_td[sid_g])
                scores_td.append(float(neg_sim))
                labels_td.append(0)

    ap = average_precision_score(labels_td, scores_td)
    auc = roc_auc_score(labels_td, scores_td)
    results["temporal_derivative"] = {"ap": round(ap, 4), "auc": round(auc, 4)}
    print(f"    TD: AP={ap:.3f}, AUC={auc:.3f}")

    # --- Attention trajectory retrieval ---
    print("  Computing attention trajectory retrieval...")
    traj_fp = TrajectoryFingerprint()

    fwd_fps_traj = {}
    rev_fps_traj = {}
    for sid in seq_ids:
        centroids = dinov3_features[sid]["centroids"]
        fwd_fps_traj[sid] = traj_fp.compute_fingerprint(centroids)
        rev_fps_traj[sid] = traj_fp.compute_fingerprint(centroids.flip(0))

    scores_traj = []
    labels_traj = []
    for i, sid_q in enumerate(seq_ids):
        pos_sim = traj_fp.compare(fwd_fps_traj[sid_q], rev_fps_traj[sid_q])
        scores_traj.append(float(pos_sim))
        labels_traj.append(1)
        for j, sid_g in enumerate(seq_ids):
            if j != i:
                neg_sim = traj_fp.compare(fwd_fps_traj[sid_q], fwd_fps_traj[sid_g])
                scores_traj.append(float(neg_sim))
                labels_traj.append(0)

    ap = average_precision_score(labels_traj, scores_traj)
    auc = roc_auc_score(labels_traj, scores_traj)
    results["attention_trajectory"] = {"ap": round(ap, 4), "auc": round(auc, 4)}
    print(f"    Traj: AP={ap:.3f}, AUC={auc:.3f}")

    # --- V-JEPA 2 retrieval ---
    if vjepa2_fwd_features and vjepa2_rev_features:
        common_ids = sorted(
            set(vjepa2_fwd_features.keys())
            & set(vjepa2_rev_features.keys())
            & set(seq_ids)
        )

        if len(common_ids) >= 3:
            print("  Computing V-JEPA 2 BoT retrieval...")
            scores_vj = []
            labels_vj = []
            for i, sid_q in enumerate(common_ids):
                query = vjepa2_fwd_features[sid_q]["mean_emb"]
                pos = vjepa2_rev_features[sid_q]["mean_emb"]
                pos_sim = F.cosine_similarity(
                    query.unsqueeze(0), pos.unsqueeze(0)
                ).item()
                scores_vj.append(pos_sim)
                labels_vj.append(1)
                for j, sid_g in enumerate(common_ids):
                    if j != i:
                        neg = vjepa2_fwd_features[sid_g]["mean_emb"]
                        neg_sim = F.cosine_similarity(
                            query.unsqueeze(0), neg.unsqueeze(0)
                        ).item()
                        scores_vj.append(neg_sim)
                        labels_vj.append(0)

            ap = average_precision_score(labels_vj, scores_vj)
            auc = roc_auc_score(labels_vj, scores_vj)
            results["vjepa2_bag_of_tokens"] = {"ap": round(ap, 4), "auc": round(auc, 4)}
            print(f"    V-JEPA 2 BoT: AP={ap:.3f}, AUC={auc:.3f}")

            print("  Computing V-JEPA 2 temporal residual retrieval...")
            scores_vj_res = []
            labels_vj_res = []
            for i, sid_q in enumerate(common_ids):
                query = vjepa2_fwd_features[sid_q]["temporal_residual"].reshape(-1)
                pos = vjepa2_rev_features[sid_q]["temporal_residual"].reshape(-1)
                pos_sim = F.cosine_similarity(
                    query.unsqueeze(0), pos.unsqueeze(0)
                ).item()
                scores_vj_res.append(pos_sim)
                labels_vj_res.append(1)
                for j, sid_g in enumerate(common_ids):
                    if j != i:
                        neg = vjepa2_fwd_features[sid_g]["temporal_residual"].reshape(
                            -1
                        )
                        neg_sim = F.cosine_similarity(
                            query.unsqueeze(0), neg.unsqueeze(0)
                        ).item()
                        scores_vj_res.append(neg_sim)
                        labels_vj_res.append(0)

            ap = average_precision_score(labels_vj_res, scores_vj_res)
            auc = roc_auc_score(labels_vj_res, scores_vj_res)
            results["vjepa2_temporal_residual"] = {
                "ap": round(ap, 4),
                "auc": round(auc, 4),
            }
            print(f"    V-JEPA 2 Res: AP={ap:.3f}, AUC={auc:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_order_sensitivity(results: dict, output_path: Path) -> None:
    """Bar chart of 1 - s_rev (order sensitivity) per method."""
    methods = []
    sensitivities = []
    ci_lows = []
    ci_highs = []

    for method, data in results.items():
        if "s_rev_mean" not in data:
            continue
        methods.append(method.replace("_", "\n"))
        sens = 1.0 - data["s_rev_mean"]
        sensitivities.append(sens)
        ci_lows.append(sens - (1.0 - data["s_rev_ci"][1]))
        ci_highs.append((1.0 - data["s_rev_ci"][0]) - sens)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        sensitivities,
        yerr=[ci_lows, ci_highs],
        capsize=4,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Order Sensitivity (1 - s_rev)")
    ax.set_title("EPIC-Kitchens: Temporal Order Sensitivity per Method")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_retrieval(results: dict, output_path: Path) -> None:
    """AP/AUC bar chart per method."""
    methods = []
    aps = []
    aucs = []

    for method, data in results.items():
        methods.append(method.replace("_", "\n"))
        aps.append(data["ap"])
        aucs.append(data["auc"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(
        x - width / 2,
        aps,
        width,
        label="AP",
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        aucs,
        width,
        label="AUC",
        color="coral",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("EPIC-Kitchens: Forward vs Reversed Retrieval")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_s_rev_distributions(
    dinov3_features: dict[str, dict],
    vjepa2_fwd: dict[str, dict] | None,
    vjepa2_rev: dict[str, dict] | None,
    output_path: Path,
) -> None:
    """Per-method s_rev histograms."""
    from video_retrieval.fingerprints import (
        TemporalDerivativeFingerprint,
        TrajectoryFingerprint,
    )

    seq_ids = sorted(dinov3_features.keys())
    td_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    distributions = {}

    # BoF
    s_rev_bof = []
    for sid in seq_ids:
        emb = dinov3_features[sid]["embeddings"]
        fwd = F.normalize(emb.mean(dim=0), dim=0)
        rev = F.normalize(emb.flip(0).mean(dim=0), dim=0)
        s_rev_bof.append(F.cosine_similarity(fwd.unsqueeze(0), rev.unsqueeze(0)).item())
    distributions["Bag-of-Frames"] = s_rev_bof

    # Temporal derivative
    s_rev_td = []
    for sid in seq_ids:
        emb = dinov3_features[sid]["embeddings"]
        d_fwd = td_fp.compute_fingerprint(emb)
        d_rev = td_fp.compute_fingerprint(emb.flip(0))
        s_rev_td.append(float(td_fp.compare(d_fwd, d_rev)))
    distributions["Temporal Derivative"] = s_rev_td

    # Attention trajectory
    s_rev_traj = []
    for sid in seq_ids:
        centroids = dinov3_features[sid]["centroids"]
        t_fwd = traj_fp.compute_fingerprint(centroids)
        t_rev = traj_fp.compute_fingerprint(centroids.flip(0))
        s_rev_traj.append(float(traj_fp.compare(t_fwd, t_rev)))
    distributions["Attention Trajectory"] = s_rev_traj

    # V-JEPA 2
    if vjepa2_fwd and vjepa2_rev:
        common = sorted(set(vjepa2_fwd.keys()) & set(vjepa2_rev.keys()))
        s_rev_vj = []
        for sid in common:
            fwd = vjepa2_fwd[sid]["mean_emb"]
            rev = vjepa2_rev[sid]["mean_emb"]
            s_rev_vj.append(
                F.cosine_similarity(fwd.unsqueeze(0), rev.unsqueeze(0)).item()
            )
        distributions["V-JEPA 2 BoT"] = s_rev_vj

        s_rev_vj_res = []
        for sid in common:
            fwd = vjepa2_fwd[sid]["temporal_residual"].reshape(-1)
            rev = vjepa2_rev[sid]["temporal_residual"].reshape(-1)
            s_rev_vj_res.append(
                F.cosine_similarity(fwd.unsqueeze(0), rev.unsqueeze(0)).item()
            )
        distributions["V-JEPA 2 Residual"] = s_rev_vj_res

    n_plots = len(distributions)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, distributions.items()):
        ax.hist(values, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(values),
            color="red",
            linestyle="--",
            label=f"mean={np.mean(values):.3f}",
        )
        ax.set_xlabel("s_rev")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)

    plt.suptitle("EPIC-Kitchens: s_rev Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_vlm_integrity(vlm_results: dict, vlm_family: str, output_path: Path) -> None:
    """Integrity accuracy + consistency bar chart with CI."""
    integrity = vlm_results["integrity"]

    conditions = []
    accs = []
    for key, val in integrity.items():
        if key.endswith("_acc"):
            cond = key.replace("_acc", "").replace("_", " ").title()
            conditions.append(cond)
            accs.append(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, accs, color="steelblue", edgecolor="black", alpha=0.8)

    # Add consistency bar
    cons = integrity["consistency_pct"]
    cons_ci = integrity["consistency_ci"]
    x_cons = len(conditions)
    ax.bar(
        x_cons,
        cons,
        color="coral",
        edgecolor="black",
        alpha=0.8,
        yerr=[[cons - cons_ci[0]], [cons_ci[1] - cons]],
        capsize=4,
    )
    conditions.append("Consistency\n(all correct)")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Accuracy / Rate")
    ax.set_title(f"EPIC-Kitchens: {vlm_family} Temporal Integrity Probe")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_vlm_prompt_variance(
    vlm_results: dict, vlm_family: str, output_path: Path
) -> None:
    """Bar chart of balanced accuracy per prompt variant."""
    per_prompt = vlm_results["per_prompt"]

    prompts = sorted(per_prompt.keys())
    accs = [per_prompt[p]["balanced_acc"] for p in prompts]
    fwd_accs = [per_prompt[p]["fwd_acc"] for p in prompts]
    rev_accs = [per_prompt[p]["rev_acc"] for p in prompts]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(prompts))
    width = 0.25
    ax.bar(
        x - width,
        fwd_accs,
        width,
        label="Fwd Acc",
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar(
        x, rev_accs, width, label="Rev Acc", color="coral", edgecolor="black", alpha=0.8
    )
    ax.bar(
        x + width,
        accs,
        width,
        label="Balanced",
        color="mediumseagreen",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ").title() for p in prompts], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"EPIC-Kitchens: {vlm_family} Prompt Sensitivity\n"
        f"(mean={vlm_results['mean_balanced_acc']:.3f} "
        f"+/- {vlm_results['std_balanced_acc']:.3f})"
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="EPIC-Kitchens Temporal Order Sensitivity + Multi-VLM Probes"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default="datasets/epic_kitchens",
        help="Path to EPIC-Kitchens data directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-vjepa2", action="store_true", help="Skip V-JEPA 2 extraction"
    )

    # VLM adapter flags
    parser.add_argument(
        "--vlm-family",
        type=str,
        default=None,
        choices=list(VLM_ADAPTERS.keys()),
        help="VLM family to use (qwen3, gemma4, llava-video)",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        help="Override default model path for the VLM family",
    )
    parser.add_argument(
        "--vlm-embeddings",
        action="store_true",
        help="Extract vision + LLM representations",
    )
    parser.add_argument(
        "--vlm-generative",
        action="store_true",
        help="Run generative probes (forward/reverse prompt variants)",
    )
    parser.add_argument(
        "--vlm-integrity-probe",
        action="store_true",
        help="Run INTACT/TAMPERED integrity probe with scramble transforms",
    )
    parser.add_argument(
        "--vlm-layer-ablation",
        action="store_true",
        help="Test last/mid/penultimate layers (with --vlm-embeddings)",
    )
    parser.add_argument(
        "--vlm-num-frames",
        type=int,
        default=16,
        help="Canonical frame count for VLMs (default 16)",
    )
    parser.add_argument(
        "--vlm-scramble-levels",
        type=int,
        nargs="+",
        default=[4, 16],
        help="Scramble chunk counts for integrity probe",
    )
    parser.add_argument(
        "--vlm-text-only-baseline",
        action="store_true",
        help="Run text-only bias measurement",
    )
    parser.add_argument(
        "--vlm-prompts-file",
        type=str,
        default=None,
        help="JSON file with causality_prompts and integrity_prompt overrides",
    )
    parser.add_argument(
        "--vlm-debug-shapes",
        action="store_true",
        help="Print pixel_values and vision tower output shapes for first sequence",
    )
    parser.add_argument(
        "--vlm-vision-token-probe",
        action="store_true",
        help="Extract vision-token-only LLM hidden-state probes "
        "(vision_only_pool, vision_only_seq, bos_token)",
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Disable feature caching"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=200,
        help="Maximum number of sequences to evaluate",
    )
    parser.add_argument(
        "--target-fps", type=float, default=3.0, help="Frame extraction rate for DINOv3"
    )
    args = parser.parse_args()

    # Validate VLM flags
    if (args.vlm_embeddings or args.vlm_generative) and not args.vlm_family:
        parser.error("--vlm-embeddings and --vlm-generative require --vlm-family")
    if args.vlm_integrity_probe and not args.vlm_family:
        parser.error("--vlm-integrity-probe requires --vlm-family")
    if args.vlm_vision_token_probe and not args.vlm_embeddings:
        parser.error("--vlm-vision-token-probe requires --vlm-embeddings")

    # Resolve prompts (file override or defaults)
    if args.vlm_prompts_file:
        causality_prompts, integrity_prompt = load_prompts_file(args.vlm_prompts_file)
        print(f"  Loaded prompts from {args.vlm_prompts_file}")
    else:
        causality_prompts = DEFAULT_CAUSALITY_PROMPTS
        integrity_prompt = DEFAULT_INTEGRITY_PROMPT

    # Resolve VLM model path
    vlm_model_path = None
    if args.vlm_family:
        vlm_model_path = args.vlm_model or VLM_DEFAULT_PATHS.get(args.vlm_family)
        if vlm_model_path is None:
            parser.error(
                f"No default model path for --vlm-family={args.vlm_family}, "
                f"use --vlm-model to specify one"
            )

    epic_dir = Path(args.epic_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = epic_dir / "feature_cache"
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    vlm_family = args.vlm_family or "none"
    print(f"Device: {device}")
    print(f"EPIC-Kitchens dir: {epic_dir}")
    print(
        f"Modes: DINOv3=yes, V-JEPA2={'no' if args.skip_vjepa2 else 'yes'}, "
        f"VLM-family={vlm_family}, "
        f"VLM-emb={'yes' if args.vlm_embeddings else 'no'}, "
        f"VLM-gen={'yes' if args.vlm_generative else 'no'}, "
        f"VLM-integrity={'yes' if args.vlm_integrity_probe else 'no'}, "
        f"VLM-layers={'yes' if args.vlm_layer_ablation else 'no'}, "
        f"VLM-vt-probe={'yes' if args.vlm_vision_token_probe else 'no'}"
    )
    t0 = time.time()

    # ====================================================================
    # Step 1: Load sequences
    # ====================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading sequences")
    print("=" * 60)
    sequences = load_sequences(epic_dir, max_sequences=args.max_sequences)

    # ====================================================================
    # Step 2: DINOv3 feature extraction
    # ====================================================================
    print("\n" + "=" * 60)
    print("Step 2: DINOv3 feature extraction")
    print("=" * 60)
    dinov3_cache = (
        cache_dir / f"dinov3_fps{args.target_fps}.pt" if not args.no_cache else None
    )
    dinov3_features = extract_dinov3_features(
        sequences,
        device,
        target_fps=args.target_fps,
        cache_path=dinov3_cache,
    )

    # ====================================================================
    # Step 3: V-JEPA 2 feature extraction (forward + reverse)
    # ====================================================================
    vjepa2_fwd = None
    vjepa2_rev = None
    if not args.skip_vjepa2:
        print("\n" + "=" * 60)
        print("Step 3: V-JEPA 2 feature extraction")
        print("=" * 60)

        vjepa2_fwd_cache = cache_dir / "vjepa2_fwd.pt" if not args.no_cache else None
        vjepa2_fwd = extract_vjepa2_features(
            sequences,
            device,
            reverse=False,
            cache_path=vjepa2_fwd_cache,
        )

        vjepa2_rev_cache = cache_dir / "vjepa2_rev.pt" if not args.no_cache else None
        vjepa2_rev = extract_vjepa2_features(
            sequences,
            device,
            reverse=True,
            cache_path=vjepa2_rev_cache,
        )

    # ====================================================================
    # Step 4: VLM embedding extraction (if --vlm-embeddings)
    # ====================================================================
    vlm_fwd = None
    vlm_rev = None
    if args.vlm_embeddings and args.vlm_family:
        assert vlm_model_path is not None
        print("\n" + "=" * 60)
        print(f"Step 4: {args.vlm_family} embedding extraction")
        print("=" * 60)

        adapter = VLM_ADAPTERS[args.vlm_family]()

        suffix = f"_{args.vlm_family}"
        if args.vlm_vision_token_probe:
            suffix += "_vtp"
        if args.vlm_layer_ablation:
            suffix += "_layers"
        vlm_fwd_cache = cache_dir / f"vlm{suffix}_fwd.pt" if not args.no_cache else None
        vlm_fwd = extract_vlm_features(
            sequences,
            adapter,
            vlm_model_path,
            device,
            n_frames=args.vlm_num_frames,
            reverse=False,
            layer_ablation=args.vlm_layer_ablation,
            vision_token_probe=args.vlm_vision_token_probe,
            cache_path=vlm_fwd_cache,
            debug_shapes=args.vlm_debug_shapes,
        )

        vlm_rev_cache = cache_dir / f"vlm{suffix}_rev.pt" if not args.no_cache else None
        vlm_rev = extract_vlm_features(
            sequences,
            adapter,
            vlm_model_path,
            device,
            n_frames=args.vlm_num_frames,
            reverse=True,
            layer_ablation=args.vlm_layer_ablation,
            vision_token_probe=args.vlm_vision_token_probe,
            cache_path=vlm_rev_cache,
            debug_shapes=args.vlm_debug_shapes,
        )

    # ====================================================================
    # Step 5: Order sensitivity evaluation
    # ====================================================================
    print("\n" + "=" * 60)
    print("Step 5: Order sensitivity (s_rev + balanced accuracy)")
    print("=" * 60)
    order_results = evaluate_order_sensitivity(
        dinov3_features,
        vjepa2_fwd,
        vjepa2_rev,
        vlm_fwd,
        vlm_rev,
        vlm_family=args.vlm_family,
    )

    for method, data in order_results.items():
        s_rev = data.get("s_rev_mean", "N/A")
        ci = data.get("s_rev_ci", ["N/A", "N/A"])
        acc = data.get("balanced_acc", "N/A")
        print(f"  {method}: s_rev={s_rev} [{ci[0]}, {ci[1]}], acc={acc}")

    # ====================================================================
    # Step 6: Retrieval evaluation
    # ====================================================================
    print("\n" + "=" * 60)
    print("Step 6: Retrieval evaluation (AP/AUC)")
    print("=" * 60)
    retrieval_results = evaluate_retrieval_task(
        dinov3_features,
        sequences,
        vjepa2_fwd,
        vjepa2_rev,
    )

    # ====================================================================
    # Step 7: VLM generative probes (if --vlm-generative)
    # ====================================================================
    vlm_generative_results = None
    if args.vlm_generative and args.vlm_family:
        assert vlm_model_path is not None
        print("\n" + "=" * 60)
        print(f"Step 7: {args.vlm_family} generative probes")
        print("=" * 60)

        adapter = VLM_ADAPTERS[args.vlm_family]()
        vlm_generative_results = evaluate_vlm_generative(
            sequences,
            adapter,
            vlm_model_path,
            device,
            n_frames=args.vlm_num_frames,
            causality_prompts=causality_prompts,
            integrity_prompt=integrity_prompt,
            run_integrity=args.vlm_integrity_probe,
            scramble_levels=args.vlm_scramble_levels,
            run_text_only=args.vlm_text_only_baseline,
        )

    # ====================================================================
    # Step 8: Save results + generate figures
    # ====================================================================
    print("\n" + "=" * 60)
    print("Step 8: Saving results and generating figures")
    print("=" * 60)

    elapsed = time.time() - t0

    # Build results dict
    all_results = {
        "metadata": {
            "vlm_family": args.vlm_family,
            "vlm_model": vlm_model_path,
            "vlm_num_frames": args.vlm_num_frames,
            "n_sequences": len(sequences),
            "n_videos": len(set(s["video_id"] for s in sequences)),
            "duration_range": [6, 15],
            "fps": args.target_fps,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "device": str(device),
            "elapsed_sec": round(elapsed, 1),
            "skip_vjepa2": args.skip_vjepa2,
            "vision_token_probe": args.vlm_vision_token_probe,
            "causality_prompts": causality_prompts,
            "integrity_prompt": integrity_prompt,
            "prompts_file": args.vlm_prompts_file,
        },
        "order_sensitivity": order_results,
        "retrieval": retrieval_results,
    }

    # VLM embedding results
    if vlm_fwd and vlm_rev and args.vlm_family:
        vlm_emb_results = {}
        family = args.vlm_family
        for key, data in order_results.items():
            if key.startswith(f"{family}_"):
                short_key = key[len(family) + 1 :]
                vlm_emb_results[short_key] = data
        all_results["vlm_embedding"] = vlm_emb_results

        # Save canonical timestamps for auditability
        sample_seq = next(iter(vlm_fwd.values()), None)
        if sample_seq and "timestamps" in sample_seq:
            all_results["metadata"]["canonical_timestamps_sample"] = sample_seq[
                "timestamps"
            ]

    # VLM generative results
    if vlm_generative_results:
        all_results["vlm_generative"] = vlm_generative_results

    # Save JSON (per-family filename)
    if args.vlm_family:
        results_path = epic_dir / f"temporal_order_results_{args.vlm_family}.json"
    else:
        results_path = epic_dir / "temporal_order_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Generate figures
    plot_order_sensitivity(
        order_results,
        figures_dir / "epic_temporal_order_sensitivity.png",
    )
    if retrieval_results:
        plot_retrieval(
            retrieval_results,
            figures_dir / "epic_temporal_order_retrieval.png",
        )
    plot_s_rev_distributions(
        dinov3_features,
        vjepa2_fwd,
        vjepa2_rev,
        figures_dir / "epic_temporal_order_s_rev_distributions.png",
    )

    if vlm_generative_results and args.vlm_family:
        plot_vlm_integrity(
            vlm_generative_results,
            args.vlm_family,
            figures_dir / f"epic_{args.vlm_family}_temporal_integrity.png",
        )
        plot_vlm_prompt_variance(
            vlm_generative_results,
            args.vlm_family,
            figures_dir / f"epic_{args.vlm_family}_prompt_variance.png",
        )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Results: {results_path}")
    print(f"  Figures: {figures_dir}/epic_*")


if __name__ == "__main__":
    main()

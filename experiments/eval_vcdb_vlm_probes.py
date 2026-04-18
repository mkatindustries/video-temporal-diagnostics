#!/usr/bin/env python3
"""VCDB VLM Generative Probes — Temporal Direction & Integrity on Diverse Internet Video.

Runs VLM generative probes on VCDB videos to test whether VLMs can detect
temporal direction and integrity on diverse internet video (not just
EPIC-Kitchens cooking). Directly addresses the "EPIC-only" reviewer critique.

For each sampled VCDB clip the script creates forward, reversed, and
scrambled (K=4, K=16) versions, then runs:

1. **Direction classification**: 3 semantically equivalent prompts asking
   "Is this video playing forward or in reverse?" — reports balanced
   accuracy per prompt and mean.
2. **Integrity probe**: forward / reversed / scrambled clips asked
   "Is this video's temporal structure INTACT or TAMPERED?" — reports
   fraction judged TAMPERED per condition.
3. **Text-only baselines**: same prompts without video to detect prior bias.
4. **Degeneracy tracking**: flags models that give constant answers.

Supports Gemma-4-31B, LLaVA-Video-7B, and Qwen3-VL-8B (same adapter
pattern as eval_epic_temporal_order.py). Uses deterministic decoding
(do_sample=False, temperature=0).

Usage:
    python experiments/eval_vcdb_vlm_probes.py \\
        --vcdb-dir datasets/vcdb/core_dataset \\
        --vlm-family gemma4 --max-clips 500

    python experiments/eval_vcdb_vlm_probes.py \\
        --vcdb-dir datasets/vcdb/core_dataset \\
        --vlm-family llava-video --max-clips 200 --skip-integrity

    python experiments/eval_vcdb_vlm_probes.py \\
        --vcdb-dir datasets/vcdb/core_dataset \\
        --vlm-family qwen3 --max-clips 500
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLM_DEFAULT_PATHS = {
    "gemma4": "google/gemma-4-31B-it",
    "llava-video": "llava-hf/LLaVA-Video-7B-Qwen2-hf",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

# 3 direction prompts — same as EPIC protocol
DIRECTION_PROMPTS = [
    "Is this video playing forward or in reverse? "
    "Answer with only FORWARD or REVERSE.",
    "Watch this video carefully. Is the temporal order normal (forward) "
    "or reversed? Reply FORWARD or REVERSE only.",
    "Determine if this video is playing in its original direction or has "
    "been reversed. Answer: FORWARD or REVERSE",
]

INTEGRITY_PROMPT = (
    "Is this video's temporal structure intact or has it been tampered with? "
    "Answer INTACT or TAMPERED only."
)


# ---------------------------------------------------------------------------
# VCDB loading (from eval_vcdb_reversal.py)
# ---------------------------------------------------------------------------


def discover_videos(vid_base_dir: str) -> list[str]:
    """Discover all video files under the VCDB core_dataset directory."""
    videos = []
    for cat in sorted(os.listdir(vid_base_dir)):
        cat_path = os.path.join(vid_base_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))
    return videos


# ---------------------------------------------------------------------------
# Frame extraction
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
# Frame scrambling (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def scramble_frames(frames: list, n_chunks: int, seed: int = 42) -> list:
    """Split frame list into n_chunks, shuffle with deterministic seed, reassemble."""
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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_forward_reverse(response: str) -> str | None:
    """Parse FORWARD or REVERSE from model response."""
    upper = response.upper()
    if "FORWARD" in upper:
        return "FORWARD"
    elif "REVERSE" in upper:
        return "REVERSE"
    return None


def parse_intact_tampered(response: str) -> str | None:
    """Parse INTACT or TAMPERED from model response."""
    upper = response.upper()
    if "INTACT" in upper:
        return "INTACT"
    elif "TAMPERED" in upper:
        return "TAMPERED"
    return None


# ---------------------------------------------------------------------------
# VLM Adapter Protocol + Implementations
# (Copied from eval_epic_temporal_order.py — self-contained)
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
# Direction classification probe
# ---------------------------------------------------------------------------


def run_direction_probe(
    clips: list[dict],
    adapter: VLMAdapter,
    model: nn.Module,
    processor: object,
    device: torch.device,
    n_frames: int,
    prompts: list[str],
) -> dict:
    """Run forward/reverse direction classification with multiple prompts.

    Returns per-prompt and mean balanced accuracy, plus per-clip results.
    """
    n_prompts = len(prompts)
    per_prompt_results = {}
    all_balanced_accs = []
    per_clip_details = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  --- Direction Prompt {prompt_idx} ---")
        fwd_correct = 0
        rev_correct = 0
        fwd_total = 0
        rev_total = 0
        all_fwd_preds = []
        all_rev_preds = []

        for clip in tqdm(clips, desc=f"Direction prompt_{prompt_idx}"):
            try:
                pil_frames = clip["frames"]
                if len(pil_frames) < 3:
                    continue

                # Forward
                inputs = adapter.prepare_inputs(
                    processor,
                    pil_frames,
                    prompt,
                    device,
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
                    per_clip_details.append(
                        {
                            "clip_id": clip["clip_id"],
                            "fwd_pred": pred_fwd,
                            "rev_pred": pred_rev,
                            "fwd_raw": resp_fwd[:200],
                            "rev_raw": resp_rev[:200],
                        }
                    )

            except Exception as e:
                if fwd_total + rev_total < 3:
                    print(f"    [error] {clip['clip_id']}: {e}")
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
        "per_clip_details": per_clip_details,
    }


# ---------------------------------------------------------------------------
# Integrity probe
# ---------------------------------------------------------------------------


def run_integrity_probe(
    clips: list[dict],
    adapter: VLMAdapter,
    model: nn.Module,
    processor: object,
    device: torch.device,
    n_frames: int,
    prompt: str,
    scramble_levels: list[int] | None = None,
) -> dict:
    """Run integrity probe: forward/reversed/scrambled -> INTACT or TAMPERED.

    Reports fraction judged TAMPERED per condition.
    """
    if scramble_levels is None:
        scramble_levels = [4, 16]

    conditions = ["forward", "reversed"] + [f"scrambled_k{k}" for k in scramble_levels]
    results = {
        c: {"tampered": 0, "intact": 0, "unparseable": 0, "total": 0}
        for c in conditions
    }

    per_clip_details = []

    for clip in tqdm(clips, desc="Integrity probe"):
        try:
            pil_frames = clip["frames"]
            if len(pil_frames) < 3:
                continue

            clip_result = {"clip_id": clip["clip_id"]}

            # Forward (expect INTACT)
            inputs = adapter.prepare_inputs(processor, pil_frames, prompt, device)
            resp = adapter.generate(model, inputs, processor)
            pred = parse_intact_tampered(resp)
            results["forward"]["total"] += 1
            if pred == "TAMPERED":
                results["forward"]["tampered"] += 1
            elif pred == "INTACT":
                results["forward"]["intact"] += 1
            else:
                results["forward"]["unparseable"] += 1
            clip_result["forward"] = pred

            # Reversed (expect TAMPERED)
            rev_frames = pil_frames[::-1]
            inputs = adapter.prepare_inputs(processor, rev_frames, prompt, device)
            resp = adapter.generate(model, inputs, processor)
            pred = parse_intact_tampered(resp)
            results["reversed"]["total"] += 1
            if pred == "TAMPERED":
                results["reversed"]["tampered"] += 1
            elif pred == "INTACT":
                results["reversed"]["intact"] += 1
            else:
                results["reversed"]["unparseable"] += 1
            clip_result["reversed"] = pred

            # Scrambled (expect TAMPERED)
            for k in scramble_levels:
                key = f"scrambled_k{k}"
                scr_frames = scramble_frames(pil_frames, k, seed=42)
                inputs = adapter.prepare_inputs(processor, scr_frames, prompt, device)
                resp = adapter.generate(model, inputs, processor)
                pred = parse_intact_tampered(resp)
                results[key]["total"] += 1
                if pred == "TAMPERED":
                    results[key]["tampered"] += 1
                elif pred == "INTACT":
                    results[key]["intact"] += 1
                else:
                    results[key]["unparseable"] += 1
                clip_result[key] = pred

            per_clip_details.append(clip_result)

        except Exception as e:
            continue

    # Compute summary
    summary = {}
    for condition in conditions:
        total = results[condition]["total"]
        tampered = results[condition]["tampered"]
        intact = results[condition]["intact"]
        frac_tampered = tampered / max(total, 1)
        frac_intact = intact / max(total, 1)
        summary[condition] = {
            "frac_tampered": round(frac_tampered, 4),
            "frac_intact": round(frac_intact, 4),
            "total": total,
            "tampered": tampered,
            "intact": intact,
            "unparseable": results[condition]["unparseable"],
        }
        print(
            f"  {condition}: TAMPERED={frac_tampered:.3f} "
            f"INTACT={frac_intact:.3f} ({total} clips)"
        )

    # Degeneracy check
    all_preds = []
    for detail in per_clip_details:
        for c in conditions:
            if c in detail and detail[c] is not None:
                all_preds.append(detail[c])
    unique_preds = set(all_preds)
    is_degenerate = len(unique_preds) <= 1 and len(all_preds) > 0
    if is_degenerate:
        print(f"  [WARNING] Degenerate output: always {list(unique_preds)[0]}")

    return {
        "per_condition": summary,
        "is_degenerate": is_degenerate,
        "per_clip_details": per_clip_details,
    }


# ---------------------------------------------------------------------------
# Text-only baseline
# ---------------------------------------------------------------------------


def run_text_only_baseline(
    adapter: VLMAdapter,
    model: nn.Module,
    processor: object,
    device: torch.device,
    direction_prompts: list[str],
    integrity_prompt: str,
    n_trials: int = 50,
) -> dict:
    """Run text-only baseline (no video) to measure prior bias.

    For direction prompts: reports FORWARD bias rate.
    For integrity prompt: reports TAMPERED bias rate.
    """
    print("\n  === Text-Only Baseline ===")

    direction_baseline = {}
    for prompt_idx, prompt in enumerate(direction_prompts):
        preds = []
        for _ in tqdm(range(n_trials), desc=f"Text-only direction prompt_{prompt_idx}"):
            try:
                inputs = adapter.prepare_text_only_inputs(processor, prompt, device)
                resp = adapter.generate(model, inputs, processor)
                pred = parse_forward_reverse(resp)
                if pred is not None:
                    preds.append(pred)
            except Exception:
                continue

        n_fwd = sum(1 for p in preds if p == "FORWARD")
        bias = n_fwd / max(len(preds), 1)
        direction_baseline[f"prompt_{prompt_idx}"] = {
            "bias_rate_forward": round(bias, 4),
            "n_trials": len(preds),
            "n_forward": n_fwd,
        }
        print(
            f"  Direction prompt {prompt_idx} text-only bias: "
            f"{bias:.3f} ({n_fwd}/{len(preds)})"
        )

    # Integrity baseline
    integrity_preds = []
    for _ in tqdm(range(n_trials), desc="Text-only integrity"):
        try:
            inputs = adapter.prepare_text_only_inputs(
                processor,
                integrity_prompt,
                device,
            )
            resp = adapter.generate(model, inputs, processor)
            pred = parse_intact_tampered(resp)
            if pred is not None:
                integrity_preds.append(pred)
        except Exception:
            continue

    n_tampered = sum(1 for p in integrity_preds if p == "TAMPERED")
    integrity_bias = n_tampered / max(len(integrity_preds), 1)
    print(
        f"  Integrity text-only bias (TAMPERED): "
        f"{integrity_bias:.3f} ({n_tampered}/{len(integrity_preds)})"
    )

    return {
        "direction": {
            "per_prompt": direction_baseline,
            "mean_bias_forward": round(
                float(
                    np.mean(
                        [v["bias_rate_forward"] for v in direction_baseline.values()]
                    )
                ),
                4,
            ),
        },
        "integrity": {
            "bias_rate_tampered": round(integrity_bias, 4),
            "n_trials": len(integrity_preds),
            "n_tampered": n_tampered,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VCDB VLM Generative Probes: Temporal Direction & Integrity"
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
        "--max-clips",
        type=int,
        default=500,
        help="Maximum number of VCDB clips to evaluate",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Number of frames to sample per clip",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Skip integrity probe (saves compute by not generating scrambled clips)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    device = torch.device(args.device)

    # Resolve model path
    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    family = args.vlm_family

    print("=" * 70)
    print("VCDB VLM GENERATIVE PROBES")
    print("=" * 70)
    print(f"  VLM family: {family}")
    print(f"  Model path: {model_path}")
    print(f"  VCDB dir:   {vcdb_dir}")
    print(f"  Max clips:  {args.max_clips}")
    print(f"  N frames:   {args.n_frames}")
    print(f"  Skip integrity: {args.skip_integrity}")
    print(f"  Device:     {device}")

    # ------------------------------------------------------------------
    # Step 1: Discover and sample VCDB videos
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering VCDB videos...")
    all_videos = discover_videos(str(vcdb_dir))
    print(f"  Total videos found: {len(all_videos)}")

    if len(all_videos) == 0:
        raise FileNotFoundError(
            f"No videos found in {vcdb_dir}. "
            "Ensure the VCDB core_dataset directory has category subdirectories "
            "containing video files."
        )

    # Deterministic subsample
    rng = np.random.RandomState(42)
    if len(all_videos) > args.max_clips:
        indices = rng.choice(len(all_videos), size=args.max_clips, replace=False)
        indices.sort()
        selected_videos = [all_videos[i] for i in indices]
    else:
        selected_videos = all_videos
    print(f"  Selected clips: {len(selected_videos)}")

    # ------------------------------------------------------------------
    # Step 2: Extract frames for all clips
    # ------------------------------------------------------------------
    print("\nStep 2: Extracting frames...")
    clips = []
    failed = 0

    for vp in tqdm(selected_videos, desc="Extracting frames"):
        video_path = os.path.join(str(vcdb_dir), vp)
        try:
            pil_frames, timestamps = sample_frames_from_video(
                video_path,
                n_frames=args.n_frames,
                max_resolution=518,
            )
            if len(pil_frames) < 3:
                failed += 1
                continue
            clips.append(
                {
                    "clip_id": vp,
                    "video_path": video_path,
                    "frames": pil_frames,
                    "timestamps": timestamps,
                }
            )
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [warn] Failed to extract {vp}: {e}")
            continue

    print(f"  Clips with frames: {len(clips)} ({failed} failed)")
    if len(clips) == 0:
        raise RuntimeError("No clips could be loaded. Check video paths and codecs.")

    # ------------------------------------------------------------------
    # Step 3: Load VLM
    # ------------------------------------------------------------------
    print(f"\nStep 3: Loading VLM ({family})...")
    adapter = VLM_ADAPTERS[family]()
    model, processor = adapter.load(model_path, device)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 4: Direction classification probe
    # ------------------------------------------------------------------
    print("\nStep 4: Direction classification probe...")
    direction_results = run_direction_probe(
        clips=clips,
        adapter=adapter,
        model=model,
        processor=processor,
        device=device,
        n_frames=args.n_frames,
        prompts=DIRECTION_PROMPTS,
    )

    # ------------------------------------------------------------------
    # Step 5: Integrity probe (unless skipped)
    # ------------------------------------------------------------------
    integrity_results = None
    if not args.skip_integrity:
        print("\nStep 5: Integrity probe...")
        integrity_results = run_integrity_probe(
            clips=clips,
            adapter=adapter,
            model=model,
            processor=processor,
            device=device,
            n_frames=args.n_frames,
            prompt=INTEGRITY_PROMPT,
            scramble_levels=[4, 16],
        )
    else:
        print("\nStep 5: Skipped (--skip-integrity)")

    # ------------------------------------------------------------------
    # Step 6: Text-only baseline
    # ------------------------------------------------------------------
    print("\nStep 6: Text-only baseline...")
    text_only_results = run_text_only_baseline(
        adapter=adapter,
        model=model,
        processor=processor,
        device=device,
        direction_prompts=DIRECTION_PROMPTS,
        integrity_prompt=INTEGRITY_PROMPT,
        n_trials=min(50, len(clips)),
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
    print("\nStep 7: Saving results...")

    # Degeneracy summary
    direction_degenerate = any(
        v.get("is_degenerate", False) for v in direction_results["per_prompt"].values()
    )

    output = {
        "metadata": {
            "vlm_family": family,
            "model_path": model_path,
            "vcdb_dir": str(vcdb_dir),
            "n_clips": len(clips),
            "n_frames": args.n_frames,
            "direction_prompts": DIRECTION_PROMPTS,
            "integrity_prompt": INTEGRITY_PROMPT,
            "skip_integrity": args.skip_integrity,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(t_elapsed, 1),
        },
        "direction": {
            "per_prompt": direction_results["per_prompt"],
            "mean_balanced_acc": direction_results["mean_balanced_acc"],
            "std_balanced_acc": direction_results["std_balanced_acc"],
            "is_degenerate": direction_degenerate,
        },
        "text_only_baseline": text_only_results,
    }

    if integrity_results is not None:
        output["integrity"] = {
            "per_condition": integrity_results["per_condition"],
            "is_degenerate": integrity_results["is_degenerate"],
        }

    # Per-clip details (direction + integrity)
    output["per_clip_direction"] = direction_results.get("per_clip_details", [])
    if integrity_results is not None:
        output["per_clip_integrity"] = integrity_results.get("per_clip_details", [])

    # Save
    results_dir = project_root / "datasets" / "vcdb"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"vlm_probes_{family}_results.json"

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
    print(f"  Clips evaluated: {len(clips)}")
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

    if integrity_results is not None:
        print(f"\n  Integrity Probe:")
        for cond, cdata in integrity_results["per_condition"].items():
            print(
                f"    {cond}: frac_TAMPERED={cdata['frac_tampered']:.3f} "
                f"frac_INTACT={cdata['frac_intact']:.3f}"
            )

    print(f"\n  Text-Only Baseline (Direction):")
    print(
        f"    Mean FORWARD bias: "
        f"{text_only_results['direction']['mean_bias_forward']:.3f}"
    )
    print(f"  Text-Only Baseline (Integrity):")
    print(
        f"    TAMPERED bias: "
        f"{text_only_results['integrity']['bias_rate_tampered']:.3f}"
    )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

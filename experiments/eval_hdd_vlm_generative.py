#!/usr/bin/env python3
"""VLM Generative Probes on Honda HDD — Forward/Reverse Direction Classification.

Runs VLM generative probes (forward/reverse direction classification) on Honda
HDD maneuver segments. Fills 3 cells in Table 8 (VLM gen x HDD for Gemma-4,
LLaVA-Video, Qwen3) with the dagger footnote.

Protocol: Same mixed-cluster selection as eval_hdd_vlm_bridge.py (128 sessions,
DBSCAN eps=0.0003, min_samples=3, 50 mixed-direction clusters). For each
maneuver segment in a mixed cluster, sample 16 frames with +/-3s context,
then run 3 direction prompts (forward vs reversed frame ordering). Reports
mean balanced accuracy across prompts.

Hypothesis: VLMs cannot detect frame reversal in dashcam driving video, so
balanced accuracy should be near chance (~0.50), consistent with EPIC and VCDB
generative probe results.

Usage:
    python experiments/eval_hdd_vlm_generative.py \\
        --hdd-dir datasets/hdd --vlm-family gemma4

    python experiments/eval_hdd_vlm_generative.py \\
        --hdd-dir datasets/hdd --vlm-family llava-video

    python experiments/eval_hdd_vlm_generative.py \\
        --hdd-dir datasets/hdd --vlm-family qwen3

    # Smoke test (2 clusters)
    python experiments/eval_hdd_vlm_generative.py \\
        --hdd-dir datasets/hdd --vlm-family gemma4 --max-clusters 2
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

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

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

DIRECTION_PROMPTS = [
    "Is this video playing forward or in reverse? "
    "Answer with only FORWARD or REVERSE.",
    "Watch this video carefully. Is the temporal order normal (forward) "
    "or reversed? Reply FORWARD or REVERSE only.",
    "Determine if this video is playing in its original direction or has "
    "been reversed. Answer: FORWARD or REVERSE",
]

MANEUVER_NAMES = {
    1: "intersection_passing",
    2: "left_turn",
    3: "right_turn",
}


# ---------------------------------------------------------------------------
# HDD Data Loading (from eval_hdd_vlm_bridge.py / eval_hdd_intersections.py)
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
    from sklearn.cluster import DBSCAN

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
# Frame Sampling (from eval_hdd_vlm_bridge.py)
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
    and uses enable_thinking=False to disable the thinking chain.
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
# Direction Classification Probe (adapted from eval_vcdb_vlm_probes.py)
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

    For each clip, presents frames in forward order (should answer FORWARD)
    and reversed order (should answer REVERSE). Reports per-prompt and mean
    balanced accuracy, plus per-clip results.
    """
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
                    fps=clip.get("fps_eff", 1.0),
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
                    fps=clip.get("fps_eff", 1.0),
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
                            "label": clip.get("label"),
                            "label_name": clip.get("label_name"),
                            "session_id": clip.get("session_id"),
                            "cluster_id": clip.get("cluster_id"),
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VLM Generative Probes on Honda HDD: Forward/Reverse Direction"
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
        help="Maximum number of mixed clusters to evaluate",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Number of frames to sample per segment",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Context seconds to add before/after each maneuver segment",
    )
    args = parser.parse_args()

    hdd_dir = Path(args.hdd_dir)
    device = torch.device(args.device)

    # Resolve model path
    model_path = args.vlm_model or VLM_DEFAULT_PATHS[args.vlm_family]
    family = args.vlm_family

    print("=" * 70)
    print("HDD VLM GENERATIVE PROBES — FORWARD/REVERSE DIRECTION")
    print("=" * 70)
    print(f"  VLM family:   {family}")
    print(f"  Model path:   {model_path}")
    print(f"  HDD dir:      {hdd_dir}")
    print(f"  Max clusters: {args.max_clusters}")
    print(f"  N frames:     {args.n_frames}")
    print(f"  Context sec:  {args.context_sec}")
    print(f"  Device:       {device}")

    # ------------------------------------------------------------------
    # Step 1: Discover HDD sessions
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering HDD sessions...")
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} sessions with labels + GPS + video")

    if len(sessions) == 0:
        raise FileNotFoundError(
            f"No valid HDD sessions found in {hdd_dir}. "
            "Ensure labels/target/*.npy, release_2019_07_08/*/camera/center/*.mp4, "
            "and release_2019_07_08/*/general/csv/rtk_pos.csv exist."
        )

    # ------------------------------------------------------------------
    # Step 2: Extract maneuver segments with GPS midpoints
    # ------------------------------------------------------------------
    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []

    for session_id, sinfo in tqdm(sessions.items(), desc="Loading sessions"):
        labels = np.load(sinfo["label_path"])
        gps_timestamps, gps_lats, gps_lngs = load_gps(sinfo["gps_path"])
        segs = extract_maneuver_segments(
            session_id=session_id,
            labels=labels,
            gps_timestamps=gps_timestamps,
            gps_lats=gps_lats,
            gps_lngs=gps_lngs,
            video_path=sinfo["video_path"],
            video_start_unix=sinfo["video_start_unix"],
        )
        all_segments.extend(segs)

    n_left = sum(1 for s in all_segments if s.label == 2)
    n_right = sum(1 for s in all_segments if s.label == 3)
    n_passing = sum(1 for s in all_segments if s.label == 1)
    print(
        f"  Total segments: {len(all_segments)} "
        f"(left={n_left}, right={n_right}, passing={n_passing})"
    )

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections (DBSCAN)
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections (DBSCAN eps=0.0003, min_samples=3)...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters (both left + right turns)
    # ------------------------------------------------------------------
    print(f"\nStep 4: Filtering for mixed clusters (max {args.max_clusters})...")
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
    n_mixed_segs = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters: {len(mixed)} with {n_mixed_segs} segments")

    if len(mixed) == 0:
        raise RuntimeError(
            "No mixed clusters found. Need clusters with both left and right turns."
        )

    # ------------------------------------------------------------------
    # Step 5: Sample frames for segments in mixed clusters
    # ------------------------------------------------------------------
    print("\nStep 5: Sampling frames for mixed-cluster segments...")
    clips = []
    failed = 0

    for cid, segs in tqdm(mixed.items(), desc="Clusters"):
        for seg in segs:
            start_sec = seg.start_frame / 3.0 - args.context_sec
            end_sec = seg.end_frame / 3.0 + args.context_sec
            start_sec = max(0.0, start_sec)

            try:
                pil_frames, timestamps = sample_canonical_frames(
                    video_path=seg.video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    n_frames=args.n_frames,
                    max_resolution=518,
                )
                if len(pil_frames) < 3:
                    failed += 1
                    continue

                fps_eff = compute_fps_eff(timestamps)

                clips.append(
                    {
                        "clip_id": f"{seg.session_id}_f{seg.start_frame}-{seg.end_frame}",
                        "session_id": seg.session_id,
                        "label": seg.label,
                        "label_name": MANEUVER_NAMES.get(seg.label, "unknown"),
                        "cluster_id": cid,
                        "frames": pil_frames,
                        "timestamps": timestamps,
                        "fps_eff": fps_eff,
                    }
                )
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(
                        f"  [warn] Failed segment {seg.session_id} "
                        f"f{seg.start_frame}-{seg.end_frame}: {e}"
                    )
                continue

    print(f"  Clips with frames: {len(clips)} ({failed} failed)")
    if len(clips) == 0:
        raise RuntimeError("No clips could be loaded. Check video paths and codecs.")

    # ------------------------------------------------------------------
    # Step 6: Load VLM
    # ------------------------------------------------------------------
    print(f"\nStep 6: Loading VLM ({family})...")
    adapter = VLM_ADAPTERS[family]()
    model, processor = adapter.load(model_path, device)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 7: Run direction probe (forward/reverse with 3 prompts)
    # ------------------------------------------------------------------
    print("\nStep 7: Direction classification probe (3 prompts)...")
    direction_results = run_direction_probe(
        clips=clips,
        adapter=adapter,
        model=model,
        processor=processor,
        device=device,
        n_frames=args.n_frames,
        prompts=DIRECTION_PROMPTS,
    )

    t_elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Step 8: Cleanup model
    # ------------------------------------------------------------------
    del model, processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 9: Assemble and save results
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
            "hdd_dir": str(hdd_dir),
            "n_sessions": len(sessions),
            "n_segments_total": len(all_segments),
            "n_mixed_clusters": len(mixed),
            "n_clips_evaluated": len(clips),
            "n_frames": args.n_frames,
            "context_sec": args.context_sec,
            "max_clusters": args.max_clusters,
            "dbscan_eps": 0.0003,
            "dbscan_min_samples": 3,
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
        "per_clip_details": direction_results.get("per_clip_details", []),
    }

    # Save
    results_path = hdd_dir / f"vlm_generative_{family}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"  Sessions: {len(sessions)}")
    print(f"  Mixed clusters: {len(mixed)}")
    print(f"  Clips evaluated: {len(clips)}")
    print(f"  Elapsed: {t_elapsed:.0f}s")

    print(f"\n  Direction Classification (Table 8, VLM gen x HDD):")
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
        print("\n  [WARNING] Degenerate output detected in one or more prompts!")
        print("  Model may be giving constant answers regardless of input ordering.")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

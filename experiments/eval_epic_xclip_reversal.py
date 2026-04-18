#!/usr/bin/env python3
"""X-CLIP Reversal Sensitivity on EPIC-Kitchens.

Evaluates whether X-CLIP's cross-frame temporal attention produces
order-sensitive clip embeddings by computing s_rev on 500 EPIC-Kitchens
sequences (same protocol as eval_epic_temporal_order.py).

X-CLIP encodes 16 uniformly-sampled frames into a single 768-dim clip
embedding using cross-frame attention. If the temporal attention is
effective, s_rev should be < 1.0 (unlike per-frame BoF which is exactly 1.0).

Usage:
    python experiments/eval_epic_xclip_reversal.py \
        --epic-dir ./data/epic_kitchens \
        --xclip-model microsoft/xclip-large-patch14-16-frames

    # Quick smoke test
    python experiments/eval_epic_xclip_reversal.py \
        --epic-dir datasets/epic_kitchens --max-sequences 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, XCLIPModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    """Resample a list of frames to a target count via uniform index selection."""
    if target_count <= 0:
        return []
    n = len(frames)
    indices = np.linspace(0, n - 1, target_count).astype(int)
    return [frames[i] for i in indices]


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

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
# Video clip extraction (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def load_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> list[np.ndarray]:
    """Extract frames from a video clip using time-based seeking."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    time_base = float(stream.time_base or 0)

    seek_sec = max(0.0, start_sec - 1.0)
    seek_pts = int(seek_sec / time_base)
    container.seek(seek_pts, stream=stream)

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


# ---------------------------------------------------------------------------
# X-CLIP Encoder (from test_video_models.py)
# ---------------------------------------------------------------------------


class XCLIPEncoder:
    """X-CLIP video encoder using cross-frame temporal attention.

    Encodes a clip of 16 frames into a single 768-dim embedding.
    """

    def __init__(
        self,
        model_name: str = "microsoft/xclip-large-patch14-16-frames",
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name

        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name,
            crop_size={"height": 336, "width": 336},
        )
        self.model = XCLIPModel.from_pretrained(model_name).eval().to(self.device)

        vision_config = getattr(self.model.config, "vision_config")
        self.num_frames = vision_config.num_frames
        self.embedding_dim = vision_config.hidden_size

    @torch.no_grad()
    def encode_clip(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Encode a list of frames as a single video clip.

        Uniformly samples to self.num_frames if needed.

        Returns:
            Clip embedding (1, embedding_dim), L2-normalized.
        """
        sampled = resample_frames(frames, self.num_frames)
        inputs = self.processor(images=sampled, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_video_features(pixel_values=inputs["pixel_values"])
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs
        return F.normalize(embeddings, p=2, dim=1)


# ---------------------------------------------------------------------------
# Sequence loading (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def load_sequences(epic_dir: Path, max_sequences: int = 500) -> list[dict]:
    """Load EPIC-Kitchens sequences from the temporal_order_sequences manifest.

    Returns list of dicts with: sequence_id, video_path, start_sec, stop_sec.
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
        raise FileNotFoundError(f"No videos found in {video_dir}.")

    if len(available) > max_sequences:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(available), size=max_sequences, replace=False)
        indices.sort()
        available = [available[i] for i in indices]

    print(
        f"  Loaded {len(available)} sequences "
        f"({len(set(s['video_id'] for s in available))} videos)"
    )
    return available


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS.mmm timestamp to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="X-CLIP reversal sensitivity on EPIC-Kitchens"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default="datasets/epic_kitchens",
        help="Path to EPIC-Kitchens dataset directory",
    )
    parser.add_argument(
        "--xclip-model",
        type=str,
        default="microsoft/xclip-large-patch14-16-frames",
        help="X-CLIP model name or path",
    )
    parser.add_argument("--max-sequences", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-fps", type=float, default=3.0)
    args = parser.parse_args()

    epic_dir = Path(args.epic_dir)

    print("=" * 60)
    print("X-CLIP REVERSAL SENSITIVITY (EPIC-Kitchens)")
    print("=" * 60)

    # Load sequences
    print("\nLoading sequences...")
    sequences = load_sequences(epic_dir, max_sequences=args.max_sequences)
    print(f"  Loaded {len(sequences)} sequences")

    if len(sequences) == 0:
        print("ERROR: No sequences found.")
        return

    # Load X-CLIP
    print(f"\nLoading X-CLIP from {args.xclip_model}...")
    encoder = XCLIPEncoder(model_name=args.xclip_model, device=args.device)
    print(f"  num_frames={encoder.num_frames}, dim={encoder.embedding_dim}")

    # Compute s_rev for each sequence
    sims = []
    failed = 0

    for seq in tqdm(sequences, desc="Computing s_rev"):
        try:
            frames = load_clip(
                seq["video_path"],
                seq["start_sec"],
                seq["stop_sec"],
                target_fps=args.target_fps,
            )
            if len(frames) < encoder.num_frames:
                failed += 1
                continue

            emb_fwd = encoder.encode_clip(frames)
            emb_rev = encoder.encode_clip(frames[::-1])

            sim = float(F.cosine_similarity(emb_fwd, emb_rev).item())
            sims.append(sim)

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  WARNING: {seq['sequence_id']} failed: {e}")
            continue

    print(f"\n  Computed: {len(sims)}/{len(sequences)} ({failed} failed)")

    if len(sims) == 0:
        print("ERROR: No successful computations.")
        del encoder
        return

    sims_arr = np.array(sims)
    mean_val, ci_lo, ci_hi = bootstrap_ci(sims_arr)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  X-CLIP s_rev = {mean_val:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  (n={len(sims)}, min={sims_arr.min():.4f}, max={sims_arr.max():.4f})")

    # Perfect separation check
    n_detected = (sims_arr < 0.95).sum()
    print(f"  Sequences with s_rev < 0.95: {n_detected}/{len(sims)}")

    # Free GPU
    del encoder
    torch.cuda.empty_cache()

    # Save results
    results = {
        "model": args.xclip_model,
        "n_sequences": len(sims),
        "s_rev_mean": mean_val,
        "s_rev_ci_low": ci_lo,
        "s_rev_ci_high": ci_hi,
        "s_rev_min": float(sims_arr.min()),
        "s_rev_max": float(sims_arr.max()),
        "s_rev_std": float(sims_arr.std()),
    }

    out_dir = epic_dir / "feature_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xclip_reversal_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

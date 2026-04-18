#!/usr/bin/env python3
"""Raw-Frame Scramble for V-JEPA 2 on VCDB.

Unlike eval_vcdb_scramble.py which shuffles extracted embeddings, this
script shuffles raw video frames before running V-JEPA 2, testing
whether the encoder itself is sensitive to temporal order.

If the encoder produces different residuals from scrambled input, the
degradation curve should be cleaner (monotonic) than the extracted-
embedding scramble.

Usage:
    python experiments/eval_vcdb_scramble_raw.py \\
        --vcdb-dir datasets/vcdb/core_dataset
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from video_retrieval.fingerprints.dtw import dtw_distance_batch

VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32
VJEPA2_SPATIAL = 256

K_VALUES = [1, 2, 4, 8, 16]


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    copy_pairs: set[tuple[str, str]] = set()
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
                    pair = (min(vid_a, vid_b), max(vid_a, vid_b))
                    copy_pairs.add(pair)
    return copy_pairs


def discover_videos(vid_base_dir: str) -> list[str]:
    videos = []
    for cat in sorted(os.listdir(vid_base_dir)):
        cat_path = os.path.join(vid_base_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))
    return videos


def load_frames_for_vjepa2(video_path: str, max_resolution: int = 256) -> list[np.ndarray]:
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base) if stream.duration else 60.0
    target_fps = VJEPA2_NUM_FRAMES / max(duration, 1.0)
    container.close()

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    sample_interval = video_fps / target_fps

    frames = []
    frame_count = 0
    next_sample = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None:
            frame_count += 1
            continue
        if frame_count >= next_sample:
            img = frame.to_ndarray(format="rgb24")
            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1] * scale), max_resolution))
            frames.append(img)
            next_sample += sample_interval
            if len(frames) >= VJEPA2_NUM_FRAMES + 10:
                break
        frame_count += 1

    container.close()
    if len(frames) == 0:
        raise ValueError("No frames extracted")
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])
    return frames[:VJEPA2_NUM_FRAMES]


def scramble_frames(frames: list[np.ndarray], n_chunks: int, seed: int) -> list[np.ndarray]:
    if n_chunks <= 1:
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
    result = []
    for p in perm:
        result.extend(chunks[p])
    return result


def build_temporal_masks(n_context_steps: int, device: torch.device):
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)
    context = grid[:n_context_steps].reshape(-1).unsqueeze(0)
    target = grid[n_context_steps:].reshape(-1).unsqueeze(0)
    return context, target


def extract_features(model, processor, frames, device, context_mask, target_mask, n_target_steps):
    """Extract BoT mean_emb and temporal residual from frames."""
    # pyrefly: ignore [not-callable]
    inputs = processor(videos=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        enc_out = model(**inputs, skip_predictor=True)
        encoder_tokens = enc_out.last_hidden_state[0]
        mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

        pred_out = model(
            **inputs, context_mask=[context_mask], target_mask=[target_mask]
        )
        predicted = pred_out.predictor_output.last_hidden_state[0]
        ground_truth = pred_out.predictor_output.target_hidden_state[0]
        predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
        ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
        residual = (predicted - ground_truth).mean(dim=1)

    return mean_emb.cpu(), residual.cpu()


def main():
    parser = argparse.ArgumentParser(description="Raw-frame scramble for V-JEPA 2 on VCDB")
    parser.add_argument("--vcdb-dir", type=str, default="datasets/vcdb/core_dataset")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = Path(args.vcdb_dir) if os.path.isabs(args.vcdb_dir) else project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    device = torch.device(args.device)

    print("=" * 70)
    print("VCDB RAW-FRAME SCRAMBLE (V-JEPA 2)")
    print("=" * 70)

    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}, Copy pairs: {len(copy_pairs)}")

    # Build pair set (same as eval_vcdb_scramble.py)
    keys = videos
    n = len(keys)
    pairs_to_compute: set[tuple[str, str]] = set()
    for a, b in copy_pairs:
        if a in set(keys) and b in set(keys):
            pairs_to_compute.add((a, b))
    n_pos = len(pairs_to_compute)

    rng = np.random.RandomState(42)
    neg_count = 0
    attempts = 0
    while neg_count < n_pos and attempts < n_pos * 20:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i == j:
            attempts += 1
            continue
        pair = (min(keys[i], keys[j]), max(keys[i], keys[j]))
        if pair not in copy_pairs and pair not in pairs_to_compute:
            pairs_to_compute.add(pair)
            neg_count += 1
        attempts += 1

    print(f"  Pairs: {len(pairs_to_compute)} ({n_pos} pos + {neg_count} neg)")

    # Load model
    print("\nLoading V-JEPA 2...")
    from transformers import AutoModel, AutoVideoProcessor

    # Resolve to local checkpoint if VTD_MODEL_DIR is set
    vjepa2_path = VJEPA2_MODEL_NAME
    model_dir = os.environ.get("VTD_MODEL_DIR")
    if model_dir:
        local = Path(model_dir) / VJEPA2_MODEL_NAME.split("/")[-1]
        if local.exists():
            vjepa2_path = str(local)

    model = AutoModel.from_pretrained(vjepa2_path, trust_remote_code=True).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(vjepa2_path, trust_remote_code=True)

    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    # Extract forward features for video A (unchanged across K)
    print("\nExtracting forward features for all videos...")
    fwd_features: dict[str, dict] = {}
    failed = 0
    for vp in tqdm(videos, desc="Forward features"):
        try:
            frames = load_frames_for_vjepa2(os.path.join(str(vid_dir), vp))
            mean_emb, residual = extract_features(
                model, processor, frames, device, context_mask, target_mask, n_target_steps
            )
            fwd_features[vp] = {"mean_emb": mean_emb, "temporal_residual": residual}
        except Exception:
            failed += 1
    print(f"  Forward: {len(fwd_features)}/{len(videos)} ({failed} failed)")

    # Sweep K values
    sweep_results: dict[int, dict] = {}

    for K in K_VALUES:
        print(f"\n  === K={K} ===")
        t0 = time.time()

        # Extract features from raw-scrambled video B
        scrambled_features: dict[str, dict] = {}
        for vp in tqdm(fwd_features.keys(), desc=f"Scramble K={K}"):
            try:
                frames = load_frames_for_vjepa2(os.path.join(str(vid_dir), vp))
                seed = int(hashlib.md5(f"{vp}_{K}".encode()).hexdigest(), 16) % (2**31)
                scrambled = scramble_frames(frames, K, seed)
                mean_emb, residual = extract_features(
                    model, processor, scrambled, device,
                    context_mask, target_mask, n_target_steps,
                )
                scrambled_features[vp] = {"mean_emb": mean_emb, "temporal_residual": residual}
            except Exception:
                continue

        # Compute similarities
        valid_pairs = [
            (a, b) for a, b in pairs_to_compute
            if a in fwd_features and b in scrambled_features
        ]

        # BoT cosine
        bot_scores = []
        bot_labels = []
        for a, b in valid_pairs:
            sim = float(torch.dot(fwd_features[a]["mean_emb"], scrambled_features[b]["mean_emb"]))
            bot_scores.append(sim)
            bot_labels.append(1 if (a, b) in copy_pairs else 0)

        # Residual DTW
        res_seqs_a = [fwd_features[a]["temporal_residual"] for a, b in valid_pairs]
        res_seqs_b = [scrambled_features[b]["temporal_residual"] for a, b in valid_pairs]
        res_dists = dtw_distance_batch(res_seqs_a, res_seqs_b, normalize=True)
        res_scores = torch.exp(-res_dists).tolist()
        res_labels = [1 if (a, b) in copy_pairs else 0 for a, b in valid_pairs]

        bot_ap = average_precision_score(bot_labels, bot_scores)
        res_ap = average_precision_score(res_labels, res_scores)

        sweep_results[K] = {
            "bot_ap": float(bot_ap),
            "res_ap": float(res_ap),
            "n_pairs": len(valid_pairs),
        }

        elapsed = time.time() - t0
        print(f"    BoT AP={bot_ap:.4f}  Residual AP={res_ap:.4f}  ({elapsed:.1f}s)")

    del model, processor
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("RAW-FRAME SCRAMBLE RESULTS")
    print("=" * 70)
    print(f"  {'K':>3s}  {'BoT AP':>8s}  {'Residual AP':>12s}")
    for K in K_VALUES:
        r = sweep_results[K]
        print(f"  {K:>3d}  {r['bot_ap']:>8.4f}  {r['res_ap']:>12.4f}")

    out_path = vcdb_dir / "raw_frame_scramble_results.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""X-CLIP Evaluation on VCDB Copy Detection.

Evaluates X-CLIP's clip-level embedding on the VCDB benchmark. X-CLIP uses
cross-frame temporal attention to produce a single 768-dim clip embedding
from 16 uniformly sampled frames. Unlike DINOv3 bag-of-frames (which pools
per-frame features), X-CLIP's temporal attention may encode ordering.

We evaluate both:
1. Standard copy detection AP (forward vs forward)
2. Reversal attack resilience (forward vs reversed AP)

Usage:
    python experiments/eval_vcdb_xclip.py \
        --vcdb-dir ./data/vcdb/core_dataset \
        --xclip-model microsoft/xclip-large-patch14-16-frames
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, XCLIPModel
from video_retrieval.utils.video import load_video


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


def bootstrap_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for AP."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    ap = average_precision_score(labels, scores)

    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot_aps[i] = ap
        else:
            boot_aps[i] = average_precision_score(l, s)

    alpha = (1 - ci) / 2
    return (
        float(ap),
        float(np.percentile(boot_aps, 100 * alpha)),
        float(np.percentile(boot_aps, 100 * (1 - alpha))),
    )


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean.

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
    return (
        point,
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# X-CLIP Encoder (from test_video_models.py)
# ---------------------------------------------------------------------------


class XCLIPEncoder:
    """X-CLIP video encoder using cross-frame temporal attention."""

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
        """Encode frames as a single clip embedding (1, D), L2-normalized."""
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
# VCDB loading (from eval_vcdb.py)
# ---------------------------------------------------------------------------


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    """Load all VCDB annotations as (videoA, videoB) pairs."""
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
                    a_s, b_s = sorted([vid_a, vid_b])
                    copy_pairs.add((a_s, b_s))
    return copy_pairs


def discover_videos(vid_base_dir: str) -> list[str]:
    """Discover all video files under VCDB core_dataset directory."""
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
# Feature extraction
# ---------------------------------------------------------------------------


def extract_xclip_features(
    encoder: XCLIPEncoder,
    vid_base_dir: str,
    video_relpaths: list[str],
    sample_rate: int = 10,
    max_frames: int = 100,
    reverse: bool = False,
) -> dict[str, torch.Tensor]:
    """Extract X-CLIP clip embeddings for all videos.

    Returns:
        Dict mapping relpath -> (1, D) L2-normalized embedding.
    """
    features = {}
    failed = 0
    direction = "reversed" if reverse else "forward"

    for vp in tqdm(video_relpaths, desc=f"X-CLIP features ({direction})"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames, fps = load_video(
                path,
                sample_rate=sample_rate,
                max_frames=max_frames,
                max_resolution=518,
            )
            if len(frames) < 3:
                failed += 1
                continue
            if reverse:
                frames = frames[::-1]
            emb = encoder.encode_clip(frames)  # (1, D)
            features[vp] = emb.cpu()
        except Exception:
            failed += 1
            continue

    print(
        f"  Extracted ({direction}): {len(features)}/{len(video_relpaths)} "
        f"({failed} failed)"
    )
    return features


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_pairwise_sims(
    features_a: dict[str, torch.Tensor],
    features_b: dict[str, torch.Tensor],
    copy_pairs: set[tuple[str, str]],
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarities for sampled pairs (all positives + matched negatives).

    features_a and features_b may be the same dict (forward-forward) or
    different (forward-reversed).

    Returns:
        (scores, labels) arrays.
    """
    key_set = set(keys)

    # Positive pairs with features
    pos_pairs = []
    for a, b in copy_pairs:
        if a in features_a and b in features_b:
            pos_pairs.append((a, b))

    n_pos = len(pos_pairs)

    # Sample negatives
    rng = np.random.RandomState(42)
    n = len(keys)
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < n_pos and attempts < n_pos * 20:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i == j:
            attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs:
            neg_pairs.append((keys[i], keys[j]))
        attempts += 1

    all_pairs = [(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]

    scores = []
    labels = []
    for (a, b), label in all_pairs:
        if a not in features_a or b not in features_b:
            continue
        sim = float(F.cosine_similarity(features_a[a], features_b[b]).item())
        scores.append(sim)
        labels.append(label)

    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="X-CLIP VCDB Evaluation")
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb/core_dataset",
        help="Path to VCDB core_dataset directory",
    )
    parser.add_argument(
        "--xclip-model",
        type=str,
        default="microsoft/xclip-large-patch14-16-frames",
        help="X-CLIP model name or path",
    )
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"

    print("=" * 60)
    print("X-CLIP VCDB EVALUATION")
    print("=" * 60)

    # Load data
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Copy pairs: {len(copy_pairs)}")

    # Load X-CLIP
    print(f"\nLoading X-CLIP from {args.xclip_model}...")
    encoder = XCLIPEncoder(model_name=args.xclip_model, device=args.device)
    print(f"  num_frames={encoder.num_frames}, dim={encoder.embedding_dim}")

    # Extract forward features
    print("\nExtracting forward features...")
    t0 = time.time()
    fwd_features = extract_xclip_features(
        encoder,
        str(vid_dir),
        videos,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        reverse=False,
    )
    print(f"  Forward extraction: {time.time() - t0:.1f}s")

    # Extract reversed features
    print("\nExtracting reversed features...")
    t0 = time.time()
    rev_features = extract_xclip_features(
        encoder,
        str(vid_dir),
        videos,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        reverse=True,
    )
    print(f"  Reversed extraction: {time.time() - t0:.1f}s")

    del encoder
    torch.cuda.empty_cache()

    # Evaluate
    keys = [v for v in videos if v in fwd_features and v in rev_features]
    print(f"\n  Common videos: {len(keys)}")

    # Forward-forward (standard copy detection)
    print("\n--- Forward vs Forward (standard copy detection) ---")
    scores_ff, labels_ff = compute_pairwise_sims(
        fwd_features, fwd_features, copy_pairs, keys
    )
    ap_ff, ci_lo_ff, ci_hi_ff = bootstrap_ap(scores_ff, labels_ff)
    auc_ff = (
        roc_auc_score(labels_ff, scores_ff) if labels_ff.sum() > 0 else float("nan")
    )
    print(f"  AP = {ap_ff:.4f}  [{ci_lo_ff:.4f}, {ci_hi_ff:.4f}]")
    print(f"  AUC = {auc_ff:.4f}")

    # Forward-reversed (reversal attack)
    print("\n--- Forward vs Reversed (reversal attack) ---")
    scores_fr, labels_fr = compute_pairwise_sims(
        fwd_features, rev_features, copy_pairs, keys
    )
    ap_fr, ci_lo_fr, ci_hi_fr = bootstrap_ap(scores_fr, labels_fr)
    auc_fr = (
        roc_auc_score(labels_fr, scores_fr) if labels_fr.sum() > 0 else float("nan")
    )
    print(f"  AP = {ap_fr:.4f}  [{ci_lo_fr:.4f}, {ci_hi_fr:.4f}]")
    print(f"  AUC = {auc_fr:.4f}")

    ap_delta = ap_fr - ap_ff
    print(f"\n  AP Delta (reversal): {ap_delta:+.4f}")

    # Per-video s_rev
    sims_rev = []
    for v in keys:
        sim = float(F.cosine_similarity(fwd_features[v], rev_features[v]).item())
        sims_rev.append(sim)
    sims_arr = np.array(sims_rev)
    s_rev_mean, s_rev_lo, s_rev_hi = bootstrap_ci(sims_arr)
    print(f"  Mean s_rev = {s_rev_mean:.4f}  [{s_rev_lo:.4f}, {s_rev_hi:.4f}]")

    # Save results
    results = {
        "model": args.xclip_model,
        "n_videos": len(keys),
        "forward_forward": {
            "ap": ap_ff,
            "ci_low": ci_lo_ff,
            "ci_high": ci_hi_ff,
            "auc": auc_ff,
        },
        "forward_reversed": {
            "ap": ap_fr,
            "ci_low": ci_lo_fr,
            "ci_high": ci_hi_fr,
            "auc": auc_fr,
        },
        "ap_delta": ap_delta,
        "s_rev_mean": s_rev_mean,
        "s_rev_ci": [s_rev_lo, s_rev_hi],
    }

    out_path = project_root / "datasets" / "vcdb" / "xclip_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

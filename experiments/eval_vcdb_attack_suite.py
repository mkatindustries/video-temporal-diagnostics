#!/usr/bin/env python3
"""VCDB Attack Suite Evaluation.

Extends the single-attack reversal evaluation (eval_vcdb_reversal.py) with a
suite of four temporal attacks that probe copy detection robustness at
different granularities:

1. **Full reversal** — reverse the entire frame sequence (baseline attack,
   same as eval_vcdb_reversal.py).
2. **Local segment reversal** — reverse a random 25% contiguous segment of
   frames, leaving the rest in forward order.  Tests whether methods detect
   partial temporal manipulation.
3. **Frame drop/duplication** — randomly drop 20% of frames and duplicate
   remaining frames to restore the original length.  Tests robustness to
   frame-level temporal jitter.
4. **Temporal cut-paste** — split the clip into 4 equal segments and randomly
   permute them.  A weaker version of full scrambling.

Expected behaviour:
- Order-invariant methods (BoF, Chamfer) should be unaffected by ALL attacks.
- Sequence-aware methods (Attn Traj, Temp Deriv, V-JEPA 2 Temp Res) should
  show degradation proportional to attack severity.
- Frame drop should minimally affect all methods (content is preserved).

Usage:
    python experiments/eval_vcdb_attack_suite.py                        # All attacks
    python experiments/eval_vcdb_attack_suite.py --skip-vjepa2          # DINOv3 only
    python experiments/eval_vcdb_attack_suite.py --attacks reversal local_reversal
"""

import argparse
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
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.trajectory import dtw_distance
from video_retrieval.models import DINOv3Encoder
from video_retrieval.utils.video import load_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32
VJEPA2_SPATIAL = 256

ATTACK_NAMES = ["reversal", "local_reversal", "frame_drop", "cut_paste"]

ATTACK_LABELS = {
    "reversal": "Full reversal",
    "local_reversal": "Local reversal",
    "frame_drop": "Frame drop/dup",
    "cut_paste": "Temporal cut-paste",
}

DINOV3_METHODS = [
    "bag_of_frames",
    "chamfer",
    "attention_trajectory",
    "temporal_derivative",
]

VJEPA2_METHODS = [
    "vjepa2_bag_of_tokens",
    "vjepa2_temporal_residual",
]

METHOD_LABELS = {
    "bag_of_frames": "DINOv3 BoF",
    "chamfer": "Chamfer",
    "attention_trajectory": "Attn Traj",
    "temporal_derivative": "Temp Deriv",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "Temp Res",
}


# ---------------------------------------------------------------------------
# Attack functions (operate on index arrays)
# ---------------------------------------------------------------------------


def attack_full_reversal(n_frames: int, rng: np.random.RandomState) -> np.ndarray:
    """Reverse the entire frame sequence."""
    return np.arange(n_frames)[::-1].copy()


def attack_local_reversal(n_frames: int, rng: np.random.RandomState) -> np.ndarray:
    """Reverse a random 25% contiguous segment; leave the rest forward."""
    indices = np.arange(n_frames)
    seg_len = max(1, n_frames // 4)
    start = rng.randint(0, n_frames - seg_len + 1)
    indices[start : start + seg_len] = indices[start : start + seg_len][::-1]
    return indices


def attack_frame_drop(n_frames: int, rng: np.random.RandomState) -> np.ndarray:
    """Drop 20% of frames randomly, duplicate remaining to restore length."""
    n_drop = max(1, n_frames // 5)
    all_idx = np.arange(n_frames)
    keep_mask = np.ones(n_frames, dtype=bool)
    drop_indices = rng.choice(n_frames, size=n_drop, replace=False)
    keep_mask[drop_indices] = False
    kept = all_idx[keep_mask]
    # Duplicate random kept frames to restore original length
    n_needed = n_frames - len(kept)
    extras = rng.choice(kept, size=n_needed, replace=True)
    result = np.concatenate([kept, extras])
    # Sort to maintain rough temporal order (duplicates scatter in)
    result.sort()
    return result


def attack_cut_paste(n_frames: int, rng: np.random.RandomState) -> np.ndarray:
    """Split into 4 equal segments and randomly permute them."""
    indices = np.arange(n_frames)
    seg_len = n_frames // 4
    # Handle remainder by making the last segment absorb extra frames
    segments = []
    for i in range(4):
        if i < 3:
            segments.append(indices[i * seg_len : (i + 1) * seg_len])
        else:
            segments.append(indices[i * seg_len :])
    perm = rng.permutation(4)
    return np.concatenate([segments[p] for p in perm])


ATTACK_FNS = {
    "reversal": attack_full_reversal,
    "local_reversal": attack_local_reversal,
    "frame_drop": attack_frame_drop,
    "cut_paste": attack_cut_paste,
}


# ---------------------------------------------------------------------------
# VCDB loading (shared with eval_vcdb_reversal.py)
# ---------------------------------------------------------------------------


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    """Load all VCDB annotations as global (videoA_path, videoB_path) pairs."""
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
# DINOv3 feature extraction
# ---------------------------------------------------------------------------


def extract_all_features(
    encoder: DINOv3Encoder,
    vid_base_dir: str,
    video_relpaths: list[str],
    sample_rate: int = 10,
    max_frames: int = 100,
) -> dict[str, dict]:
    """Extract DINOv3 features for all videos.

    Returns:
        Dict mapping relpath -> {
            'embeddings': (T, 1024) CLS embeddings,
            'centroids': (T, 2) attention centroids,
            'mean_emb': (1024,) L2-normalized mean embedding,
        }
    """
    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc="Extracting DINOv3 features"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames, fps = load_video(
                path, sample_rate=sample_rate, max_frames=max_frames, max_resolution=518
            )
            if len(frames) < 3:
                failed += 1
                continue
            emb = encoder.encode_frames(frames)
            centroids = encoder.get_attention_centroids(frames)
            mean_emb = F.normalize(emb.mean(dim=0), dim=0)
            features[vp] = {
                "embeddings": emb,
                "centroids": centroids,
                "mean_emb": mean_emb,
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(video_relpaths)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# V-JEPA 2 feature extraction
# ---------------------------------------------------------------------------


def build_temporal_masks(
    n_context_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context/target masks for V-JEPA 2 temporal prediction."""
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)
    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)
    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def load_frames_for_vjepa2(
    video_path: str,
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base) if stream.duration else 60.0
    target_fps = VJEPA2_NUM_FRAMES / max(duration, 1.0)
    container.close()

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps_actual = float(stream.average_rate or 30)
    sample_interval = video_fps_actual / target_fps

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
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, new_h))
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


def extract_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    vid_base_dir: str,
    video_relpaths: list[str],
    device: torch.device,
    frame_indices: np.ndarray | None = None,
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for all videos.

    Args:
        frame_indices: If provided, reorder extracted frames using these
            indices before feeding to the model.  This changes the
            representation because V-JEPA 2 uses positional encodings.

    Returns:
        Dict mapping relpath -> {'mean_emb': ..., 'temporal_residual': ...}
    """
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    label = "reordered" if frame_indices is not None else "forward"
    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc=f"V-JEPA 2 features ({label})"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames = load_frames_for_vjepa2(path)

            if frame_indices is not None:
                # Clip indices to valid range for this video
                idx = np.clip(frame_indices, 0, len(frames) - 1)
                # If attack produced different length, adapt
                if len(idx) != len(frames):
                    idx = idx[: len(frames)]
                frames = [frames[i] for i in idx]

            # pyrefly: ignore [not-callable]
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

            features[vp] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1

    print(
        f"  V-JEPA 2 ({label}): {len(features)}/{len(video_relpaths)} ({failed} failed)"
    )
    return features


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------


def sample_pairs(
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[int]]:
    """Build balanced set of positive + negative pairs.

    Returns:
        (pairs, labels) where labels[i] = 1 for positive, 0 for negative.
    """
    key_set = set(keys)
    n = len(keys)

    pos_pairs = []
    for a, b in copy_pairs:
        if a in key_set and b in key_set:
            pos_pairs.append((a, b))
    n_pos = len(pos_pairs)

    rng = np.random.RandomState(42)
    neg_pairs = []
    seen = set(copy_pairs)
    neg_attempts = 0
    while len(neg_pairs) < n_pos and neg_attempts < n_pos * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        a, b = sorted([keys[i], keys[j]])
        pair = (a, b)
        if pair not in seen:
            seen.add(pair)
            neg_pairs.append(pair)
        neg_attempts += 1

    pairs = pos_pairs + neg_pairs
    labels = [1] * n_pos + [0] * len(neg_pairs)
    return pairs, labels


# ---------------------------------------------------------------------------
# DINOv3 similarity with attacks (tensor-level)
# ---------------------------------------------------------------------------


def compute_dinov3_similarities(
    features: dict[str, dict],
    pairs: list[tuple[str, str]],
    labels: list[int],
    attack_name: str,
    rng: np.random.RandomState,
) -> dict[str, dict]:
    """Compute DINOv3 similarities for a given attack.

    For the "normal" baseline, similarities are computed on unmodified
    tensors.  For attacks, the SECOND clip's embedding/centroid indices are
    reordered according to the attack function.

    Negatives are always unmodified (attack is only applied to positive
    pairs' second clip).

    Returns:
        method_name -> {'scores': list[float], 'labels': list[int]}
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()
    attack_fn = ATTACK_FNS.get(attack_name)

    results = {m: {"scores": [], "labels": []} for m in DINOV3_METHODS}

    for idx, (a, b) in enumerate(
        tqdm(pairs, desc=f"  DINOv3 [{attack_name}]", leave=False)
    ):
        if a not in features or b not in features:
            continue

        ea = features[a]["embeddings"]
        ca = features[a]["centroids"]
        eb = features[b]["embeddings"]
        cb = features[b]["centroids"]
        label = labels[idx]

        # Apply attack to second clip (only for positive pairs)
        if attack_fn is not None and label == 1:
            n_b = eb.shape[0]
            atk_idx = attack_fn(n_b, rng)
            eb = eb[atk_idx]
            cb = cb[atk_idx]

        # Bag-of-frames: mean embedding cosine (order-invariant)
        m1 = F.normalize(ea.mean(dim=0), dim=0)
        m2 = F.normalize(eb.mean(dim=0), dim=0)
        bof_sim = float(torch.dot(m1, m2).item())

        # Chamfer: max-matching (order-invariant)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sim = (max_ab + max_ba) / 2

        # Temporal derivative DTW (order-sensitive)
        fp_a = deriv_fp.compute_fingerprint(ea)
        fp_b = deriv_fp.compute_fingerprint(eb)
        deriv_sim = deriv_fp.compare(fp_a, fp_b)

        # Attention trajectory DTW (order-sensitive)
        tfp_a = traj_fp.compute_fingerprint(ca)
        tfp_b = traj_fp.compute_fingerprint(cb)
        traj_sim = traj_fp.compare(tfp_a, tfp_b)

        for method, sim in [
            ("bag_of_frames", bof_sim),
            ("chamfer", chamfer_sim),
            ("temporal_derivative", deriv_sim),
            ("attention_trajectory", traj_sim),
        ]:
            results[method]["scores"].append(sim)
            results[method]["labels"].append(label)

    return results


# ---------------------------------------------------------------------------
# V-JEPA 2 similarity with attacks
# ---------------------------------------------------------------------------


def compute_vjepa2_similarities(
    vjepa2_fwd: dict[str, dict],
    vjepa2_attacked: dict[str, dict],
    pairs: list[tuple[str, str]],
    labels: list[int],
    attack_name: str,
) -> dict[str, dict]:
    """Compute V-JEPA 2 similarities for a given attack.

    vjepa2_fwd: forward features for all videos.
    vjepa2_attacked: features for second-clip videos re-extracted with
        attacked frame order (only for positive pairs' B clips).  For
        negative pairs, forward features are used.

    Returns:
        method_name -> {'scores': list[float], 'labels': list[int]}
    """
    results = {m: {"scores": [], "labels": []} for m in VJEPA2_METHODS}

    for idx, (a, b) in enumerate(pairs):
        label = labels[idx]

        # Determine which feature set to use for B
        if label == 1 and b in vjepa2_attacked:
            vb = vjepa2_attacked[b]
        elif b in vjepa2_fwd:
            vb = vjepa2_fwd[b]
        else:
            continue

        if a not in vjepa2_fwd:
            continue

        va = vjepa2_fwd[a]

        # Bag-of-tokens (order-invariant)
        bot_sim = float(torch.dot(va["mean_emb"], vb["mean_emb"]).item())

        # Temporal residual DTW (order-sensitive)
        res_dist = dtw_distance(
            va["temporal_residual"],
            vb["temporal_residual"],
            normalize=True,
        )
        tres_sim = float(torch.exp(torch.tensor(-res_dist)).item())

        results["vjepa2_bag_of_tokens"]["scores"].append(bot_sim)
        results["vjepa2_bag_of_tokens"]["labels"].append(label)
        results["vjepa2_temporal_residual"]["scores"].append(tres_sim)
        results["vjepa2_temporal_residual"]["labels"].append(label)

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ap_ci(
    scores: list[float],
    labels: list[int],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AP with 95% bootstrap CI.

    Returns:
        (ap, ci_lower, ci_upper)
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n = len(scores_arr)

    if labels_arr.sum() == 0 or labels_arr.sum() == n:
        return float("nan"), float("nan"), float("nan")

    ap = average_precision_score(labels_arr, scores_arr)

    rng = np.random.RandomState(seed)
    boot_aps = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_true_b = labels_arr[idx]
        y_score_b = scores_arr[idx]
        if y_true_b.sum() == 0 or y_true_b.sum() == len(y_true_b):
            continue
        boot_aps.append(average_precision_score(y_true_b, y_score_b))

    if len(boot_aps) < 10:
        return float(ap), float("nan"), float("nan")

    boot_aps = np.array(boot_aps)
    ci_lo = float(np.percentile(boot_aps, 2.5))
    ci_hi = float(np.percentile(boot_aps, 97.5))
    return float(ap), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VCDB Attack Suite Evaluation")
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb/core_dataset",
        help="Path to VCDB core_dataset directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-vjepa2",
        action="store_true",
        help="Skip V-JEPA 2 methods (DINOv3 only, faster)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["all"],
        choices=["all"] + ATTACK_NAMES,
        help="Which attacks to run (default: all)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=10, help="Frame sampling rate for DINOv3"
    )
    parser.add_argument(
        "--max-frames", type=int, default=100, help="Max frames per video for DINOv3"
    )
    args = parser.parse_args()

    attacks = ATTACK_NAMES if "all" in args.attacks else args.attacks

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"

    print("=" * 80)
    print("VCDB ATTACK SUITE EVALUATION")
    print("=" * 80)
    print(f"  VCDB dir:    {vcdb_dir}")
    print(f"  Attacks:     {', '.join(attacks)}")
    print(f"  Skip V-JEPA: {args.skip_vjepa2}")
    print(f"  Device:      {args.device}")

    # ------------------------------------------------------------------
    # Step 1: Discover videos + load annotations
    # ------------------------------------------------------------------
    print("\nStep 1: Loading dataset...")
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Annotated copy pairs: {len(copy_pairs)}")

    # ------------------------------------------------------------------
    # Step 2: Extract DINOv3 features (once, shared across all attacks)
    # ------------------------------------------------------------------
    print("\nStep 2: Extracting DINOv3 features...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)
    t0 = time.time()
    features = extract_all_features(
        encoder,
        str(vid_dir),
        videos,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )
    print(f"  DINOv3 extraction: {time.time() - t0:.1f}s")
    del encoder
    torch.cuda.empty_cache()

    keys = sorted(features.keys())
    print(f"  Videos with features: {len(keys)}")

    # ------------------------------------------------------------------
    # Step 3: Build shared evaluation pairs
    # ------------------------------------------------------------------
    print("\nStep 3: Sampling evaluation pairs...")
    pairs, labels = sample_pairs(keys, copy_pairs)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Pairs: {len(pairs)} ({n_pos} pos + {n_neg} neg)")

    # Identify B-clip videos from positive pairs (for V-JEPA 2 re-extraction)
    pos_b_videos = sorted({b for (a, b), lbl in zip(pairs, labels) if lbl == 1})

    # ------------------------------------------------------------------
    # Step 4: Extract V-JEPA 2 features
    # ------------------------------------------------------------------
    vjepa2_fwd = None
    vjepa2_attacked_cache = {}  # attack_name -> {relpath: features}

    if not args.skip_vjepa2:
        print("\nStep 4: Loading V-JEPA 2 model...")
        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        device = torch.device(args.device)

        # Forward features (all videos)
        print("  Extracting V-JEPA 2 forward features...")
        t0 = time.time()
        vjepa2_fwd = extract_vjepa2_features(
            vjepa2_model, vjepa2_processor, str(vid_dir), keys, device
        )
        print(f"  Forward extraction: {time.time() - t0:.1f}s")

        # Attacked features: re-extract B-clips with each attack's frame order
        rng_vjepa = np.random.RandomState(42)
        for attack_name in attacks:
            print(f"  Extracting V-JEPA 2 [{attack_name}] features for B-clips...")
            t0 = time.time()
            # Generate attack indices for VJEPA2_NUM_FRAMES
            atk_indices = ATTACK_FNS[attack_name](VJEPA2_NUM_FRAMES, rng_vjepa)
            vjepa2_attacked_cache[attack_name] = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                str(vid_dir),
                pos_b_videos,
                device,
                frame_indices=atk_indices,
            )
            print(f"  {attack_name} extraction: {time.time() - t0:.1f}s")

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()
    else:
        print("\nStep 4: Skipping V-JEPA 2 (--skip-vjepa2)")

    # ------------------------------------------------------------------
    # Step 5: Compute original (no-attack) baseline
    # ------------------------------------------------------------------
    print("\nStep 5: Computing original (no-attack) baseline...")
    rng_baseline = np.random.RandomState(42)
    baseline_dinov3 = compute_dinov3_similarities(
        features, pairs, labels, attack_name="none", rng=rng_baseline
    )
    baseline_vjepa2 = None
    if vjepa2_fwd is not None:
        baseline_vjepa2 = compute_vjepa2_similarities(
            vjepa2_fwd, vjepa2_fwd, pairs, labels, "none"
        )

    # ------------------------------------------------------------------
    # Step 6: Run each attack and collect results
    # ------------------------------------------------------------------
    print("\nStep 6: Running attacks...")

    # Structure: attack_name -> method -> {ap, ci_lo, ci_hi}
    all_results = {}

    # Baseline
    all_results["none"] = {}
    methods = list(DINOV3_METHODS)
    if vjepa2_fwd is not None:
        methods.extend(VJEPA2_METHODS)

    for m in DINOV3_METHODS:
        ap, ci_lo, ci_hi = bootstrap_ap_ci(
            baseline_dinov3[m]["scores"], baseline_dinov3[m]["labels"]
        )
        all_results["none"][m] = {"ap": ap, "ci_lo": ci_lo, "ci_hi": ci_hi}

    if baseline_vjepa2 is not None:
        for m in VJEPA2_METHODS:
            ap, ci_lo, ci_hi = bootstrap_ap_ci(
                baseline_vjepa2[m]["scores"], baseline_vjepa2[m]["labels"]
            )
            all_results["none"][m] = {"ap": ap, "ci_lo": ci_lo, "ci_hi": ci_hi}

    # Each attack
    for attack_name in attacks:
        print(f"\n  --- Attack: {ATTACK_LABELS[attack_name]} ---")
        rng_atk = np.random.RandomState(42)

        # DINOv3
        atk_dinov3 = compute_dinov3_similarities(
            features, pairs, labels, attack_name, rng_atk
        )

        all_results[attack_name] = {}
        for m in DINOV3_METHODS:
            ap, ci_lo, ci_hi = bootstrap_ap_ci(
                atk_dinov3[m]["scores"], atk_dinov3[m]["labels"]
            )
            all_results[attack_name][m] = {"ap": ap, "ci_lo": ci_lo, "ci_hi": ci_hi}

        # V-JEPA 2
        if vjepa2_fwd is not None and attack_name in vjepa2_attacked_cache:
            atk_vjepa2 = compute_vjepa2_similarities(
                vjepa2_fwd,
                vjepa2_attacked_cache[attack_name],
                pairs,
                labels,
                attack_name,
            )
            for m in VJEPA2_METHODS:
                ap, ci_lo, ci_hi = bootstrap_ap_ci(
                    atk_vjepa2[m]["scores"], atk_vjepa2[m]["labels"]
                )
                all_results[attack_name][m] = {"ap": ap, "ci_lo": ci_lo, "ci_hi": ci_hi}

    # ------------------------------------------------------------------
    # Step 7: Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ATTACK SUITE RESULTS")
    print("=" * 80)

    header = (
        f"  {'Attack':<22s} | {'Method':<20s} | {'Original AP':>11s} | "
        f"{'Attacked AP':>11s} | {'Delta AP':>10s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for attack_name in attacks:
        atk_label = ATTACK_LABELS[attack_name]
        for m in methods:
            m_label = METHOD_LABELS.get(m, m)
            orig_ap = all_results["none"].get(m, {}).get("ap", float("nan"))
            atk_ap = all_results[attack_name].get(m, {}).get("ap", float("nan"))
            if np.isnan(orig_ap) or np.isnan(atk_ap):
                delta = float("nan")
                delta_str = f"{'nan':>10s}"
            else:
                delta = atk_ap - orig_ap
                delta_str = f"{delta:>+10.3f}"
            print(
                f"  {atk_label:<22s} | {m_label:<20s} | "
                f"{orig_ap:>11.3f} | {atk_ap:>11.3f} | {delta_str}"
            )
        # Separator between attacks
        if attack_name != attacks[-1]:
            print("  " + "-" * (len(header) - 2))

    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 8: Save results JSON
    # ------------------------------------------------------------------
    results_path = project_root / "datasets" / "vcdb" / "attack_suite_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for attack_name in ["none"] + attacks:
        serializable[attack_name] = {}
        for m in methods:
            if m in all_results.get(attack_name, {}):
                entry = all_results[attack_name][m]
                serializable[attack_name][m] = {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in entry.items()
                }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

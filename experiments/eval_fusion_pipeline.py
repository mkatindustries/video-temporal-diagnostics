#!/usr/bin/env python3
"""Two-Stage Fusion Pipeline for VCDB Copy Detection.

Demonstrates the paper's core contribution: a deployable two-stage system
with hard separation between semantic retrieval and temporal discrimination.

Architecture (hard stage separation):
    Stage 1: Semantic pre-filter (BoF/Chamfer, order-invariant)
             Answer: "Is this visually similar?" → candidate pairs
    Stage 2: Temporal classifier (DTW-based, order-aware)
             Answer: "Is this temporally consistent?" → accept/reject

    Duplicate = Stage 1 YES + Stage 2 YES
    Attack    = Stage 1 YES + Stage 2 NO

    4 scenarios: standard, +reversal, +scramble, +speed-change

The first-run failure (v1 used all 6 features in Stage 2, logistic regression
put 95%+ weight on order-invariant features → 0% attack rejection) motivates
this hard separation. A single ensemble cannot optimize for both copy detection
AND temporal discrimination — the Triad thesis in action.

Usage:
    python experiments/eval_fusion_pipeline.py                          # Full run
    python experiments/eval_fusion_pipeline.py --skip-vjepa2            # DINOv3 only
    python experiments/eval_fusion_pipeline.py --fusion-mode learned    # Logistic regression
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder
from video_retrieval.utils.video import load_video


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def save_feature_cache(features: dict, cache_path: Path) -> None:
    """Save extracted features to disk as a .pt file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_features = {}
    for k, v in features.items():
        cpu_features[k] = {
            fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv for fk, fv in v.items()
        }
    torch.save(cpu_features, cache_path)
    print(f"  Cache saved to {cache_path}")


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load cached features from disk, or return None if not found."""
    if not cache_path.exists():
        return None
    print(f"  Loading cache from {cache_path}")
    return torch.load(cache_path, weights_only=False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32
VJEPA2_SPATIAL = 256

DINOV3_METHODS = [
    "bag_of_frames",
    "chamfer",
    "temporal_derivative",
    "attention_trajectory",
]

VJEPA2_METHODS = [
    "vjepa2_bag_of_tokens",
    "vjepa2_temporal_residual",
]

# Hard stage separation: semantic (order-invariant) vs temporal (order-aware)
SEMANTIC_METHODS = [
    "bag_of_frames",
    "chamfer",
    "vjepa2_bag_of_tokens",
]

TEMPORAL_METHODS = [
    "temporal_derivative",
    "attention_trajectory",
    "vjepa2_temporal_residual",
]

METHOD_LABELS = {
    "bag_of_frames": "Bag of Frames",
    "chamfer": "Chamfer",
    "temporal_derivative": "Temporal Derivative",
    "attention_trajectory": "Attention Trajectory",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "V-JEPA 2 Temporal Res.",
}

SCENARIO_COLORS = {
    "standard": "#2ecc71",
    "reversal": "#e74c3c",
    "scramble": "#3498db",
    "speed_change": "#f39c12",
}

STAGE1_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# VCDB loading
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
    """Extract DINOv3 features for all videos."""
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

    print(f"  Extracted: {len(features)}/{len(video_relpaths)} " f"({failed} failed)")
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
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for all videos."""
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc="V-JEPA 2 features"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames = load_frames_for_vjepa2(path)

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

    print(f"  V-JEPA 2: {len(features)}/{len(video_relpaths)} " f"({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    scores: dict[tuple[str, str], float],
    copy_pairs: set[tuple[str, str]],
) -> dict[str, float]:
    """Compute AP and AUC for a method."""
    y_true = []
    y_score = []

    for pair, sim in scores.items():
        y_true.append(1 if pair in copy_pairs else 0)
        y_score.append(sim)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"ap": float("nan"), "auc": float("nan"), "n_pos": n_pos, "n_neg": n_neg}

    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return {"ap": float(ap), "auc": float(auc), "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# Attack transforms
# ---------------------------------------------------------------------------


def scramble_tensor(tensor: torch.Tensor, n_chunks: int, seed: int) -> torch.Tensor:
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


def speed_change_tensor(tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample tensor along dim 0 to simulate speed change.

    Args:
        tensor: (T, ...) tensor to resample.
        factor: Speed factor. >1 = faster (fewer frames), <1 = slower (more frames).

    Returns:
        Resampled tensor with new_T = max(3, int(T / factor)) frames.
    """
    new_T = max(3, int(len(tensor) / factor))
    indices = torch.linspace(0, len(tensor) - 1, new_T).long()
    return tensor[indices]


def apply_attack(
    features: dict[str, dict],
    vjepa2_features: dict[str, dict] | None,
    video_key: str,
    attack: str,
    speed_factor: float = 1.0,
) -> tuple[dict, dict | None]:
    """Apply an attack to a video's features, returning modified copies.

    Args:
        features: DINOv3 features dict (full dataset).
        vjepa2_features: V-JEPA 2 features dict (full dataset), or None.
        video_key: Key of the video to attack.
        attack: One of "reversal", "scramble", "speed".
        speed_factor: Speed factor for "speed" attack.

    Returns:
        (attacked_dinov3_dict, attacked_vjepa2_dict) for the single video.
        V-JEPA 2 reversal flips temporal_residual as an approximation
        (true reversal requires re-extraction with reversed frames).
    """
    feat = features[video_key]
    emb = feat["embeddings"]
    cen = feat["centroids"]

    if attack == "reversal":
        emb_att = emb.flip(0)
        cen_att = cen.flip(0)
        mean_att = feat["mean_emb"]  # order-invariant
    elif attack == "scramble":
        seed = int(hashlib.md5(f"{video_key}_scramble".encode()).hexdigest(), 16) % (
            2**31
        )
        emb_att = scramble_tensor(emb, 8, seed)
        cen_att = scramble_tensor(cen, 8, seed)
        mean_att = feat["mean_emb"]  # order-invariant
    elif attack == "speed":
        emb_att = speed_change_tensor(emb, speed_factor)
        cen_att = speed_change_tensor(cen, speed_factor)
        mean_att = F.normalize(emb_att.mean(dim=0), dim=0)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    dinov3_attacked = {
        "embeddings": emb_att,
        "centroids": cen_att,
        "mean_emb": mean_att,
    }

    vjepa2_attacked = None
    if vjepa2_features is not None and video_key in vjepa2_features:
        vf = vjepa2_features[video_key]
        res = vf["temporal_residual"]

        if attack == "reversal":
            # Approximation: flip temporal residual. True reversal requires
            # re-extraction with reversed frame input (positional encodings
            # and predictor context/target split change with frame order).
            res_att = res.flip(0)
            mean_att_v = vf["mean_emb"]
        elif attack == "scramble":
            seed = int(
                hashlib.md5(f"{video_key}_scramble".encode()).hexdigest(), 16
            ) % (2**31)
            res_att = scramble_tensor(res, 8, seed)
            mean_att_v = vf["mean_emb"]
        elif attack == "speed":
            res_att = speed_change_tensor(res, speed_factor)
            mean_att_v = vf["mean_emb"]  # BoT mean is roughly speed-invariant
        else:
            raise ValueError(f"Unknown attack: {attack}")

        vjepa2_attacked = {
            "mean_emb": mean_att_v,
            "temporal_residual": res_att,
        }

    return dinov3_attacked, vjepa2_attacked


# ---------------------------------------------------------------------------
# Stage 1: BoF cosine pre-filter
# ---------------------------------------------------------------------------


def stage1_prefilter(
    features: dict[str, dict],
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    """Sweep BoF cosine thresholds, select operating point with recall >= 0.99.

    Returns:
        Dict with per-threshold stats and selected threshold.
    """
    # Build mean_emb matrix (N, 1024)
    mean_embs = torch.stack([features[k]["mean_emb"] for k in keys])
    sim_matrix = mean_embs @ mean_embs.T  # (N, N)

    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Identify positive pairs that exist in features
    pos_indices = []
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pos_indices.append((key_to_idx[a], key_to_idx[b]))

    n_pos = len(pos_indices)
    if n_pos == 0:
        print("  WARNING: No positive pairs found in features")
        return {"sweep": {}, "selected_threshold": 0.50}

    sweep = {}
    selected_threshold = STAGE1_THRESHOLDS[0]

    for threshold in STAGE1_THRESHOLDS:
        # Count how many positive pairs pass the threshold
        pos_recall_count = 0
        for i, j in pos_indices:
            if sim_matrix[i, j].item() >= threshold:
                pos_recall_count += 1

        recall = pos_recall_count / n_pos

        # Count total candidate pairs above threshold (upper triangle)
        mask = torch.triu(sim_matrix >= threshold, diagonal=1)
        n_candidates = int(mask.sum().item())

        sweep[threshold] = {
            "recall": recall,
            "n_candidates": n_candidates,
            "pos_recalled": pos_recall_count,
        }

    # Select highest threshold where recall >= 0.99
    for threshold in reversed(STAGE1_THRESHOLDS):
        if sweep[threshold]["recall"] >= 0.99:
            selected_threshold = threshold
            break

    return {"sweep": sweep, "selected_threshold": selected_threshold}


def get_candidate_pairs(
    features: dict[str, dict],
    keys: list[str],
    threshold: float,
    extra_pairs: set[tuple[str, str]] | None = None,
) -> set[tuple[str, str]]:
    """Return pairs above BoF cosine threshold, plus any extra forced pairs.

    Args:
        features: DINOv3 features with mean_emb.
        keys: Sorted video keys.
        threshold: BoF cosine similarity threshold.
        extra_pairs: Additional pairs to always include (e.g., attack pairs).
    """
    mean_embs = torch.stack([features[k]["mean_emb"] for k in keys])
    sim_matrix = mean_embs @ mean_embs.T

    key_to_idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    candidates = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j].item() >= threshold:
                candidates.add((keys[i], keys[j]))

    if extra_pairs:
        for a, b in extra_pairs:
            pair = tuple(sorted([a, b]))
            if pair[0] in key_to_idx and pair[1] in key_to_idx:
                # pyrefly: ignore [bad-argument-type]
                candidates.add(pair)

    return candidates


# ---------------------------------------------------------------------------
# Stage 2: Compute all method similarities for candidate pairs
# ---------------------------------------------------------------------------


def compute_pair_similarities(
    features_a: dict[str, dict],
    features_b: dict[str, dict],
    pairs: set[tuple[str, str]],
    methods: list[str],
    vjepa2_a: dict[str, dict] | None = None,
    vjepa2_b: dict[str, dict] | None = None,
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute all method similarities for given pairs.

    features_a and features_b can differ (e.g., when B is attacked).
    For standard evaluation, pass the same dict for both.

    Uses batched DTW for temporal_derivative, attention_trajectory,
    and vjepa2_temporal_residual.
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints for A
    deriv_fps_a = {}
    traj_fps_a = {}
    for k in features_a:
        deriv_fps_a[k] = deriv_fp.compute_fingerprint(features_a[k]["embeddings"])
        traj_fps_a[k] = traj_fp.compute_fingerprint(features_a[k]["centroids"])

    results = {m: {} for m in methods}

    # Filter to valid pairs
    valid_pairs = [(a, b) for a, b in pairs if a in features_a and b in features_b]

    # --- Non-DTW methods ---
    for a, b in valid_pairs:
        ea = features_a[a]["embeddings"]
        eb = features_b[b]["embeddings"]

        if "bag_of_frames" in results:
            m1 = features_a[a]["mean_emb"]
            m2 = features_b[b]["mean_emb"]
            results["bag_of_frames"][(a, b)] = float(torch.dot(m1, m2).item())

        if "chamfer" in results:
            sim_matrix = torch.mm(ea, eb.t())
            max_ab = sim_matrix.max(dim=1).values.mean().item()
            max_ba = sim_matrix.max(dim=0).values.mean().item()
            results["chamfer"][(a, b)] = (max_ab + max_ba) / 2

        if vjepa2_a is not None and vjepa2_b is not None:
            if a in vjepa2_a and b in vjepa2_b:
                if "vjepa2_bag_of_tokens" in results:
                    results["vjepa2_bag_of_tokens"][(a, b)] = float(
                        torch.dot(
                            vjepa2_a[a]["mean_emb"],
                            vjepa2_b[b]["mean_emb"],
                        ).item()
                    )

    # --- Batched DTW: temporal_derivative ---
    if "temporal_derivative" in results:
        dtw_pairs_td = []
        seqs_a_td = []
        seqs_b_td = []
        for a, b in valid_pairs:
            fp_a = deriv_fps_a[a]
            fp_b = deriv_fp.compute_fingerprint(features_b[b]["embeddings"])
            if fp_a.shape[0] > 0 and fp_b.shape[0] > 0:
                dtw_pairs_td.append((a, b))
                seqs_a_td.append(fp_a)
                seqs_b_td.append(fp_b)

        if dtw_pairs_td:
            dists = dtw_distance_batch(seqs_a_td, seqs_b_td, normalize=False)
            sims = torch.exp(-dists)
            for idx, (a, b) in enumerate(dtw_pairs_td):
                results["temporal_derivative"][(a, b)] = float(sims[idx].item())

    # --- Batched DTW: attention_trajectory ---
    if "attention_trajectory" in results:
        dtw_pairs_at = []
        seqs_a_at = []
        seqs_b_at = []
        for a, b in valid_pairs:
            fp_a = traj_fps_a[a]
            fp_b = traj_fp.compute_fingerprint(features_b[b]["centroids"])
            if fp_a.shape[0] > 0 and fp_b.shape[0] > 0:
                dtw_pairs_at.append((a, b))
                seqs_a_at.append(fp_a)
                seqs_b_at.append(fp_b)

        if dtw_pairs_at:
            dists = dtw_distance_batch(seqs_a_at, seqs_b_at, normalize=True)
            sims = torch.exp(-dists * 5)
            for idx, (a, b) in enumerate(dtw_pairs_at):
                results["attention_trajectory"][(a, b)] = float(sims[idx].item())

    # --- Batched DTW: vjepa2_temporal_residual ---
    if (
        "vjepa2_temporal_residual" in results
        and vjepa2_a is not None
        and vjepa2_b is not None
    ):
        dtw_pairs_vj = []
        seqs_a_vj = []
        seqs_b_vj = []
        for a, b in valid_pairs:
            if a in vjepa2_a and b in vjepa2_b:
                dtw_pairs_vj.append((a, b))
                seqs_a_vj.append(vjepa2_a[a]["temporal_residual"])
                seqs_b_vj.append(vjepa2_b[b]["temporal_residual"])

        if dtw_pairs_vj:
            dists = dtw_distance_batch(seqs_a_vj, seqs_b_vj, normalize=True)
            sims = torch.exp(-dists)
            for idx, (a, b) in enumerate(dtw_pairs_vj):
                results["vjepa2_temporal_residual"][(a, b)] = float(sims[idx].item())

    return results


# ---------------------------------------------------------------------------
# Stage 1 scoring (semantic-only)
# ---------------------------------------------------------------------------


def stage1_semantic_score(
    pair_sims: dict[str, dict[tuple[str, str], float]],
    semantic_methods: list[str],
    copy_pairs: set[tuple[str, str]],
) -> dict:
    """Compute semantic-only fused score (Stage 1 quality metric).

    Uses only order-invariant features (BoF, Chamfer, V-JEPA 2 BoT).
    This is what determines if a pair is a visual near-duplicate.
    """
    all_pairs = set()
    for m in semantic_methods:
        if m in pair_sims:
            all_pairs.update(pair_sims[m].keys())

    fused_scores = {}
    for pair in all_pairs:
        scores = []
        for m in semantic_methods:
            if m in pair_sims and pair in pair_sims[m]:
                scores.append(pair_sims[m][pair])
        if scores:
            fused_scores[pair] = sum(scores) / len(scores)

    fused_metrics = evaluate_method(fused_scores, copy_pairs)

    per_method = {}
    for m in semantic_methods:
        if m in pair_sims and pair_sims[m]:
            per_method[m] = evaluate_method(pair_sims[m], copy_pairs)
        else:
            per_method[m] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": 0,
                "n_neg": 0,
            }

    return {
        "semantic_ap": fused_metrics["ap"],
        "semantic_auc": fused_metrics["auc"],
        "n_pos": fused_metrics["n_pos"],
        "n_neg": fused_metrics["n_neg"],
        "per_method": per_method,
    }


# ---------------------------------------------------------------------------
# Stage 2 fusion (temporal-only — hard separation)
# ---------------------------------------------------------------------------


def stage2_threshold_fusion(
    pair_sims: dict[str, dict[tuple[str, str], float]],
    temporal_methods: list[str],
    copy_pairs: set[tuple[str, str]],
) -> dict:
    """Equal-weight mean of temporal-only method scores.

    Stage 2 uses ONLY order-aware features (temporal derivative, attention
    trajectory, V-JEPA 2 temporal residual). This is the hard separation:
    Stage 1 handles semantic similarity, Stage 2 handles temporal consistency.

    Returns dict with pipeline AP/AUC and per-method AP/AUC.
    """
    all_pairs = set()
    for m in temporal_methods:
        if m in pair_sims:
            all_pairs.update(pair_sims[m].keys())

    fused_scores = {}
    for pair in all_pairs:
        scores = []
        for m in temporal_methods:
            if m in pair_sims and pair in pair_sims[m]:
                scores.append(pair_sims[m][pair])
        if scores:
            fused_scores[pair] = sum(scores) / len(scores)

    fused_metrics = evaluate_method(fused_scores, copy_pairs)

    per_method = {}
    for m in temporal_methods:
        if m in pair_sims and pair_sims[m]:
            per_method[m] = evaluate_method(pair_sims[m], copy_pairs)
        else:
            per_method[m] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": 0,
                "n_neg": 0,
            }

    return {
        "pipeline_ap": fused_metrics["ap"],
        "pipeline_auc": fused_metrics["auc"],
        "n_pos": fused_metrics["n_pos"],
        "n_neg": fused_metrics["n_neg"],
        "per_method": per_method,
    }


def stage2_learned_fusion(
    pair_sims: dict[str, dict[tuple[str, str], float]],
    temporal_methods: list[str],
    copy_pairs: set[tuple[str, str]],
) -> dict:
    """5-fold stratified CV with logistic regression on temporal-only scores.

    Hard separation: only temporal features are used. The classifier learns
    to distinguish true copies (temporally consistent) from attacks (temporally
    disrupted) using order-aware features only.
    """
    all_pairs = set()
    for m in temporal_methods:
        if m in pair_sims:
            all_pairs.update(pair_sims[m].keys())

    pair_list = sorted(all_pairs)
    X_rows = []
    y_list = []
    valid_pairs = []

    for pair in pair_list:
        row = []
        has_all = True
        for m in temporal_methods:
            if m in pair_sims and pair in pair_sims[m]:
                row.append(pair_sims[m][pair])
            else:
                has_all = False
                break
        if has_all:
            X_rows.append(row)
            y_list.append(1 if pair in copy_pairs else 0)
            valid_pairs.append(pair)

    X = np.array(X_rows)
    y = np.array(y_list)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if n_pos < 5 or n_neg < 5:
        print("  WARNING: Too few samples for cross-validation")
        return {
            "pipeline_ap": float("nan"),
            "pipeline_auc": float("nan"),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "per_method": {},
            "learned_weights": {},
        }

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.zeros(len(y))
    fold_weights = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        clf = LogisticRegression(
            l1_ratio=0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        cv_scores[test_idx] = clf.predict_proba(X_test)[:, 1]
        fold_weights.append(clf.coef_[0].tolist())

    pipeline_ap = float(average_precision_score(y, cv_scores))
    pipeline_auc = float(roc_auc_score(y, cv_scores))

    mean_weights = np.mean(fold_weights, axis=0).tolist()
    learned_weights = {m: w for m, w in zip(temporal_methods, mean_weights)}

    per_method = {}
    for j, m in enumerate(temporal_methods):
        method_scores = {}
        for idx, pair in enumerate(valid_pairs):
            method_scores[pair] = X[idx, j]
        per_method[m] = evaluate_method(method_scores, copy_pairs)

    return {
        "pipeline_ap": pipeline_ap,
        "pipeline_auc": pipeline_auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "per_method": per_method,
        "learned_weights": learned_weights,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: str,
    features: dict[str, dict],
    vjepa2_features: dict[str, dict] | None,
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
    neg_pairs: set[tuple[str, str]],
    stage1_threshold: float,
    all_methods: list[str],
    fusion_mode: str,
) -> dict:
    """Run a single scenario with hard stage separation.

    Hard separation architecture:
        Stage 1: Semantic score (BoF/Chamfer/V-JEPA BoT) — "is this visually similar?"
        Stage 2: Temporal score (temp deriv/attn traj/V-JEPA res) — "is this temporally consistent?"
        Duplicate = Stage 1 pass AND Stage 2 pass
        Attack    = Stage 1 pass AND Stage 2 fail

    For attack scenarios, attacked copies of positive-pair B videos are
    created and added as negatives. The temporal-only Stage 2 classifier
    must reject them (they pass Stage 1 because BoF/Chamfer are order-invariant).

    Returns:
        Dict with pipeline AP, stage1 semantic AP, per-method AP,
        attack rejection rate, etc.
    """
    print(f"\n  --- Scenario: {scenario} ---")
    t0 = time.time()

    # Determine which temporal methods are available
    temporal_methods = [m for m in TEMPORAL_METHODS if m in all_methods]
    semantic_methods = [m for m in SEMANTIC_METHODS if m in all_methods]

    # Base pair set: all positive + negative pairs
    base_pairs = set()
    for a, b in copy_pairs:
        if a in features and b in features:
            base_pairs.add((a, b))
    base_pairs.update(neg_pairs)

    if scenario == "standard":
        # Stage 1: BoF pre-filter
        candidate_pairs = get_candidate_pairs(
            features,
            keys,
            stage1_threshold,
            extra_pairs=base_pairs,
        )
        eval_pairs = base_pairs & candidate_pairs

        # Compute ALL method similarities
        pair_sims = compute_pair_similarities(
            features,
            features,
            eval_pairs,
            all_methods,
            vjepa2_a=vjepa2_features,
            vjepa2_b=vjepa2_features,
        )

        # Stage 1 semantic score (for reporting)
        semantic_result = stage1_semantic_score(
            pair_sims,
            semantic_methods,
            copy_pairs,
        )

        # Stage 2 temporal fusion (the pipeline score)
        if fusion_mode == "learned":
            result = stage2_learned_fusion(pair_sims, temporal_methods, copy_pairs)
        else:
            result = stage2_threshold_fusion(pair_sims, temporal_methods, copy_pairs)

        result["semantic_ap"] = semantic_result["semantic_ap"]
        result["semantic_auc"] = semantic_result["semantic_auc"]
        result["attack_rejection_rate"] = None
        result["n_attack_pairs"] = 0

        # Also include semantic per-method in the per_method dict
        for m, v in semantic_result["per_method"].items():
            result["per_method"][m] = v

        elapsed = time.time() - t0
        print(
            f"  Semantic AP: {result['semantic_ap']:.4f} | "
            f"Temporal AP: {result['pipeline_ap']:.4f} "
            f"(n_pos={result['n_pos']}, n_neg={result['n_neg']}) "
            f"[{elapsed:.1f}s]"
        )
        return result

    # --- Attack scenarios ---
    pos_pairs_with_features = [
        (a, b) for a, b in copy_pairs if a in features and b in features
    ]

    if scenario == "speed_change":
        speed_factors = [0.5, 1.5, 2.0]
    else:
        speed_factors = [1.0]

    # Build attacked features
    attack_dinov3 = {}
    attack_vjepa2 = {}
    attack_pairs = set()

    for a, b in pos_pairs_with_features:
        for sf in speed_factors:
            if scenario == "speed_change":
                attack_key = f"__attack__{b}__speed_{sf}"
                attack = "speed"
            elif scenario == "reversal":
                attack_key = f"__attack__{b}__rev"
                attack = "reversal"
            elif scenario == "scramble":
                attack_key = f"__attack__{b}__scr"
                attack = "scramble"
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            d_att, v_att = apply_attack(
                features,
                vjepa2_features,
                b,
                attack,
                speed_factor=sf,
            )
            attack_dinov3[attack_key] = d_att
            if v_att is not None:
                attack_vjepa2[attack_key] = v_att
            attack_pairs.add((a, attack_key))

    n_attack = len(attack_pairs)
    print(f"  Attack pairs: {n_attack}")

    # Build augmented feature dicts
    features_b = dict(features)
    features_b.update(attack_dinov3)

    vjepa2_b = None
    if vjepa2_features is not None:
        vjepa2_b = dict(vjepa2_features)
        vjepa2_b.update(attack_vjepa2)

    # Evaluation set: original positives + original negatives + attack negatives
    eval_pairs = set(base_pairs)
    eval_pairs.update(attack_pairs)

    # Compute ALL method similarities
    pair_sims = compute_pair_similarities(
        features,
        features_b,
        eval_pairs,
        all_methods,
        vjepa2_a=vjepa2_features,
        vjepa2_b=vjepa2_b,
    )

    # Stage 1 semantic score (for reporting)
    semantic_result = stage1_semantic_score(
        pair_sims,
        semantic_methods,
        copy_pairs,
    )

    # Stage 2 temporal fusion (the pipeline score — temporal only)
    if fusion_mode == "learned":
        result = stage2_learned_fusion(pair_sims, temporal_methods, copy_pairs)
    else:
        result = stage2_threshold_fusion(pair_sims, temporal_methods, copy_pairs)

    result["semantic_ap"] = semantic_result["semantic_ap"]
    result["semantic_auc"] = semantic_result["semantic_auc"]

    # Include semantic per-method
    for m, v in semantic_result["per_method"].items():
        result["per_method"][m] = v

    # --- Attack rejection rate ---
    # Rejection = attack pair has temporal score below the temporal threshold.
    # Threshold = score that achieves 95% recall on true positive pairs.
    # This simulates: "Stage 1 passed (visually similar), does Stage 2 reject?"
    temporal_pos_scores = []
    temporal_attack_scores = []

    for pair in eval_pairs:
        scores = []
        for m in temporal_methods:
            if m in pair_sims and pair in pair_sims[m]:
                scores.append(pair_sims[m][pair])
        if not scores:
            continue
        fused = sum(scores) / len(scores)

        if pair in copy_pairs:
            temporal_pos_scores.append(fused)
        elif pair in attack_pairs:
            temporal_attack_scores.append(fused)

    rejected = 0
    if temporal_pos_scores and temporal_attack_scores:
        # Find threshold at 95th percentile of positive scores (low end)
        # i.e., 95% of true copies score above this threshold
        temporal_threshold = float(np.percentile(temporal_pos_scores, 5))
        for score in temporal_attack_scores:
            if score < temporal_threshold:
                rejected += 1

    rejection_rate = rejected / max(n_attack, 1)
    result["attack_rejection_rate"] = rejection_rate
    result["n_attack_pairs"] = n_attack
    result["temporal_threshold"] = (
        float(np.percentile(temporal_pos_scores, 5)) if temporal_pos_scores else None
    )

    elapsed = time.time() - t0
    print(
        f"  Semantic AP: {semantic_result['semantic_ap']:.4f} | "
        f"Temporal AP: {result['pipeline_ap']:.4f} | "
        f"Attack rejection: {rejection_rate:.3f} "
        f"({rejected}/{n_attack}) [{elapsed:.1f}s]"
    )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_threshold_sweep(
    sweep_results: dict,
    fig_dir: Path,
):
    """Plot Stage 1 recall vs threshold."""
    thresholds = sorted(sweep_results.keys())
    recalls = [sweep_results[t]["recall"] for t in thresholds]
    n_candidates = [sweep_results[t]["n_candidates"] for t in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(thresholds, recalls, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax1.axhline(y=0.99, color="red", linestyle="--", alpha=0.5, label="99% recall")
    ax1.set_xlabel("BoF Cosine Threshold", fontsize=12)
    ax1.set_ylabel("Copy Pair Recall", fontsize=12)
    ax1.set_title("Stage 1: Pre-filter Recall vs Threshold", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(thresholds, n_candidates, "o-", color="#3498db", linewidth=2, markersize=8)
    ax2.set_xlabel("BoF Cosine Threshold", fontsize=12)
    ax2.set_ylabel("Candidate Pairs", fontsize=12)
    ax2.set_title("Stage 1: Candidate Count vs Threshold", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Two-Stage Pipeline: Stage 1 Threshold Sweep", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    plot_path = fig_dir / "fusion_pipeline_threshold_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


def plot_scenario_results(
    scenario_results: dict[str, dict],
    fig_dir: Path,
):
    """Plot pipeline AP and attack rejection rates by scenario."""
    scenarios = list(scenario_results.keys())
    pipeline_aps = [scenario_results[s]["pipeline_ap"] for s in scenarios]
    rejection_rates = [
        scenario_results[s].get("attack_rejection_rate") or 0 for s in scenarios
    ]
    colors = [SCENARIO_COLORS.get(s, "#95a5a6") for s in scenarios]
    labels = [s.replace("_", " ").title() for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pipeline AP by scenario
    bars = ax1.bar(
        range(len(scenarios)),
        pipeline_aps,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(labels, fontsize=11)
    for bar, val in zip(bars, pipeline_aps):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax1.set_ylabel("Pipeline AP", fontsize=12)
    ax1.set_title("Pipeline AP by Scenario", fontsize=13)
    ax1.set_ylim(0, 1.15)

    # Attack rejection rates (skip standard which has no attacks)
    attack_scenarios = [s for s in scenarios if s != "standard"]
    attack_labels = [s.replace("_", " ").title() for s in attack_scenarios]
    attack_rates = [
        scenario_results[s].get("attack_rejection_rate") or 0 for s in attack_scenarios
    ]
    attack_colors = [SCENARIO_COLORS.get(s, "#95a5a6") for s in attack_scenarios]

    if attack_scenarios:
        bars = ax2.bar(
            range(len(attack_scenarios)),
            attack_rates,
            color=attack_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xticks(range(len(attack_scenarios)))
        ax2.set_xticklabels(attack_labels, fontsize=11)
        for bar, val in zip(bars, attack_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax2.set_ylabel("Rejection Rate", fontsize=12)
        ax2.set_title("Attack Rejection Rate", fontsize=13)
        ax2.set_ylim(0, 1.15)
    else:
        ax2.text(
            0.5,
            0.5,
            "No attack scenarios",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=14,
        )

    fig.suptitle(
        "Two-Stage Fusion Pipeline: Scenario Comparison", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    plot_path = fig_dir / "fusion_pipeline_rejection.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Fusion Pipeline for VCDB Copy Detection"
    )
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb/core_dataset",
        help="Path to VCDB core_dataset directory",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=10, help="Frame sampling rate"
    )
    parser.add_argument(
        "--max-frames", type=int, default=100, help="Max frames per video"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-vjepa2", action="store_true", help="Skip V-JEPA 2 (DINOv3 only)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction even if cached features exist",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="threshold",
        choices=["threshold", "learned"],
        help="Fusion mode: threshold (equal-weight) or learned (logistic)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = (
        Path(args.vcdb_dir)
        if os.path.isabs(args.vcdb_dir)
        else project_root / args.vcdb_dir
    )
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    if not ann_dir.exists():
        print(f"ERROR: Annotation dir not found: {ann_dir}")
        return
    if not vid_dir.exists():
        print(f"ERROR: Video dir not found: {vid_dir}")
        return

    print("=" * 70)
    print("TWO-STAGE FUSION PIPELINE")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  Sample rate: {args.sample_rate}, max frames: {args.max_frames}")
    print(f"  Fusion mode: {args.fusion_mode}")
    print(f"  Skip V-JEPA 2: {args.skip_vjepa2}")

    # ------------------------------------------------------------------
    # Step 1: Discover videos + load annotations
    # ------------------------------------------------------------------
    print("\nStep 1: Loading dataset...")
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Annotated copy pairs: {len(copy_pairs)}")

    # ------------------------------------------------------------------
    # Step 2: Extract/load DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 2: Loading DINOv3 features...")
    cache_dir = vcdb_dir / "feature_cache"
    dinov3_cache = cache_dir / f"dinov3_sr{args.sample_rate}_mf{args.max_frames}.pt"

    features = load_feature_cache(dinov3_cache) if not args.no_cache else None
    if features is None:
        print("  Loading DINOv3 encoder...")
        encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

        print("  Extracting DINOv3 features...")
        t0 = time.time()
        features = extract_all_features(
            encoder,
            str(vid_dir),
            videos,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames,
        )
        print(f"  DINOv3 feature extraction: {time.time() - t0:.1f}s")

        del encoder
        torch.cuda.empty_cache()

        save_feature_cache(features, dinov3_cache)

    keys = sorted(features.keys())
    print(f"  Videos with features: {len(keys)}")

    # ------------------------------------------------------------------
    # Step 3: Extract/load V-JEPA 2 features
    # ------------------------------------------------------------------
    vjepa2_features = None

    if not args.skip_vjepa2:
        print("\nStep 3: Loading V-JEPA 2 features...")
        vjepa2_cache = cache_dir / "vjepa2.pt"

        vjepa2_features = (
            load_feature_cache(vjepa2_cache) if not args.no_cache else None
        )
        if vjepa2_features is None:
            print("  Loading V-JEPA 2 model...")
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )

            print("  Extracting V-JEPA 2 features...")
            t0 = time.time()
            vjepa2_features = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                str(vid_dir),
                keys,
                torch.device(args.device),
            )
            print(f"  V-JEPA 2 extraction: {time.time() - t0:.1f}s")

            del vjepa2_model, vjepa2_processor
            torch.cuda.empty_cache()

            save_feature_cache(vjepa2_features, vjepa2_cache)
    else:
        print("\nStep 3: Skipping V-JEPA 2 (--skip-vjepa2)")

    # ------------------------------------------------------------------
    # Step 4: Stage 1 threshold sweep
    # ------------------------------------------------------------------
    print("\nStep 4: Stage 1 BoF threshold sweep...")
    stage1_result = stage1_prefilter(features, keys, copy_pairs)
    sweep = stage1_result["sweep"]
    stage1_threshold = stage1_result["selected_threshold"]

    print(f"\n  {'Threshold':>10s}  {'Recall':>8s}  {'Candidates':>12s}")
    print("  " + "-" * 34)
    for t in STAGE1_THRESHOLDS:
        s = sweep[t]
        marker = " <--" if t == stage1_threshold else ""
        print(f"  {t:>10.2f}  {s['recall']:>8.4f}  {s['n_candidates']:>12d}{marker}")
    print(
        f"\n  Selected threshold: {stage1_threshold:.2f} "
        f"(recall={sweep[stage1_threshold]['recall']:.4f}, "
        f"candidates={sweep[stage1_threshold]['n_candidates']})"
    )

    # ------------------------------------------------------------------
    # Step 5: Build balanced pair set
    # ------------------------------------------------------------------
    print("\nStep 5: Building evaluation pair set...")
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    pos_pairs = set()
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pos_pairs.add((a, b))
    n_pos = len(pos_pairs)

    neg_pairs = set()
    rng = np.random.RandomState(42)
    neg_attempts = 0
    while len(neg_pairs) < n_pos and neg_attempts < n_pos * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs and pair not in neg_pairs:
            # pyrefly: ignore [bad-argument-type]
            neg_pairs.add(pair)
        neg_attempts += 1

    print(
        f"  Evaluation pairs: {n_pos} pos + {len(neg_pairs)} neg = "
        f"{n_pos + len(neg_pairs)} total"
    )

    # ------------------------------------------------------------------
    # Step 6: Run scenarios
    # ------------------------------------------------------------------
    methods = list(DINOV3_METHODS)
    if vjepa2_features is not None:
        methods.extend(VJEPA2_METHODS)

    scenarios = ["standard", "reversal", "scramble", "speed_change"]
    scenario_results = {}

    print("\nStep 6: Running scenarios...")
    for scenario in scenarios:
        result = run_scenario(
            scenario=scenario,
            features=features,
            vjepa2_features=vjepa2_features,
            keys=keys,
            copy_pairs=copy_pairs,
            neg_pairs=neg_pairs,
            stage1_threshold=stage1_threshold,
            all_methods=methods,
            fusion_mode=args.fusion_mode,
        )
        scenario_results[scenario] = result

    # ------------------------------------------------------------------
    # Step 7: Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: TWO-STAGE FUSION PIPELINE (HARD SEPARATION)")
    print("=" * 70)

    print(f"\n  Fusion mode: {args.fusion_mode}")
    print(f"  Stage 1 threshold: {stage1_threshold:.2f}")
    print(f"  Stage 1 features: {', '.join(SEMANTIC_METHODS)}")
    print(
        f"  Stage 2 features: {', '.join(m for m in TEMPORAL_METHODS if m in methods)}"
    )

    print(
        f"\n  {'Scenario':<16s}  {'Semantic AP':>11s}  {'Temporal AP':>11s}  "
        f"{'Rejection':>10s}  {'Attack Pairs':>12s}"
    )
    print("  " + "-" * 65)

    for scenario in scenarios:
        r = scenario_results[scenario]
        sem_ap = r.get("semantic_ap", float("nan"))
        rej = r.get("attack_rejection_rate")
        rej_str = f"{rej:.3f}" if rej is not None else "N/A"
        n_att = r.get("n_attack_pairs", 0)
        print(
            f"  {scenario:<16s}  {sem_ap:>11.4f}  {r['pipeline_ap']:>11.4f}  "
            f"{rej_str:>10s}  {n_att:>12d}"
        )

    # Per-method breakdown for standard scenario
    print(f"\n  Per-method AP (standard scenario):")
    std_pm = scenario_results["standard"].get("per_method", {})
    print(f"    {'--- Semantic (Stage 1) ---'}")
    for m in SEMANTIC_METHODS:
        if m in std_pm:
            print(f"    {METHOD_LABELS[m]:<26s}  AP={std_pm[m]['ap']:.4f}")
    print(f"    {'--- Temporal (Stage 2) ---'}")
    for m in TEMPORAL_METHODS:
        if m in std_pm:
            print(f"    {METHOD_LABELS[m]:<26s}  AP={std_pm[m]['ap']:.4f}")

    # Learned weights (if applicable)
    if args.fusion_mode == "learned":
        for scenario in scenarios:
            lw = scenario_results[scenario].get("learned_weights", {})
            if lw:
                print(f"\n  Learned weights ({scenario}, temporal-only):")
                for m, w in lw.items():
                    print(f"    {METHOD_LABELS[m]:<26s}  w={w:.4f}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nStep 8: Generating figures...")
    plot_threshold_sweep(sweep, fig_dir)
    plot_scenario_results(scenario_results, fig_dir)

    # ------------------------------------------------------------------
    # Step 9: Save results JSON
    # ------------------------------------------------------------------
    results_path = project_root / "datasets" / "vcdb" / "fusion_pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if obj is None:
            return None
        return str(obj)

    output = {
        "config": {
            "fusion_mode": args.fusion_mode,
            "stage1_threshold": stage1_threshold,
            "sample_rate": args.sample_rate,
            "max_frames": args.max_frames,
            "skip_vjepa2": args.skip_vjepa2,
            "n_videos": len(keys),
            "n_methods": len(methods),
            "methods": methods,
        },
        "stage1_sweep": make_serializable(sweep),
        "scenarios": make_serializable(scenario_results),
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

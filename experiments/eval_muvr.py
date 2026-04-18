#!/usr/bin/env python3
"""MUVR Multi-Level Video Correspondence Evaluation.

Evaluates whether DINOv3 temporal fingerprints and V-JEPA 2 can distinguish
different correspondence levels in the MUVR benchmark (NeurIPS 2025).

MUVR provides pairwise video relationships at multiple levels:
- news partition: copy, event, copy+event, independent
- dance partition: reasoning (=action match), independent

This tests our core hypothesis across correspondence types:
- Copy detection: bag-of-frames/Chamfer should dominate (same as VCDB)
- Event matching: temporal methods may help (same event, different recording)
- Action matching: V-JEPA 2 may help (same dance, different performer)

Pipeline:
1. Load MUVR annotations (topics, videos, relationships)
2. Discover video files from extracted tar archives
3. Extract DINOv3 + V-JEPA 2 features per video
4. Compute 6-method pairwise similarities for annotated pairs
5. Evaluate AP/AUC per correspondence level

Usage:
    bash scripts/download_muvr.sh           # Download first
    python experiments/eval_muvr.py          # Run evaluation
    python experiments/eval_muvr.py --partition news --max-topics 10
"""

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance, dtw_distance_batch
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32
VJEPA2_SPATIAL = 256

# MUVR videos are pre-processed at 336px/6fps, but we re-sample for our models
DINOV3_FPS = 3.0
DINOV3_MAX_RES = 518
DINOV3_MAX_FRAMES = 150  # cap for long clips

METHODS = [
    "bag_of_frames",
    "chamfer",
    "temporal_derivative",
    "attention_trajectory",
    "vjepa2_bag_of_tokens",
    "vjepa2_temporal_residual",
]

METHOD_COLORS = {
    "bag_of_frames": "#e74c3c",
    "chamfer": "#1abc9c",
    "temporal_derivative": "#2ecc71",
    "attention_trajectory": "#3498db",
    "vjepa2_bag_of_tokens": "#9b59b6",
    "vjepa2_temporal_residual": "#f39c12",
}

METHOD_LABELS = {
    "bag_of_frames": "Bag of\nFrames",
    "chamfer": "Chamfer",
    "temporal_derivative": "Temporal\nDerivative",
    "attention_trajectory": "Attention\nTrajectory",
    "vjepa2_bag_of_tokens": "V-JEPA 2\nBag of Tokens",
    "vjepa2_temporal_residual": "V-JEPA 2\nTemporal Res.",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MUVRVideo:
    """A video entry from MUVR."""

    video_id: int
    video_name: str  # Bilibili BV ID
    topic_id: int
    is_query: bool
    video_path: str  # resolved path to .mp4


@dataclass
class MUVRPair:
    """An annotated pair from MUVR."""

    video1_id: int
    video2_id: int
    topic_id: int
    relationship: str  # "copy", "event", "copy and event", "reasoning", "independent"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_video_file(
    video_name: str, timestamp: int, video_dirs: list[Path]
) -> str | None:
    """Find a video file across possible directory layouts.

    MUVR videos are split into 2-minute segments with filenames like:
    - BV..._00.00.00-00.02.00.mp4 (timestamp=0)
    - BV..._00.02.00-00.04.00.mp4 (timestamp=120)
    Or unsegmented: BV....mp4 (short videos, timestamp=0)

    Args:
        video_name: Bilibili BV ID (e.g., "BV13bzrYkEUi").
        timestamp: Start time in seconds from annotations.
        video_dirs: Directories to search.
    """
    # Convert timestamp to HH.MM.SS format used in filenames
    h = timestamp // 3600
    m = (timestamp % 3600) // 60
    s = timestamp % 60
    ts_prefix = f"{h:02d}.{m:02d}.{s:02d}"

    # Build candidate filenames in priority order
    candidates = []

    # Segmented: BV..._HH.MM.SS-*.mp4
    segment_pattern = f"{video_name}_{ts_prefix}-*.mp4"

    # Unsegmented: BV....mp4 (only valid for timestamp=0)
    if timestamp == 0:
        candidates.append(f"{video_name}.mp4")

    for vdir in video_dirs:
        # Try segment pattern first (glob for end timestamp)
        matches = list(vdir.glob(segment_pattern))
        if not matches:
            matches = list(vdir.glob(f"**/{segment_pattern}"))
        if matches:
            return str(matches[0])

        # Try unsegmented candidates
        for candidate in candidates:
            path = vdir / candidate
            if path.exists():
                return str(path)
            # Check subdirectories
            sub_matches = list(vdir.glob(f"**/{candidate}"))
            if sub_matches:
                return str(sub_matches[0])

    return None


def load_annotations(
    ann_dir: Path,
    video_dirs: list[Path],
    max_topics: int | None = None,
) -> tuple[dict[int, MUVRVideo], list[MUVRPair], dict[int, str]]:
    """Load MUVR annotations and resolve video paths.

    Args:
        ann_dir: Path to annotations/{partition}/.
        video_dirs: List of directories to search for video files.
        max_topics: If set, limit to the first N topics.

    Returns:
        Tuple of (videos_by_id, pairs, topics_by_id).
    """
    # Load topics
    with open(ann_dir / "topics.json") as f:
        topics_raw = json.load(f)
    topics_by_id = {t["id"]: t.get("description", t.get("name", str(t["id"])))
                    for t in topics_raw}

    if max_topics:
        keep_topic_ids = set(sorted(topics_by_id.keys())[:max_topics])
        topics_by_id = {k: v for k, v in topics_by_id.items() if k in keep_topic_ids}
    else:
        keep_topic_ids = None

    # Load videos
    with open(ann_dir / "videos.json") as f:
        videos_raw = json.load(f)

    videos_by_id: dict[int, MUVRVideo] = {}
    missing_videos = 0

    for v in videos_raw:
        if keep_topic_ids and v["topic_id"] not in keep_topic_ids:
            continue

        video_path = find_video_file(v["video_name"], v.get("timestamp", 0), video_dirs)
        if video_path is None:
            missing_videos += 1
            continue

        videos_by_id[v["id"]] = MUVRVideo(
            video_id=v["id"],
            video_name=v["video_name"],
            topic_id=v["topic_id"],
            is_query=bool(v.get("is_query", 0)),
            video_path=video_path,
        )

    if missing_videos:
        print(f"  Warning: {missing_videos} videos not found on disk")

    # Load relationships
    with open(ann_dir / "relationships.json") as f:
        rels_raw = json.load(f)

    pairs: list[MUVRPair] = []
    for r in rels_raw:
        if keep_topic_ids and r.get("topic1_id", r.get("topic_id")) not in keep_topic_ids:
            continue
        # Only keep pairs where both videos are available
        if r["video1_id"] not in videos_by_id or r["video2_id"] not in videos_by_id:
            continue
        pairs.append(MUVRPair(
            video1_id=r["video1_id"],
            video2_id=r["video2_id"],
            topic_id=r.get("topic1_id", r.get("topic_id", 0)),
            relationship=r["relationship"],
        ))

    return videos_by_id, pairs, topics_by_id


# ---------------------------------------------------------------------------
# Video clip extraction
# ---------------------------------------------------------------------------


def load_clip(
    video_path: str,
    target_fps: float = 3.0,
    max_resolution: int = 518,
    max_frames: int = 150,
) -> list[np.ndarray]:
    """Extract frames from a full video file.

    MUVR videos are pre-cropped to ~2 minutes, so we load the entire clip.

    Args:
        video_path: Path to video file.
        target_fps: Target frame rate for extraction.
        max_resolution: Maximum height (preserves aspect ratio).
        max_frames: Maximum number of frames to extract.

    Returns:
        List of RGB frames as numpy arrays.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [bad-argument-type]
    time_base = float(stream.time_base)

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
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)
            next_sample += sample_interval

            if len(frames) >= max_frames:
                break

        frame_count += 1

    container.close()
    return frames


def load_clip_vjepa2(
    video_path: str,
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Samples uniformly across the full video duration.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base) if stream.duration else 60.0

    target_fps = VJEPA2_NUM_FRAMES / max(duration, 1.0)

    container.close()

    frames = load_clip(
        video_path,
        target_fps=target_fps,
        max_resolution=max_resolution,
        max_frames=VJEPA2_NUM_FRAMES + 10,
    )

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])

    return frames[:VJEPA2_NUM_FRAMES]


# ---------------------------------------------------------------------------
# Feature caching
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
# Feature extraction
# ---------------------------------------------------------------------------


def extract_dinov3_features(
    encoder: DINOv3Encoder,
    videos: dict[int, MUVRVideo],
) -> dict[int, dict]:
    """Extract DINOv3 features for all videos.

    Returns:
        Dict mapping video_id -> {
            'embeddings': (T, 1024),
            'centroids': (T, 2),
            'mean_emb': (1024,),
        }
    """
    features = {}
    failed = 0
    sorted_videos = sorted(videos.items())

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Prefetch first clip
        future = pool.submit(
            load_clip, sorted_videos[0][1].video_path,
            DINOV3_FPS, DINOV3_MAX_RES, DINOV3_MAX_FRAMES,
        )

        for idx, (vid, video) in enumerate(tqdm(sorted_videos, desc="DINOv3 features")):
            try:
                frames = future.result()

                # Prefetch next while GPU works
                if idx + 1 < len(sorted_videos):
                    next_video = sorted_videos[idx + 1][1]
                    future = pool.submit(
                        load_clip, next_video.video_path,
                        DINOV3_FPS, DINOV3_MAX_RES, DINOV3_MAX_FRAMES,
                    )

                if len(frames) < 3:
                    failed += 1
                    continue

                emb = encoder.encode_frames(frames)
                centroids = encoder.get_attention_centroids(frames)
                mean_emb = F.normalize(emb.mean(dim=0), dim=0)

                features[vid] = {
                    "embeddings": emb,
                    "centroids": centroids,
                    "mean_emb": mean_emb,
                }
            except Exception:
                failed += 1
                # Ensure prefetch is still queued for next iteration
                if idx + 1 < len(sorted_videos):
                    next_video = sorted_videos[idx + 1][1]
                    future = pool.submit(
                        load_clip, next_video.video_path,
                        DINOV3_FPS, DINOV3_MAX_RES, DINOV3_MAX_FRAMES,
                    )

    print(f"  DINOv3: {len(features)}/{len(videos)} ({failed} failed)")
    return features


def build_temporal_masks(
    n_context_steps: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context/target masks for V-JEPA 2 temporal prediction."""
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)
    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)
    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def extract_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    videos: dict[int, MUVRVideo],
    device: torch.device,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for all videos.

    Returns:
        Dict mapping video_id -> {'mean_emb': ..., 'temporal_residual': ...}
    """
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0
    sorted_videos = sorted(videos.items())

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Prefetch first clip
        future = pool.submit(load_clip_vjepa2, sorted_videos[0][1].video_path)

        for idx, (vid, video) in enumerate(tqdm(sorted_videos, desc="V-JEPA 2 features")):
            try:
                frames = future.result()

                # Prefetch next while GPU works
                if idx + 1 < len(sorted_videos):
                    next_video = sorted_videos[idx + 1][1]
                    future = pool.submit(load_clip_vjepa2, next_video.video_path)

                if len(frames) < VJEPA2_NUM_FRAMES:
                    failed += 1
                    continue

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

                features[vid] = {
                    "mean_emb": mean_emb.cpu(),
                    "temporal_residual": residual.cpu(),
                }
            except Exception:
                failed += 1
                # Ensure prefetch is still queued for next iteration
                if idx + 1 < len(sorted_videos):
                    next_video = sorted_videos[idx + 1][1]
                    future = pool.submit(load_clip_vjepa2, next_video.video_path)

    print(f"  V-JEPA 2: {len(features)}/{len(videos)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_pair_similarities(
    pair: MUVRPair,
    dinov3_features: dict[int, dict],
    vjepa2_features: dict[int, dict],
    deriv_fps: dict[int, object],
    traj_fps: dict[int, object],
    deriv_fp: TemporalDerivativeFingerprint,
    traj_fp: TrajectoryFingerprint,
) -> dict[str, float] | None:
    """Compute all 6 similarity scores for one pair.

    Returns dict of method -> score, or None if features missing.
    """
    a, b = pair.video1_id, pair.video2_id

    if a not in dinov3_features or b not in dinov3_features:
        return None

    fa = dinov3_features[a]
    fb = dinov3_features[b]

    scores: dict[str, float] = {}

    # Bag-of-frames
    scores["bag_of_frames"] = float(torch.dot(fa["mean_emb"], fb["mean_emb"]).item())

    # Chamfer
    sim_matrix = torch.mm(fa["embeddings"], fb["embeddings"].t())
    max_ab = sim_matrix.max(dim=1).values.mean().item()
    max_ba = sim_matrix.max(dim=0).values.mean().item()
    scores["chamfer"] = (max_ab + max_ba) / 2

    # Temporal derivative DTW
    if a in deriv_fps and b in deriv_fps:
        # pyrefly: ignore [bad-argument-type]
        scores["temporal_derivative"] = deriv_fp.compare(deriv_fps[a], deriv_fps[b])

    # Attention trajectory DTW
    if a in traj_fps and b in traj_fps:
        # pyrefly: ignore [bad-argument-type]
        scores["attention_trajectory"] = traj_fp.compare(traj_fps[a], traj_fps[b])

    # V-JEPA 2 methods
    if a in vjepa2_features and b in vjepa2_features:
        va = vjepa2_features[a]
        vb = vjepa2_features[b]

        scores["vjepa2_bag_of_tokens"] = float(
            torch.dot(va["mean_emb"], vb["mean_emb"]).item()
        )

        res_dist = dtw_distance(
            va["temporal_residual"], vb["temporal_residual"], normalize=True
        )
        scores["vjepa2_temporal_residual"] = float(
            torch.exp(torch.tensor(-res_dist)).item()
        )

    return scores


def compute_all_similarities(
    pairs: list[MUVRPair],
    dinov3_features: dict[int, dict],
    vjepa2_features: dict[int, dict],
) -> dict[str, dict[str, tuple[list[float], list[int]]]]:
    """Compute pairwise similarities for all annotated pairs.

    Uses batched DTW on GPU for temporal derivative, attention trajectory,
    and V-JEPA 2 temporal residual comparisons.

    Returns:
        Dict mapping relationship_type -> method_name -> (scores, labels).
        "independent" pairs get label=0, all other relationships get label=1.
        Additionally, an "_all" key aggregates all pairs with binary
        match/non-match labels.
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for vid in dinov3_features:
        deriv_fps[vid] = deriv_fp.compute_fingerprint(
            dinov3_features[vid]["embeddings"]
        )
        traj_fps[vid] = traj_fp.compute_fingerprint(
            dinov3_features[vid]["centroids"]
        )

    # Collect relationship types
    rel_types = sorted(set(p.relationship for p in pairs))
    print(f"  Relationship types: {rel_types}")
    print(f"  Total annotated pairs: {len(pairs)}")

    # --- Pass 1: non-DTW similarities + collect DTW inputs ---
    print("  Computing BoF/Chamfer and collecting DTW pairs...")

    # Track per-pair results for bucketing
    pair_bof = []
    pair_chamfer = []
    pair_meta = []  # (pair_index, is_match, relationship)
    valid_mask = []  # which pairs had features

    # DTW input collectors
    deriv_pairs_a, deriv_pairs_b = [], []
    traj_pairs_a, traj_pairs_b = [], []
    deriv_valid = []  # indices into pair_meta for deriv DTW
    traj_valid = []   # indices into pair_meta for traj DTW
    res_pairs_a, res_pairs_b = [], []
    res_valid = []    # indices into pair_meta for vjepa2 residual
    bot_scores = []   # bag-of-tokens (no DTW)
    bot_valid = []

    skipped = 0
    for pair in tqdm(pairs, desc="  Pass 1: BoF/Chamfer"):
        a, b = pair.video1_id, pair.video2_id
        if a not in dinov3_features or b not in dinov3_features:
            skipped += 1
            continue

        idx = len(pair_meta)
        is_match = 0 if pair.relationship == "independent" else 1
        pair_meta.append((idx, is_match, pair.relationship))

        fa = dinov3_features[a]
        fb = dinov3_features[b]

        # Bag-of-frames
        pair_bof.append(float(torch.dot(fa["mean_emb"], fb["mean_emb"]).item()))

        # Chamfer
        sim_matrix = torch.mm(fa["embeddings"], fb["embeddings"].t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        pair_chamfer.append((max_ab + max_ba) / 2)

        # Temporal derivative DTW inputs
        if a in deriv_fps and b in deriv_fps:
            deriv_pairs_a.append(deriv_fps[a])
            deriv_pairs_b.append(deriv_fps[b])
            deriv_valid.append(idx)

        # Attention trajectory DTW inputs
        if a in traj_fps and b in traj_fps:
            traj_pairs_a.append(traj_fps[a])
            traj_pairs_b.append(traj_fps[b])
            traj_valid.append(idx)

        # V-JEPA 2 inputs
        if a in vjepa2_features and b in vjepa2_features:
            va = vjepa2_features[a]
            vb = vjepa2_features[b]
            bot_scores.append(float(torch.dot(va["mean_emb"], vb["mean_emb"]).item()))
            bot_valid.append(idx)
            res_pairs_a.append(va["temporal_residual"])
            res_pairs_b.append(vb["temporal_residual"])
            res_valid.append(idx)

    computed = len(pair_meta)
    print(f"  Computed: {computed}, Skipped: {skipped}")

    # --- Pass 2: batched DTW ---
    print(f"  Batching temporal derivative DTW ({len(deriv_pairs_a)} pairs)...")
    deriv_dists = dtw_distance_batch(deriv_pairs_a, deriv_pairs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).tolist()

    print(f"  Batching attention trajectory DTW ({len(traj_pairs_a)} pairs)...")
    traj_dists = dtw_distance_batch(traj_pairs_a, traj_pairs_b, normalize=True)
    traj_sims = torch.exp(-traj_dists * 5).tolist()

    if res_pairs_a:
        print(f"  Batching V-JEPA 2 temporal residual DTW ({len(res_pairs_a)} pairs)...")
        res_dists = dtw_distance_batch(res_pairs_a, res_pairs_b, normalize=True)
        res_sims = torch.exp(-res_dists).tolist()
    else:
        res_sims = []

    # Build per-pair score dicts
    pair_scores_list: list[dict[str, float]] = [{} for _ in range(computed)]
    for i in range(computed):
        pair_scores_list[i]["bag_of_frames"] = pair_bof[i]
        pair_scores_list[i]["chamfer"] = pair_chamfer[i]

    for local_i, global_i in enumerate(deriv_valid):
        pair_scores_list[global_i]["temporal_derivative"] = deriv_sims[local_i]

    for local_i, global_i in enumerate(traj_valid):
        pair_scores_list[global_i]["attention_trajectory"] = traj_sims[local_i]

    for local_i, global_i in enumerate(bot_valid):
        pair_scores_list[global_i]["vjepa2_bag_of_tokens"] = bot_scores[local_i]

    for local_i, global_i in enumerate(res_valid):
        pair_scores_list[global_i]["vjepa2_temporal_residual"] = res_sims[local_i]

    # --- Pass 3: bucket into relationship types ---
    all_scores: dict[str, dict[str, tuple[list[float], list[int]]]] = {}
    all_scores["_all"] = {m: ([], []) for m in METHODS}
    for rel in rel_types:
        if rel != "independent":
            all_scores[rel] = {m: ([], []) for m in METHODS}

    for idx, is_match, relationship in pair_meta:
        scores = pair_scores_list[idx]

        # Add to aggregate
        for method in METHODS:
            if method in scores:
                all_scores["_all"][method][0].append(scores[method])
                all_scores["_all"][method][1].append(is_match)

        # Add to per-relationship type (positive=this type vs independent)
        if relationship != "independent":
            for method in METHODS:
                if method in scores:
                    all_scores[relationship][method][0].append(scores[method])
                    all_scores[relationship][method][1].append(1)
        else:
            # Independent pairs contribute as negatives to each type bucket
            for rel in rel_types:
                if rel != "independent" and rel in all_scores:
                    for method in METHODS:
                        if method in scores:
                            all_scores[rel][method][0].append(scores[method])
                            all_scores[rel][method][1].append(0)

    return all_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_scores(
    scores_list: list[float], labels_list: list[int]
) -> dict:
    """Compute AP, AUC, and gap for a set of scores."""
    scores = np.array(scores_list)
    labels = np.array(labels_list)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"ap": float("nan"), "auc": float("nan"),
                "n_pos": n_pos, "n_neg": n_neg}

    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    same_mean = float(scores[labels == 1].mean())
    diff_mean = float(scores[labels == 0].mean())

    return {
        "ap": ap, "auc": auc,
        "n_pos": n_pos, "n_neg": n_neg,
        "same_mean": same_mean, "diff_mean": diff_mean,
        "gap": same_mean - diff_mean,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrimination(
    results: dict[str, dict[str, dict]],
    partition: str,
    fig_dir: Path,
):
    """Generate AP bar chart, one group per relationship type."""
    rel_types = [k for k in results.keys() if k != "_all"]
    all_types = ["_all"] + sorted(rel_types)

    methods = [m for m in METHODS if m in results["_all"]]
    n_methods = len(methods)
    n_groups = len(all_types)

    fig, ax = plt.subplots(figsize=(3 * n_groups + 2, 6))

    bar_width = 0.8 / n_methods
    group_labels = ["All"] + [t.replace("copy and event", "copy+event").title()
                              for t in sorted(rel_types)]

    for i, method in enumerate(methods):
        positions = []
        values = []
        for j, rel in enumerate(all_types):
            if method in results.get(rel, {}):
                r = results[rel][method]
                positions.append(j + i * bar_width)
                values.append(r["ap"] if not np.isnan(r.get("ap", float("nan"))) else 0)

        ax.bar(
            positions, values,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.3,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        for pos, val in zip(positions, values):
            if val > 0:
                ax.text(pos, val + 0.005, f"{val:.2f}", ha="center",
                        va="bottom", fontsize=7, rotation=45)

    ax.set_xticks([j + bar_width * (n_methods - 1) / 2 for j in range(n_groups)])
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("Average Precision", fontsize=12)
    ax.set_title(f"MUVR {partition.title()}: Discrimination by Correspondence Level",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    path = fig_dir / f"muvr_{partition}_discrimination.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_similarity_distributions(
    all_scores: dict[str, dict[str, tuple[list[float], list[int]]]],
    partition: str,
    fig_dir: Path,
):
    """Generate similarity distributions for the aggregate (_all) scores."""
    methods = [m for m in METHODS if m in all_scores["_all"]
               and all_scores["_all"][m][0]]
    n_methods = len(methods)
    ncols = 3 if n_methods > 4 else 2
    nrows = (n_methods + ncols - 1) // ncols

    color_match = "#3498db"
    color_indep = "#e74c3c"

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, method in zip(axes, methods):
        scores_list, labels_list = all_scores["_all"][method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)

        match = scores[labels == 1]
        indep = scores[labels == 0]

        lo = min(scores.min(), 0)
        hi = max(scores.max(), 1)
        bins = np.linspace(lo, hi, 40)

        ax.hist(match, bins=bins, alpha=0.6, color=color_match,
                label=f"Match (n={len(match)})", density=True)
        ax.hist(indep, bins=bins, alpha=0.6, color=color_indep,
                label=f"Independent (n={len(indep)})", density=True)
        ax.set_xlabel("Similarity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        title = METHOD_LABELS[method].replace("\n", " ")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        if len(match) > 0:
            ax.axvline(match.mean(), color=color_match, linestyle="--", alpha=0.8)
        if len(indep) > 0:
            ax.axvline(indep.mean(), color=color_indep, linestyle="--", alpha=0.8)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"MUVR {partition.title()}: Similarity Distributions\n"
        "(Match vs Independent)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / f"muvr_{partition}_similarity_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MUVR Multi-Level Video Correspondence Evaluation"
    )
    parser.add_argument(
        "--muvr-dir", type=str, default="datasets/muvr",
        help="Path to MUVR dataset directory",
    )
    parser.add_argument(
        "--partition", type=str, default="dance",
        choices=["dance", "news", "instance", "others", "region"],
        help="Which partition to evaluate",
    )
    parser.add_argument(
        "--max-topics", type=int, default=None,
        help="Limit to first N topics (for quick testing)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass cache load (still saves after extraction)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    muvr_dir = project_root / args.muvr_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    ann_dir = muvr_dir / "annotations" / args.partition
    video_dir = muvr_dir / "videos" / args.partition

    print("=" * 70)
    print(f"MUVR: {args.partition.upper()} PARTITION EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load annotations
    # ------------------------------------------------------------------
    print("\nStep 1: Loading annotations...")
    if not ann_dir.exists():
        print(f"  ERROR: Annotations not found at {ann_dir}")
        print("  Run: bash scripts/download_muvr.sh")
        return
    if not video_dir.exists():
        print(f"  ERROR: Videos not found at {video_dir}")
        print("  Run: bash scripts/download_muvr.sh")
        return

    # Search for videos in the partition directory and any subdirectories
    video_search_dirs = [video_dir]
    for subdir in video_dir.iterdir():
        if subdir.is_dir():
            video_search_dirs.append(subdir)

    videos, pairs, topics = load_annotations(
        ann_dir, video_search_dirs, max_topics=args.max_topics,
    )

    # Count relationships
    rel_counts: dict[str, int] = defaultdict(int)
    for p in pairs:
        rel_counts[p.relationship] += 1

    print(f"  Videos found: {len(videos)}")
    print(f"  Topics: {len(topics)}")
    print(f"  Annotated pairs: {len(pairs)}")
    for rel, count in sorted(rel_counts.items()):
        print(f"    {rel}: {count}")

    if not pairs:
        print("\nERROR: No annotated pairs found.")
        return

    # ------------------------------------------------------------------
    # Step 2: Extract DINOv3 features (with caching)
    # ------------------------------------------------------------------
    dinov3_cache = muvr_dir / "cache" / f"dinov3_{args.partition}_features.pt"
    dinov3_features = None if args.no_cache else load_feature_cache(dinov3_cache)

    if dinov3_features is None:
        print("\nStep 2: Loading DINOv3 encoder...")
        encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

        print("  Extracting DINOv3 features...")
        t0 = time.time()
        dinov3_features = extract_dinov3_features(encoder, videos)
        print(f"  DINOv3 time: {time.time() - t0:.1f}s")

        save_feature_cache(dinov3_features, dinov3_cache)

        del encoder
        torch.cuda.empty_cache()
    else:
        print("\nStep 2: DINOv3 features loaded from cache")

    # ------------------------------------------------------------------
    # Step 3: Extract V-JEPA 2 features (with caching)
    # ------------------------------------------------------------------
    vjepa2_cache = muvr_dir / "cache" / f"vjepa2_{args.partition}_features.pt"
    vjepa2_features = None if args.no_cache else load_feature_cache(vjepa2_cache)

    if vjepa2_features is None:
        print("\nStep 3: Loading V-JEPA 2 model...")
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
            vjepa2_model, vjepa2_processor, videos, torch.device(args.device),
        )
        print(f"  V-JEPA 2 time: {time.time() - t0:.1f}s")

        save_feature_cache(vjepa2_features, vjepa2_cache)

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()
    else:
        print("\nStep 3: V-JEPA 2 features loaded from cache")

    # ------------------------------------------------------------------
    # Step 4: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 4: Computing pairwise similarities...")
    t0 = time.time()
    all_scores = compute_all_similarities(pairs, dinov3_features, vjepa2_features)
    print(f"  Similarity time: {time.time() - t0:.1f}s")

    # Dump pair scores for bootstrap CI computation
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores["_all"].items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }
    pair_path = muvr_dir / f"pair_scores_{args.partition}.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair scores saved to {pair_path}")

    # ------------------------------------------------------------------
    # Step 5: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: MUVR {args.partition.upper()}")
    print("=" * 70)

    results: dict[str, dict[str, dict]] = {}

    for rel_type in sorted(all_scores.keys()):
        print(f"\n  --- {rel_type} ---")
        results[rel_type] = {}

        for method in METHODS:
            if method not in all_scores[rel_type]:
                continue
            scores_list, labels_list = all_scores[rel_type][method]
            if not scores_list:
                continue

            r = evaluate_scores(scores_list, labels_list)
            results[rel_type][method] = r

            if not np.isnan(r["ap"]):
                print(
                    f"    {method:<28s}  AP={r['ap']:.4f}  AUC={r['auc']:.4f}  "
                    f"gap={r['gap']:+.4f}  (pos={r['n_pos']}, neg={r['n_neg']})"
                )

    print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # Step 6: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_discrimination(results, args.partition, fig_dir)
    plot_similarity_distributions(all_scores, args.partition, fig_dir)

    # Summary
    print(f"\nSummary:")
    print(f"  Partition: {args.partition}")
    print(f"  Videos: {len(videos)}")
    print(f"  Pairs evaluated: {len(pairs)}")
    print(f"  Relationship types: {sorted(rel_counts.keys())}")

    print("\nDone.")


if __name__ == "__main__":
    main()

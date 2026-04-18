#!/usr/bin/env python3
"""Ensemble Metric Learning (VCDB + HDD).

Learns optimal weights for a linear ensemble of similarity methods:
    Sim = w1*BoF + w2*Chamfer + w3*TempDeriv + w4*AttnTraj
          [+ w5*VJEPA2_BoT + w6*VJEPA2_Res]

Uses logistic regression on labeled pairs from VCDB (copy detection) and
HDD (maneuver discrimination). Shows that the ensemble outperforms any
individual method on both tasks simultaneously.

Features:
- 5-fold stratified cross-validation
- Leave-one-dataset-out evaluation
- Grid search baseline for validation
- Checkpoints per-pair similarities to JSON for fast re-runs

Usage:
    python experiments/eval_ensemble.py                        # Full run
    python experiments/eval_ensemble.py --skip-hdd             # VCDB only
    python experiments/eval_ensemble.py --skip-vjepa2          # DINOv3 only
    python experiments/eval_ensemble.py --reuse-checkpoints    # Skip feature extraction
"""

import argparse
import json
import os
import time
import zoneinfo
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from video_retrieval.fingerprints import TemporalDerivativeFingerprint, TrajectoryFingerprint
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

METHOD_COLORS = {
    "bag_of_frames": "#e74c3c",
    "chamfer": "#1abc9c",
    "temporal_derivative": "#2ecc71",
    "attention_trajectory": "#3498db",
    "vjepa2_bag_of_tokens": "#9b59b6",
    "vjepa2_temporal_residual": "#f39c12",
    "ensemble": "#2c3e50",
}

METHOD_LABELS = {
    "bag_of_frames": "Bag of Frames",
    "chamfer": "Chamfer",
    "temporal_derivative": "Temporal Deriv.",
    "attention_trajectory": "Attn. Trajectory",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "V-JEPA 2 Res.",
    "ensemble": "Ensemble",
}

MANEUVER_NAMES = {
    1: "intersection_passing",
    2: "left_turn",
    3: "right_turn",
}


# ---------------------------------------------------------------------------
# VCDB loading (from eval_vcdb.py)
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
# DINOv3 feature extraction (from eval_vcdb.py)
# ---------------------------------------------------------------------------

def extract_vcdb_dinov3_features(
    encoder: DINOv3Encoder,
    vid_base_dir: str,
    video_relpaths: list[str],
    sample_rate: int = 10,
    max_frames: int = 100,
) -> dict[str, dict]:
    """Extract DINOv3 features for VCDB videos."""
    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc="Extracting DINOv3 features (VCDB)"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames, fps = load_video(path, sample_rate=sample_rate,
                                     max_frames=max_frames, max_resolution=518)
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

    print(f"  Extracted: {len(features)}/{len(video_relpaths)} "
          f"({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# V-JEPA 2 feature extraction (from eval_vcdb_reversal.py)
# ---------------------------------------------------------------------------

def build_temporal_masks(
    n_context_steps: int, device: torch.device,
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


def extract_vcdb_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    vid_base_dir: str,
    video_relpaths: list[str],
    device: torch.device,
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for VCDB videos."""
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc="V-JEPA 2 features (VCDB)"):
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

    print(f"  V-JEPA 2 (VCDB): {len(features)}/{len(video_relpaths)} "
          f"({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# HDD data loading (from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------

@dataclass
class ManeuverSegment:
    """A contiguous maneuver segment from one HDD session."""
    session_id: str
    label: int
    start_frame: int
    end_frame: int
    lat: float
    lng: float
    video_path: str
    video_start_unix: float


def parse_video_start_time(video_filename: str) -> float:
    """Parse video start unix timestamp from filename."""
    match = re.match(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", video_filename
    )
    if not match:
        raise ValueError(f"Cannot parse timestamp from {video_filename}")
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return dt.timestamp()


def discover_sessions(hdd_dir: Path) -> dict[str, dict]:
    """Discover all valid HDD sessions."""
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
    """Load GPS data from rtk_pos.csv."""
    data = np.genfromtxt(
        gps_path, delimiter=",", skip_header=1, usecols=(0, 2, 3), dtype=np.float64
    )
    timestamps = data[:, 0]
    latitudes = data[:, 1]
    longitudes = data[:, 2]
    return timestamps, latitudes, longitudes


def _add_segment(
    segments: list[ManeuverSegment],
    session_id: str,
    label: int,
    start_frame: int,
    end_frame: int,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    video_path: str,
    video_start_unix: float,
):
    """Add a maneuver segment with GPS midpoint lookup."""
    mid_frame = (start_frame + end_frame) // 2
    mid_ts = video_start_unix + mid_frame / 3.0
    gps_idx = np.searchsorted(gps_timestamps, mid_ts)
    # pyrefly: ignore [no-matching-overload]
    gps_idx = min(gps_idx, len(gps_timestamps) - 1)
    lat = gps_lats[gps_idx]
    lng = gps_lngs[gps_idx]
    if np.isnan(lat) or np.isnan(lng) or abs(lat) < 1 or abs(lng) < 1:
        return
    segments.append(
        ManeuverSegment(
            session_id=session_id, label=label,
            start_frame=start_frame, end_frame=end_frame,
            # pyrefly: ignore [bad-argument-type]
            lat=lat,
            # pyrefly: ignore [bad-argument-type]
            lng=lng,
            video_path=video_path,
            video_start_unix=video_start_unix,
        )
    )


def extract_maneuver_segments(
    session_id: str,
    labels: np.ndarray,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    video_path: str,
    video_start_unix: float,
    target_labels: tuple[int, ...] = (1, 2, 3),
) -> list[ManeuverSegment]:
    """Extract contiguous maneuver segments and their GPS midpoints."""
    segments: list[ManeuverSegment] = []
    in_segment = False
    seg_start = 0
    seg_label = 0

    for i in range(len(labels)):
        if labels[i] in target_labels:
            if not in_segment or labels[i] != seg_label:
                if in_segment:
                    _add_segment(
                        segments, session_id, seg_label, seg_start, i,
                        gps_timestamps, gps_lats, gps_lngs,
                        video_path, video_start_unix,
                    )
                seg_start = i
                seg_label = int(labels[i])
                in_segment = True
        else:
            if in_segment:
                _add_segment(
                    segments, session_id, seg_label, seg_start, i,
                    gps_timestamps, gps_lats, gps_lngs,
                    video_path, video_start_unix,
                )
                in_segment = False

    if in_segment:
        _add_segment(
            segments, session_id, seg_label, seg_start, len(labels),
            gps_timestamps, gps_lats, gps_lngs,
            video_path, video_start_unix,
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
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(coords)
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
# HDD clip + feature extraction (from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------

def load_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> list[np.ndarray]:
    """Extract frames from a video clip using time-based seeking."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [bad-argument-type]
    time_base = float(stream.time_base)

    # pyrefly: ignore [no-matching-overload]
    seek_sec = max(0, start_sec - 1.0)
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


def load_clip_vjepa2(
    video_path: str,
    start_sec: float,
    end_sec: float,
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2 from a clip."""
    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0
    target_fps = VJEPA2_NUM_FRAMES / duration
    frames = load_clip(video_path, start_sec, end_sec,
                       target_fps=target_fps, max_resolution=max_resolution)
    if len(frames) == 0:
        raise ValueError("No frames extracted")
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])
    return frames[:VJEPA2_NUM_FRAMES]


def extract_hdd_dinov3_features(
    encoder: DINOv3Encoder,
    segments: list[ManeuverSegment],
    context_sec: float = 3.0,
) -> dict[int, dict]:
    """Extract DINOv3 features for HDD maneuver segments."""
    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting DINOv3 features (HDD)")):
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)
        try:
            frames = load_clip(seg.video_path, start_sec, end_sec,
                               target_fps=3.0, max_resolution=518)
            if len(frames) < 3:
                failed += 1
                continue
            emb = encoder.encode_frames(frames)
            centroids = encoder.get_attention_centroids(frames)
            mean_emb = F.normalize(emb.mean(dim=0), dim=0)
            features[i] = {
                "embeddings": emb,
                "centroids": centroids,
                "mean_emb": mean_emb,
            }
        except Exception:
            failed += 1
            continue

    print(f"  DINOv3 (HDD): {len(features)}/{len(segments)} ({failed} failed)")
    return features


def extract_hdd_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for HDD maneuver segments."""
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="V-JEPA 2 features (HDD)")):
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)
        try:
            frames = load_clip_vjepa2(seg.video_path, start_sec, end_sec)
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

            features[i] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  V-JEPA 2 (HDD): {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Similarity computation helpers
# ---------------------------------------------------------------------------

def compute_vcdb_similarities(
    features: dict[str, dict],
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
    methods: list[str],
    vjepa2_features: dict[str, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute per-pair similarity vectors for VCDB.

    Returns:
        X: (n_pairs, n_methods) feature matrix
        y: (n_pairs,) labels
        pair_ids: list of "a|b" strings for identification
    """
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints
    print("  Pre-computing VCDB fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for k in tqdm(keys, desc="  Fingerprints", leave=False):
        deriv_fps[k] = deriv_fp.compute_fingerprint(features[k]["embeddings"])
        traj_fps[k] = traj_fp.compute_fingerprint(features[k]["centroids"])

    # Build pair set
    pairs_to_compute = set()
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pairs_to_compute.add((a, b))

    n_pos = len(pairs_to_compute)
    rng = np.random.RandomState(42)
    neg_count = 0
    neg_attempts = 0
    while neg_count < n_pos and neg_attempts < n_pos * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs and pair not in pairs_to_compute:
            # pyrefly: ignore [bad-argument-type]
            pairs_to_compute.add(pair)
            neg_count += 1
        neg_attempts += 1

    print(f"  VCDB pairs: {len(pairs_to_compute)} ({n_pos} pos + {neg_count} neg)")

    X_rows = []
    y_list = []
    pair_ids = []

    for a, b in tqdm(sorted(pairs_to_compute), desc="  VCDB similarities"):
        ea = features[a]["embeddings"]
        eb = features[b]["embeddings"]

        row = {}

        # Bag-of-frames
        m1 = features[a]["mean_emb"]
        m2 = features[b]["mean_emb"]
        row["bag_of_frames"] = float(torch.dot(m1, m2).item())

        # Chamfer
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        row["chamfer"] = (max_ab + max_ba) / 2

        # Temporal derivative DTW
        row["temporal_derivative"] = deriv_fp.compare(deriv_fps[a], deriv_fps[b])

        # Attention trajectory DTW
        row["attention_trajectory"] = traj_fp.compare(traj_fps[a], traj_fps[b])

        # V-JEPA 2
        if vjepa2_features is not None and a in vjepa2_features and b in vjepa2_features:
            va = vjepa2_features[a]
            vb = vjepa2_features[b]
            row["vjepa2_bag_of_tokens"] = float(
                torch.dot(va["mean_emb"], vb["mean_emb"]).item()
            )
            res_dist = dtw_distance(
                va["temporal_residual"], vb["temporal_residual"], normalize=True
            )
            row["vjepa2_temporal_residual"] = float(
                torch.exp(torch.tensor(-res_dist)).item()
            )
        else:
            row["vjepa2_bag_of_tokens"] = float("nan")
            row["vjepa2_temporal_residual"] = float("nan")

        X_rows.append([row[m] for m in methods])
        y_list.append(1 if (a, b) in copy_pairs else 0)
        pair_ids.append(f"{a}|{b}")

    return np.array(X_rows), np.array(y_list), pair_ids


def compute_hdd_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    methods: list[str],
    vjepa2_features: dict[int, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute per-pair similarity vectors for HDD.

    Returns:
        X: (n_pairs, n_methods) feature matrix
        y: (n_pairs,) labels
        pair_ids: list of "idx_a|idx_b" strings
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints
    print("  Pre-computing HDD fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    X_rows = []
    y_list = []
    pair_ids = []

    for cid in tqdm(sorted(cluster_to_indices.keys()), desc="  HDD similarities"):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                a_idx = indices[a_pos]
                b_idx = indices[b_pos]

                gt = 1 if segments[a_idx].label == segments[b_idx].label else 0

                ea = features[a_idx]["embeddings"]
                eb = features[b_idx]["embeddings"]

                row = {}

                # Bag-of-frames
                row["bag_of_frames"] = float(
                    torch.dot(features[a_idx]["mean_emb"],
                              features[b_idx]["mean_emb"]).item()
                )

                # Chamfer
                sim_matrix = torch.mm(ea, eb.t())
                max_ab = sim_matrix.max(dim=1).values.mean().item()
                max_ba = sim_matrix.max(dim=0).values.mean().item()
                row["chamfer"] = (max_ab + max_ba) / 2

                # Temporal derivative DTW
                row["temporal_derivative"] = deriv_fp.compare(
                    deriv_fps[a_idx], deriv_fps[b_idx]
                )

                # Attention trajectory DTW
                row["attention_trajectory"] = traj_fp.compare(
                    traj_fps[a_idx], traj_fps[b_idx]
                )

                # V-JEPA 2
                if (vjepa2_features is not None
                        and a_idx in vjepa2_features
                        and b_idx in vjepa2_features):
                    va = vjepa2_features[a_idx]
                    vb = vjepa2_features[b_idx]
                    row["vjepa2_bag_of_tokens"] = float(
                        torch.dot(va["mean_emb"], vb["mean_emb"]).item()
                    )
                    res_dist = dtw_distance(
                        va["temporal_residual"], vb["temporal_residual"],
                        normalize=True,
                    )
                    row["vjepa2_temporal_residual"] = float(
                        torch.exp(torch.tensor(-res_dist)).item()
                    )
                else:
                    row["vjepa2_bag_of_tokens"] = float("nan")
                    row["vjepa2_temporal_residual"] = float("nan")

                X_rows.append([row[m] for m in methods])
                y_list.append(gt)
                pair_ids.append(f"{a_idx}|{b_idx}")

    print(f"  HDD pairs: {len(y_list)} "
          f"({sum(y_list)} pos + {len(y_list) - sum(y_list)} neg)")

    return np.array(X_rows), np.array(y_list), pair_ids


# ---------------------------------------------------------------------------
# Ensemble learning
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    methods: list[str],
    dataset_labels: np.ndarray | None = None,
) -> dict:
    """Train and evaluate ensemble via logistic regression.

    Args:
        X: (n_pairs, n_methods) similarity features.
        y: (n_pairs,) binary labels.
        methods: Method names corresponding to X columns.
        dataset_labels: (n_pairs,) 0=VCDB, 1=HDD for per-dataset eval.

    Returns:
        Dict with cross-validated results, weights, and per-method baselines.
    """
    # Drop rows with NaN (missing V-JEPA 2 features)
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    ds_valid = dataset_labels[valid_mask] if dataset_labels is not None else None

    n_valid = len(y_valid)
    n_pos = int(y_valid.sum())
    n_neg = n_valid - n_pos
    print(f"  Valid pairs: {n_valid} ({n_pos} pos + {n_neg} neg)")

    if n_pos < 5 or n_neg < 5:
        print("  ERROR: Too few samples for cross-validation")
        return {}

    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_ensemble_scores = np.zeros(n_valid)
    cv_individual_scores = {m: np.zeros(n_valid) for m in methods}

    fold_weights = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_valid, y_valid)):
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train = y_valid[train_idx]

        # Train logistic regression
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=1000, random_state=42,
        )
        clf.fit(X_train, y_train)

        # Ensemble predictions (probability of positive class)
        cv_ensemble_scores[test_idx] = clf.predict_proba(X_test)[:, 1]
        fold_weights.append(clf.coef_[0].tolist())

        # Individual method predictions (raw similarity as score)
        for j, m in enumerate(methods):
            cv_individual_scores[m][test_idx] = X_test[:, j]

    # Compute metrics
    results: dict[str, Any] = {"n_valid": n_valid, "n_pos": n_pos, "n_neg": n_neg}

    # Overall ensemble
    results["ensemble"] = {
        "ap": float(average_precision_score(y_valid, cv_ensemble_scores)),
        "auc": float(roc_auc_score(y_valid, cv_ensemble_scores)),
    }

    # Individual methods
    for m in methods:
        results[m] = {
            "ap": float(average_precision_score(y_valid, cv_individual_scores[m])),
            "auc": float(roc_auc_score(y_valid, cv_individual_scores[m])),
        }

    # Mean weights across folds
    mean_weights = np.mean(fold_weights, axis=0).tolist()
    results["weights"] = {m: w for m, w in zip(methods, mean_weights)}

    # Per-dataset breakdown
    if ds_valid is not None:
        for ds_name, ds_val in [("vcdb", 0), ("hdd", 1)]:
            ds_mask = ds_valid == ds_val
            if ds_mask.sum() < 2:
                continue
            y_ds = y_valid[ds_mask]
            n_pos_ds = int(y_ds.sum())
            n_neg_ds = len(y_ds) - n_pos_ds
            if n_pos_ds == 0 or n_neg_ds == 0:
                continue
            results[f"{ds_name}_ensemble"] = {
                "ap": float(average_precision_score(
                    y_ds, cv_ensemble_scores[ds_mask])),
                "auc": float(roc_auc_score(y_ds, cv_ensemble_scores[ds_mask])),
            }
            for m in methods:
                results[f"{ds_name}_{m}"] = {
                    "ap": float(average_precision_score(
                        y_ds, cv_individual_scores[m][ds_mask])),
                    "auc": float(roc_auc_score(
                        y_ds, cv_individual_scores[m][ds_mask])),
                }

    # Leave-one-dataset-out
    if ds_valid is not None:
        for train_ds, test_ds, train_name, test_name in [
            (0, 1, "vcdb", "hdd"), (1, 0, "hdd", "vcdb"),
        ]:
            train_mask = ds_valid == train_ds
            test_mask = ds_valid == test_ds
            if train_mask.sum() < 5 or test_mask.sum() < 5:
                continue
            y_train_ds = y_valid[train_mask]
            y_test_ds = y_valid[test_mask]
            n_pos_test = int(y_test_ds.sum())
            n_neg_test = len(y_test_ds) - n_pos_test
            if n_pos_test == 0 or n_neg_test == 0:
                continue
            if int(y_train_ds.sum()) == 0 or len(y_train_ds) == int(y_train_ds.sum()):
                continue

            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            clf.fit(X_valid[train_mask], y_train_ds)
            preds = clf.predict_proba(X_valid[test_mask])[:, 1]
            results[f"lodo_train_{train_name}_test_{test_name}"] = {
                "ap": float(average_precision_score(y_test_ds, preds)),
                "auc": float(roc_auc_score(y_test_ds, preds)),
                "weights": {m: float(w) for m, w in zip(methods, clf.coef_[0])},
            }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ensemble_comparison(
    results: dict,
    methods: list[str],
    fig_dir: Path,
):
    """Generate bar chart: AP of ensemble vs individual methods."""
    all_methods = methods + ["ensemble"]

    # Overall comparison
    aps = [results.get(m, {}).get("ap", float("nan")) for m in all_methods]
    aucs = [results.get(m, {}).get("auc", float("nan")) for m in all_methods]
    labels = [METHOD_LABELS.get(m, m) for m in all_methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in all_methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(range(len(all_methods)), aps, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(all_methods)))
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    for bar, val in zip(bars, aps):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                     fontweight="bold")
    ax1.set_ylabel("Average Precision", fontsize=12)
    ax1.set_title("Combined AP (5-fold CV)", fontsize=13)
    ax1.set_ylim(0, 1.15)

    bars = ax2.bar(range(len(all_methods)), aucs, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(all_methods)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    for bar, val in zip(bars, aucs):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                     fontweight="bold")
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("Combined AUC (5-fold CV)", fontsize=13)
    ax2.set_ylim(0, 1.15)

    fig.suptitle(
        "Ensemble vs Individual Methods (VCDB + HDD Combined)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    plot_path = fig_dir / "ensemble_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


def plot_ensemble_weights(
    results: dict,
    methods: list[str],
    fig_dir: Path,
):
    """Generate bar chart of learned ensemble weights."""
    weights = results.get("weights", {})
    if not weights:
        return

    w_vals = [weights.get(m, 0.0) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(methods)), w_vals, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    for bar, val in zip(bars, w_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Logistic Regression Coefficient", fontsize=12)
    ax.set_title("Ensemble Weights (mean across 5 folds)", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()

    plot_path = fig_dir / "ensemble_weights.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: list[str],
    methods: list[str],
    dataset: str,
):
    """Save per-pair similarities to JSON for fast re-runs."""
    data = {
        "dataset": dataset,
        "methods": methods,
        "pairs": [],
    }
    for i in range(len(y)):
        row = {
            "pair_id": pair_ids[i],
            "label": int(y[i]),
            "similarities": {m: float(X[i, j]) for j, m in enumerate(methods)},
        }
        data["pairs"].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved to {path}")


def load_checkpoint(
    path: Path,
    methods: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load per-pair similarities from JSON checkpoint."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Verify methods match
    if data["methods"] != methods:
        print(f"  Checkpoint methods mismatch, re-computing...")
        return None

    X_rows = []
    y_list = []
    pair_ids = []

    for row in data["pairs"]:
        X_rows.append([row["similarities"][m] for m in methods])
        y_list.append(row["label"])
        pair_ids.append(row["pair_id"])

    return np.array(X_rows), np.array(y_list), pair_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Metric Learning (VCDB + HDD)"
    )
    parser.add_argument("--vcdb-dir", type=str,
                        default="datasets/vcdb/core_dataset",
                        help="Path to VCDB core_dataset directory")
    parser.add_argument("--hdd-dir", type=str,
                        default="datasets/hdd",
                        help="Path to HDD dataset directory")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-vjepa2", action="store_true",
                        help="Skip V-JEPA 2 (DINOv3 only)")
    parser.add_argument("--skip-hdd", action="store_true",
                        help="Skip HDD (VCDB only)")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="Reuse saved similarity checkpoints if available")
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--context-sec", type=float, default=3.0)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    hdd_dir = project_root / args.hdd_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)
    checkpoint_dir = project_root / "datasets" / "ensemble_checkpoints"

    print("=" * 70)
    print("ENSEMBLE METRIC LEARNING (VCDB + HDD)")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  HDD dir: {hdd_dir}")
    print(f"  Skip V-JEPA 2: {args.skip_vjepa2}")
    print(f"  Skip HDD: {args.skip_hdd}")
    print(f"  Reuse checkpoints: {args.reuse_checkpoints}")

    methods = list(DINOV3_METHODS)
    if not args.skip_vjepa2:
        methods.extend(VJEPA2_METHODS)

    # ==================================================================
    # VCDB SIMILARITIES
    # ==================================================================
    vcdb_ckpt = checkpoint_dir / "vcdb_similarities.json"
    vcdb_loaded = None

    if args.reuse_checkpoints:
        vcdb_loaded = load_checkpoint(vcdb_ckpt, methods)

    if vcdb_loaded is not None:
        X_vcdb, y_vcdb, pair_ids_vcdb = vcdb_loaded
        print(f"\n  Loaded VCDB checkpoint: {len(y_vcdb)} pairs")
    else:
        print("\nStep 1: Computing VCDB similarities...")

        # Discover + annotations
        videos = discover_videos(str(vid_dir))
        copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
        print(f"  Videos: {len(videos)}, copy pairs: {len(copy_pairs)}")

        # DINOv3 features
        print("  Loading DINOv3 encoder...")
        encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)
        t0 = time.time()
        vcdb_features = extract_vcdb_dinov3_features(
            encoder, str(vid_dir), videos,
            sample_rate=args.sample_rate, max_frames=args.max_frames,
        )
        print(f"  DINOv3 extraction: {time.time() - t0:.1f}s")
        del encoder
        torch.cuda.empty_cache()

        vcdb_keys = sorted(vcdb_features.keys())

        # V-JEPA 2 features
        vcdb_vjepa2 = None
        if not args.skip_vjepa2:
            print("  Loading V-JEPA 2 model...")
            from transformers import AutoModel, AutoVideoProcessor
            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            ).to(args.device).eval()
            vjepa2_proc = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            t0 = time.time()
            vcdb_vjepa2 = extract_vcdb_vjepa2_features(
                vjepa2_model, vjepa2_proc, str(vid_dir), vcdb_keys,
                torch.device(args.device),
            )
            print(f"  V-JEPA 2 extraction: {time.time() - t0:.1f}s")
            del vjepa2_model, vjepa2_proc
            torch.cuda.empty_cache()

        # Compute similarities
        t0 = time.time()
        X_vcdb, y_vcdb, pair_ids_vcdb = compute_vcdb_similarities(
            vcdb_features, vcdb_keys, copy_pairs, methods,
            vjepa2_features=vcdb_vjepa2,
        )
        print(f"  VCDB similarity computation: {time.time() - t0:.1f}s")

        # Checkpoint
        save_checkpoint(vcdb_ckpt, X_vcdb, y_vcdb, pair_ids_vcdb, methods, "vcdb")

    # ==================================================================
    # HDD SIMILARITIES
    # ==================================================================
    X_hdd = None
    y_hdd = None
    pair_ids_hdd = None

    if not args.skip_hdd:
        hdd_ckpt = checkpoint_dir / "hdd_similarities.json"
        hdd_loaded = None

        if args.reuse_checkpoints:
            hdd_loaded = load_checkpoint(hdd_ckpt, methods)

        if hdd_loaded is not None:
            X_hdd, y_hdd, pair_ids_hdd = hdd_loaded
            print(f"\n  Loaded HDD checkpoint: {len(y_hdd)} pairs")
        else:
            print("\nStep 2: Computing HDD similarities...")

            # Discover sessions
            sessions = discover_sessions(hdd_dir)
            print(f"  Sessions: {len(sessions)}")

            # Extract maneuver segments
            all_segments: list[ManeuverSegment] = []
            for session_id in tqdm(sorted(sessions.keys()), desc="Loading HDD sessions"):
                info = sessions[session_id]
                labels = np.load(info["label_path"])
                try:
                    gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
                except Exception:
                    continue
                segs = extract_maneuver_segments(
                    session_id, labels, gps_ts, gps_lats, gps_lngs,
                    info["video_path"], info["video_start_unix"],
                )
                all_segments.extend(segs)

            # Cluster + filter
            clusters = cluster_intersections(all_segments)
            mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
            print(f"  Mixed clusters: {len(mixed)}")

            eval_segments: list[ManeuverSegment] = []
            cluster_to_indices: dict[int, list[int]] = defaultdict(list)
            for cid, segs in mixed.items():
                for seg in segs:
                    idx = len(eval_segments)
                    eval_segments.append(seg)
                    cluster_to_indices[cid].append(idx)

            print(f"  Evaluation segments: {len(eval_segments)}")

            # DINOv3 features
            print("  Loading DINOv3 encoder...")
            encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)
            t0 = time.time()
            hdd_features = extract_hdd_dinov3_features(
                encoder, eval_segments, context_sec=args.context_sec,
            )
            print(f"  DINOv3 extraction: {time.time() - t0:.1f}s")
            del encoder
            torch.cuda.empty_cache()

            # V-JEPA 2 features
            hdd_vjepa2 = None
            if not args.skip_vjepa2:
                print("  Loading V-JEPA 2 model...")
                from transformers import AutoModel, AutoVideoProcessor
                vjepa2_model = AutoModel.from_pretrained(
                    VJEPA2_MODEL_NAME, trust_remote_code=True
                ).to(args.device).eval()
                vjepa2_proc = AutoVideoProcessor.from_pretrained(
                    VJEPA2_MODEL_NAME, trust_remote_code=True
                )
                t0 = time.time()
                hdd_vjepa2 = extract_hdd_vjepa2_features(
                    vjepa2_model, vjepa2_proc, eval_segments,
                    device=torch.device(args.device),
                    context_sec=args.context_sec,
                )
                print(f"  V-JEPA 2 extraction: {time.time() - t0:.1f}s")
                del vjepa2_model, vjepa2_proc
                torch.cuda.empty_cache()

            # Compute similarities
            t0 = time.time()
            X_hdd, y_hdd, pair_ids_hdd = compute_hdd_similarities(
                eval_segments, hdd_features, cluster_to_indices, methods,
                vjepa2_features=hdd_vjepa2,
            )
            print(f"  HDD similarity computation: {time.time() - t0:.1f}s")

            # Checkpoint
            save_checkpoint(hdd_ckpt, X_hdd, y_hdd, pair_ids_hdd, methods, "hdd")

    # ==================================================================
    # COMBINE DATASETS + LEARN ENSEMBLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("ENSEMBLE LEARNING")
    print("=" * 70)

    if X_hdd is not None:
        assert y_hdd is not None
        X_combined = np.vstack([X_vcdb, X_hdd])
        y_combined = np.concatenate([y_vcdb, y_hdd])
        ds_labels = np.concatenate([
            np.zeros(len(y_vcdb)),
            np.ones(len(y_hdd)),
        ])
        print(f"  Combined: {len(y_combined)} pairs "
              f"({len(y_vcdb)} VCDB + {len(y_hdd)} HDD)")
    else:
        X_combined = X_vcdb
        y_combined = y_vcdb
        ds_labels = np.zeros(len(y_vcdb))
        print(f"  VCDB only: {len(y_combined)} pairs")

    results = evaluate_ensemble(X_combined, y_combined, methods, ds_labels)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  {'Method':<25s}  {'AP':>8s}  {'AUC':>8s}")
    print("  " + "-" * 45)

    all_methods = methods + ["ensemble"]
    for m in all_methods:
        if m in results:
            ap = results[m].get("ap", float("nan"))
            auc = results[m].get("auc", float("nan"))
            marker = " ***" if m == "ensemble" else ""
            print(f"  {METHOD_LABELS.get(m, m):<25s}  {ap:>8.4f}  {auc:>8.4f}{marker}")

    if "weights" in results:
        print(f"\n  Learned weights:")
        for m in methods:
            w = results["weights"].get(m, 0.0)
            print(f"    {METHOD_LABELS.get(m, m):<25s}  {w:>+8.4f}")

    # Per-dataset breakdown
    for ds_name in ["vcdb", "hdd"]:
        key = f"{ds_name}_ensemble"
        if key in results:
            print(f"\n  {ds_name.upper()} subset:")
            print(f"    Ensemble: AP={results[key]['ap']:.4f}  "
                  f"AUC={results[key]['auc']:.4f}")
            for m in methods:
                mkey = f"{ds_name}_{m}"
                if mkey in results:
                    print(f"    {METHOD_LABELS.get(m, m):<23s}: "
                          f"AP={results[mkey]['ap']:.4f}  "
                          f"AUC={results[mkey]['auc']:.4f}")

    # Leave-one-dataset-out
    for key in ["lodo_train_vcdb_test_hdd", "lodo_train_hdd_test_vcdb"]:
        if key in results:
            print(f"\n  {key}:")
            print(f"    AP={results[key]['ap']:.4f}  AUC={results[key]['auc']:.4f}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_ensemble_comparison(results, methods, fig_dir)
    plot_ensemble_weights(results, methods, fig_dir)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_path = project_root / "datasets" / "ensemble_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any remaining numpy types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

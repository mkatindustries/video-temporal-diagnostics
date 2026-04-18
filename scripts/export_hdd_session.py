#!/usr/bin/env python3
"""Export a single HDD session to HDF5 with DINOv3 and/or V-JEPA 2 features.

Extracts video frames at a configurable fps, aligns them with GPS timestamps,
runs the specified model(s), and writes everything to an HDF5 file sorted by
timestamp for sequential downstream consumption.

HDF5 schema:
    <session_id>.h5
    ├── dino/                        # DINOv3 group (one record per sampled frame)
    │   ├── timestamps               # float64, (N,) — epoch seconds
    │   ├── lat                      # float64, (N,) — latitude
    │   ├── lng                      # float64, (N,) — longitude
    │   ├── embeddings               # float32, (N, 1024) — CLS token embeddings
    │   └── attention_maps           # float32, (N, H, W) — CLS→patch spatial attention
    │
    ├── jepa/                        # V-JEPA 2 group (one record per 64-frame window)
    │   ├── timestamps               # float64, (M,) — epoch seconds (window center)
    │   ├── lat                      # float64, (M,) — latitude
    │   ├── lng                      # float64, (M,) — longitude
    │   ├── embeddings               # float32, (M, 1024) — mean-pooled encoder tokens
    │   └── prediction_maps          # float32, (M, 16, 16) — spatial prediction error
    │
    └── gps/                         # native-rate RTK GPS (~100Hz)
        ├── timestamps               # float64, (K,) — epoch seconds
        ├── lat                      # float64, (K,) — latitude
        └── lng                      # float64, (K,) — longitude

Usage:
    python scripts/export_hdd_session.py 201702271017                   # both models (default)
    python scripts/export_hdd_session.py 201702271017 --groups dino     # DINOv3 only
    python scripts/export_hdd_session.py 201702271017 --groups jepa     # V-JEPA 2 only
    python scripts/export_hdd_session.py --stats                        # list sessions
"""

import argparse
import re
import shutil
import time
import zoneinfo
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import av
import cv2
import h5py  # type: ignore[import]
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# V-JEPA 2 constants
# ---------------------------------------------------------------------------

VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32   # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256     # 16h × 16w
VJEPA2_SPATIAL_H = 16
VJEPA2_SPATIAL_W = 16


# ---------------------------------------------------------------------------
# Session discovery (adapted from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------


def parse_video_start_time(video_filename: str) -> float:
    """Parse video start unix timestamp from filename.

    Filename format: 2017-02-27-10-17-27_new_0.75.mp4
    Timestamps are in US/Pacific local time (PST or PDT depending on date).
    Using America/Los_Angeles handles DST transitions automatically —
    DST started March 12 2017, so Feb/early-Mar sessions are PST (UTC-8)
    and Apr+ sessions are PDT (UTC-7).
    """
    match = re.match(
        r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", video_filename
    )
    if not match:
        raise ValueError(f"Cannot parse timestamp from {video_filename}")

    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return dt.timestamp()


def discover_session(hdd_dir: Path, session_id: str) -> dict:
    """Discover a single HDD session with video and GPS.

    Returns:
        Dict with 'video_path', 'gps_path', 'video_start_unix'.

    Raises:
        FileNotFoundError: If session files cannot be found.
    """
    release_dir = hdd_dir / "release_2019_07_08"

    # Search for the session directory under date dirs
    for date_dir in sorted(release_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        session_dir = date_dir / session_id
        if not session_dir.is_dir():
            continue

        # Find video
        camera_dir = session_dir / "camera" / "center"
        if not camera_dir.exists():
            raise FileNotFoundError(
                f"Camera directory not found: {camera_dir}"
            )
        video_files = list(camera_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError(
                f"No .mp4 files in {camera_dir}"
            )
        video_path = video_files[0]

        # Find GPS
        gps_path = session_dir / "general" / "csv" / "rtk_pos.csv"
        if not gps_path.exists():
            raise FileNotFoundError(f"GPS file not found: {gps_path}")

        video_start_unix = parse_video_start_time(video_path.name)

        return {
            "video_path": str(video_path),
            "gps_path": str(gps_path),
            "video_start_unix": video_start_unix,
        }

    raise FileNotFoundError(
        f"Session {session_id} not found under {release_dir}"
    )


def discover_sessions(hdd_dir: Path) -> dict[str, dict]:
    """Discover all valid HDD sessions with labels, GPS, and video.

    Returns:
        Dict mapping session_id -> {
            'label_path': Path to label .npy,
            'video_path': str path to .mp4,
            'gps_path': str path to rtk_pos.csv,
            'video_start_unix': float,
        }
    """
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


MANEUVER_NAMES = {
    1: "intersection_passing",
    2: "left_turn",
    3: "right_turn",
}


def print_session_stats(hdd_dir: Path, sort_by: str = "turns"):
    """Print per-session stats with turn event counts and GPS integrity.

    Args:
        hdd_dir: Path to HDD dataset directory.
        sort_by: Sort key — "turns" (total left+right), "left", "right",
                 "duration", or "session".
    """
    sessions = discover_sessions(hdd_dir)
    print(f"Found {len(sessions)} valid sessions\n")

    rows = []
    gps_bad = []
    for session_id in sorted(sessions.keys()):
        info = sessions[session_id]
        labels = np.load(info["label_path"])
        duration_sec = len(labels) / 3.0  # labels at 3 fps

        counts = {1: 0, 2: 0, 3: 0}
        # Count contiguous segments (not individual frames)
        prev = 0
        for lbl in labels:
            if lbl in (1, 2, 3) and lbl != prev:
                counts[int(lbl)] += 1
            prev = lbl

        # GPS integrity checks
        gps_flag = ""
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])

            n_gps = len(gps_ts)
            # Valid = not NaN and not near-zero
            valid = (
                ~np.isnan(gps_lats) & ~np.isnan(gps_lngs)
                & (np.abs(gps_lats) > 1) & (np.abs(gps_lngs) > 1)
            )
            n_valid = int(valid.sum())
            pct_valid = n_valid / n_gps * 100 if n_gps > 0 else 0

            # Check for stuck GPS (all same coordinate)
            if n_valid > 0:
                valid_lats = gps_lats[valid]
                valid_lngs = gps_lngs[valid]
                lat_range = valid_lats.max() - valid_lats.min()
                lng_range = valid_lngs.max() - valid_lngs.min()
                # ~0.0001 degrees ≈ 11m — if range is less, GPS is stuck
                if lat_range < 0.0001 and lng_range < 0.0001:
                    gps_flag = "STUCK"
                elif pct_valid < 50:
                    gps_flag = "LOW"
            else:
                gps_flag = "EMPTY"

            # Check video-GPS time alignment
            if not gps_flag:
                video_start = info["video_start_unix"]
                video_end = video_start + duration_sec
                # Overlap = max(0, min(video_end, gps_end) - max(video_start, gps_start))
                overlap = max(0, min(video_end, gps_ts[-1]) - max(video_start, gps_ts[0]))
                overlap_pct = overlap / duration_sec * 100 if duration_sec > 0 else 0
                if overlap_pct < 10:
                    gps_flag = "DESYNC"
                elif overlap_pct < 50:
                    gps_flag = "PARTIAL"

        except Exception:
            n_gps = 0
            pct_valid = 0
            gps_flag = "FAIL"

        row = {
            "session": session_id,
            "duration_min": duration_sec / 60.0,
            "passing": counts[1],
            "left": counts[2],
            "right": counts[3],
            "turns": counts[2] + counts[3],
            "gps_pct": pct_valid,
            "gps_flag": gps_flag,
        }
        rows.append(row)
        if gps_flag:
            gps_bad.append(row)

    # Sort
    sort_keys = {
        "turns": lambda r: r["turns"],
        "left": lambda r: r["left"],
        "right": lambda r: r["right"],
        "duration": lambda r: r["duration_min"],
        "session": lambda r: r["session"],
    }
    key_fn = sort_keys.get(sort_by, sort_keys["turns"])
    rows.sort(key=key_fn, reverse=(sort_by != "session"))

    # Print table
    print(f"{'Session':<16s} {'Dur(min)':>8s} {'Passing':>8s} "
          f"{'Left':>6s} {'Right':>6s} {'Turns':>6s} "
          f"{'GPS%':>5s} {'GPS':>5s}")
    print("-" * 72)
    total_turns = 0
    for r in rows:
        flag_str = r["gps_flag"] if r["gps_flag"] else "OK"
        print(
            f"{r['session']:<16s} {r['duration_min']:8.1f} {r['passing']:8d} "
            f"{r['left']:6d} {r['right']:6d} {r['turns']:6d} "
            f"{r['gps_pct']:5.0f} {flag_str:>5s}"
        )
        total_turns += r["turns"]
    print("-" * 72)
    print(f"{'Total':<16s} {'':>8s} {'':>8s} "
          f"{'':>6s} {'':>6s} {total_turns:6d}")

    # Summary of GPS issues
    if gps_bad:
        print(f"\nGPS issues ({len(gps_bad)} sessions):")
        for r in gps_bad:
            print(f"  {r['session']}  {r['gps_flag']:<5s}  "
                  f"valid={r['gps_pct']:.0f}%  turns={r['turns']}")
    else:
        print("\nAll sessions have valid GPS.")


# ---------------------------------------------------------------------------
# GPS loading (adapted from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------


def load_gps(gps_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GPS data from rtk_pos.csv.

    Note: CSV headers are swapped -- column labeled 'lng' contains latitude
    (~37.39), column labeled 'lat' contains longitude (~-122.05).

    Returns:
        (timestamps, latitudes, longitudes) as numpy arrays.
    """
    data = np.genfromtxt(
        gps_path, delimiter=",", skip_header=1, usecols=(0, 2, 3),
        dtype=np.float64,
    )
    timestamps = data[:, 0]
    # Headers say lng,lat but values are swapped
    latitudes = data[:, 1]   # column labeled 'lng' is actually latitude
    longitudes = data[:, 2]  # column labeled 'lat' is actually longitude
    return timestamps, latitudes, longitudes


# ---------------------------------------------------------------------------
# Frame extraction with GPS alignment
# ---------------------------------------------------------------------------


def extract_frames_with_gps(
    video_path: str,
    video_start_unix: float,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract frames from video and align with GPS.

    For each sampled frame, computes its unix timestamp and finds the nearest
    GPS entry. Frames with invalid GPS (NaN or near-zero) are skipped.

    Returns:
        (frames, timestamps, latitudes, longitudes) — all aligned and filtered.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [bad-argument-type]
    time_base = float(stream.time_base)

    sample_interval = video_fps / target_fps

    frames = []
    timestamps = []
    lats = []
    lngs = []

    frame_count = 0
    next_sample = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        if frame_count >= next_sample:
            frame_time_sec = float(frame.pts) * time_base
            frame_ts = video_start_unix + frame_time_sec

            # Find nearest GPS entry
            gps_idx = np.searchsorted(gps_timestamps, frame_ts)
            # pyrefly: ignore [no-matching-overload]
            gps_idx = min(gps_idx, len(gps_timestamps) - 1)

            lat = gps_lats[gps_idx]
            lng = gps_lngs[gps_idx]

            # Skip frames with invalid GPS
            if np.isnan(lat) or np.isnan(lng) or abs(lat) < 1 or abs(lng) < 1:
                frame_count += 1
                continue

            img = frame.to_ndarray(format="rgb24")

            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)
            timestamps.append(frame_ts)
            lats.append(lat)
            lngs.append(lng)

            next_sample += sample_interval

        frame_count += 1

    container.close()

    return (
        frames,
        np.array(timestamps, dtype=np.float64),
        np.array(lats, dtype=np.float64),
        np.array(lngs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# DINOv3 feature extraction (combined embeddings + attention maps)
# ---------------------------------------------------------------------------


def extract_dino_features(
    frames: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 16,
    layer: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract DINOv3 CLS embeddings and spatial attention maps.

    Runs a single forward pass per batch to get both outputs without
    redundant inference.

    Args:
        frames: List of RGB frames.
        device: Torch device.
        batch_size: Batch size for processing.
        layer: Attention layer to use (-1 = last).

    Returns:
        (embeddings (N, 1024), attention_maps (N, H, W)) as numpy arrays.
    """
    from video_retrieval.models import DINOv3Encoder

    encoder = DINOv3Encoder(device=device)
    skip_tokens = 1 + encoder.num_register_tokens  # CLS + register tokens

    all_embeddings = []
    all_attn_maps = []

    for i in tqdm(range(0, len(frames), batch_size), desc="DINOv3 inference"):
        batch_frames = frames[i : i + batch_size]
        inputs = encoder._preprocess(batch_frames)

        with torch.no_grad():
            outputs = encoder.model(**inputs)

        # CLS embeddings
        embeddings = outputs.last_hidden_state[:, 0]  # (B, D)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

        # Attention maps
        attn = outputs.attentions[layer]   # (B, heads, N, N)
        attn = attn.mean(dim=1)            # (B, N, N)
        cls_attn = attn[:, 0, skip_tokens:]  # (B, num_patches)

        num_patches = cls_attn.shape[1]
        h = w = int(num_patches ** 0.5)
        attn_maps = cls_attn.view(-1, h, w)  # (B, h, w)

        # Normalize to probability distribution
        attn_maps = attn_maps / (attn_maps.sum(dim=(1, 2), keepdim=True) + 1e-8)
        all_attn_maps.append(attn_maps.cpu())

    del encoder
    torch.cuda.empty_cache()

    embeddings_np = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    attn_maps_np = torch.cat(all_attn_maps, dim=0).numpy().astype(np.float32)

    return embeddings_np, attn_maps_np


# ---------------------------------------------------------------------------
# V-JEPA 2 feature extraction (sliding window)
# ---------------------------------------------------------------------------


def _build_temporal_masks(
    n_context_steps: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context/target masks that split along the temporal axis.

    Context = all spatial positions at time steps 0..n_context_steps-1
    Target  = all spatial positions at time steps n_context_steps..T_PATCHES-1

    Returns position-index tensors of shape (1, n_positions).
    """
    all_indices = torch.arange(
        VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device,
    )
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)

    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)

    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def extract_vjepa2_features(
    frames: list[np.ndarray],
    timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    device: str = "cuda",
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract V-JEPA 2 embeddings and prediction error maps.

    V-JEPA 2 expects 64 frames per input. We slide a window of 64 frames
    across the session with the given stride, assigning each embedding to
    the center timestamp of its window.

    For each window, also runs the predictor with a temporal split (context =
    first half, target = second half) and computes the spatial prediction error
    map: ||predicted - ground_truth|| per spatial position, averaged over
    target time steps. This produces a (16, 16) "surprise map" per window.

    Args:
        frames: List of RGB frames (extracted at target fps).
        timestamps: Unix timestamps for each frame.
        gps_lats: Latitudes for each frame.
        gps_lngs: Longitudes for each frame.
        device: Torch device.

    Returns:
        (embeddings (M, 1024), prediction_maps (M, 16, 16),
         timestamps (M,), lats (M,), lngs (M,))
        where M is the number of windows.
    """
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )
    model = model.to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    # Temporal split: context = first half, target = second half
    n_context_steps = VJEPA2_T_PATCHES // 2  # 16
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps  # 16
    context_mask, target_mask = _build_temporal_masks(
        n_context_steps, torch.device(device),
    )

    n_frames = len(frames)

    all_embeddings = []
    all_pred_maps = []
    all_timestamps = []
    all_lats = []
    all_lngs = []

    # If fewer than 64 frames, pad and produce one embedding
    if n_frames < VJEPA2_NUM_FRAMES:
        starts = [0]
    else:
        starts = list(range(0, n_frames - VJEPA2_NUM_FRAMES + 1, stride))
        # Ensure last window is included
        if starts[-1] + VJEPA2_NUM_FRAMES < n_frames:
            starts.append(n_frames - VJEPA2_NUM_FRAMES)

    for start in tqdm(starts, desc="V-JEPA 2 inference"):
        end = start + VJEPA2_NUM_FRAMES
        window_frames = frames[start:end]

        # Pad if needed (only for the n_frames < 64 case)
        while len(window_frames) < VJEPA2_NUM_FRAMES:
            window_frames.append(window_frames[-1])

        center = start + VJEPA2_NUM_FRAMES // 2
        center = min(center, n_frames - 1)

        inputs = processor(videos=window_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Encoder pass: mean-pooled embedding
            enc_out = model(**inputs, skip_predictor=True)
            encoder_tokens = enc_out.last_hidden_state[0]  # (T*S, 1024)
            mean_emb = F.normalize(
                encoder_tokens.mean(dim=0), dim=0
            )  # (1024,)

            # Predictor pass: prediction error maps
            pred_out = model(
                **inputs,
                context_mask=[context_mask],
                target_mask=[target_mask],
            )
            predicted = pred_out.predictor_output.last_hidden_state[0]
            ground_truth = pred_out.predictor_output.target_hidden_state[0]

            # Reshape to (n_target_steps, SPATIAL, D)
            predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
            ground_truth = ground_truth.reshape(
                n_target_steps, VJEPA2_SPATIAL, -1,
            )

            # Per-spatial-position error, averaged over target time steps
            error = (predicted - ground_truth).norm(dim=-1)  # (n_target, 256)
            error_map = error.mean(dim=0)  # (256,)
            error_map = error_map.reshape(
                VJEPA2_SPATIAL_H, VJEPA2_SPATIAL_W,
            )  # (16, 16)

        all_embeddings.append(mean_emb.cpu().numpy())
        all_pred_maps.append(error_map.cpu().numpy())
        all_timestamps.append(timestamps[center])
        all_lats.append(gps_lats[center])
        all_lngs.append(gps_lngs[center])

    del model, processor
    torch.cuda.empty_cache()

    return (
        np.stack(all_embeddings).astype(np.float32),
        np.stack(all_pred_maps).astype(np.float32),
        np.array(all_timestamps, dtype=np.float64),
        np.array(all_lats, dtype=np.float64),
        np.array(all_lngs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------


def write_hdf5(
    output_path: Path,
    group_name: str,
    timestamps: np.ndarray,
    lats: np.ndarray,
    lngs: np.ndarray,
    embeddings: np.ndarray,
    attention_maps: np.ndarray | None = None,
    prediction_maps: np.ndarray | None = None,
):
    """Write features to HDF5 file.

    Creates or appends to the file, creating the specified group.
    All datasets within the group share the same N (one record per frame).

    Args:
        output_path: Path to .h5 file.
        group_name: HDF5 group name (e.g. "dino", "jepa").
        timestamps: (N,) float64 epoch seconds.
        lats: (N,) float64 latitudes.
        lngs: (N,) float64 longitudes.
        embeddings: (N, D) float32 model embeddings.
        attention_maps: (N, H, W) float32 spatial attention (DINOv3, optional).
        prediction_maps: (N, H, W) float32 prediction error maps (V-JEPA 2, optional).
    """
    # Sort by timestamp
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    lats = lats[order]
    lngs = lngs[order]
    embeddings = embeddings[order]
    if attention_maps is not None:
        attention_maps = attention_maps[order]
    if prediction_maps is not None:
        prediction_maps = prediction_maps[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "a") as f:
        if group_name in f:
            del f[group_name]

        g = f.create_group(group_name)
        g.create_dataset("timestamps", data=timestamps)
        g.create_dataset("lat", data=lats)
        g.create_dataset("lng", data=lngs)
        g.create_dataset("embeddings", data=embeddings)

        if attention_maps is not None:
            g.create_dataset("attention_maps", data=attention_maps)

        if prediction_maps is not None:
            g.create_dataset("prediction_maps", data=prediction_maps)


def write_gps_group(
    output_path: Path,
    gps_ts: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
):
    """Write standalone gps/ group to HDF5 at native rate.

    Uses append mode and deletes existing group if present.
    """
    if len(gps_ts) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "a") as f:
        if "gps" in f:
            del f["gps"]
        g = f.create_group("gps")
        g.create_dataset("timestamps", data=gps_ts)
        g.create_dataset("lat", data=gps_lats)
        g.create_dataset("lng", data=gps_lngs)


def main():
    parser = argparse.ArgumentParser(
        description="Export a single HDD session to HDF5 with model features."
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        type=str,
        help="HDD session ID (e.g. 201702271017)",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dino", "jepa"],
        help="HDF5 group names to export (default: dino jepa). "
        "Starts with 'dino' → DINOv3, starts with 'jepa' → V-JEPA 2.",
    )
    parser.add_argument("--fps", type=float, default=3.0, help="Frame sampling rate (default: 3)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--output-dir", type=str, default="exports/")
    parser.add_argument("--max-resolution", type=int, default=518, help="Max frame height for DINOv3")
    parser.add_argument(
        "--jepa-stride", type=int, default=8,
        help="V-JEPA 2 sliding window stride in frames (default: 8)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="List all sessions with turn event counts and exit.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="turns",
        choices=["turns", "left", "right", "duration", "session"],
        help="Sort key for --stats (default: turns).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir

    # ------------------------------------------------------------------
    # Stats mode
    # ------------------------------------------------------------------
    if args.stats:
        print_session_stats(hdd_dir, sort_by=args.sort_by)
        return

    if not args.session_id:
        parser.error("session_id is required (unless using --stats)")

    output_dir = project_root / args.output_dir / args.session_id
    output_path = output_dir / "features.h5"

    model_labels = ", ".join(
        "V-JEPA 2" if g.startswith("jepa") else "DINOv3" for g in args.groups
    )
    print("=" * 70)
    print(f"HDD SESSION EXPORT: {args.session_id} → {model_labels}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover session
    # ------------------------------------------------------------------
    print(f"\nStep 1: Discovering session {args.session_id}...")
    session = discover_session(hdd_dir, args.session_id)
    print(f"  Video: {session['video_path']}")
    print(f"  GPS:   {session['gps_path']}")

    # ------------------------------------------------------------------
    # Copy source video to output directory
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    src_video = Path(session["video_path"])
    dst_video = output_dir / "video.mp4"
    if dst_video.exists():
        print(f"  Video already copied: {dst_video}")
    else:
        print(f"  Copying video to {dst_video}...")
        shutil.copy2(src_video, dst_video)
        print(f"  Copied ({dst_video.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Step 2: Load GPS
    # ------------------------------------------------------------------
    print("\nStep 2: Loading GPS data...")
    gps_ts, gps_lats, gps_lngs = load_gps(session["gps_path"])
    print(f"  GPS entries: {len(gps_ts)}")

    # ------------------------------------------------------------------
    # Step 3: Extract frames with GPS alignment
    # ------------------------------------------------------------------
    # Use the highest resolution needed across all groups
    needs_dino = any(not g.startswith("jepa") for g in args.groups)
    max_res = args.max_resolution if needs_dino else 256
    print(f"\nStep 3: Extracting frames at {args.fps} fps (max_resolution={max_res})...")
    t0 = time.time()
    frames, timestamps, lats, lngs = extract_frames_with_gps(
        session["video_path"],
        session["video_start_unix"],
        gps_ts,
        gps_lats,
        gps_lngs,
        target_fps=args.fps,
        max_resolution=max_res,
    )
    print(f"  Extracted {len(frames)} frames with valid GPS ({time.time() - t0:.1f}s)")

    if len(frames) == 0:
        print("ERROR: No frames extracted. Exiting.")
        return

    print(f"  Time range: {timestamps[0]:.1f} – {timestamps[-1]:.1f}")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Run models and write HDF5
    # ------------------------------------------------------------------
    for group_name in args.groups:
        use_jepa = group_name.startswith("jepa")

        if use_jepa:
            print(f"\nStep 4 [{group_name}]: Running V-JEPA 2 "
                  f"(sliding window of {VJEPA2_NUM_FRAMES} frames, "
                  f"stride {args.jepa_stride})...")
            # Downscale frames to 256 for V-JEPA 2 if extracted at higher res
            if max_res > 256:
                jepa_frames = []
                for img in frames:
                    scale = 256 / img.shape[0]
                    new_w = int(img.shape[1] * scale)
                    jepa_frames.append(cv2.resize(img, (new_w, 256)))
            else:
                jepa_frames = frames

            t0 = time.time()
            embeddings, prediction_maps, ts_out, lats_out, lngs_out = extract_vjepa2_features(
                jepa_frames, timestamps, lats, lngs, device=args.device,
                stride=args.jepa_stride,
            )
            attention_maps = None
            print(f"  {len(embeddings)} embeddings ({embeddings.shape[1]}D), "
                  f"prediction maps {prediction_maps.shape[1]}x{prediction_maps.shape[2]} "
                  f"in {time.time() - t0:.1f}s")
        else:
            print(f"\nStep 4 [{group_name}]: Running DINOv3...")
            t0 = time.time()
            embeddings, attention_maps = extract_dino_features(
                frames, device=args.device,
            )
            prediction_maps = None
            ts_out, lats_out, lngs_out = timestamps, lats, lngs
            print(
                f"  {len(embeddings)} embeddings ({embeddings.shape[1]}D), "
                f"attention maps {attention_maps.shape[1]}x{attention_maps.shape[2]} "
                f"in {time.time() - t0:.1f}s"
            )

        print(f"  Writing group '{group_name}' to {output_path}...")
        write_hdf5(
            output_path,
            group_name,
            ts_out,
            lats_out,
            lngs_out,
            embeddings,
            attention_maps=attention_maps,
            prediction_maps=prediction_maps,
        )

    # ------------------------------------------------------------------
    # Step 5: Write standalone GPS group
    # ------------------------------------------------------------------
    print("\nStep 5: Writing standalone GPS group...")
    write_gps_group(output_path, gps_ts, gps_lats, gps_lngs)
    print(f"  gps/: {len(gps_ts)} entries at native rate")

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    print(f"\nVerification:")
    with h5py.File(output_path, "r") as f:
        print(f"  Groups in file: {list(f.keys())}")
        for gname in f:
            g = f[gname]
            print(f"  Group '{gname}':")
            for k in g:
                print(f"    {k}: shape={g[k].shape}, dtype={g[k].dtype}")
            if "timestamps" in g:
                ts = g["timestamps"][:]
                sorted_ok = all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
                print(f"    Sorted by timestamp: {sorted_ok}")

    print("\nDone.")


if __name__ == "__main__":
    main()

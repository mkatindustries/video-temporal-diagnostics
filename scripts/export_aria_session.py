#!/usr/bin/env python3
"""Export an Aria glasses recording to HDF5 with DINOv3 and/or V-JEPA 2 features.

Extracts video frames at a configurable fps, aligns them with GPS and IMU sensor
data, runs the specified model(s), and writes everything to an HDF5 file sorted by
timestamp for sequential downstream consumption.

HDF5 schema:
    <recording_id>/features.h5
    +-- dino/
    |   +-- timestamps          # float64, (N,) -- epoch seconds
    |   +-- lat                 # float64, (N,) -- latitude
    |   +-- lng                 # float64, (N,) -- longitude
    |   +-- accel               # float32, (N, 3) -- accelerometer (x,y,z) nearest to frame
    |   +-- gyro                # float32, (N, 3) -- gyroscope (x,y,z) nearest to frame
    |   +-- embeddings          # float32, (N, 1024) -- CLS token embeddings
    |   +-- attention_maps      # float32, (N, H, W) -- CLS->patch spatial attention
    |
    +-- jepa/
    |   +-- timestamps          # float64, (M,) -- epoch seconds
    |   +-- lat                 # float64, (M,) -- latitude
    |   +-- lng                 # float64, (M,) -- longitude
    |   +-- accel               # float32, (M, 3) -- accelerometer at window center
    |   +-- gyro                # float32, (M, 3) -- gyroscope at window center
    |   +-- embeddings          # float32, (M, 1024) -- mean-pooled encoder tokens
    |   +-- prediction_maps     # float32, (M, 16, 16) -- spatial prediction error
    |
    +-- imu/                    # standalone high-rate IMU (default 100Hz)
    |   +-- timestamps          # float64, (P,) -- epoch seconds
    |   +-- accel               # float32, (P, 3) -- accelerometer (x,y,z)
    |   +-- gyro                # float32, (P, 3) -- gyroscope (x,y,z)
    |
    +-- gps/                    # native 1Hz GPS
        +-- timestamps          # float64, (K,) -- epoch seconds
        +-- lat                 # float64, (K,) -- latitude
        +-- lng                 # float64, (K,) -- longitude

Usage:
    python scripts/export_aria_session.py /tmp/1501677363692556/1501677363692556
    python scripts/export_aria_session.py /tmp/1501677363692556/1501677363692556 --groups dino
    python scripts/export_aria_session.py /tmp/1501677363692556/1501677363692556 --groups jepa
    python scripts/export_aria_session.py /tmp/1501677363692556/1501677363692556 --fps 5 --output-dir exports/
"""

import argparse
import json
import time
from pathlib import Path

import av
import cv2
import h5py  # type: ignore[import]
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# V-JEPA 2 constants (shared with export_hdd_session.py)
# ---------------------------------------------------------------------------

VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32   # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256     # 16h x 16w
VJEPA2_SPATIAL_H = 16
VJEPA2_SPATIAL_W = 16


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_metadata(session_dir: Path) -> dict:
    """Load Aria metadata.json and extract timing info.

    Returns:
        Dict with 'recording_id', 'recording_start_epoch', 'duration_seconds'.
    """
    meta_path = session_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    recording_id = meta["recording_id"]
    duration = meta["duration_seconds"]

    # Extract recording start time from inner metadata
    inner = json.loads(meta["tags"]["metadata"])
    recording_start_epoch = inner["start_time"]

    return {
        "recording_id": recording_id,
        "recording_start_epoch": float(recording_start_epoch),
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------------
# GPS loading
# ---------------------------------------------------------------------------


def load_gps(
    session_dir: Path,
    accuracy_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GPS data from Aria gps.json.

    Uses stream 0 (281-1). Filters to entries with non-zero lat/lng and
    accuracy <= threshold. Uses utc_time_ms for epoch timestamps directly.

    Returns:
        (epoch_timestamps, latitudes, longitudes) as numpy arrays, sorted by
        timestamp. Empty arrays if no valid GPS.
    """
    gps_path = session_dir / "gps.json"
    with open(gps_path) as f:
        gps_data = json.load(f)

    # Use stream 0 (281-1)
    samples = gps_data[0]["samples"]

    timestamps = []
    lats = []
    lngs = []

    for s in samples:
        lat = s.get("latitude", 0.0)
        lng = s.get("longitude", 0.0)
        utc_ms = s.get("utc_time_ms", 0)
        accuracy = s.get("accuracy", float("inf"))

        # Skip invalid: zero coords, no UTC time, or poor accuracy
        if lat == 0.0 and lng == 0.0:
            continue
        if utc_ms <= 0:
            continue
        if accuracy > accuracy_max:
            continue

        timestamps.append(utc_ms / 1000.0)
        lats.append(lat)
        lngs.append(lng)

    timestamps = np.array(timestamps, dtype=np.float64)
    lats = np.array(lats, dtype=np.float64)
    lngs = np.array(lngs, dtype=np.float64)

    # Sort by timestamp
    if len(timestamps) > 0:
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        lats = lats[order]
        lngs = lngs[order]

    return timestamps, lats, lngs


def compute_epoch_offset(session_dir: Path) -> float:
    """Compute device-clock to epoch offset from GPS utc_time_ms.

    Returns:
        epoch_offset such that epoch_time = device_timestamp + epoch_offset.
    """
    gps_path = session_dir / "gps.json"
    with open(gps_path) as f:
        gps_data = json.load(f)

    samples = gps_data[0]["samples"]
    offsets = []
    for s in samples:
        utc_ms = s.get("utc_time_ms", 0)
        dev_ts = s.get("timestamp", 0.0)
        if utc_ms > 0 and dev_ts > 1.0:
            offsets.append(utc_ms / 1000.0 - dev_ts)

    if not offsets:
        raise ValueError("No valid GPS entries with utc_time_ms to compute epoch offset")

    return float(np.median(offsets))


# ---------------------------------------------------------------------------
# IMU loading
# ---------------------------------------------------------------------------


def load_imu(
    session_dir: Path,
    epoch_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load IMU data from Aria imu.json.

    Uses stream 0 (1202-1, ~1kHz accelerometer + gyroscope). Converts device
    timestamps to epoch using the provided offset.

    Args:
        session_dir: Path to session directory.
        epoch_offset: Offset to convert device timestamps to epoch seconds.

    Returns:
        (epoch_timestamps, accel (N,3), gyro (N,3)) as numpy arrays.
    """
    imu_path = session_dir / "imu.json"

    print("  Loading IMU JSON (this may take a moment for large files)...")
    with open(imu_path) as f:
        imu_data = json.load(f)

    samples = imu_data[0]["samples"]
    n = len(samples)

    timestamps = np.empty(n, dtype=np.float64)
    accel = np.empty((n, 3), dtype=np.float32)
    gyro = np.empty((n, 3), dtype=np.float32)

    for i, s in enumerate(samples):
        timestamps[i] = s["timestamp"] + epoch_offset
        accel[i] = s["accel"]
        gyro[i] = s["gyro"]

    return timestamps, accel, gyro


def downsample_imu(
    timestamps: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    target_rate: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample 1kHz IMU data to target_rate by averaging windows.

    Args:
        timestamps: (N,) epoch timestamps at ~1kHz.
        accel: (N, 3) accelerometer data.
        gyro: (N, 3) gyroscope data.
        target_rate: Target sampling rate in Hz (default 100).

    Returns:
        (timestamps_ds, accel_ds, gyro_ds) downsampled arrays.
    """
    window = round(1000 / target_rate)
    n = len(timestamps)
    n_trunc = (n // window) * window

    timestamps = timestamps[:n_trunc].reshape(-1, window)
    accel = accel[:n_trunc].reshape(-1, window, 3)
    gyro = gyro[:n_trunc].reshape(-1, window, 3)

    return (
        timestamps.mean(axis=1),
        accel.mean(axis=1).astype(np.float32),
        gyro.mean(axis=1).astype(np.float32),
    )


def extract_frames_with_sensors(
    video_path: str,
    recording_start_epoch: float,
    gps_timestamps: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
    imu_timestamps: np.ndarray,
    imu_accel: np.ndarray,
    imu_gyro: np.ndarray,
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> tuple[
    list[np.ndarray], np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Extract frames from Aria video and align with GPS and IMU.

    Frame epoch time = recording_start_epoch + PTS_seconds. For each sampled
    frame, finds the nearest GPS and IMU entries via searchsorted.

    Returns:
        (frames, timestamps, latitudes, longitudes, accel, gyro)
        All aligned and filtered to frames with valid GPS.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    time_base = float(stream.time_base or 0)

    sample_interval = video_fps / target_fps
    has_gps = len(gps_timestamps) > 0
    has_imu = len(imu_timestamps) > 0

    frames = []
    timestamps = []
    lats = []
    lngs = []
    accels = []
    gyros = []

    frame_count = 0
    next_sample = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        if frame_count >= next_sample:
            frame_time_sec = float(frame.pts) * time_base
            frame_ts = recording_start_epoch + frame_time_sec

            # GPS alignment
            if has_gps:
                gps_idx = np.searchsorted(gps_timestamps, frame_ts)
                gps_idx = min(int(gps_idx), len(gps_timestamps) - 1)
                # Also check the entry before
                if gps_idx > 0:
                    d_after = abs(gps_timestamps[gps_idx] - frame_ts)
                    d_before = abs(gps_timestamps[gps_idx - 1] - frame_ts)
                    if d_before < d_after:
                        gps_idx -= 1
                lat = gps_lats[gps_idx]
                lng = gps_lngs[gps_idx]
            else:
                lat = 0.0
                lng = 0.0

            # IMU alignment
            if has_imu:
                imu_idx = np.searchsorted(imu_timestamps, frame_ts)
                imu_idx = min(int(imu_idx), len(imu_timestamps) - 1)
                if imu_idx > 0:
                    d_after = abs(imu_timestamps[imu_idx] - frame_ts)
                    d_before = abs(imu_timestamps[imu_idx - 1] - frame_ts)
                    if d_before < d_after:
                        imu_idx -= 1
                accel_val = imu_accel[imu_idx]
                gyro_val = imu_gyro[imu_idx]
            else:
                accel_val = np.zeros(3, dtype=np.float32)
                gyro_val = np.zeros(3, dtype=np.float32)

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
            accels.append(accel_val)
            gyros.append(gyro_val)

            next_sample += sample_interval

        frame_count += 1

    container.close()

    return (
        frames,
        np.array(timestamps, dtype=np.float64),
        np.array(lats, dtype=np.float64),
        np.array(lngs, dtype=np.float64),
        np.array(accels, dtype=np.float32),
        np.array(gyros, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# DINOv3 feature extraction
# ---------------------------------------------------------------------------


def extract_dino_features(
    frames: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 16,
    layer: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract DINOv3 CLS embeddings and spatial attention maps.

    Returns:
        (embeddings (N, 1024), attention_maps (N, H, W)) as numpy arrays.
    """
    from video_retrieval.models import DINOv3Encoder

    encoder = DINOv3Encoder(device=device)
    skip_tokens = 1 + encoder.num_register_tokens

    all_embeddings = []
    all_attn_maps = []

    for i in tqdm(range(0, len(frames), batch_size), desc="DINOv3 inference"):
        batch_frames = frames[i : i + batch_size]
        inputs = encoder._preprocess(batch_frames)

        with torch.no_grad():
            outputs = encoder.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())

        attn = outputs.attentions[layer]
        attn = attn.mean(dim=1)
        cls_attn = attn[:, 0, skip_tokens:]

        num_patches = cls_attn.shape[1]
        h = w = int(num_patches ** 0.5)
        attn_maps = cls_attn.view(-1, h, w)
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
    """Build context/target masks that split along the temporal axis."""
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
    lats: np.ndarray,
    lngs: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
    device: str = "cuda",
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Extract V-JEPA 2 embeddings and prediction error maps.

    Slides a window of 64 frames with the given stride. Each embedding is
    assigned to the center timestamp of its window. GPS/IMU values at window
    center are taken from the pre-aligned input arrays.

    Returns:
        (embeddings, prediction_maps, timestamps, lats, lngs, accel, gyro)
    """
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )
    model = model.to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = _build_temporal_masks(
        n_context_steps, torch.device(device),
    )

    n_frames = len(frames)

    all_embeddings = []
    all_pred_maps = []
    all_timestamps = []
    all_lats = []
    all_lngs = []
    all_accels = []
    all_gyros = []

    if n_frames < VJEPA2_NUM_FRAMES:
        starts = [0]
    else:
        starts = list(range(0, n_frames - VJEPA2_NUM_FRAMES + 1, stride))
        if starts[-1] + VJEPA2_NUM_FRAMES < n_frames:
            starts.append(n_frames - VJEPA2_NUM_FRAMES)

    for start in tqdm(starts, desc="V-JEPA 2 inference"):
        end = start + VJEPA2_NUM_FRAMES
        window_frames = frames[start:end]

        while len(window_frames) < VJEPA2_NUM_FRAMES:
            window_frames.append(window_frames[-1])

        center = min(start + VJEPA2_NUM_FRAMES // 2, n_frames - 1)

        inputs = processor(videos=window_frames, return_tensors="pt")
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
            ground_truth = ground_truth.reshape(
                n_target_steps, VJEPA2_SPATIAL, -1,
            )

            error = (predicted - ground_truth).norm(dim=-1)
            error_map = error.mean(dim=0)
            error_map = error_map.reshape(VJEPA2_SPATIAL_H, VJEPA2_SPATIAL_W)

        all_embeddings.append(mean_emb.cpu().numpy())
        all_pred_maps.append(error_map.cpu().numpy())
        all_timestamps.append(timestamps[center])
        all_lats.append(lats[center])
        all_lngs.append(lngs[center])
        all_accels.append(accels[center])
        all_gyros.append(gyros[center])

    del model, processor
    torch.cuda.empty_cache()

    return (
        np.stack(all_embeddings).astype(np.float32),
        np.stack(all_pred_maps).astype(np.float32),
        np.array(all_timestamps, dtype=np.float64),
        np.array(all_lats, dtype=np.float64),
        np.array(all_lngs, dtype=np.float64),
        np.array(all_accels, dtype=np.float32),
        np.array(all_gyros, dtype=np.float32),
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
    accel: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
):
    """Write features to HDF5 file.

    Creates or appends to the file, creating the specified group. All datasets
    within the group share the same N (one record per frame/window).
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
    if accel is not None:
        accel = accel[order]
    if gyro is not None:
        gyro = gyro[order]

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
        if accel is not None:
            g.create_dataset("accel", data=accel)
        if gyro is not None:
            g.create_dataset("gyro", data=gyro)


def write_sensor_groups(
    output_path: Path,
    imu_ts: np.ndarray,
    imu_accel: np.ndarray,
    imu_gyro: np.ndarray,
    gps_ts: np.ndarray,
    gps_lats: np.ndarray,
    gps_lngs: np.ndarray,
):
    """Write standalone imu/ and gps/ groups to HDF5 at their native rates.

    Uses append mode and deletes existing groups if present.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "a") as f:
        # IMU group
        if len(imu_ts) > 0:
            if "imu" in f:
                del f["imu"]
            g = f.create_group("imu")
            g.create_dataset("timestamps", data=imu_ts)
            g.create_dataset("accel", data=imu_accel)
            g.create_dataset("gyro", data=imu_gyro)

        # GPS group
        if len(gps_ts) > 0:
            if "gps" in f:
                del f["gps"]
            g = f.create_group("gps")
            g.create_dataset("timestamps", data=gps_ts)
            g.create_dataset("lat", data=gps_lats)
            g.create_dataset("lng", data=gps_lngs)


def main():
    parser = argparse.ArgumentParser(
        description="Export an Aria recording to HDF5 with model features."
    )
    parser.add_argument(
        "session_dir",
        type=str,
        help="Path to Aria session directory containing data.mp4, gps.json, "
        "imu.json, metadata.json",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dino", "jepa"],
        help="HDF5 group names to export (default: dino jepa).",
    )
    parser.add_argument(
        "--fps", type=float, default=3.0,
        help="Frame sampling rate (default: 3)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str, default="exports/",
        help="Output directory (default: exports/)",
    )
    parser.add_argument(
        "--max-resolution", type=int, default=518,
        help="Max frame height for DINOv3 (default: 518)",
    )
    parser.add_argument(
        "--gps-accuracy-max", type=float, default=100.0,
        help="Reject GPS fixes with accuracy > this in meters (default: 100)",
    )
    parser.add_argument(
        "--imu-rate", type=int, default=100,
        help="Target IMU sampling rate in Hz; raw 1kHz is downsampled (default: 100)",
    )
    parser.add_argument(
        "--jepa-stride", type=int, default=8,
        help="V-JEPA 2 sliding window stride in frames (default: 8)",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    project_root = Path(__file__).parent.parent
    output_dir_base = project_root / args.output_dir

    # ------------------------------------------------------------------
    # Step 1: Load metadata
    # ------------------------------------------------------------------
    print("=" * 70)
    meta = load_metadata(session_dir)
    recording_id = meta["recording_id"]
    recording_start_epoch = meta["recording_start_epoch"]

    output_dir = output_dir_base / recording_id
    output_path = output_dir / "features.h5"

    model_labels = ", ".join(
        "V-JEPA 2" if g.startswith("jepa") else "DINOv3" for g in args.groups
    )
    print(f"ARIA SESSION EXPORT: {recording_id} -> {model_labels}")
    print("=" * 70)

    print(f"\nStep 1: Metadata")
    print(f"  Recording ID: {recording_id}")
    print(f"  Recording start: {recording_start_epoch:.0f} (epoch)")
    print(f"  Duration: {meta['duration_seconds']:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Load GPS
    # ------------------------------------------------------------------
    print(f"\nStep 2: Loading GPS data (accuracy <= {args.gps_accuracy_max}m)...")
    gps_ts, gps_lats, gps_lngs = load_gps(
        session_dir, accuracy_max=args.gps_accuracy_max,
    )
    print(f"  Valid GPS entries: {len(gps_ts)}")
    if len(gps_ts) > 0:
        print(f"  GPS time range: {gps_ts[0]:.1f} - {gps_ts[-1]:.1f}")
        print(f"  Lat range: {gps_lats.min():.6f} - {gps_lats.max():.6f}")
        print(f"  Lng range: {gps_lngs.min():.6f} - {gps_lngs.max():.6f}")
    else:
        print("  WARNING: No valid GPS entries. Coordinates will be zero.")

    # ------------------------------------------------------------------
    # Step 3: Load IMU
    # ------------------------------------------------------------------
    print("\nStep 3: Loading IMU data...")
    try:
        epoch_offset = compute_epoch_offset(session_dir)
        print(f"  Epoch offset: {epoch_offset:.3f}")
        imu_ts, imu_accel, imu_gyro = load_imu(session_dir, epoch_offset)
        print(f"  IMU samples: {len(imu_ts)}")
        print(f"  IMU time range: {imu_ts[0]:.1f} - {imu_ts[-1]:.1f}")
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"  WARNING: Could not load IMU: {e}")
        imu_ts = np.array([], dtype=np.float64)
        imu_accel = np.empty((0, 3), dtype=np.float32)
        imu_gyro = np.empty((0, 3), dtype=np.float32)

    # Downsample IMU for standalone group
    if len(imu_ts) > 0:
        imu_ds_ts, imu_ds_accel, imu_ds_gyro = downsample_imu(
            imu_ts, imu_accel, imu_gyro, target_rate=args.imu_rate,
        )
        print(f"  Downsampled IMU: {len(imu_ts)} -> {len(imu_ds_ts)} "
              f"samples ({args.imu_rate} Hz)")
    else:
        imu_ds_ts = imu_ts
        imu_ds_accel = imu_accel
        imu_ds_gyro = imu_gyro

    # ------------------------------------------------------------------
    # Step 4: Extract frames with sensor alignment
    # ------------------------------------------------------------------
    video_path = str(session_dir / "data.mp4")
    needs_dino = any(not g.startswith("jepa") for g in args.groups)
    max_res = args.max_resolution if needs_dino else 256

    print(f"\nStep 4: Extracting frames at {args.fps} fps (max_resolution={max_res})...")
    t0 = time.time()
    frames, timestamps, lats, lngs, accels, gyros = extract_frames_with_sensors(
        video_path,
        recording_start_epoch,
        gps_ts, gps_lats, gps_lngs,
        imu_ts, imu_accel, imu_gyro,
        target_fps=args.fps,
        max_resolution=max_res,
    )
    print(f"  Extracted {len(frames)} frames ({time.time() - t0:.1f}s)")

    if len(frames) == 0:
        print("ERROR: No frames extracted. Exiting.")
        return

    print(f"  Time range: {timestamps[0]:.1f} - {timestamps[-1]:.1f}")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.1f}s")
    print(f"  Frame size: {frames[0].shape[1]}x{frames[0].shape[0]}")

    # ------------------------------------------------------------------
    # Step 5: Run models and write HDF5
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name in args.groups:
        use_jepa = group_name.startswith("jepa")

        if use_jepa:
            print(f"\nStep 5 [{group_name}]: Running V-JEPA 2 "
                  f"(sliding window of {VJEPA2_NUM_FRAMES} frames, "
                  f"stride {args.jepa_stride})...")

            if max_res > 256:
                jepa_frames = []
                for img in frames:
                    scale = 256 / img.shape[0]
                    new_w = int(img.shape[1] * scale)
                    jepa_frames.append(cv2.resize(img, (new_w, 256)))
            else:
                jepa_frames = frames

            t0 = time.time()
            (embeddings, prediction_maps, ts_out, lats_out, lngs_out,
             accels_out, gyros_out) = extract_vjepa2_features(
                jepa_frames, timestamps, lats, lngs, accels, gyros,
                device=args.device,
                stride=args.jepa_stride,
            )
            attention_maps = None
            print(f"  {len(embeddings)} embeddings ({embeddings.shape[1]}D), "
                  f"prediction maps {prediction_maps.shape[1]}x{prediction_maps.shape[2]} "
                  f"in {time.time() - t0:.1f}s")
        else:
            print(f"\nStep 5 [{group_name}]: Running DINOv3...")
            t0 = time.time()
            embeddings, attention_maps = extract_dino_features(
                frames, device=args.device,
            )
            prediction_maps = None
            ts_out, lats_out, lngs_out = timestamps, lats, lngs
            accels_out, gyros_out = accels, gyros
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
            accel=accels_out,
            gyro=gyros_out,
        )

    # ------------------------------------------------------------------
    # Step 6: Write standalone sensor groups
    # ------------------------------------------------------------------
    print("\nStep 6: Writing standalone sensor groups...")
    write_sensor_groups(
        output_path,
        imu_ds_ts, imu_ds_accel, imu_ds_gyro,
        gps_ts, gps_lats, gps_lngs,
    )
    if len(imu_ds_ts) > 0:
        print(f"  imu/: {len(imu_ds_ts)} samples at {args.imu_rate} Hz")
    else:
        print("  imu/: skipped (no IMU data)")
    if len(gps_ts) > 0:
        print(f"  gps/: {len(gps_ts)} entries at native 1 Hz")
    else:
        print("  gps/: skipped (no GPS data)")

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

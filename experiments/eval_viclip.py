#!/usr/bin/env python3
"""ViCLIP Evaluation Across All Benchmarks.

Evaluates ViCLIP (ViT-L, InternVid-10M) as a video-native single-vector
baseline. Tests whether contrastive video-text pretraining changes the
temporal blindness story compared to image encoders applied per-frame.

Benchmarks: VCDB, HDD, nuScenes, EPIC-Kitchens s_rev, Nymeria, MUVR News.

Hypothesis: ViCLIP will match DINOv3 BoF on copy detection, fail on HDD
(order-invariant single vector), and show s_rev ~ 1.0.

Usage:
    python experiments/eval_viclip.py \\
        --vcdb-dir datasets/vcdb/core_dataset \\
        --hdd-dir datasets/hdd \\
        --epic-dir datasets/epic_kitchens \\
        --benchmarks vcdb hdd epic

    # All benchmarks
    python experiments/eval_viclip.py --benchmarks all
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# ViCLIP default config
VICLIP_NUM_FRAMES = 8
VICLIP_RESOLUTION = 224
VICLIP_EMBED_DIM = 768


# ---------------------------------------------------------------------------
# ViCLIP model loading
# ---------------------------------------------------------------------------


def load_viclip(model_dir: str, device: torch.device):
    """Load ViCLIP model from local directory.

    Args:
        model_dir: Path to OpenGVLab/ViCLIP directory.
        device: Torch device.

    Returns:
        ViCLIP model in eval mode.
    """
    # Add model dir to path so imports work
    sys.path.insert(0, model_dir)

    from viclip import ViCLIP

    weight_path = os.path.join(model_dir, "ViClip-InternVid-10M-FLT.pth")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(model_dir, "ViCLIP-L_InternVid-FLT-10M.pth")

    model = ViCLIP(pretrain=weight_path)
    model = model.to(device).eval()

    # Remove model dir from path
    sys.path.pop(0)

    return model


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames_uniform(
    video_path: str,
    n_frames: int = VICLIP_NUM_FRAMES,
    resolution: int = VICLIP_RESOLUTION,
) -> list[np.ndarray]:
    """Extract n_frames uniformly spaced frames from a video.

    Returns list of (H, W, 3) RGB numpy arrays resized to resolution x resolution.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        # Estimate from duration
        duration = float(stream.duration * stream.time_base) if stream.duration else 10.0
        fps = float(stream.average_rate or 30)
        total_frames = int(duration * fps)

    # Uniform frame indices
    indices = np.linspace(0, max(total_frames - 1, 1), n_frames, dtype=int)

    frames = []
    frame_idx = 0
    target_set = set(indices.tolist())

    for frame in container.decode(video=0):
        if frame_idx in target_set:
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (resolution, resolution))
            frames.append(img)
            if len(frames) >= n_frames:
                break
        frame_idx += 1

    container.close()

    # Pad if not enough
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((resolution, resolution, 3), dtype=np.uint8))

    return frames[:n_frames]


def extract_frames_timed(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = VICLIP_NUM_FRAMES,
    resolution: int = VICLIP_RESOLUTION,
) -> list[np.ndarray]:
    """Extract n_frames uniformly from a time window."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base or 1 / 30)

    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0
    target_times = [start_sec + i * duration / max(n_frames - 1, 1) for i in range(n_frames)]

    seek_sec = max(0.0, start_sec - 1.0)
    container.seek(int(seek_sec / time_base), stream=stream)

    all_frames = []
    all_times = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        t = float(frame.pts) * time_base
        if t < start_sec - 0.1:
            continue
        if t > end_sec + 0.1:
            break
        img = frame.to_ndarray(format="rgb24")
        img = cv2.resize(img, (resolution, resolution))
        all_frames.append(img)
        all_times.append(t)

    container.close()

    if not all_frames:
        return [np.zeros((resolution, resolution, 3), dtype=np.uint8)] * n_frames

    # Pick closest to target times
    all_times_arr = np.array(all_times)
    selected = []
    for t in target_times:
        idx = int(np.argmin(np.abs(all_times_arr - t)))
        selected.append(all_frames[idx])

    return selected


def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """Convert list of (H, W, 3) numpy arrays to (1, T, C, H, W) tensor.

    Normalizes to [0, 1] range as expected by ViCLIP.
    """
    tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
    video = torch.stack(tensors)  # (T, C, H, W)
    return video.unsqueeze(0)  # (1, T, C, H, W)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_video(model, frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Encode frames into a single L2-normalized 768-d vector."""
    video = frames_to_tensor(frames).to(device)
    emb = model.encode_vision(video, test=True)
    # Handle tuple return (some versions return (features, pooled))
    if isinstance(emb, tuple):
        emb = emb[-1]  # pooled
    if emb.ndim == 3:
        emb = emb.mean(dim=1)  # mean-pool if per-frame
    emb = F.normalize(emb.squeeze(0), dim=-1)
    return emb.cpu()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ap(scores, labels, n_resamples=1000, seed=42):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    ap = average_precision_score(labels, scores)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot.append(ap)
        else:
            boot.append(average_precision_score(l, s))
    boot = np.array(boot)
    return float(ap), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# VCDB evaluation
# ---------------------------------------------------------------------------


def eval_vcdb(model, vcdb_dir: Path, device: torch.device) -> dict:
    """VCDB copy detection + reversal + scramble."""
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"

    # Load annotations
    copy_pairs: set[tuple[str, str]] = set()
    for fname in sorted(os.listdir(str(ann_dir))):
        if not fname.endswith(".txt"):
            continue
        cat = fname.replace(".txt", "")
        with open(os.path.join(str(ann_dir), fname)) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 6:
                    continue
                va = os.path.join(cat, parts[0].strip())
                vb = os.path.join(cat, parts[1].strip())
                if va != vb:
                    copy_pairs.add((min(va, vb), max(va, vb)))

    # Discover videos
    videos = []
    for cat in sorted(os.listdir(str(vid_dir))):
        cat_path = os.path.join(str(vid_dir), cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))

    print(f"  VCDB: {len(videos)} videos, {len(copy_pairs)} copy pairs")

    # Extract features
    features = {}
    failed = 0
    for vp in tqdm(videos, desc="ViCLIP VCDB"):
        try:
            frames = extract_frames_uniform(os.path.join(str(vid_dir), vp))
            features[vp] = encode_video(model, frames, device)
        except Exception:
            failed += 1

    print(f"  Extracted: {len(features)}/{len(videos)} ({failed} failed)")

    # Build pair set
    keys = sorted(features.keys())
    key_set = set(keys)
    n = len(keys)

    pairs = []
    labels = []
    for a, b in copy_pairs:
        if a in key_set and b in key_set:
            pairs.append((a, b))
            labels.append(1)

    n_pos = len(pairs)
    rng = np.random.RandomState(42)
    neg_count = 0
    attempts = 0
    while neg_count < n_pos and attempts < n_pos * 20:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i == j:
            attempts += 1
            continue
        pair = (min(keys[i], keys[j]), max(keys[i], keys[j]))
        if pair not in copy_pairs and pair not in set(pairs):
            pairs.append(pair)
            labels.append(0)
            neg_count += 1
        attempts += 1

    # Compute similarities
    scores = []
    for a, b in pairs:
        sim = float(torch.dot(features[a], features[b]).item())
        scores.append(sim)

    scores = np.array(scores)
    labels = np.array(labels)
    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
    auc = roc_auc_score(labels, scores)

    # Reversal s_rev
    rev_sims = []
    for vp in tqdm(list(features.keys())[:200], desc="ViCLIP reversal"):
        try:
            frames = extract_frames_uniform(os.path.join(str(vid_dir), vp))
            fwd_emb = features[vp]
            rev_frames = frames[::-1]
            rev_emb = encode_video(model, rev_frames, device)
            s_rev = float(torch.dot(fwd_emb, rev_emb).item())
            rev_sims.append(s_rev)
        except Exception:
            continue

    mean_s_rev = float(np.mean(rev_sims)) if rev_sims else float("nan")

    result = {
        "ap": ap, "ci_low": ci_lo, "ci_high": ci_hi, "auc": float(auc),
        "s_rev": mean_s_rev, "n_pairs": len(pairs), "n_pos": n_pos,
    }
    print(f"  VCDB AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  s_rev={mean_s_rev:.4f}")
    return result


# ---------------------------------------------------------------------------
# HDD evaluation
# ---------------------------------------------------------------------------


def eval_hdd(model, hdd_dir: Path, device: torch.device) -> dict:
    """HDD maneuver discrimination."""
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_hdd_intersections import (
        ManeuverSegment,
        cluster_intersections,
        discover_sessions,
        extract_maneuver_segments,
        filter_mixed_clusters,
        load_gps,
    )

    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="HDD sessions"):
        info = sessions[sid]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            sid, labels, gps_ts, gps_lats, gps_lngs,
            info["video_path"], info["video_start_unix"],
        )
        all_segments.extend(segs)

    clusters = cluster_intersections(all_segments)
    mixed = filter_mixed_clusters(clusters, max_clusters=50)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    print(f"  HDD: {len(eval_segments)} segments, {len(mixed)} clusters")

    # Extract features
    features = {}
    failed = 0
    for i, seg in enumerate(tqdm(eval_segments, desc="ViCLIP HDD")):
        start_sec = seg.start_frame / 3.0 - 3.0
        end_sec = seg.end_frame / 3.0 + 3.0
        start_sec = max(0.0, start_sec)
        try:
            frames = extract_frames_timed(seg.video_path, start_sec, end_sec)
            features[i] = encode_video(model, frames, device)
        except Exception:
            failed += 1

    print(f"  Extracted: {len(features)}/{len(eval_segments)} ({failed} failed)")

    # Pairwise evaluation
    pair_scores = []
    pair_labels = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                sim = float(torch.dot(features[indices[a]], features[indices[b]]).item())
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                pair_scores.append(sim)
                pair_labels.append(gt)

    scores = np.array(pair_scores)
    labels = np.array(pair_labels)
    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)

    result = {
        "ap": ap, "ci_low": ci_lo, "ci_high": ci_hi,
        "n_pairs": len(labels), "n_pos": int(labels.sum()),
    }
    print(f"  HDD AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    return result


# ---------------------------------------------------------------------------
# EPIC s_rev evaluation
# ---------------------------------------------------------------------------


def eval_epic(model, epic_dir: Path, device: torch.device) -> dict:
    """EPIC-Kitchens forward/reverse s_rev."""
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_epic_temporal_order import load_sequences

    sequences = load_sequences(epic_dir, max_sequences=200)

    s_revs = []
    failed = 0
    for seq in tqdm(sequences, desc="ViCLIP EPIC s_rev"):
        try:
            frames = extract_frames_timed(
                seq["video_path"], seq["start_sec"], seq["stop_sec"]
            )
            fwd_emb = encode_video(model, frames, device)
            rev_emb = encode_video(model, frames[::-1], device)
            s_rev = float(torch.dot(fwd_emb, rev_emb).item())
            s_revs.append(s_rev)
        except Exception:
            failed += 1

    mean_s_rev = float(np.mean(s_revs))
    std_s_rev = float(np.std(s_revs))

    result = {
        "mean_s_rev": mean_s_rev,
        "std_s_rev": std_s_rev,
        "n_sequences": len(s_revs),
        "failed": failed,
    }
    print(f"  EPIC s_rev={mean_s_rev:.4f} ± {std_s_rev:.4f} ({len(s_revs)} sequences)")
    return result


# ---------------------------------------------------------------------------
# nuScenes evaluation
# ---------------------------------------------------------------------------


def eval_nuscenes(model, nuscenes_dir: Path, device: torch.device, version: str = "v1.0-trainval") -> dict:
    """nuScenes maneuver discrimination."""
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_nuscenes_intersections import (
        ManeuverSegment as NuScenesSegment,
        assign_keyframes_to_segments,
        cluster_intersections as nuscenes_cluster,
        filter_mixed_clusters as nuscenes_filter,
        get_scene_keyframes,
        load_can_bus,
        load_nuscenes_metadata,
        segment_maneuvers,
    )

    print(f"  Loading nuScenes metadata (version={version})...")
    metadata = load_nuscenes_metadata(nuscenes_dir, version)

    all_segments: list = []
    scene_keyframes: dict = {}
    for scene in tqdm(metadata.scenes, desc="nuScenes scenes"):
        scene_name = scene["name"]
        can_data = load_can_bus(nuscenes_dir / "can_bus" / "can_bus", scene_name)
        if can_data is None:
            continue
        timestamps, steering, yaw_rates, positions, speeds = can_data
        segs = segment_maneuvers(timestamps, steering, yaw_rates, positions, speeds, scene_name)
        all_segments.extend(segs)
        kfs = get_scene_keyframes(scene, metadata, nuscenes_dir)
        if kfs:
            scene_keyframes[scene_name] = kfs

    clusters = nuscenes_cluster(all_segments, eps=30.0, min_samples=2)
    mixed = nuscenes_filter(clusters, max_clusters=50)

    eval_segments: list = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    assign_keyframes_to_segments(eval_segments, scene_keyframes)
    print(f"  nuScenes: {len(eval_segments)} segments, {len(mixed)} clusters")

    # Extract features from keyframe images
    features = {}
    failed = 0
    for i, seg in enumerate(tqdm(eval_segments, desc="ViCLIP nuScenes")):
        if len(seg.keyframe_paths) < 3:
            failed += 1
            continue
        try:
            frames = []
            for path in seg.keyframe_paths[:VICLIP_NUM_FRAMES]:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (VICLIP_RESOLUTION, VICLIP_RESOLUTION))
                frames.append(img)

            if len(frames) < 3:
                failed += 1
                continue

            while len(frames) < VICLIP_NUM_FRAMES:
                frames.append(frames[-1])
            frames = frames[:VICLIP_NUM_FRAMES]

            features[i] = encode_video(model, frames, device)
        except Exception:
            failed += 1

    print(f"  Extracted: {len(features)}/{len(eval_segments)} ({failed} failed)")

    # Pairwise evaluation
    pair_scores = []
    pair_labels = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                sim = float(torch.dot(features[indices[a]], features[indices[b]]).item())
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                pair_scores.append(sim)
                pair_labels.append(gt)

    scores = np.array(pair_scores)
    labels = np.array(pair_labels)
    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)

    result = {
        "ap": ap, "ci_low": ci_lo, "ci_high": ci_hi,
        "n_pairs": len(labels), "n_pos": int(labels.sum()),
    }
    print(f"  nuScenes AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ViCLIP evaluation across all benchmarks")
    parser.add_argument("--viclip-dir", type=str,
                        default=None,
                        help="Path to ViCLIP model directory. Falls back to VTD_MODEL_DIR/OpenGVLab/ViCLIP")
    parser.add_argument("--vcdb-dir", type=str, default="datasets/vcdb/core_dataset")
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--epic-dir", type=str, default="datasets/epic_kitchens")
    parser.add_argument("--nuscenes-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--benchmarks", nargs="+", default=["vcdb", "hdd", "epic"],
                        help="Benchmarks to run (vcdb, hdd, epic, nuscenes, or all)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    device = torch.device(args.device)

    # Resolve ViCLIP model path
    viclip_dir = args.viclip_dir
    if viclip_dir is None:
        model_base = os.environ.get("VTD_MODEL_DIR")
        if model_base:
            viclip_dir = os.path.join(model_base, "OpenGVLab", "ViCLIP")
        else:
            viclip_dir = str(project_root / "models" / "OpenGVLab" / "ViCLIP")

    if not os.path.exists(viclip_dir):
        print(f"ERROR: ViCLIP model not found at {viclip_dir}")
        return

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["vcdb", "hdd", "epic", "nuscenes"]

    print("=" * 70)
    print("ViCLIP EVALUATION (ViT-L, InternVid-10M)")
    print("=" * 70)
    print(f"  Model: {viclip_dir}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Frames: {VICLIP_NUM_FRAMES} @ {VICLIP_RESOLUTION}x{VICLIP_RESOLUTION}")
    print(f"  Embedding: {VICLIP_EMBED_DIM}-d, cosine similarity")

    # Load model
    print("\nLoading ViCLIP...")
    t0 = time.time()
    model = load_viclip(viclip_dir, device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    results = {}

    if "vcdb" in benchmarks:
        print(f"\n{'=' * 50}")
        print("VCDB")
        print("=" * 50)
        vcdb_dir = Path(args.vcdb_dir) if os.path.isabs(args.vcdb_dir) else project_root / args.vcdb_dir
        results["vcdb"] = eval_vcdb(model, vcdb_dir, device)

    if "hdd" in benchmarks:
        print(f"\n{'=' * 50}")
        print("HDD")
        print("=" * 50)
        hdd_dir = project_root / args.hdd_dir
        results["hdd"] = eval_hdd(model, hdd_dir, device)

    if "epic" in benchmarks:
        print(f"\n{'=' * 50}")
        print("EPIC-Kitchens s_rev")
        print("=" * 50)
        epic_dir = project_root / args.epic_dir
        results["epic"] = eval_epic(model, epic_dir, device)

    if "nuscenes" in benchmarks:
        print(f"\n{'=' * 50}")
        print("nuScenes")
        print("=" * 50)
        nuscenes_dir = Path(args.nuscenes_dir) if args.nuscenes_dir else project_root / "datasets" / "nuscenes"
        results["nuscenes"] = eval_nuscenes(model, nuscenes_dir, device)

    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    if "vcdb" in results:
        r = results["vcdb"]
        print(f"  VCDB:     AP={r['ap']:.4f}  s_rev={r['s_rev']:.4f}")
    if "hdd" in results:
        r = results["hdd"]
        print(f"  HDD:      AP={r['ap']:.4f}")
    if "nuscenes" in results:
        r = results["nuscenes"]
        print(f"  nuScenes: AP={r['ap']:.4f}")
    if "epic" in results:
        r = results["epic"]
        print(f"  EPIC:     s_rev={r['mean_s_rev']:.4f}")

    # Save
    out_path = project_root / "datasets" / "viclip_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

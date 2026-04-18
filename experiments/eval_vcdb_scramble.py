#!/usr/bin/env python3
"""VCDB Temporal Scramble Gradient.

Measures how sensitive each copy detection method is to partial temporal
disruption. For each scramble level K, video B's frame embeddings are
divided into K equal chunks, shuffled, then reassembled. BoF/Chamfer
(order-invariant) should remain flat; temporal derivative and attention
trajectory should degrade with more chunks.

This is a more nuanced generalization of the binary reversal test —
instead of asking "does reversal break detection?", it asks "how much
temporal disruption can each method tolerate?"

Usage:
    python experiments/eval_vcdb_scramble.py                # Full run
    python experiments/eval_vcdb_scramble.py --skip-vjepa2  # DINOv3 only
"""

import argparse
import hashlib
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
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance, dtw_distance_batch
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

SCRAMBLE_LEVELS = [1, 2, 3, 4, 6, 8, 12, 16]

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
}

METHOD_LABELS = {
    "bag_of_frames": "Bag of Frames",
    "chamfer": "Chamfer",
    "temporal_derivative": "Temporal Derivative",
    "attention_trajectory": "Attention Trajectory",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "V-JEPA 2 Temporal Res.",
}


# ---------------------------------------------------------------------------
# VCDB loading (from eval_vcdb.py)
# ---------------------------------------------------------------------------


def parse_timestamp(ts: str) -> float:
    """Parse HH:MM:SS timestamp to seconds."""
    parts = ts.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


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
# V-JEPA 2 feature extraction (from eval_vcdb_reversal.py)
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
# Scramble logic
# ---------------------------------------------------------------------------


def scramble_tensor(tensor: torch.Tensor, n_chunks: int, seed: int) -> torch.Tensor:
    """Split tensor along dim 0 into n_chunks, shuffle chunks, reassemble.

    Args:
        tensor: (T, ...) tensor to scramble.
        n_chunks: Number of chunks to split into.
        seed: Random seed for reproducible permutation.

    Returns:
        Scrambled tensor with same shape as input.
    """
    if n_chunks <= 1:
        return tensor

    T = tensor.shape[0]
    chunk_size = T // n_chunks
    if chunk_size < 1:
        chunk_size = 1
        n_chunks = T

    # Split into chunks (last chunk may be larger to absorb remainder)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else T
        chunks.append(tensor[start:end])

    # Shuffle chunks
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(chunks))
    shuffled = [chunks[p] for p in perm]

    return torch.cat(shuffled, dim=0)


def scramble_features(
    features: dict[str, dict],
    n_chunks: int,
    video_keys: list[str],
) -> dict[str, dict]:
    """Scramble B's features (embeddings + centroids) for all videos.

    Uses a deterministic seed per video (hashlib MD5 of key + n_chunks)
    for reproducibility across runs. mean_emb is recomputed from scrambled
    embeddings (should be identical to original since mean is order-invariant).
    """
    scrambled = {}
    for idx, vp in enumerate(video_keys):
        if vp not in features:
            continue
        feat = features[vp]
        seed = int(hashlib.md5(f"{vp}_{n_chunks}".encode()).hexdigest(), 16) % (2**31)

        emb_s = scramble_tensor(feat["embeddings"], n_chunks, seed)
        cen_s = scramble_tensor(feat["centroids"], n_chunks, seed)
        mean_emb_s = F.normalize(emb_s.mean(dim=0), dim=0)

        scrambled[vp] = {
            "embeddings": emb_s,
            "centroids": cen_s,
            "mean_emb": mean_emb_s,
        }
    return scrambled


def scramble_vjepa2_residual(
    residual: torch.Tensor,
    n_chunks: int,
    seed: int,
) -> torch.Tensor:
    """Scramble V-JEPA 2 temporal residual chunks."""
    return scramble_tensor(residual, n_chunks, seed)


# ---------------------------------------------------------------------------
# Evaluation (from eval_vcdb.py)
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
    return {"ap": ap, "auc": auc, "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# Similarity computation at one scramble level
# ---------------------------------------------------------------------------


def compute_scrambled_similarities(
    features_a: dict[str, dict],
    features_b: dict[str, dict],
    pairs_to_compute: set[tuple[str, str]],
    methods: list[str],
    vjepa2_a: dict[str, dict] | None = None,
    vjepa2_b: dict[str, dict] | None = None,
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute similarities: A forward vs B scrambled.

    Uses batched DTW for temporal_derivative, attention_trajectory, and
    vjepa2_temporal_residual methods. BoF and Chamfer are already fast
    (no DTW) so they stay per-pair.

    Args:
        features_a: DINOv3 features for A (forward).
        features_b: DINOv3 features for B (scrambled).
        pairs_to_compute: Set of (a, b) pairs.
        methods: List of method names to compute.
        vjepa2_a: V-JEPA 2 features for A (forward).
        vjepa2_b: V-JEPA 2 features for B (scrambled residuals).
    """
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints for A (forward)
    deriv_fps_a = {}
    traj_fps_a = {}
    for k in features_a:
        deriv_fps_a[k] = deriv_fp.compute_fingerprint(features_a[k]["embeddings"])
        traj_fps_a[k] = traj_fp.compute_fingerprint(features_a[k]["centroids"])

    results = {m: {} for m in methods}

    # Filter to valid pairs
    valid_pairs = [
        (a, b) for a, b in pairs_to_compute if a in features_a and b in features_b
    ]

    # --- Non-DTW methods (already fast, compute per-pair) ---
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
# Plotting
# ---------------------------------------------------------------------------


def plot_scramble_gradient(
    sweep_results: dict[int, dict[str, dict[str, float]]],
    methods: list[str],
    fig_dir: Path,
):
    """Generate line plot: x=chunks, y=AP/AUC, one line per method."""
    levels = sorted(sweep_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        color = METHOD_COLORS.get(method, "#95a5a6")
        label = METHOD_LABELS.get(method, method)

        aps = [sweep_results[k][method]["ap"] for k in levels]
        aucs = [sweep_results[k][method]["auc"] for k in levels]

        ax1.plot(levels, aps, "o-", color=color, label=label, linewidth=2, markersize=6)
        ax2.plot(
            levels, aucs, "o-", color=color, label=label, linewidth=2, markersize=6
        )

    ax1.set_xlabel("Number of Chunks (scramble level)", fontsize=12)
    ax1.set_ylabel("Average Precision", fontsize=12)
    ax1.set_title("Average Precision vs Temporal Scramble", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(levels)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of Chunks (scramble level)", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("ROC-AUC vs Temporal Scramble", fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(levels)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "VCDB Temporal Scramble Gradient\n"
        "(K=1: no scramble, K=16: fine-grained chunk shuffle)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    plot_path = fig_dir / "vcdb_scramble_gradient.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VCDB Temporal Scramble Gradient")
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
        "--skip-vjepa2", action="store_true", help="Skip V-JEPA 2 (DINOv3 only, faster)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction even if cached features exist",
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

    # Validate paths before proceeding
    if not ann_dir.exists():
        print(f"ERROR: Annotation dir not found: {ann_dir}")
        return
    if not vid_dir.exists():
        print(f"ERROR: Video dir not found: {vid_dir}")
        return

    print("=" * 70)
    print("VCDB TEMPORAL SCRAMBLE GRADIENT")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  Sample rate: {args.sample_rate}, max frames: {args.max_frames}")
    print(f"  Scramble levels: {SCRAMBLE_LEVELS}")
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
    # Step 2: Extract DINOv3 features (forward only — scrambling is on tensors)
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
        t_feat = time.time() - t0
        print(f"  DINOv3 feature extraction: {t_feat:.1f}s")

        del encoder
        torch.cuda.empty_cache()

        save_feature_cache(features, dinov3_cache)

    keys = sorted(features.keys())
    print(f"  Videos with features: {len(keys)}")

    # ------------------------------------------------------------------
    # Step 3: Extract V-JEPA 2 features (forward only)
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
    # Step 4: Build shared pair set (same as eval_vcdb_reversal.py)
    # ------------------------------------------------------------------
    print("\nStep 4: Building shared pair set...")
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    pairs_to_compute = set()
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pairs_to_compute.add((a, b))

    n_pos = len(pairs_to_compute)

    n_neg_target = n_pos
    rng = np.random.RandomState(42)
    neg_count = 0
    neg_attempts = 0
    while neg_count < n_neg_target and neg_attempts < n_neg_target * 20:
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

    print(f"  Total pairs: {len(pairs_to_compute)} " f"({n_pos} pos + {neg_count} neg)")

    # ------------------------------------------------------------------
    # Step 5: Sweep scramble levels
    # ------------------------------------------------------------------
    methods = list(DINOV3_METHODS)
    if vjepa2_features is not None:
        methods.extend(VJEPA2_METHODS)

    sweep_results: dict[int, dict[str, dict[str, float]]] = {}

    print("\nStep 5: Sweeping scramble levels...", flush=True)
    for n_chunks in SCRAMBLE_LEVELS:
        print(f"\n  --- Scramble K={n_chunks} ---", flush=True)
        t0 = time.time()

        # Scramble B's features
        features_b = scramble_features(features, n_chunks, keys)

        # Scramble V-JEPA 2 residuals (mean_emb is order-invariant)
        vjepa2_b = None
        if vjepa2_features is not None:
            vjepa2_b = {}
            for vp in vjepa2_features:
                seed = int(hashlib.md5(f"{vp}_{n_chunks}".encode()).hexdigest(), 16) % (
                    2**31
                )
                vf = vjepa2_features[vp]
                vjepa2_b[vp] = {
                    "mean_emb": vf["mean_emb"],  # order-invariant
                    "temporal_residual": scramble_vjepa2_residual(
                        vf["temporal_residual"], n_chunks, seed
                    ),
                }

        # Compute similarities: A forward vs B scrambled
        sims = compute_scrambled_similarities(
            features,
            features_b,
            pairs_to_compute,
            methods,
            vjepa2_a=vjepa2_features,
            vjepa2_b=vjepa2_b,
        )

        # Evaluate each method
        level_results = {}
        for method in methods:
            if sims[method]:
                metrics = evaluate_method(sims[method], copy_pairs)
            else:
                metrics = {
                    "ap": float("nan"),
                    "auc": float("nan"),
                    "n_pos": 0,
                    "n_neg": 0,
                }
            level_results[method] = metrics

        sweep_results[n_chunks] = level_results
        elapsed = time.time() - t0

        # Print row
        parts = [f"K={n_chunks:>2d}"]
        for method in methods:
            ap = level_results[method]["ap"]
            parts.append(f"{METHOD_LABELS[method]}: {ap:.4f}")
        print(f"  {' | '.join(parts)}  ({elapsed:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: TEMPORAL SCRAMBLE GRADIENT")
    print("=" * 70)

    header = f"  {'K':>3s}"
    for method in methods:
        label = METHOD_LABELS[method][:16]
        header += f"  {label:>16s}"
    print(header)
    print("  " + "-" * (4 + 18 * len(methods)))

    for n_chunks in SCRAMBLE_LEVELS:
        row = f"  {n_chunks:>3d}"
        for method in methods:
            ap = sweep_results[n_chunks][method]["ap"]
            row += f"  {ap:>16.4f}"
        print(row)

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 7: Generate figure
    # ------------------------------------------------------------------
    print("\nStep 7: Generating figure...")
    plot_scramble_gradient(sweep_results, methods, fig_dir)

    # ------------------------------------------------------------------
    # Step 8: Save results JSON
    # ------------------------------------------------------------------
    results_path = project_root / "datasets" / "vcdb" / "scramble_gradient_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for n_chunks in SCRAMBLE_LEVELS:
        serializable[str(n_chunks)] = {}
        for method in methods:
            serializable[str(n_chunks)][method] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in sweep_results[n_chunks][method].items()
            }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

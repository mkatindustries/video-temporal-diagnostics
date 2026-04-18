#!/usr/bin/env python3
"""VCDB Reversal Attack Evaluation.

Tests video reversal as an attack vector against copy detection methods.
No prior work (including Fojcik & Syga 2025) has tested this attack.

The experiment exposes a fundamental tension:
- Order-invariant methods (bag-of-frames, Chamfer) are blind to reversal
  (identical scores for forward and reversed) but work for copy detection.
- Order-aware methods (temporal derivative, attention trajectory, V-JEPA 2
  temporal residual) detect reversal but fail to match reversed copies.

For DINOv3 methods, reversal is done by flipping already-extracted tensors
(embeddings.flip(0), centroids.flip(0)) — zero re-extraction cost.
V-JEPA 2 requires re-extraction with reversed frame input because positional
encodings and predictor context/target split change with frame order.

Usage:
    python experiments/eval_vcdb_reversal.py                # Full run
    python experiments/eval_vcdb_reversal.py --skip-vjepa2  # DINOv3 only
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
    "bag_of_frames": "Bag of\nFrames",
    "chamfer": "Chamfer",
    "temporal_derivative": "Temporal\nDerivative",
    "attention_trajectory": "Attention\nTrajectory",
    "vjepa2_bag_of_tokens": "V-JEPA 2\nBag of Tokens",
    "vjepa2_temporal_residual": "V-JEPA 2\nTemporal Res.",
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
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2.

    Samples uniformly across the full video duration.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base) if stream.duration else 60.0

    target_fps = VJEPA2_NUM_FRAMES / max(duration, 1.0)
    container.close()

    # Use a simple frame extraction loop
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

    # Pad to VJEPA2_NUM_FRAMES if needed
    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])

    return frames[:VJEPA2_NUM_FRAMES]


def extract_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    vid_base_dir: str,
    video_relpaths: list[str],
    device: torch.device,
    reverse: bool = False,
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for all videos.

    Args:
        reverse: If True, reverse frame order before processing.
            This changes the representation because V-JEPA 2 uses
            positional encodings and the predictor context/target split
            is affected by frame order.

    Returns:
        Dict mapping relpath -> {'mean_emb': ..., 'temporal_residual': ...}
    """
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    direction = "reversed" if reverse else "forward"
    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc=f"V-JEPA 2 features ({direction})"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames = load_frames_for_vjepa2(path)

            if reverse:
                frames = frames[::-1]

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
        f"  V-JEPA 2 ({direction}): {len(features)}/{len(video_relpaths)} "
        f"({failed} failed)"
    )
    return features


# ---------------------------------------------------------------------------
# Similarity computation: normal vs reversed
# ---------------------------------------------------------------------------


def compute_reversal_similarities(
    features: dict[str, dict],
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
    vjepa2_fwd: dict[str, dict] | None,
    vjepa2_rev: dict[str, dict] | None,
) -> dict[str, dict[str, dict[tuple[str, str], float]]]:
    """Compute similarities for both normal and reversed conditions.

    For each annotated pair (A, B):
    - Normal:   sim(A_forward, B_forward)
    - Reversed: sim(A_forward, B_reversed)

    For DINOv3, reversal is done by tensor flipping (zero cost).
    For V-JEPA 2, uses pre-extracted forward/reversed features.

    Returns:
        Dict mapping condition ("normal", "reversed") ->
            method_name -> {(vidA, vidB): similarity}
    """
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute forward fingerprints for all videos
    print("  Pre-computing forward fingerprints...")
    deriv_fps_fwd = {}
    traj_fps_fwd = {}
    for k in tqdm(keys, desc="  Forward fingerprints", leave=False):
        emb = features[k]["embeddings"]
        cen = features[k]["centroids"]
        deriv_fps_fwd[k] = deriv_fp.compute_fingerprint(emb)
        traj_fps_fwd[k] = traj_fp.compute_fingerprint(cen)

    # Identify which pairs to compute:
    # All positive pairs that we have features for
    pairs_to_compute = set()
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pairs_to_compute.add((a, b))

    n_pos = len(pairs_to_compute)
    print(f"  Positive pairs with features: {n_pos}")

    # Sample negative pairs — match the positive count for balanced eval
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

    print(
        f"  Total pairs to evaluate: {len(pairs_to_compute)} "
        f"({n_pos} pos + {neg_count} neg)"
    )

    # Initialize results: condition -> method -> {pair: sim}
    methods = list(DINOV3_METHODS)
    if vjepa2_fwd is not None and vjepa2_rev is not None:
        methods.extend(VJEPA2_METHODS)

    results = {
        "normal": {m: {} for m in methods},
        "reversed": {m: {} for m in methods},
    }

    for a, b in tqdm(pairs_to_compute, desc="  Computing similarities"):
        ea = features[a]["embeddings"]
        eb = features[b]["embeddings"]

        # ---- NORMAL CONDITION: sim(A_forward, B_forward) ----

        # Bag-of-frames (forward)
        m1 = F.normalize(ea.mean(dim=0), dim=0)
        m2 = F.normalize(eb.mean(dim=0), dim=0)
        results["normal"]["bag_of_frames"][(a, b)] = float(torch.dot(m1, m2).item())

        # Chamfer (forward)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        results["normal"]["chamfer"][(a, b)] = (max_ab + max_ba) / 2

        # Temporal derivative DTW (forward)
        results["normal"]["temporal_derivative"][(a, b)] = deriv_fp.compare(
            deriv_fps_fwd[a], deriv_fps_fwd[b]
        )

        # Attention trajectory DTW (forward)
        results["normal"]["attention_trajectory"][(a, b)] = traj_fp.compare(
            traj_fps_fwd[a], traj_fps_fwd[b]
        )

        # ---- REVERSED CONDITION: sim(A_forward, B_reversed) ----

        # Reverse B's embeddings and centroids
        eb_rev = eb.flip(0)
        cb_rev = features[b]["centroids"].flip(0)

        # Bag-of-frames (reversed) — mean is order-invariant, should be identical
        m2_rev = F.normalize(eb_rev.mean(dim=0), dim=0)
        results["reversed"]["bag_of_frames"][(a, b)] = float(
            torch.dot(m1, m2_rev).item()
        )

        # Chamfer (reversed) — max-matching is order-invariant, should be identical
        sim_matrix_rev = torch.mm(ea, eb_rev.t())
        max_ab_rev = sim_matrix_rev.max(dim=1).values.mean().item()
        max_ba_rev = sim_matrix_rev.max(dim=0).values.mean().item()
        results["reversed"]["chamfer"][(a, b)] = (max_ab_rev + max_ba_rev) / 2

        # Temporal derivative DTW (reversed) — different derivative sequences
        deriv_b_rev = deriv_fp.compute_fingerprint(eb_rev)
        results["reversed"]["temporal_derivative"][(a, b)] = deriv_fp.compare(
            deriv_fps_fwd[a], deriv_b_rev
        )

        # Attention trajectory DTW (reversed) — reversed trajectory
        traj_b_rev = traj_fp.compute_fingerprint(cb_rev)
        results["reversed"]["attention_trajectory"][(a, b)] = traj_fp.compare(
            traj_fps_fwd[a], traj_b_rev
        )

        # ---- V-JEPA 2 methods ----
        if vjepa2_fwd is not None and vjepa2_rev is not None:
            if a in vjepa2_fwd and b in vjepa2_fwd and b in vjepa2_rev:
                va_fwd = vjepa2_fwd[a]
                vb_fwd = vjepa2_fwd[b]
                vb_rev = vjepa2_rev[b]

                # Normal: V-JEPA 2 bag-of-tokens
                results["normal"]["vjepa2_bag_of_tokens"][(a, b)] = float(
                    torch.dot(va_fwd["mean_emb"], vb_fwd["mean_emb"]).item()
                )

                # Normal: V-JEPA 2 temporal residual
                res_dist_fwd = dtw_distance(
                    va_fwd["temporal_residual"],
                    vb_fwd["temporal_residual"],
                    normalize=True,
                )
                results["normal"]["vjepa2_temporal_residual"][(a, b)] = float(
                    torch.exp(torch.tensor(-res_dist_fwd)).item()
                )

                # Reversed: V-JEPA 2 bag-of-tokens
                results["reversed"]["vjepa2_bag_of_tokens"][(a, b)] = float(
                    torch.dot(va_fwd["mean_emb"], vb_rev["mean_emb"]).item()
                )

                # Reversed: V-JEPA 2 temporal residual
                res_dist_rev = dtw_distance(
                    va_fwd["temporal_residual"],
                    vb_rev["temporal_residual"],
                    normalize=True,
                )
                results["reversed"]["vjepa2_temporal_residual"][(a, b)] = float(
                    torch.exp(torch.tensor(-res_dist_rev)).item()
                )

    return results


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
# Plotting
# ---------------------------------------------------------------------------


def plot_reversal_comparison(
    eval_results: dict[str, dict[str, dict[str, float]]],
    methods: list[str],
    fig_dir: Path,
):
    """Generate grouped bar chart comparing normal vs reversed conditions.

    Args:
        eval_results: condition -> method -> {'ap': ..., 'auc': ...}
        methods: List of method names in display order.
        fig_dir: Directory to save figure.
    """
    labels = [METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods]
    normal_aps = [eval_results["normal"][m]["ap"] for m in methods]
    reversed_aps = [eval_results["reversed"][m]["ap"] for m in methods]
    normal_aucs = [eval_results["normal"][m]["auc"] for m in methods]
    reversed_aucs = [eval_results["reversed"][m]["auc"] for m in methods]

    x = np.arange(len(methods))
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # AP subplot
    bars_n = ax1.bar(
        x - bar_width / 2,
        normal_aps,
        bar_width,
        color=[METHOD_COLORS.get(m, "#95a5a6") for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    bars_r = ax1.bar(
        x + bar_width / 2,
        reversed_aps,
        bar_width,
        color=[METHOD_COLORS.get(m, "#95a5a6") for m in methods],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.5,
        hatch="//",
    )

    for bar, val in zip(bars_n, normal_aps):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    for bar, val in zip(bars_r, reversed_aps):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Average Precision", fontsize=12)
    ax1.set_title("Average Precision", fontsize=13)
    ax1.set_ylim(0, 1.15)
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor="#cccccc", edgecolor="black", linewidth=0.8, label="Normal"),
        Patch(
            facecolor="#cccccc",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.5,
            hatch="//",
            label="Reversed",
        ),
    ]
    ax1.legend(handles=legend_patches, fontsize=10)

    # AUC subplot
    bars_n = ax2.bar(
        x - bar_width / 2,
        normal_aucs,
        bar_width,
        color=[METHOD_COLORS.get(m, "#95a5a6") for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    bars_r = ax2.bar(
        x + bar_width / 2,
        reversed_aucs,
        bar_width,
        color=[METHOD_COLORS.get(m, "#95a5a6") for m in methods],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.5,
        hatch="//",
    )

    for bar, val in zip(bars_n, normal_aucs):
        if not np.isnan(val):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    for bar, val in zip(bars_r, reversed_aucs):
        if not np.isnan(val):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("ROC-AUC", fontsize=13)
    ax2.set_ylim(0, 1.15)
    ax2.legend(handles=legend_patches, fontsize=10)

    fig.suptitle(
        "VCDB Reversal Attack: Normal vs Reversed Copy Detection",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    plot_path = fig_dir / "vcdb_reversal_attack.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VCDB Reversal Attack Evaluation")
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
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("VCDB REVERSAL ATTACK EVALUATION")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  Sample rate: {args.sample_rate}, max frames: {args.max_frames}")
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
    # Step 2: Extract DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 2: Loading DINOv3 encoder...")
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

    keys = sorted(features.keys())
    print(f"  Videos with features: {len(keys)}")

    # ------------------------------------------------------------------
    # Step 3: Extract V-JEPA 2 features (forward + reversed)
    # ------------------------------------------------------------------
    vjepa2_fwd = None
    vjepa2_rev = None

    if not args.skip_vjepa2:
        print("\nStep 3: Loading V-JEPA 2 model...")
        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )

        # Extract forward features
        print("  Extracting V-JEPA 2 forward features...")
        t0 = time.time()
        vjepa2_fwd = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            str(vid_dir),
            keys,
            torch.device(args.device),
            reverse=False,
        )
        print(f"  Forward extraction: {time.time() - t0:.1f}s")

        # Extract reversed features
        print("  Extracting V-JEPA 2 reversed features...")
        t0 = time.time()
        vjepa2_rev = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            str(vid_dir),
            keys,
            torch.device(args.device),
            reverse=True,
        )
        print(f"  Reversed extraction: {time.time() - t0:.1f}s")

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()
    else:
        print("\nStep 3: Skipping V-JEPA 2 (--skip-vjepa2)")

    # ------------------------------------------------------------------
    # Step 4: Compute similarities for normal and reversed conditions
    # ------------------------------------------------------------------
    print("\nStep 4: Computing similarities (normal + reversed)...")
    t0 = time.time()
    all_sims = compute_reversal_similarities(
        features,
        keys,
        copy_pairs,
        vjepa2_fwd,
        vjepa2_rev,
    )
    print(f"  Similarity computation: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Evaluate AP/AUC per method per condition
    # ------------------------------------------------------------------
    methods = list(DINOV3_METHODS)
    if vjepa2_fwd is not None:
        methods.extend(VJEPA2_METHODS)

    print("\n" + "=" * 70)
    print("RESULTS: NORMAL vs REVERSED")
    print("=" * 70)

    eval_results = {"normal": {}, "reversed": {}}

    for condition in ["normal", "reversed"]:
        for method in methods:
            scores = all_sims[condition][method]
            if not scores:
                eval_results[condition][method] = {
                    "ap": float("nan"),
                    "auc": float("nan"),
                    "n_pos": 0,
                    "n_neg": 0,
                }
                continue
            metrics = evaluate_method(scores, copy_pairs)
            eval_results[condition][method] = metrics

    # Print comparison table
    print(
        f"\n  {'Method':<28s}  {'Normal AP':>10s}  {'Reversed AP':>11s}  "
        f"{'Delta AP':>9s}  {'Normal AUC':>10s}  {'Reversed AUC':>12s}  "
        f"{'Delta AUC':>9s}"
    )
    print("  " + "-" * 100)

    for method in methods:
        n_ap = eval_results["normal"][method]["ap"]
        r_ap = eval_results["reversed"][method]["ap"]
        n_auc = eval_results["normal"][method]["auc"]
        r_auc = eval_results["reversed"][method]["auc"]
        d_ap = r_ap - n_ap if not (np.isnan(n_ap) or np.isnan(r_ap)) else float("nan")
        d_auc = (
            r_auc - n_auc if not (np.isnan(n_auc) or np.isnan(r_auc)) else float("nan")
        )

        label = METHOD_LABELS.get(method, method).replace("\n", " ")
        print(
            f"  {label:<28s}  {n_ap:>10.4f}  {r_ap:>11.4f}  "
            f"{d_ap:>+9.4f}  {n_auc:>10.4f}  {r_auc:>12.4f}  "
            f"{d_auc:>+9.4f}"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 6: Generate figure
    # ------------------------------------------------------------------
    print("\nStep 6: Generating figure...")
    plot_reversal_comparison(eval_results, methods, fig_dir)

    # ------------------------------------------------------------------
    # Step 7: Save results JSON
    # ------------------------------------------------------------------
    results_path = project_root / "datasets" / "vcdb" / "reversal_attack_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for condition in ["normal", "reversed"]:
        serializable[condition] = {}
        for method in methods:
            serializable[condition][method] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in eval_results[condition][method].items()
            }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

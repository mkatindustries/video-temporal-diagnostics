#!/usr/bin/env python3
"""VCDB Benchmark Evaluation.

Evaluates temporal fingerprinting methods on the VCDB (Video Copy Detection
Benchmark) core dataset. VCDB contains 528 videos across 28 categories with
9,236 annotated partial copy pairs.

Evaluation approach (balanced pair sampling):
- Extract features for ALL 528 videos.
- Build a global set of positive copy pairs from all annotations.
- Sample an equal number of negative pairs (1:1 ratio, seed=42).
- Compute binary AP (sklearn.average_precision_score) and ROC-AUC
  over the pooled pair set.

Note: this is *binary AP on sampled pairs*, not per-query mAP over the
full corpus. The 1:1 ratio is validated by a sensitivity analysis across
1:1, 1:5, and all-pairs ratios (see eval_vcdb_neg_sensitivity.py) that
confirms rank stability across sampling strategies.

Methods evaluated:
1. DINOv3 bag-of-frames (mean embedding cosine)
2. DINOv3 temporal derivative DTW
3. DINOv3 attention trajectory DTW
4. ViSiL Chamfer similarity (order-agnostic frame matching)

Usage:
    python experiments/eval_vcdb.py [--sample-rate 10] [--device cuda]
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

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
from video_retrieval.models import DINOv3Encoder
from video_retrieval.utils.video import load_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# ---------------------------------------------------------------------------
# VCDB loading
# ---------------------------------------------------------------------------


def parse_timestamp(ts: str) -> float:
    """Parse HH:MM:SS timestamp to seconds."""
    parts = ts.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    """Load all VCDB annotations as global (videoA_path, videoB_path) pairs.

    Video paths are relative to vid_base_dir, e.g. "baggio_penalty_1994/abc.mp4".

    Returns:
        Set of (vidA_relpath, vidB_relpath) tuples (sorted order).
    """
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
    """Discover all video files under the VCDB core_dataset directory.

    Returns:
        List of relative paths like "baggio_penalty_1994/abc.mp4".
    """
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


def extract_all_features(
    encoder: DINOv3Encoder,
    vid_base_dir: str,
    video_relpaths: list[str],
    sample_rate: int = 10,
    max_frames: int = 60,
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

    for vp in tqdm(video_relpaths, desc="Extracting features"):
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
        except Exception as e:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(video_relpaths)} " f"({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_chamfer_sim(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """ViSiL-style Chamfer similarity (order-agnostic frame matching)."""
    sim_matrix = torch.mm(emb1, emb2.t())
    max_1to2 = sim_matrix.max(dim=1).values.mean().item()
    max_2to1 = sim_matrix.max(dim=0).values.mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_pairwise_dtw(
    features: dict[str, dict],
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute all method similarities on a shared set of sampled pairs.

    DTW is O(T1*T2) per pair, so computing all N*(N-1)/2 pairs for 528
    videos would be ~140K DTW calls. Instead we sample a shared evaluation
    set: all positive pairs + sampled negatives. All methods (including
    bag-of-frames and Chamfer) are evaluated on this same set for fair
    comparison.

    Returns:
        Dict mapping method name -> {(vidA, vidB): similarity}.
        All methods share the same pair set.
    """
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints for all videos
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for k in tqdm(keys, desc="  Fingerprints", leave=False):
        emb = features[k]["embeddings"]
        cen = features[k]["centroids"]
        deriv_fps[k] = deriv_fp.compute_fingerprint(emb)
        traj_fps[k] = traj_fp.compute_fingerprint(cen)

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

    # Compute all method similarities on the shared pair set
    bof_sims = {}
    chamfer_sims = {}
    deriv_sims = {}
    traj_sims = {}

    for a, b in tqdm(pairs_to_compute, desc="  Computing similarities"):
        ea = features[a]["embeddings"]
        eb = features[b]["embeddings"]

        # Bag-of-frames
        m1 = F.normalize(ea.mean(dim=0), dim=0)
        m2 = F.normalize(eb.mean(dim=0), dim=0)
        bof_sims[(a, b)] = float(torch.dot(m1, m2).item())

        # Chamfer
        sim_matrix = torch.mm(ea, eb.t())
        max_1to2 = sim_matrix.max(dim=1).values.mean().item()
        max_2to1 = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims[(a, b)] = (max_1to2 + max_2to1) / 2

        # DTW methods
        deriv_sims[(a, b)] = deriv_fp.compare(deriv_fps[a], deriv_fps[b])
        traj_sims[(a, b)] = traj_fp.compare(traj_fps[a], traj_fps[b])

    return {
        "bag_of_frames": bof_sims,
        "chamfer": chamfer_sims,
        "temporal_derivative": deriv_sims,
        "attention_trajectory": traj_sims,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    scores: dict[tuple[str, str], float] | np.ndarray,
    copy_pairs: set[tuple[str, str]],
    keys: list[str] | None = None,
) -> dict[str, float]:
    """Compute AP and AUC for a method.

    Args:
        scores: Either a dict of {(vidA, vidB): sim} or a full NxN matrix.
        copy_pairs: Set of positive pairs.
        keys: Required if scores is a matrix (maps index to video key).

    Returns:
        Dict with 'ap', 'auc', 'n_pos', 'n_neg'.
    """
    y_true = []
    y_score = []

    if isinstance(scores, np.ndarray):
        # Full matrix mode
        n = scores.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                # pyrefly: ignore [unsupported-operation]
                pair = tuple(sorted([keys[i], keys[j]]))
                y_true.append(1 if pair in copy_pairs else 0)
                y_score.append(scores[i, j])
    else:
        # Dict mode (sampled pairs)
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VCDB Benchmark Evaluation")
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
        "--max-frames",
        type=int,
        default=100,
        help="Max frames per video (uniformly sampled across full duration)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("VCDB BENCHMARK EVALUATION (cross-category retrieval)")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  Sample rate: {args.sample_rate}, max frames: {args.max_frames}")

    # Discover videos and load annotations
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Annotated copy pairs: {len(copy_pairs)}")

    # Extract features for all videos
    print("\nLoading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    print("\nExtracting features for all videos...")
    t0 = time.time()
    features = extract_all_features(
        encoder,
        str(vid_dir),
        videos,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )
    t_feat = time.time() - t0
    print(f"  Total feature extraction: {t_feat:.1f}s")

    keys = sorted(features.keys())
    n = len(keys)
    print(f"  Videos with features: {n}")

    # --- Compute all similarities on shared pair set ---
    print("\nComputing similarities (all methods on shared pair set)...")
    t0 = time.time()
    all_sims = compute_pairwise_dtw(features, keys, copy_pairs)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # --- Evaluate all methods on the same pairs ---
    print("\n" + "=" * 70)
    print("RESULTS (all methods evaluated on same pair set)")
    print("=" * 70)

    results = {}
    method_order = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]

    for name in method_order:
        metrics = evaluate_method(all_sims[name], copy_pairs)
        results[name] = metrics
        print(
            f"  {name:<25s}  AP={metrics['ap']:.4f}  AUC={metrics['auc']:.4f}  "
            f"(pos={metrics['n_pos']}, neg={metrics['n_neg']})"
        )

    print("=" * 70)

    # Save results
    results_path = project_root / "datasets" / "vcdb" / "eval_results.json"
    serializable = {
        m: {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in vals.items()
        }
        for m, vals in results.items()
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Dump pair-level scores for bootstrap CI computation
    pair_data = {}
    for method_name, sims_dict in all_sims.items():
        scores_list = []
        labels_list = []
        for pair, sim in sims_dict.items():
            scores_list.append(float(sim))
            labels_list.append(1 if pair in copy_pairs else 0)
        pair_data[method_name] = {"scores": scores_list, "labels": labels_list}

    pair_path = project_root / "datasets" / "vcdb" / "pair_scores.json"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    # Figure
    if results:
        methods = [m for m in method_order if m in results]
        aps = [results[m]["ap"] for m in methods]
        aucs = [results[m]["auc"] for m in methods]
        labels = [m.replace("_", " ").title() for m in methods]
        colors = {
            "bag_of_frames": "#e74c3c",
            "temporal_derivative": "#2ecc71",
            "attention_trajectory": "#3498db",
            "chamfer": "#1abc9c",
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # AP
        bars = ax1.bar(
            range(len(methods)),
            aps,
            color=[colors.get(m, "#95a5a6") for m in methods],
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        for bar, val in zip(bars, aps):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax1.set_ylabel("Average Precision", fontsize=12)
        ax1.set_title("VCDB: Average Precision", fontsize=13)
        ax1.set_ylim(0, min(1.0, max(aps) * 1.15 + 0.05))

        # AUC
        bars = ax2.bar(
            range(len(methods)),
            aucs,
            color=[colors.get(m, "#95a5a6") for m in methods],
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        for bar, val in zip(bars, aucs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax2.set_ylabel("ROC-AUC", fontsize=12)
        ax2.set_title("VCDB: ROC-AUC", fontsize=13)
        ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))

        fig.suptitle(
            "VCDB Copy Detection Benchmark (528 videos, cross-category)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        plot_path = fig_dir / "vcdb_benchmark.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Figure saved to {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

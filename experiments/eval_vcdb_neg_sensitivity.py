#!/usr/bin/env python3
"""VCDB Negative-Sampling Sensitivity Analysis.

Runs the VCDB copy detection benchmark at multiple negative-sampling ratios
(1:1, 1:5, all-pairs) to verify that diagnostic conclusions are robust to
the choice of negative set. Reuses features from eval_vcdb.py.

This addresses Reviewer #2's concern that the 1:1 ratio is non-standard
and may inflate or distort AP/AUC values.

Usage:
    # Run with default ratios (1:1, 1:5, all-pairs)
    python experiments/eval_vcdb_neg_sensitivity.py

    # Run specific ratios
    python experiments/eval_vcdb_neg_sensitivity.py --neg-ratios 1 5 10

    # All-pairs only (slow: ~140K DTW comparisons)
    python experiments/eval_vcdb_neg_sensitivity.py --neg-ratios 0
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder
from video_retrieval.utils.video import load_video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# ---------------------------------------------------------------------------
# Shared with eval_vcdb.py — self-contained per project convention
# ---------------------------------------------------------------------------


def parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
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
                    a_s, b_s = sorted([vid_a, vid_b])
                    copy_pairs.add((a_s, b_s))
    return copy_pairs


def discover_videos(vid_base_dir: str) -> list[str]:
    videos = []
    for cat in sorted(os.listdir(vid_base_dir)):
        cat_path = os.path.join(vid_base_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))
    return videos


def extract_all_features(
    encoder: DINOv3Encoder,
    vid_base_dir: str,
    video_relpaths: list[str],
    sample_rate: int = 10,
    max_frames: int = 60,
) -> dict[str, dict]:
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
        except Exception:
            failed += 1
            continue
    print(f"  Extracted: {len(features)}/{len(video_relpaths)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Pair sampling with configurable negative ratio
# ---------------------------------------------------------------------------


def sample_pairs(
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
    neg_ratio: int,
    seed: int = 42,
) -> tuple[set[tuple[str, str]], int, int]:
    """Sample evaluation pairs with a given negative ratio.

    Args:
        keys: List of video keys that have features.
        copy_pairs: Set of positive pairs.
        neg_ratio: Number of negatives per positive (0 = all pairs).
        seed: Random seed.

    Returns:
        (pairs_to_compute, n_pos, n_neg)
    """
    n = len(keys)
    key_set = set(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # All positive pairs with features
    pairs_to_compute = set()
    for a, b in copy_pairs:
        if a in key_set and b in key_set:
            pairs_to_compute.add((a, b))

    n_pos = len(pairs_to_compute)

    if neg_ratio == 0:
        # All pairs mode
        for i in range(n):
            for j in range(i + 1, n):
                a_s, b_s = sorted([keys[i], keys[j]])
                pair = (a_s, b_s)
                if pair not in pairs_to_compute:
                    pairs_to_compute.add(pair)
        n_neg = len(pairs_to_compute) - n_pos
    else:
        # Sampled negatives
        n_neg_target = n_pos * neg_ratio
        rng = np.random.RandomState(seed)
        neg_count = 0
        neg_attempts = 0
        while neg_count < n_neg_target and neg_attempts < n_neg_target * 20:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i == j:
                neg_attempts += 1
                continue
            a_s, b_s = sorted([keys[i], keys[j]])
            pair = (a_s, b_s)
            if pair not in copy_pairs and pair not in pairs_to_compute:
                pairs_to_compute.add(pair)
                neg_count += 1
            neg_attempts += 1
        n_neg = neg_count

    return pairs_to_compute, n_pos, n_neg


def compute_similarities(
    features: dict[str, dict],
    pairs: set[tuple[str, str]],
    deriv_fps: dict,
    traj_fps: dict,
    deriv_fp: TemporalDerivativeFingerprint,
    traj_fp: TrajectoryFingerprint,
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute all method similarities for a given pair set.

    Uses batched DTW for temporal_derivative and attention_trajectory methods.
    """
    pairs_list = list(pairs)
    bof_sims = {}
    chamfer_sims = {}

    # BoF and Chamfer are fast — keep serial
    for a, b in tqdm(pairs_list, desc="  Computing BoF/Chamfer"):
        ea = features[a]["embeddings"]
        eb = features[b]["embeddings"]

        m1 = F.normalize(ea.mean(dim=0), dim=0)
        m2 = F.normalize(eb.mean(dim=0), dim=0)
        bof_sims[(a, b)] = float(torch.dot(m1, m2).item())

        sim_matrix = torch.mm(ea, eb.t())
        max_1to2 = sim_matrix.max(dim=1).values.mean().item()
        max_2to1 = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims[(a, b)] = (max_1to2 + max_2to1) / 2

    # Batched DTW for temporal derivative
    print("  Computing temporal derivative (batched DTW)...")
    deriv_seqs_a = [deriv_fps[a] for a, b in pairs_list]
    deriv_seqs_b = [deriv_fps[b] for a, b in pairs_list]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=True)
    deriv_sims = {}
    for i, (a, b) in enumerate(pairs_list):
        deriv_sims[(a, b)] = float(torch.exp(-deriv_dists[i]).item())

    # Batched DTW for attention trajectory
    print("  Computing attention trajectory (batched DTW)...")
    traj_seqs_a = [traj_fps[a] for a, b in pairs_list]
    traj_seqs_b = [traj_fps[b] for a, b in pairs_list]
    traj_dists = dtw_distance_batch(traj_seqs_a, traj_seqs_b, normalize=True)
    traj_sims = {}
    for i, (a, b) in enumerate(pairs_list):
        traj_sims[(a, b)] = float(torch.exp(-traj_dists[i] * 5).item())

    return {
        "bag_of_frames": bof_sims,
        "chamfer": chamfer_sims,
        "temporal_derivative": deriv_sims,
        "attention_trajectory": traj_sims,
    }


def evaluate_method(
    scores: dict[tuple[str, str], float],
    copy_pairs: set[tuple[str, str]],
) -> dict[str, float]:
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VCDB Negative-Sampling Sensitivity Analysis"
    )
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb/core_dataset",
        help="Path to VCDB core_dataset directory",
    )
    parser.add_argument(
        "--neg-ratios",
        type=int,
        nargs="+",
        default=[1, 5, 0],
        help="Negative-to-positive ratios (0 = all pairs)",
    )
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = project_root / args.vcdb_dir
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"

    print("=" * 70)
    print("VCDB NEGATIVE-SAMPLING SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  Neg ratios: {args.neg_ratios}")
    print(f"  (0 = all-pairs = ~139K comparisons)")

    # Discover and load
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}, Copy pairs: {len(copy_pairs)}")

    # Extract features (once, shared across all ratios)
    print("\nLoading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)
    print("Extracting features...")
    features = extract_all_features(
        encoder,
        str(vid_dir),
        videos,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )
    keys = sorted(features.keys())
    print(f"  Videos with features: {len(keys)}")

    # Pre-compute fingerprints (once)
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()
    deriv_fps = {}
    traj_fps = {}
    for k in tqdm(keys, desc="Pre-computing fingerprints"):
        deriv_fps[k] = deriv_fp.compute_fingerprint(features[k]["embeddings"])
        traj_fps[k] = traj_fp.compute_fingerprint(features[k]["centroids"])

    # Run at each ratio
    all_results = {}
    method_order = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]

    for ratio in args.neg_ratios:
        ratio_label = f"1:{ratio}" if ratio > 0 else "all_pairs"
        print(f"\n{'=' * 70}")
        print(f"RATIO: {ratio_label}")
        print(f"{'=' * 70}")

        pairs, n_pos, n_neg = sample_pairs(keys, copy_pairs, ratio, seed=42)
        print(f"  Pairs: {len(pairs)} ({n_pos} pos + {n_neg} neg)")

        if ratio == 0 and len(pairs) > 100000:
            print(
                f"  WARNING: {len(pairs)} pairs — this will take a while for DTW methods"
            )

        t0 = time.time()
        all_sims = compute_similarities(
            features, pairs, deriv_fps, traj_fps, deriv_fp, traj_fp
        )
        elapsed = time.time() - t0
        print(f"  Similarity computation: {elapsed:.1f}s")

        results = {}
        for name in method_order:
            metrics = evaluate_method(all_sims[name], copy_pairs)
            results[name] = metrics
            print(
                f"  {name:<25s}  AP={metrics['ap']:.4f}  AUC={metrics['auc']:.4f}  "
                f"(pos={metrics['n_pos']}, neg={metrics['n_neg']})"
            )
        all_results[ratio_label] = results

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: AP by negative-sampling ratio")
    print(f"{'=' * 70}")
    ratio_labels = [f"1:{r}" if r > 0 else "all_pairs" for r in args.neg_ratios]
    header = f"  {'Method':<25s}  " + "  ".join(f"{r:>10s}" for r in ratio_labels)
    print(header)
    print("  " + "-" * (25 + 12 * len(ratio_labels)))
    for name in method_order:
        vals = "  ".join(f"{all_results[r][name]['ap']:>10.4f}" for r in ratio_labels)
        print(f"  {name:<25s}  {vals}")

    # Check rank stability
    print(f"\n{'=' * 70}")
    print("RANK STABILITY CHECK")
    print(f"{'=' * 70}")
    for r in ratio_labels:
        ranking = sorted(
            method_order, key=lambda m: all_results[r][m]["ap"], reverse=True
        )
        print(f"  {r}: {' > '.join(ranking)}")

    # Save results
    results_path = project_root / "datasets" / "vcdb" / "neg_sensitivity_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        ratio_label: {
            m: {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in vals.items()
            }
            for m, vals in results.items()
        }
        for ratio_label, results in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate LaTeX table rows for paper
    print("\nLaTeX table rows:")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    cols = " & ".join(f"\\textbf{{{r}}}" for r in ratio_labels)
    print(f"\\textbf{{Method}} & {cols} \\\\")
    print("\\midrule")
    display = {
        "bag_of_frames": "DINOv3 BoF",
        "chamfer": "DINOv3 Chamfer",
        "temporal_derivative": "DINOv3 Temporal Deriv.",
        "attention_trajectory": "DINOv3 Attn.\\ Trajectory",
    }
    for name in method_order:
        vals = " & ".join(f"{all_results[r][name]['ap']:.3f}" for r in ratio_labels)
        print(f"{display[name]} & {vals} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()

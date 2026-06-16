#!/usr/bin/env python3
"""Honda HDD Intersection Clustering + Maneuver Discrimination Experiment.

Evaluates whether DINOv3 attention trajectory fingerprints can distinguish
different maneuvers (left turn vs right turn) at the same intersection,
where bag-of-frames fails because the visual background is identical.

Pipeline:
1. Load HDD session data (labels, GPS, video paths)
2. Extract contiguous maneuver segments with GPS midpoints
3. Cluster intersections using DBSCAN on GPS coordinates
4. Extract video clips for qualifying maneuver segments
5. Compute DINOv3 features (embeddings + attention centroids)
6. Compare all pairs within each cluster using 4 methods
7. Evaluate discrimination (AP, AUC) between same-maneuver and
   different-maneuver pairs

Hypothesis: Bag-of-frames and Chamfer will have low discrimination
(same background -> high similarity regardless of maneuver). Attention
trajectory will discriminate (gaze direction differs between left and
right turns).

Usage:
    python experiments/eval_hdd_intersections.py [--device cuda]
    python experiments/eval_hdd_intersections.py --fps-downsample 2 5 10 15
"""

import argparse
import json
import logging
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
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.fingerprints.trajectory import dtw_distance
from video_retrieval.models import DINOv3Encoder

from common import (  # noqa: F401 -- re-exported for backward compatibility
    DINOV3_MODEL_NAME,
    MANEUVER_NAMES,
    ManeuverSegment,
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
    bootstrap_ap,
    build_temporal_masks,
    cluster_intersections,
    discover_sessions,
    extract_clip_features,
    extract_maneuver_segments,
    extract_vjepa2_features,
    filter_mixed_clusters,
    load_clip,
    load_clip_vjepa2,
    load_gps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_all_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    vjepa2_features: dict[int, dict] | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each cluster, aggregated.

    Uses batched GPU DTW for temporal derivative, attention trajectory,
    and V-JEPA 2 temporal residual comparisons.

    For all pairs within each cluster:
    - Ground truth: same-maneuver label = positive (1), different = negative (0)
    - Compute 4 DINOv3 similarity methods + optional 2 V-JEPA 2 methods

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute all fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    # Enumerate all pairs across clusters
    print("  Enumerating pairs...")
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                pair_a_indices.append(indices[a_pos])
                pair_b_indices.append(indices[b_pos])
                gt = (
                    1
                    if segments[indices[a_pos]].label == segments[indices[b_pos]].label
                    else 0
                )
                pair_gts.append(gt)

    total_pairs = len(pair_gts)
    print(f"  Total pairs to compute: {total_pairs}")

    # --- Vectorized bag-of-frames (batch dot product) ---
    print("  Computing bag-of-frames similarities...")
    mean_embs_a = torch.stack([features[i]["mean_emb"] for i in pair_a_indices]).to(
        device
    )
    mean_embs_b = torch.stack([features[i]["mean_emb"] for i in pair_b_indices]).to(
        device
    )
    bof_sims = (mean_embs_a * mean_embs_b).sum(dim=1).cpu().tolist()

    # --- Vectorized Chamfer (loop but on GPU) ---
    print("  Computing Chamfer similarities...")
    chamfer_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        ea = features[a_idx]["embeddings"].to(device)
        eb = features[b_idx]["embeddings"].to(device)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims.append((max_ab + max_ba) / 2)

    # --- Batched DTW: temporal derivatives ---
    print("  Computing temporal derivative DTW (batched GPU)...")
    deriv_seqs_a = [deriv_fps[i].to(device) for i in pair_a_indices]
    deriv_seqs_b = [deriv_fps[i].to(device) for i in pair_b_indices]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).cpu().tolist()

    # --- Batched DTW: attention trajectories ---
    print("  Computing attention trajectory DTW (batched GPU)...")
    traj_seqs_a = [traj_fps[i].to(device) for i in pair_a_indices]
    traj_seqs_b = [traj_fps[i].to(device) for i in pair_b_indices]
    traj_dists = dtw_distance_batch(traj_seqs_a, traj_seqs_b, normalize=True)
    traj_sims = torch.exp(-5.0 * traj_dists).cpu().tolist()

    # Assemble results
    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": (bof_sims, list(pair_gts)),
        "chamfer": (chamfer_sims, list(pair_gts)),
        "temporal_derivative": (deriv_sims, list(pair_gts)),
        "attention_trajectory": (traj_sims, list(pair_gts)),
    }

    # --- V-JEPA 2 methods ---
    if vjepa2_features:
        # Find pairs where both have V-JEPA 2 features
        vjepa2_mask = [
            (a_idx in vjepa2_features and b_idx in vjepa2_features)
            for a_idx, b_idx in zip(pair_a_indices, pair_b_indices)
        ]
        v_a_indices = [a for a, m in zip(pair_a_indices, vjepa2_mask) if m]
        v_b_indices = [b for b, m in zip(pair_b_indices, vjepa2_mask) if m]
        v_gts = [g for g, m in zip(pair_gts, vjepa2_mask) if m]

        if v_a_indices:
            # Bag-of-tokens (batch dot product)
            print("  Computing V-JEPA 2 bag-of-tokens similarities...")
            v_mean_a = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_a_indices]
            ).to(device)
            v_mean_b = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_b_indices]
            ).to(device)
            bot_sims = (v_mean_a * v_mean_b).sum(dim=1).cpu().tolist()

            # Temporal residual DTW (batched GPU)
            print("  Computing V-JEPA 2 temporal residual DTW (batched GPU)...")
            res_seqs_a = [
                vjepa2_features[i]["temporal_residual"].to(device) for i in v_a_indices
            ]
            res_seqs_b = [
                vjepa2_features[i]["temporal_residual"].to(device) for i in v_b_indices
            ]
            res_dists = dtw_distance_batch(res_seqs_a, res_seqs_b, normalize=True)
            res_sims = torch.exp(-res_dists).cpu().tolist()

            all_scores["vjepa2_bag_of_tokens"] = (bot_sims, list(v_gts))
            all_scores["vjepa2_temporal_residual"] = (res_sims, list(v_gts))
        else:
            all_scores["vjepa2_bag_of_tokens"] = ([], [])
            all_scores["vjepa2_temporal_residual"] = ([], [])

    return all_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrimination(results: dict, fig_dir: Path):
    """Generate AP/AUC bar chart."""
    methods = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    labels = [
        "Bag of\nFrames",
        "Chamfer",
        "Temporal\nDerivative",
        "Attention\nTrajectory",
    ]
    colors = {
        "bag_of_frames": "#e74c3c",
        "chamfer": "#1abc9c",
        "temporal_derivative": "#2ecc71",
        "attention_trajectory": "#3498db",
        "vjepa2_bag_of_tokens": "#9b59b6",
        "vjepa2_temporal_residual": "#f39c12",
    }

    # Add V-JEPA 2 methods if present
    if "vjepa2_bag_of_tokens" in results:
        methods.append("vjepa2_bag_of_tokens")
        labels.append("V-JEPA 2\nBag of Tokens")
    if "vjepa2_temporal_residual" in results:
        methods.append("vjepa2_temporal_residual")
        labels.append("V-JEPA 2\nTemporal Res.")

    aps = [results[m]["ap"] for m in methods]
    aucs = [results[m]["auc"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AP
    bars = ax1.bar(
        range(len(methods)),
        aps,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, fontsize=10)
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
    ax1.set_title("Maneuver Discrimination: AP", fontsize=13)
    ax1.set_ylim(0, min(1.0, max(aps) * 1.15 + 0.05))
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax1.legend()

    # AUC
    bars = ax2.bar(
        range(len(methods)),
        aucs,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, fontsize=10)
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
    ax2.set_title("Maneuver Discrimination: AUC", fontsize=13)
    ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.legend()

    fig.suptitle(
        "Honda HDD: Left Turn vs Right Turn at Same Intersection",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "hdd_maneuver_discrimination.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_similarity_distributions(
    all_scores: dict[str, tuple[list[float], list[int]]], fig_dir: Path
):
    """Generate same-vs-different maneuver similarity histograms."""
    methods = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    titles = ["Bag of Frames", "Chamfer", "Temporal Derivative", "Attention Trajectory"]

    # Add V-JEPA 2 methods if present
    if "vjepa2_bag_of_tokens" in all_scores and all_scores["vjepa2_bag_of_tokens"][0]:
        methods.append("vjepa2_bag_of_tokens")
        titles.append("V-JEPA 2 Bag of Tokens")
    if (
        "vjepa2_temporal_residual" in all_scores
        and all_scores["vjepa2_temporal_residual"][0]
    ):
        methods.append("vjepa2_temporal_residual")
        titles.append("V-JEPA 2 Temporal Residual")

    n_methods = len(methods)
    ncols = 3 if n_methods > 4 else 2
    nrows = (n_methods + ncols - 1) // ncols

    color_same = "#3498db"
    color_diff = "#e74c3c"

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, method, title in zip(axes, methods, titles):
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)

        same = scores[labels == 1]
        diff = scores[labels == 0]

        lo = min(scores.min(), 0)
        hi = max(scores.max(), 1)
        bins = np.linspace(lo, hi, 40)

        ax.hist(
            same,
            bins=bins,
            alpha=0.6,
            color=color_same,
            label=f"Same maneuver (n={len(same)})",
            density=True,
        )
        ax.hist(
            diff,
            bins=bins,
            alpha=0.6,
            color=color_diff,
            label=f"Different maneuver (n={len(diff)})",
            density=True,
        )
        ax.set_xlabel("Similarity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # Mean lines
        if len(same) > 0:
            ax.axvline(same.mean(), color=color_same, linestyle="--", alpha=0.8)
        if len(diff) > 0:
            ax.axvline(diff.mean(), color=color_diff, linestyle="--", alpha=0.8)

    # Hide unused subplot axes
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Honda HDD: Similarity Distributions\n"
        "(Same vs Different Maneuver at Same Intersection)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "hdd_similarity_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Honda HDD Intersection Clustering + Maneuver Discrimination"
    )
    parser.add_argument(
        "--hdd-dir",
        type=str,
        default="datasets/hdd",
        help="Path to HDD dataset directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of mixed clusters to evaluate",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Seconds of context before/after maneuver",
    )
    parser.add_argument(
        "--fps-downsample",
        type=float,
        nargs="+",
        default=None,
        help="V-JEPA 2 fps cap values for frame-rate downsample experiment "
        "(e.g. --fps-downsample 2 5 10 15). Runs V-JEPA 2 extraction "
        "at each fps cap and reports AP with bootstrap CIs.",
    )
    parser.add_argument(
        "--context-sec-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Context window durations (seconds before/after maneuver) for "
        "density sweep experiment. Always uses 64 frames; varying "
        "context_sec changes effective FPS. "
        "(e.g. --context-sec-sweep 1.0 2.0 3.0 4.0 6.5 8.0)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    fig_dir = project_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("HONDA HDD: INTERSECTION CLUSTERING + MANEUVER DISCRIMINATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover sessions
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    t0 = time.time()
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} valid sessions (with labels, GPS, video)")
    print(f"  Discovery time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Extract maneuver segments with GPS midpoints
    # ------------------------------------------------------------------
    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []
    sessions_with_segments = 0

    for session_id in tqdm(sorted(sessions.keys()), desc="Loading sessions"):
        info = sessions[session_id]

        labels = np.load(info["label_path"])

        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception as e:
            logger.warning("Failed to load GPS for session %s: %s", session_id, e)
            continue

        segs = extract_maneuver_segments(
            session_id,
            labels,
            gps_ts,
            gps_lats,
            gps_lngs,
            info["video_path"],
            info["video_start_unix"],
            target_labels=(1, 2, 3),
        )

        if segs:
            sessions_with_segments += 1
        all_segments.extend(segs)

    # Count by type
    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1

    print(
        f"  Total segments: {len(all_segments)} "
        f"from {sessions_with_segments} sessions"
    )
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(label_val, 0)}")

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections (DBSCAN eps=0.0003)...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters
    # ------------------------------------------------------------------
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    total_segs_in_mixed = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print(
            "\nERROR: No mixed clusters found. "
            "Cannot evaluate maneuver discrimination."
        )
        return

    # Build flat list of segments in qualifying clusters, with cluster mapping
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    eval_label_counts: dict[int, int] = defaultdict(int)
    for seg in eval_segments:
        eval_label_counts[seg.label] += 1

    print(f"\n  Segments for evaluation: {len(eval_segments)}")
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {eval_label_counts.get(label_val, 0)}")

    # ------------------------------------------------------------------
    # Step 5: Extract video clips and DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 4: Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    print("\nStep 5: Extracting video clips and DINOv3 features...")
    t_feat_start = time.time()
    features = extract_clip_features(
        encoder,
        eval_segments,
        context_sec=args.context_sec,
        target_fps=3.0,
        max_resolution=518,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    # Free encoder memory
    del encoder
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5b: Extract V-JEPA 2 features
    # ------------------------------------------------------------------
    print("\nStep 5b: Loading V-JEPA 2 model...")
    from transformers import AutoModel, AutoVideoProcessor

    vjepa2_model = AutoModel.from_pretrained(VJEPA2_MODEL_NAME, trust_remote_code=True)
    vjepa2_model = vjepa2_model.to(args.device).eval()
    vjepa2_processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    print("  Extracting V-JEPA 2 features...")
    t_vjepa_start = time.time()
    vjepa2_features, vjepa2_clip_stats = extract_vjepa2_features(
        vjepa2_model,
        vjepa2_processor,
        eval_segments,
        device=torch.device(args.device),
        context_sec=args.context_sec,
    )
    t_vjepa = time.time() - t_vjepa_start
    print(f"  V-JEPA 2 feature extraction time: {t_vjepa:.1f}s")

    del vjepa2_model, vjepa2_processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Compute similarities across all clusters
    # ------------------------------------------------------------------
    print("\nStep 6: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_all_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        vjepa2_features=vjepa2_features,
        device=torch.device(args.device),
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: MANEUVER DISCRIMINATION AT SAME INTERSECTION")
    print("=" * 70)

    results = {}
    method_order = [
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ]
    if "vjepa2_bag_of_tokens" in all_scores:
        method_order.append("vjepa2_bag_of_tokens")
    if "vjepa2_temporal_residual" in all_scores:
        method_order.append("vjepa2_temporal_residual")

    for method in method_order:
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap = average_precision_score(labels, scores)
        auc = roc_auc_score(labels, scores)

        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())
        gap = same_mean - diff_mean

        results[method] = {
            "ap": float(ap),
            "auc": float(auc),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": gap,
        }

        print(
            f"  {method:<25s}  AP={ap:.4f}  AUC={auc:.4f}  "
            f"gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    # Dump pair-level scores for bootstrap CI computation
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }

    pair_path = hdd_dir / "pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir)
    plot_similarity_distributions(all_scores, fig_dir)

    # Summary
    n_total_pairs = len(all_scores["bag_of_frames"][0])
    print("\nSummary:")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Mixed intersection clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    print(f"  Total pairs evaluated: {n_total_pairs}")

    print("\nDone.")

    # ------------------------------------------------------------------
    # Optional: FPS downsample experiment
    # ------------------------------------------------------------------
    if args.fps_downsample:
        print("\n" + "=" * 70)
        print("FPS DOWNSAMPLE EXPERIMENT (V-JEPA 2 Temporal Residual)")
        print("=" * 70)

        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )

        downsample_results: dict[str, list] = {}
        for fps_val in args.fps_downsample:
            print(f"\n  --- fps_cap = {fps_val} ---")
            ds_features, _ = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                eval_segments,
                device=torch.device(args.device),
                context_sec=args.context_sec,
                fps_cap=fps_val,
            )

            # Compute similarities for V-JEPA 2 methods only
            ds_scores = compute_all_similarities(
                eval_segments,
                features,
                cluster_to_indices,
                vjepa2_features=ds_features,
                device=torch.device(args.device),
            )

            for method_key in ["vjepa2_temporal_residual", "vjepa2_bag_of_tokens"]:
                if method_key not in ds_scores:
                    continue
                s, l = ds_scores[method_key]
                s_arr, l_arr = np.array(s), np.array(l)
                if l_arr.sum() == 0 or l_arr.sum() == len(l_arr):
                    continue
                ap_val, ci_lo, ci_hi = bootstrap_ap(s_arr, l_arr)
                print(
                    f"    {method_key:<30s}  AP={ap_val:.4f}  "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                )
                downsample_results.setdefault(method_key, []).append(
                    {
                        "fps_cap": fps_val,
                        "ap": ap_val,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                    }
                )

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

        # Save downsample results
        ds_out = fig_dir.parent / "datasets" / "hdd" / "fps_downsample_results.json"
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_out, "w") as f:
            json.dump(downsample_results, f, indent=2)
        print(f"\n  Downsample results saved to {ds_out}")

    # ------------------------------------------------------------------
    # Optional: Context window sweep (density experiment)
    # ------------------------------------------------------------------
    if args.context_sec_sweep:
        print("\n" + "=" * 70)
        print("CONTEXT WINDOW SWEEP (V-JEPA 2 Temporal Residual)")
        print("=" * 70)
        print(f"  Windows: {args.context_sec_sweep}")
        print(f"  Fixed 64-frame input; varying context changes fps_eff")

        # Load V-JEPA 2 if not already loaded by fps_downsample
        if not args.fps_downsample:
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
        else:
            # Re-load since fps_downsample deleted it
            from transformers import AutoModel, AutoVideoProcessor

            vjepa2_model = AutoModel.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )
            vjepa2_model = vjepa2_model.to(args.device).eval()
            vjepa2_processor = AutoVideoProcessor.from_pretrained(
                VJEPA2_MODEL_NAME, trust_remote_code=True
            )

        sweep_out = (
            fig_dir.parent / "datasets" / "hdd" / "context_sec_sweep_results.json"
        )
        sweep_out.parent.mkdir(parents=True, exist_ok=True)

        # Resume from partial results if they exist
        sweep_results = {}
        completed_windows: set[float] = set()
        if sweep_out.exists():
            with open(sweep_out) as f:
                sweep_results = json.load(f)
            for method_entries in sweep_results.values():
                for entry in method_entries:
                    completed_windows.add(entry["context_sec"])
            if completed_windows:
                print(
                    f"  Resuming: skipping {len(completed_windows)} completed windows: "
                    f"{sorted(completed_windows)}"
                )

        for ctx_sec in args.context_sec_sweep:
            if ctx_sec in completed_windows:
                print(
                    f"\n  --- context_sec = {ctx_sec} --- SKIPPED (already completed)"
                )
                continue

            print(f"\n  --- context_sec = {ctx_sec} ---")
            ctx_features, ctx_stats = extract_vjepa2_features(
                vjepa2_model,
                vjepa2_processor,
                eval_segments,
                device=torch.device(args.device),
                context_sec=ctx_sec,
            )

            # Compute similarities for V-JEPA 2 methods only
            ctx_scores = compute_all_similarities(
                eval_segments,
                features,
                cluster_to_indices,
                vjepa2_features=ctx_features,
                device=torch.device(args.device),
            )

            # Aggregate clip stats
            fps_effs = [s["fps_eff"] for s in ctx_stats]
            dup_ratios = [s["dup_ratio"] for s in ctx_stats]
            durations = [s["duration"] for s in ctx_stats]
            stats_summary = {
                "fps_eff_median": float(np.median(fps_effs)),
                "fps_eff_mean": float(np.mean(fps_effs)),
                "dup_ratio_mean": float(np.mean(dup_ratios)),
                "duration_median": float(np.median(durations)),
                "unique_frames_mean": float(
                    np.mean([s["unique_frames"] for s in ctx_stats])
                ),
            }
            print(
                f"    fps_eff={stats_summary['fps_eff_median']:.1f} (median), "
                f"dup_ratio={stats_summary['dup_ratio_mean']:.3f}, "
                f"unique_frames={stats_summary['unique_frames_mean']:.1f}"
            )

            for method_key in ["vjepa2_temporal_residual", "vjepa2_bag_of_tokens"]:
                if method_key not in ctx_scores:
                    continue
                s, l = ctx_scores[method_key]
                s_arr, l_arr = np.array(s), np.array(l)
                if l_arr.sum() == 0 or l_arr.sum() == len(l_arr):
                    continue
                ap_val, ci_lo, ci_hi = bootstrap_ap(s_arr, l_arr)
                print(
                    f"    {method_key:<30s}  AP={ap_val:.4f}  "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                )
                sweep_results.setdefault(method_key, []).append(
                    {
                        "context_sec": ctx_sec,
                        "ap": ap_val,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                        **stats_summary,
                    }
                )

            # Save incrementally after each window
            with open(sweep_out, "w") as f:
                json.dump(sweep_results, f, indent=2)
            n_done = len(completed_windows) + 1
            completed_windows.add(ctx_sec)
            print(
                f"    [checkpoint saved: {n_done}/{len(args.context_sec_sweep)} windows]"
            )

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

        print(f"\n  Context sweep results saved to {sweep_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two-Stage Reranking Pipeline: BoF Retrieval + Temporal Residual Reranking.

Implements the reviewer-requested "proposed solution" that transforms the paper
from pure diagnosis to diagnosis + prescription. Demonstrates that a practical
system can combine the speed of BoF (cosine, indexable) with the motion
discrimination of V-JEPA 2 temporal residuals (DTW, expensive but only on k
candidates).

Pipeline for each query segment in HDD:
  Stage 1: DINOv3 BoF cosine similarity → top-k candidates (fast, O(1) per pair)
  Stage 2: V-JEPA 2 temporal residual DTW → rerank those k candidates (accurate)

Evaluation:
  - AP for reranked pipeline at k ∈ {10, 25, 50, 100, all}
  - Baselines: BoF-only, BoT-only, residual-only, oracle routing
  - Wall-clock timing for each stage
  - Bootstrap CIs for all metrics

Protocol: 128 sessions, 1,687 maneuvers, 50 mixed-direction clusters,
DBSCAN eps=0.0003, min_samples=3, +/-3 s context, 1,000 bootstrap resamples,
seed=42.

Usage:
    python experiments/eval_reranking_pipeline.py [--device cuda] \\
        [--hdd-dir datasets/hdd] [--output-dir datasets/hdd]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

# Add experiments/ to sys.path so sibling imports work from any cwd
sys.path.insert(0, str(Path(__file__).parent))
from eval_hdd_intersections import (  # noqa: E402
    build_temporal_masks,
    cluster_intersections,
    discover_sessions,
    extract_clip_features,
    extract_maneuver_segments,
    extract_vjepa2_features,
    filter_mixed_clusters,
    load_gps,
    MANEUVER_NAMES,
    ManeuverSegment,
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
)
from tqdm import tqdm
from video_retrieval.fingerprints import TemporalDerivativeFingerprint
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# Reranking depths to evaluate
K_VALUES = [10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Bootstrap CI for AP
# ---------------------------------------------------------------------------


def bootstrap_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for Average Precision.

    Returns:
        (ap, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    ap = average_precision_score(labels, scores)

    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot_aps[i] = ap  # degenerate sample
        else:
            boot_aps[i] = average_precision_score(l, s)

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_aps, 100 * alpha))
    ci_high = float(np.percentile(boot_aps, 100 * (1 - alpha)))

    return float(ap), ci_low, ci_high


# ---------------------------------------------------------------------------
# Pairwise similarity computation
# ---------------------------------------------------------------------------


def compute_pairwise_bof(
    features: dict[int, dict],
    pair_a: list[int],
    pair_b: list[int],
    device: torch.device,
) -> np.ndarray:
    """Compute bag-of-frames cosine similarity for all pairs.

    Args:
        features: Dict mapping segment index -> {'mean_emb': (D,), ...}.
        pair_a: List of first-segment indices.
        pair_b: List of second-segment indices.
        device: Torch device.

    Returns:
        Similarity scores array of shape (n_pairs,).
    """
    mean_a = torch.stack([features[i]["mean_emb"] for i in pair_a]).to(device)
    mean_b = torch.stack([features[i]["mean_emb"] for i in pair_b]).to(device)
    sims = (mean_a * mean_b).sum(dim=1).cpu().numpy()
    return sims


def compute_pairwise_bot(
    vjepa2_features: dict[int, dict],
    pair_a: list[int],
    pair_b: list[int],
    device: torch.device,
) -> np.ndarray:
    """Compute V-JEPA 2 bag-of-tokens cosine similarity for all pairs.

    Args:
        vjepa2_features: Dict mapping segment index -> {'mean_emb': (D,), ...}.
        pair_a: List of first-segment indices.
        pair_b: List of second-segment indices.
        device: Torch device.

    Returns:
        Similarity scores array of shape (n_pairs,).
    """
    mean_a = torch.stack([vjepa2_features[i]["mean_emb"] for i in pair_a]).to(device)
    mean_b = torch.stack([vjepa2_features[i]["mean_emb"] for i in pair_b]).to(device)
    sims = (mean_a * mean_b).sum(dim=1).cpu().numpy()
    return sims


def compute_pairwise_residual_dtw(
    vjepa2_features: dict[int, dict],
    pair_a: list[int],
    pair_b: list[int],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Compute V-JEPA 2 temporal residual DTW similarity for all pairs.

    similarity = exp(-dtw_distance)

    Args:
        vjepa2_features: Dict mapping segment index -> {'temporal_residual': ...}.
        pair_a: List of first-segment indices.
        pair_b: List of second-segment indices.
        device: Torch device.

    Returns:
        Tuple of (similarity scores (n_pairs,), wall_clock_seconds).
    """
    t_start = time.time()
    seqs_a = [vjepa2_features[i]["temporal_residual"].to(device) for i in pair_a]
    seqs_b = [vjepa2_features[i]["temporal_residual"].to(device) for i in pair_b]
    dists = dtw_distance_batch(seqs_a, seqs_b, normalize=True)
    sims = torch.exp(-dists).cpu().numpy()
    elapsed = time.time() - t_start
    return sims, elapsed


# ---------------------------------------------------------------------------
# Reranking pipeline
# ---------------------------------------------------------------------------


def rerank_pipeline_ap(
    bof_scores: np.ndarray,
    residual_scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Compute AP for the 2-stage reranking pipeline.

    Stage 1: Select top-k by BoF score.
    Stage 2: Rerank those k by residual DTW score.

    Non-selected items receive -inf score, so they rank below all top-k
    candidates in the final ranking.

    Args:
        bof_scores: (N,) BoF cosine similarities for one query.
        residual_scores: (N,) residual DTW similarities for the same query.
        labels: (N,) binary ground truth (1=same maneuver, 0=different).
        k: Number of candidates to shortlist from Stage 1.

    Returns:
        Average Precision on the full reranked list.
    """
    n = len(bof_scores)
    effective_k = min(k, n)

    # Stage 1: top-k by BoF
    top_k_idx = np.argsort(bof_scores)[::-1][:effective_k]

    # Stage 2: rerank those k by residual score
    # Non-selected items get -inf so they rank last
    reranked_scores = np.full(n, -np.inf)
    reranked_scores[top_k_idx] = residual_scores[top_k_idx]

    # Handle edge case: if all labels in top-k are the same class
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")

    return float(average_precision_score(labels, reranked_scores))


def oracle_routing_ap(
    bof_scores: np.ndarray,
    residual_scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute AP using oracle routing: max(BoF, residual) per pair.

    This is an upper bound on any routing/fusion strategy.

    Args:
        bof_scores: (N,) BoF cosine similarities.
        residual_scores: (N,) residual DTW similarities.
        labels: (N,) binary ground truth.

    Returns:
        Average Precision with oracle routing.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    oracle_scores = np.maximum(bof_scores, residual_scores)
    return float(average_precision_score(labels, oracle_scores))


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def evaluate_reranking(
    segments: list[ManeuverSegment],
    dino_features: dict[int, dict],
    vjepa2_features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    device: torch.device,
    k_values: list[int],
) -> dict:
    """Run the full reranking evaluation across all clusters.

    Computes pairwise similarities within each cluster, then evaluates
    BoF-only, BoT-only, residual-only, oracle routing, and reranking
    pipeline at each k.

    Args:
        segments: Evaluation segments with maneuver labels.
        dino_features: DINOv3 features per segment index.
        vjepa2_features: V-JEPA 2 features per segment index.
        cluster_to_indices: Cluster ID -> list of segment indices.
        device: Torch device.
        k_values: List of k values for reranking.

    Returns:
        Dict with all results, metrics, and timing.
    """
    # Enumerate all pairs across clusters
    print("  Enumerating pairs...")
    pair_a_indices: list[int] = []
    pair_b_indices: list[int] = []
    pair_gts: list[int] = []

    for cid in sorted(cluster_to_indices.keys()):
        indices = [
            i
            for i in cluster_to_indices[cid]
            if i in dino_features and i in vjepa2_features
        ]
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
    labels = np.array(pair_gts)
    n_pos = int(labels.sum())
    n_neg = total_pairs - n_pos

    print(f"  Total pairs: {total_pairs} (pos={n_pos}, neg={n_neg})")

    if total_pairs == 0:
        print("  ERROR: No valid pairs found.")
        return {}

    # --- Compute all pairwise similarities ---

    # Stage 1: BoF cosine (fast)
    print("  Computing BoF cosine similarities...")
    t_bof_start = time.time()
    bof_scores = compute_pairwise_bof(
        dino_features, pair_a_indices, pair_b_indices, device
    )
    t_bof = time.time() - t_bof_start
    print(f"    BoF time: {t_bof:.3f}s ({total_pairs} pairs)")

    # BoT cosine (V-JEPA 2 bag-of-tokens)
    print("  Computing V-JEPA 2 BoT cosine similarities...")
    t_bot_start = time.time()
    bot_scores = compute_pairwise_bot(
        vjepa2_features, pair_a_indices, pair_b_indices, device
    )
    t_bot = time.time() - t_bot_start
    print(f"    BoT time: {t_bot:.3f}s ({total_pairs} pairs)")

    # Stage 2: Residual DTW (expensive)
    print("  Computing V-JEPA 2 residual DTW similarities...")
    residual_scores, t_residual = compute_pairwise_residual_dtw(
        vjepa2_features, pair_a_indices, pair_b_indices, device
    )
    print(f"    Residual DTW time: {t_residual:.3f}s ({total_pairs} pairs)")

    # --- Baseline APs with bootstrap CIs ---
    print("\n  Computing baseline APs...")
    bof_ap, bof_ci_lo, bof_ci_hi = bootstrap_ap(bof_scores, labels)
    bot_ap, bot_ci_lo, bot_ci_hi = bootstrap_ap(bot_scores, labels)
    res_ap, res_ci_lo, res_ci_hi = bootstrap_ap(residual_scores, labels)
    oracle_ap_val = oracle_routing_ap(bof_scores, residual_scores, labels)
    _, oracle_ci_lo, oracle_ci_hi = bootstrap_ap(
        np.maximum(bof_scores, residual_scores), labels
    )

    print(f"    BoF-only:         AP={bof_ap:.4f} [{bof_ci_lo:.4f}, {bof_ci_hi:.4f}]")
    print(f"    BoT-only:         AP={bot_ap:.4f} [{bot_ci_lo:.4f}, {bot_ci_hi:.4f}]")
    print(f"    Residual-only:    AP={res_ap:.4f} [{res_ci_lo:.4f}, {res_ci_hi:.4f}]")
    print(
        f"    Oracle routing:   AP={oracle_ap_val:.4f} "
        f"[{oracle_ci_lo:.4f}, {oracle_ci_hi:.4f}]"
    )

    # --- Reranking at each k ---
    print("\n  Computing reranking pipeline APs...")
    reranking_results: dict[int, dict] = {}

    for k in k_values:
        # For each pair, compute reranking AP
        # We aggregate across all pairs (not per-query), matching the
        # binary AP evaluation protocol from eval_hdd_intersections.py
        effective_k = min(k, total_pairs)
        reranked_scores_full = np.full(total_pairs, -np.inf)

        # Top-k by BoF score, then rerank by residual
        top_k_idx = np.argsort(bof_scores)[::-1][:effective_k]
        reranked_scores_full[top_k_idx] = residual_scores[top_k_idx]

        rr_ap, rr_ci_lo, rr_ci_hi = bootstrap_ap(reranked_scores_full, labels)

        # Timing estimate: Stage 1 is free (precomputed cosine),
        # Stage 2 is proportional to k / total_pairs
        t_stage2_estimate = t_residual * (effective_k / total_pairs)

        reranking_results[k] = {
            "k": k,
            "effective_k": effective_k,
            "ap": rr_ap,
            "ci_low": rr_ci_lo,
            "ci_high": rr_ci_hi,
            "t_stage2_estimate_s": t_stage2_estimate,
            "speedup_vs_full": t_residual / max(t_stage2_estimate, 1e-9),
        }

        print(
            f"    k={k:>4d}:  AP={rr_ap:.4f} [{rr_ci_lo:.4f}, {rr_ci_hi:.4f}]  "
            f"Stage2 est: {t_stage2_estimate:.3f}s  "
            f"(speedup: {reranking_results[k]['speedup_vs_full']:.1f}x)"
        )

    # Also evaluate k=all (full residual reranking, same as residual-only)
    reranking_results[total_pairs] = {
        "k": total_pairs,
        "effective_k": total_pairs,
        "ap": res_ap,
        "ci_low": res_ci_lo,
        "ci_high": res_ci_hi,
        "t_stage2_estimate_s": t_residual,
        "speedup_vs_full": 1.0,
    }

    # --- Per-cluster reranking evaluation ---
    # Provides a more granular view: for each query in each cluster,
    # compute reranked AP within the cluster
    print("\n  Computing per-cluster reranking metrics...")
    per_cluster_stats: dict[int, dict] = {}

    for cid in sorted(cluster_to_indices.keys()):
        indices = [
            i
            for i in cluster_to_indices[cid]
            if i in dino_features and i in vjepa2_features
        ]
        n = len(indices)
        if n < 3:
            continue

        # Compute pairwise similarities within this cluster
        cluster_bof = compute_pairwise_bof(
            dino_features,
            [indices[a] for a in range(n) for b in range(a + 1, n)],
            [indices[b] for a in range(n) for b in range(a + 1, n)],
            device,
        )
        cluster_res, _ = compute_pairwise_residual_dtw(
            vjepa2_features,
            [indices[a] for a in range(n) for b in range(a + 1, n)],
            [indices[b] for a in range(n) for b in range(a + 1, n)],
            device,
        )
        cluster_labels = np.array(
            [
                1 if segments[indices[a]].label == segments[indices[b]].label else 0
                for a in range(n)
                for b in range(a + 1, n)
            ]
        )

        if cluster_labels.sum() == 0 or cluster_labels.sum() == len(cluster_labels):
            continue

        cluster_bof_ap = float(average_precision_score(cluster_labels, cluster_bof))
        cluster_res_ap = float(average_precision_score(cluster_labels, cluster_res))

        per_cluster_stats[cid] = {
            "n_segments": n,
            "n_pairs": len(cluster_labels),
            "n_pos": int(cluster_labels.sum()),
            "bof_ap": cluster_bof_ap,
            "residual_ap": cluster_res_ap,
        }

    # --- Assemble results ---
    results = {
        "protocol": {
            "dataset": "HDD",
            "n_pairs": total_pairs,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "k_values": k_values + [total_pairs],
            "n_bootstrap": 1000,
            "seed": 42,
        },
        "baselines": {
            "bof": {
                "ap": bof_ap,
                "ci_low": bof_ci_lo,
                "ci_high": bof_ci_hi,
                "time_s": t_bof,
            },
            "bot": {
                "ap": bot_ap,
                "ci_low": bot_ci_lo,
                "ci_high": bot_ci_hi,
                "time_s": t_bot,
            },
            "residual": {
                "ap": res_ap,
                "ci_low": res_ci_lo,
                "ci_high": res_ci_hi,
                "time_s": t_residual,
            },
            "oracle_routing": {
                "ap": oracle_ap_val,
                "ci_low": oracle_ci_lo,
                "ci_high": oracle_ci_hi,
            },
        },
        "reranking": {str(k): v for k, v in reranking_results.items()},
        "timing": {
            "bof_total_s": t_bof,
            "bot_total_s": t_bot,
            "residual_total_s": t_residual,
            "per_pair_bof_us": (t_bof / max(total_pairs, 1)) * 1e6,
            "per_pair_residual_us": (t_residual / max(total_pairs, 1)) * 1e6,
        },
        "per_cluster": per_cluster_stats,
    }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_reranking_curve(results: dict, fig_dir: Path) -> None:
    """Generate AP vs k line plot with baseline horizontal lines.

    Shows how reranking AP changes as k increases, with baselines for
    BoF-only, BoT-only, residual-only, and oracle routing.

    Args:
        results: Output from evaluate_reranking().
        fig_dir: Directory to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Extract reranking curve data
    reranking = results["reranking"]
    k_vals = []
    ap_vals = []
    ci_lo_vals = []
    ci_hi_vals = []

    for k_str in sorted(reranking.keys(), key=lambda x: int(x)):
        entry = reranking[k_str]
        k_vals.append(entry["k"])
        ap_vals.append(entry["ap"])
        ci_lo_vals.append(entry["ci_low"])
        ci_hi_vals.append(entry["ci_high"])

    k_vals = np.array(k_vals)
    ap_vals = np.array(ap_vals)
    ci_lo_vals = np.array(ci_lo_vals)
    ci_hi_vals = np.array(ci_hi_vals)

    # Plot reranking curve with CIs
    ax.plot(
        k_vals,
        ap_vals,
        "o-",
        color="#2ecc71",
        linewidth=2.5,
        markersize=8,
        label="BoF + Residual Reranking",
        zorder=5,
    )
    ax.fill_between(
        k_vals,
        ci_lo_vals,
        ci_hi_vals,
        alpha=0.2,
        color="#2ecc71",
    )

    # Baselines as horizontal lines
    baselines = results["baselines"]

    ax.axhline(
        baselines["bof"]["ap"],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label=f"BoF-only (AP={baselines['bof']['ap']:.3f})",
    )
    ax.axhline(
        baselines["bot"]["ap"],
        color="#9b59b6",
        linestyle="--",
        linewidth=1.5,
        label=f"BoT-only (AP={baselines['bot']['ap']:.3f})",
    )
    ax.axhline(
        baselines["residual"]["ap"],
        color="#3498db",
        linestyle="--",
        linewidth=1.5,
        label=f"Residual-only (AP={baselines['residual']['ap']:.3f})",
    )
    ax.axhline(
        baselines["oracle_routing"]["ap"],
        color="#f39c12",
        linestyle=":",
        linewidth=1.5,
        label=f"Oracle routing (AP={baselines['oracle_routing']['ap']:.3f})",
    )

    ax.set_xlabel("k (number of candidates from Stage 1)", fontsize=12)
    ax.set_ylabel("Average Precision", fontsize=12)
    ax.set_title(
        "HDD: Two-Stage Reranking Pipeline\n"
        "Stage 1: DINOv3 BoF top-k   |   Stage 2: V-JEPA 2 Residual DTW rerank",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate each k point
    for k, ap in zip(k_vals, ap_vals):
        if k < k_vals[-1]:  # skip the "all" point (it overlaps residual-only)
            ax.annotate(
                f"{ap:.3f}",
                (k, ap),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    path = fig_dir / "hdd_reranking_pipeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def plot_speedup(results: dict, fig_dir: Path) -> None:
    """Generate speedup bar chart showing Stage 2 cost reduction.

    Args:
        results: Output from evaluate_reranking().
        fig_dir: Directory to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    reranking = results["reranking"]
    total_pairs = results["protocol"]["n_pairs"]

    # Filter to non-"all" k values
    k_entries = [
        (int(k_str), entry)
        for k_str, entry in reranking.items()
        if int(k_str) < total_pairs
    ]
    k_entries.sort(key=lambda x: x[0])

    k_labels = [str(k) for k, _ in k_entries]
    speedups = [entry["speedup_vs_full"] for _, entry in k_entries]
    stage2_times = [entry["t_stage2_estimate_s"] for _, entry in k_entries]
    aps = [entry["ap"] for _, entry in k_entries]

    # Left: speedup bar chart
    bars = ax1.bar(
        k_labels,
        speedups,
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, speedups):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.set_xlabel("k (Stage 1 shortlist size)", fontsize=12)
    ax1.set_ylabel("Speedup over full DTW", fontsize=12)
    ax1.set_title("Stage 2 Speedup", fontsize=13, fontweight="bold")

    # Right: AP vs Stage 2 time scatter
    full_time = results["timing"]["residual_total_s"]
    full_ap = results["baselines"]["residual"]["ap"]

    ax2.scatter(
        stage2_times,
        aps,
        s=100,
        color="#2ecc71",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    for k_val, t, ap in zip([k for k, _ in k_entries], stage2_times, aps):
        ax2.annotate(
            f"k={k_val}",
            (t, ap),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    # Full residual baseline
    ax2.scatter(
        [full_time],
        [full_ap],
        s=120,
        color="#3498db",
        marker="D",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
        label=f"Full residual (AP={full_ap:.3f})",
    )

    ax2.set_xlabel("Stage 2 estimated time (s)", fontsize=12)
    ax2.set_ylabel("Average Precision", fontsize=12)
    ax2.set_title("AP vs Compute Cost", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "HDD: Reranking Pipeline Efficiency",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / "hdd_reranking_speedup.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def print_latex_table(results: dict) -> None:
    """Print LaTeX table rows for inclusion in the paper.

    Args:
        results: Output from evaluate_reranking().
    """
    print("\n% LaTeX table rows for reranking pipeline results")
    print("% \\begin{tabular}{lcccc}")
    print("% Method & AP & 95\\% CI & Stage 2 Time (s) & Speedup \\\\")
    print("% \\midrule")

    baselines = results["baselines"]

    print(
        f"DINOv3 BoF (Stage 1 only) & "
        f"{baselines['bof']['ap']:.3f} & "
        f"[{baselines['bof']['ci_low']:.3f}, {baselines['bof']['ci_high']:.3f}] & "
        f"{baselines['bof']['time_s']:.2f} & --- \\\\"
    )
    print(
        f"V-JEPA 2 BoT & "
        f"{baselines['bot']['ap']:.3f} & "
        f"[{baselines['bot']['ci_low']:.3f}, {baselines['bot']['ci_high']:.3f}] & "
        f"{baselines['bot']['time_s']:.2f} & --- \\\\"
    )
    print(
        f"V-JEPA 2 Residual (full) & "
        f"{baselines['residual']['ap']:.3f} & "
        f"[{baselines['residual']['ci_low']:.3f}, "
        f"{baselines['residual']['ci_high']:.3f}] & "
        f"{baselines['residual']['time_s']:.2f} & 1.0$\\times$ \\\\"
    )
    print("\\midrule")

    reranking = results["reranking"]
    total_pairs = results["protocol"]["n_pairs"]

    for k_str in sorted(reranking.keys(), key=lambda x: int(x)):
        entry = reranking[k_str]
        k = entry["k"]
        if k >= total_pairs:
            continue
        print(
            f"BoF top-{k} + Residual rerank & "
            f"{entry['ap']:.3f} & "
            f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] & "
            f"{entry['t_stage2_estimate_s']:.2f} & "
            f"{entry['speedup_vs_full']:.1f}$\\times$ \\\\"
        )

    print("\\midrule")
    print(
        f"Oracle routing & "
        f"{baselines['oracle_routing']['ap']:.3f} & "
        f"[{baselines['oracle_routing']['ci_low']:.3f}, "
        f"{baselines['oracle_routing']['ci_high']:.3f}] & "
        f"--- & --- \\\\"
    )
    print("% \\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="HDD Two-Stage Reranking Pipeline: BoF + Residual Reranking"
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
        "--output-dir",
        type=str,
        default="datasets/hdd",
        help="Directory for output JSON results",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    output_dir = project_root / args.output_dir
    fig_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    print("=" * 70)
    print("HDD: TWO-STAGE RERANKING PIPELINE")
    print("Stage 1: DINOv3 BoF top-k  |  Stage 2: V-JEPA 2 Residual DTW rerank")
    print("=" * 70)
    print(f"  k values: {K_VALUES}")
    print(f"  Context: +/-{args.context_sec}s")
    print(f"  Max clusters: {args.max_clusters}")
    print(f"  Output dir: {output_dir}")

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
        except Exception:
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

    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1
    print(
        f"  Total segments: {len(all_segments)} "
        f"from {sessions_with_segments} sessions"
    )
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(lv, 0)}")

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
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {eval_label_counts.get(lv, 0)}")

    # ------------------------------------------------------------------
    # Step 5a: Extract DINOv3 features
    # ------------------------------------------------------------------
    print("\nStep 5a: Loading DINOv3 encoder...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    print("  Extracting DINOv3 features...")
    t_dino_start = time.time()
    dino_features = extract_clip_features(
        encoder,
        eval_segments,
        context_sec=args.context_sec,
        target_fps=3.0,
        max_resolution=518,
    )
    t_dino = time.time() - t_dino_start
    print(f"  DINOv3 extraction time: {t_dino:.1f}s")

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
        device=device,
        context_sec=args.context_sec,
    )
    t_vjepa = time.time() - t_vjepa_start
    print(f"  V-JEPA 2 extraction time: {t_vjepa:.1f}s")

    del vjepa2_model, vjepa2_processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Run reranking evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RERANKING EVALUATION")
    print("=" * 70)

    results = evaluate_reranking(
        eval_segments,
        dino_features,
        vjepa2_features,
        cluster_to_indices,
        device,
        K_VALUES,
    )

    if not results:
        print("\nERROR: Evaluation returned no results.")
        return

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    results_path = output_dir / "hdd_reranking_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Step 8: Generate figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_reranking_curve(results, fig_dir)
    plot_speedup(results, fig_dir)

    # ------------------------------------------------------------------
    # Step 9: Print LaTeX table
    # ------------------------------------------------------------------
    print_latex_table(results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baselines = results["baselines"]
    print(f"  Baselines:")
    print(f"    BoF-only AP:       {baselines['bof']['ap']:.4f}")
    print(f"    BoT-only AP:       {baselines['bot']['ap']:.4f}")
    print(f"    Residual-only AP:  {baselines['residual']['ap']:.4f}")
    print(f"    Oracle routing AP: {baselines['oracle_routing']['ap']:.4f}")

    print(f"\n  Reranking pipeline:")
    reranking = results["reranking"]
    total_pairs = results["protocol"]["n_pairs"]
    for k_str in sorted(reranking.keys(), key=lambda x: int(x)):
        entry = reranking[k_str]
        k = entry["k"]
        if k >= total_pairs:
            label = "all"
        else:
            label = str(k)
        print(
            f"    k={label:<5s}  AP={entry['ap']:.4f}  "
            f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
        )

    print(f"\n  Timing:")
    timing = results["timing"]
    print(f"    BoF total:     {timing['bof_total_s']:.3f}s")
    print(f"    Residual total: {timing['residual_total_s']:.3f}s")
    print(f"    Per-pair BoF:   {timing['per_pair_bof_us']:.1f} us")
    print(f"    Per-pair residual: {timing['per_pair_residual_us']:.1f} us")

    print("\nDone.")


if __name__ == "__main__":
    main()

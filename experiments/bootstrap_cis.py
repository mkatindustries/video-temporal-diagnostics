#!/usr/bin/env python3
"""
Bootstrap confidence intervals for all benchmark headline AP/AUC numbers.

Usage:
    python experiments/bootstrap_cis.py --benchmark vcdb   --pairs-json datasets/vcdb/pair_scores.json
    python experiments/bootstrap_cis.py --benchmark hdd    --pairs-json datasets/hdd/pair_scores.json
    python experiments/bootstrap_cis.py --benchmark nuscenes --pairs-json datasets/nuscenes/pair_scores.json

Input format (pair_scores.json):
    {
        "method_name": {
            "scores": [0.95, 0.12, ...],   # similarity scores, one per pair
            "labels": [1, 0, ...]           # binary labels (1=positive, 0=negative)
        },
        ...
    }

Pair-score dumping is built into the eval scripts (no patches needed).
See BOOTSTRAP_INSTRUCTIONS.md for re-run instructions.

Output: Formatted LaTeX table rows with 95% bootstrap CIs.

Author: Auto-generated for reviewer response.
"""

import argparse
import json
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _ap_gpu_batch(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute average precision for a batch of bootstrap samples on GPU.

    Args:
        scores: (B, N) tensor of similarity scores.
        labels: (B, N) tensor of binary labels.

    Returns:
        (B,) tensor of AP values.
    """
    B, N = scores.shape

    # Sort each row by descending score
    sorted_indices = scores.argsort(dim=1, descending=True)
    sorted_labels = labels.gather(1, sorted_indices)  # (B, N)

    # Cumulative TP and total predictions
    tp_cumsum = sorted_labels.cumsum(dim=1)  # (B, N)
    positions = torch.arange(1, N + 1, device=scores.device, dtype=scores.dtype).unsqueeze(0)  # (1, N)

    precision = tp_cumsum / positions  # (B, N)
    n_pos = labels.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
    recall_change = sorted_labels / n_pos  # (B, N)

    # AP = sum(precision * recall_change) per row
    ap = (precision * recall_change).sum(dim=1)  # (B,)
    return ap


def _auc_gpu_batch(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute ROC AUC for a batch of bootstrap samples on GPU.

    Uses the Mann-Whitney U statistic: AUC = U / (n_pos * n_neg).
    For each resample, count how many (positive_score > negative_score) pairs.

    For large N this is memory-intensive, so we use the rank-based formula:
    AUC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Args:
        scores: (B, N) tensor of similarity scores.
        labels: (B, N) tensor of binary labels.

    Returns:
        (B,) tensor of AUC values.
    """
    B, N = scores.shape

    # Rank scores (1-based, average ties)
    sorted_indices = scores.argsort(dim=1)
    ranks = torch.zeros_like(scores)
    batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand_as(sorted_indices)
    ranks[batch_idx, sorted_indices] = torch.arange(1, N + 1, device=scores.device, dtype=scores.dtype).unsqueeze(0).expand(B, -1)

    # Handle ties by averaging ranks
    # For bootstrap CIs the effect of ties is negligible, skip for speed

    n_pos = labels.sum(dim=1)  # (B,)
    n_neg = N - n_pos  # (B,)

    # Sum of ranks for positive samples
    pos_rank_sum = (ranks * labels).sum(dim=1)  # (B,)

    # AUC via Mann-Whitney
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg).clamp(min=1)
    return auc


def bootstrap_ap(scores, labels, n_resamples=2000, ci=0.95, seed=42, device=None):
    """Bootstrap 95% CI for average precision, GPU-accelerated when available.

    Matches the bootstrap_ap implementation in eval_hdd_intersections.py.
    Uses percentile method with 2000 resamples.
    """
    scores_np = np.asarray(scores)
    labels_np = np.asarray(labels)
    n = len(scores_np)

    # Point estimate (sklearn for reference accuracy)
    point_ap = average_precision_score(labels_np, scores_np)

    if device is not None and device.type == "cuda":
        # GPU path: generate all bootstrap indices and compute AP in batch
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        scores_t = torch.tensor(scores_np, device=device, dtype=torch.float32)
        labels_t = torch.tensor(labels_np, device=device, dtype=torch.float32)

        # Generate all bootstrap indices at once: (B, N)
        boot_indices = torch.randint(0, n, (n_resamples, n), device=device, generator=gen)

        boot_scores = scores_t[boot_indices]  # (B, N)
        boot_labels = labels_t[boot_indices]  # (B, N)

        # Check for degenerate samples
        pos_counts = boot_labels.sum(dim=1)
        degenerate = (pos_counts == 0) | (pos_counts == n)

        boot_aps = _ap_gpu_batch(boot_scores, boot_labels)
        boot_aps[degenerate] = float(point_ap)

        boot_aps_np = boot_aps.cpu().numpy()
    else:
        # CPU path (original)
        rng = np.random.RandomState(seed)
        boot_aps_list = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n, size=n)
            s_boot = scores_np[idx]
            l_boot = labels_np[idx]
            if l_boot.sum() == 0 or l_boot.sum() == n:
                boot_aps_list.append(point_ap)
                continue
            boot_aps_list.append(average_precision_score(l_boot, s_boot))
        boot_aps_np = np.array(boot_aps_list)

    alpha = 1 - ci
    ci_low = np.percentile(boot_aps_np, 100 * alpha / 2)
    ci_high = np.percentile(boot_aps_np, 100 * (1 - alpha / 2))
    return point_ap, ci_low, ci_high


def bootstrap_auc(scores, labels, n_resamples=2000, ci=0.95, seed=42, device=None):
    """Bootstrap 95% CI for ROC AUC, GPU-accelerated when available."""
    scores_np = np.asarray(scores)
    labels_np = np.asarray(labels)
    n = len(scores_np)

    point_auc = roc_auc_score(labels_np, scores_np)

    if device is not None and device.type == "cuda":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 1)  # Different seed from AP to avoid correlation

        scores_t = torch.tensor(scores_np, device=device, dtype=torch.float32)
        labels_t = torch.tensor(labels_np, device=device, dtype=torch.float32)

        boot_indices = torch.randint(0, n, (n_resamples, n), device=device, generator=gen)
        boot_scores = scores_t[boot_indices]
        boot_labels = labels_t[boot_indices]

        pos_counts = boot_labels.sum(dim=1)
        degenerate = (pos_counts == 0) | (pos_counts == n)

        boot_aucs = _auc_gpu_batch(boot_scores, boot_labels)
        boot_aucs[degenerate] = float(point_auc)

        boot_aucs_np = boot_aucs.cpu().numpy()
    else:
        rng = np.random.RandomState(seed)
        boot_aucs_list = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n, size=n)
            s_boot = scores_np[idx]
            l_boot = labels_np[idx]
            if l_boot.sum() == 0 or l_boot.sum() == n:
                boot_aucs_list.append(point_auc)
                continue
            boot_aucs_list.append(roc_auc_score(l_boot, s_boot))
        boot_aucs_np = np.array(boot_aucs_list)

    alpha = 1 - ci
    ci_low = np.percentile(boot_aucs_np, 100 * alpha / 2)
    ci_high = np.percentile(boot_aucs_np, 100 * (1 - alpha / 2))
    return point_auc, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for benchmark AP/AUC")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["vcdb", "hdd", "nuscenes", "nymeria", "muvr"],
        help="Which benchmark",
    )
    parser.add_argument(
        "--pairs-json",
        required=True,
        help="Path to pair_scores.json with per-method scores and labels",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json", default=None, help="Optional: save results to JSON"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for GPU-accelerated bootstrap (e.g. 'cuda'). Default: auto-detect.",
    )
    args = parser.parse_args()

    # Auto-detect GPU
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    with open(args.pairs_json) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(
        f"Bootstrap CIs for {args.benchmark.upper()} ({args.n_resamples} resamples, seed={args.seed})"
    )
    print(f"{'='*70}\n")

    results = {}
    for method, vals in data.items():
        scores = vals["scores"]
        labels = vals["labels"]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            print(f"  {method}: skipped (n_pos={n_pos}, n_neg={n_neg})")
            continue

        ap, ap_lo, ap_hi = bootstrap_ap(
            scores, labels, args.n_resamples, seed=args.seed, device=device,
        )
        auc, auc_lo, auc_hi = bootstrap_auc(
            scores, labels, args.n_resamples, seed=args.seed, device=device,
        )

        results[method] = {
            "ap": ap,
            "ap_ci_low": ap_lo,
            "ap_ci_high": ap_hi,
            "auc": auc,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        print(f"  {method}:")
        print(f"    AP  = {ap:.3f}  [{ap_lo:.3f}, {ap_hi:.3f}]")
        print(f"    AUC = {auc:.3f}  [{auc_lo:.3f}, {auc_hi:.3f}]")
        print(f"    (n_pos={n_pos}, n_neg={n_neg})")
        print()

    # LaTeX table rows
    print(f"\n{'='*70}")
    print("LaTeX table rows (paste into paper):")
    print(f"{'='*70}\n")

    METHOD_DISPLAY = {
        "bag_of_frames": "DINOv3 BoF",
        "chamfer": "DINOv3 Chamfer",
        "temporal_derivative": "DINOv3 Temp.\\ Deriv.",
        "attention_trajectory": "DINOv3 Attn.\\ Traj.",
        "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
        "vjepa2_temporal_residual": "V-JEPA 2 Temp.\\ Res.",
    }

    for method, r in results.items():
        display = METHOD_DISPLAY.get(method, method)
        ap_str = f"{r['ap']:.3f}"
        ci_str = f"\\tiny{{[{r['ap_ci_low']:.3f}, {r['ap_ci_high']:.3f}]}}"
        print(f"  {display} & {ap_str} {ci_str} \\\\")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pooling Ablation: Mean, Max, Attention-Weighted on Cached VLM Vision Sequences.

Demonstrates that ALL symmetric pooling operators produce order-invariant
representations (s_rev ≈ 1.0), validating Proposition 1 empirically.

Loads cached vision_seq features from the EPIC-Kitchens feature cache
(produced by eval_epic_temporal_order.py with --vlm-embeddings), computes
three pooling variants (mean, max, attention-weighted), and measures s_rev
with bootstrap CIs for each.

Expected result: all ≈ 1.0 (all permutation-invariant by construction).

Usage:
    # From cached Gemma-4 features
    python experiments/eval_pooling_ablation.py \\
        --epic-dir ./data/epic_kitchens \\
        --vlm-family gemma4

    # From cached LLaVA features
    python experiments/eval_pooling_ablation.py \\
        --epic-dir ./data/epic_kitchens \\
        --vlm-family llava-video

    # Both (run twice, or specify --vlm-family multiple times is not supported,
    # run separately)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature caching (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load cached features from disk, or return None if not found."""
    if not cache_path.exists():
        return None
    print(f"  Loading cache from {cache_path}")
    return torch.load(cache_path, weights_only=False)


# ---------------------------------------------------------------------------
# Bootstrap CI (from eval_epic_temporal_order.py)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)

    point = float(values.mean())

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Pooling operators
# ---------------------------------------------------------------------------


def pool_mean(seq: torch.Tensor) -> torch.Tensor:
    """Mean pooling over sequence dimension. seq: (T, D) -> (D,)"""
    return F.normalize(seq.mean(dim=0), dim=0)


def pool_max(seq: torch.Tensor) -> torch.Tensor:
    """Max pooling over sequence dimension. seq: (T, D) -> (D,)"""
    return F.normalize(seq.max(dim=0).values, dim=0)


def pool_attention_weighted(seq: torch.Tensor) -> torch.Tensor:
    """Attention-weighted pooling (self-attention with learned-free query).

    Uses the mean token as a query, computing dot-product attention weights
    over the sequence, then weighted-summing. This is still a symmetric
    function (permutation-invariant) because the query is the mean and
    softmax(dot(mean, x_i)) is invariant to reordering.

    seq: (T, D) -> (D,)
    """
    query = seq.mean(dim=0, keepdim=True)  # (1, D)
    scores = torch.mm(query, seq.t())  # (1, T)
    weights = F.softmax(scores, dim=-1)  # (1, T)
    pooled = torch.mm(weights, seq)  # (1, D)
    return F.normalize(pooled.squeeze(0), dim=0)


POOLING_OPS = {
    "mean": pool_mean,
    "max": pool_max,
    "attention_weighted": pool_attention_weighted,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pooling ablation on cached VLM vision sequences"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default="datasets/epic_kitchens",
        help="Path to EPIC-Kitchens dataset directory",
    )
    parser.add_argument(
        "--vlm-family",
        type=str,
        required=True,
        choices=["gemma4", "llava-video"],
        help="VLM family whose cached features to use",
    )
    args = parser.parse_args()

    epic_dir = Path(args.epic_dir)
    cache_dir = epic_dir / "feature_cache"

    suffix = f"_{args.vlm_family}"
    fwd_cache_path = cache_dir / f"vlm{suffix}_fwd.pt"
    rev_cache_path = cache_dir / f"vlm{suffix}_rev.pt"

    print("=" * 60)
    print(f"POOLING ABLATION: {args.vlm_family}")
    print("=" * 60)

    # Load cached features
    fwd_features = load_feature_cache(fwd_cache_path)
    rev_features = load_feature_cache(rev_cache_path)

    if fwd_features is None:
        print(f"ERROR: Forward cache not found at {fwd_cache_path}")
        print("Run eval_epic_temporal_order.py with --vlm-embeddings first.")
        return
    if rev_features is None:
        print(f"ERROR: Reverse cache not found at {rev_cache_path}")
        print("Run eval_epic_temporal_order.py with --vlm-embeddings first.")
        return

    # Find sequences that have vision_seq in both forward and reverse
    common_ids = []
    for seq_id in fwd_features:
        if seq_id not in rev_features:
            continue
        fwd_vs = fwd_features[seq_id].get("vision_seq")
        rev_vs = rev_features[seq_id].get("vision_seq")
        if fwd_vs is not None and rev_vs is not None:
            common_ids.append(seq_id)

    print(f"\n  Sequences with vision_seq in both fwd/rev: {len(common_ids)}")
    if len(common_ids) == 0:
        print("ERROR: No sequences with vision_seq found in cache.")
        return

    # Compute s_rev for each pooling operator
    results = {}
    for pool_name, pool_fn in POOLING_OPS.items():
        sims = []
        for seq_id in common_ids:
            fwd_seq = fwd_features[seq_id]["vision_seq"]  # (T, D)
            rev_seq = rev_features[seq_id]["vision_seq"]  # (T, D)

            if fwd_seq.dim() != 2 or rev_seq.dim() != 2:
                continue

            fwd_pooled = pool_fn(fwd_seq.float())
            rev_pooled = pool_fn(rev_seq.float())

            sim = float(torch.dot(fwd_pooled, rev_pooled).item())
            sims.append(sim)

        sims_arr = np.array(sims)
        mean_val, ci_lo, ci_hi = bootstrap_ci(sims_arr)

        results[pool_name] = {
            "s_rev": mean_val,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_sequences": len(sims),
        }

        print(
            f"\n  {pool_name:>20s}:  s_rev = {mean_val:.4f}  "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]  (n={len(sims)})"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Pooling':<22s}  {'s_rev':>8s}  {'95% CI':>20s}")
    print("-" * 55)
    for pool_name in POOLING_OPS:
        r = results[pool_name]
        print(
            f"{pool_name:<22s}  {r['s_rev']:>8.4f}  "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )
    print()

    # Save results
    out_path = cache_dir / f"pooling_ablation_{args.vlm_family}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "vlm_family": args.vlm_family,
                "n_sequences": len(common_ids),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()

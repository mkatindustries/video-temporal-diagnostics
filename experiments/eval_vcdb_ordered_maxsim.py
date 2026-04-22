#!/usr/bin/env python3
"""VCDB OrderedMaxSim ablation on cached features (both backbones).

Runs the full comparator suite on VCDB copy detection using pre-cached
DINOv3 and V-JEPA 2 features. No feature extraction needed — loads from:
  - vcdb/core_dataset/feature_cache/dinov3_sr10_mf100.pt
  - vcdb/core_dataset/feature_cache/vjepa2.pt

Comparators per backbone:
  DINOv3:  BoF, Chamfer, MaxSim, OrderedMaxSim (soft+hard), DTW on embeddings
  V-JEPA 2: BoT, Chamfer, MaxSim, OrderedMaxSim (soft+hard), DTW on residuals

Uses the paper's 1:1 negative sampling protocol (seed=42).

Usage:
    python experiments/eval_vcdb_ordered_maxsim.py \
        --vcdb-dir /path/to/vcdb/core_dataset
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_retrieval.fingerprints.dtw import dtw_distance_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5]


# ---------------------------------------------------------------------------
# VCDB data loading (matching eval_vcdb.py protocol)
# ---------------------------------------------------------------------------


def load_vcdb_annotations(ann_dir: str) -> set[tuple[str, str]]:
    """Load VCDB copy pair annotations.

    Returns set of (vidA_relpath, vidB_relpath) sorted tuples.
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
    return copy_pairs


def sample_pairs(
    keys: list[str],
    copy_pairs: set[tuple[str, str]],
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Sample balanced positive + negative pairs (1:1, seed=42).

    Returns:
        (pairs_list, labels_array) where labels are 1=copy, 0=negative.
    """
    key_set = set(keys)

    # Positive pairs with features
    pos_pairs = []
    for a, b in copy_pairs:
        if a in key_set and b in key_set:
            pos_pairs.append((a, b))

    n_pos = len(pos_pairs)
    n = len(keys)

    # Sample negatives (matching eval_vcdb.py protocol)
    neg_pairs = []
    sampled = set(pos_pairs)
    rng = np.random.RandomState(42)
    neg_attempts = 0

    while len(neg_pairs) < n_pos and neg_attempts < n_pos * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs and pair not in sampled:
            sampled.add(pair)
            neg_pairs.append(pair)
        neg_attempts += 1

    all_pairs = pos_pairs + neg_pairs
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

    return all_pairs, labels


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------


def compare_bof(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bag-of-frames: mean-pool -> cosine."""
    m1 = F.normalize(a.mean(dim=0, keepdim=True), dim=-1)
    m2 = F.normalize(b.mean(dim=0, keepdim=True), dim=-1)
    return (m1 @ m2.T).item()


def compare_chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bidirectional MaxSim."""
    sim = a @ b.T
    fwd = sim.max(dim=1).values.mean().item()
    bwd = sim.max(dim=0).values.mean().item()
    return 0.5 * (fwd + bwd)


def compare_maxsim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Unidirectional MaxSim."""
    sim = a @ b.T
    return sim.max(dim=1).values.mean().item()


def _oms_soft_dir(a: torch.Tensor, b: torch.Tensor, lam: float) -> float:
    """One-directional soft OrderedMaxSim."""
    sim = a @ b.T
    max_sims, assignments = sim.max(dim=1)
    base = max_sims.mean().item()
    violations = sum(
        1 for i in range(1, len(assignments))
        if assignments[i] < assignments[i - 1]
    )
    return base - lam * violations / len(assignments)


def compare_oms_soft(a: torch.Tensor, b: torch.Tensor, lam: float = 0.1) -> float:
    """Symmetrized soft OrderedMaxSim."""
    return 0.5 * (_oms_soft_dir(a, b, lam) + _oms_soft_dir(b, a, lam))


def _oms_hard_dir(a: torch.Tensor, b: torch.Tensor) -> float:
    """One-directional hard OrderedMaxSim."""
    sim = a @ b.T
    Ka, Kb = sim.shape
    total, prev_j = 0.0, 0
    for i in range(Ka):
        if prev_j >= Kb:
            total += sim[i].max().item()
        else:
            best_j = prev_j + sim[i, prev_j:].argmax().item()
            total += sim[i, best_j].item()
            prev_j = best_j
    return total / Ka


def compare_oms_hard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetrized hard OrderedMaxSim."""
    return 0.5 * (_oms_hard_dir(a, b) + _oms_hard_dir(b, a))


def compare_dtw(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> float:
    """DTW distance -> exp(-d) similarity."""
    dist = dtw_distance_batch([a.to(device)], [b.to(device)], normalize=True)
    return torch.exp(-dist[0]).item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def bootstrap_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for AP."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    ap = average_precision_score(labels, scores)

    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot_aps[i] = ap
        else:
            boot_aps[i] = average_precision_score(l, s)

    return float(ap), float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5))


def run_comparator(
    name: str,
    compare_fn,
    features: dict,
    seq_key: str,
    pairs: list[tuple[str, str]],
    labels: np.ndarray,
) -> dict:
    """Run a comparator on all pairs. Returns metrics dict."""
    scores = np.zeros(len(pairs))
    for k, (a, b) in enumerate(tqdm(pairs, desc=f"  {name}", leave=False)):
        fa = features.get(a)
        fb = features.get(b)
        if fa is None or fb is None:
            continue
        sa = fa[seq_key]
        sb = fb[seq_key]
        # L2-normalize for cosine-based comparators
        sa = F.normalize(sa, dim=-1)
        sb = F.normalize(sb, dim=-1)
        scores[k] = compare_fn(sa, sb)

    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "ap": round(ap, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "auc": round(auc, 4),
    }


def run_bof(
    features: dict,
    emb_key: str,
    pairs: list[tuple[str, str]],
    labels: np.ndarray,
) -> dict:
    """BoF/BoT: mean-pool cosine on pre-computed mean_emb."""
    scores = np.zeros(len(pairs))
    for k, (a, b) in enumerate(tqdm(pairs, desc="  bof/bot", leave=False)):
        fa = features.get(a)
        fb = features.get(b)
        if fa is None or fb is None:
            continue
        scores[k] = float(torch.dot(fa[emb_key], fb[emb_key]).item())

    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "ap": round(ap, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "auc": round(auc, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VCDB OrderedMaxSim ablation on cached features",
    )
    parser.add_argument(
        "--vcdb-dir", type=str,
        default=None,
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    vcdb_dir = Path(args.vcdb_dir)
    device = torch.device(args.device)
    ann_dir = vcdb_dir / "annotation"
    cache_dir = vcdb_dir / "feature_cache"

    print("=" * 70)
    print("VCDB OrderedMaxSim Ablation (Cached Features)")
    print("=" * 70)
    print(f"  VCDB dir:    {vcdb_dir}")
    print(f"  Device:      {device}")
    print()

    # ---- Load annotations ----
    print("Loading VCDB annotations...")
    copy_pairs = load_vcdb_annotations(str(ann_dir))
    print(f"  Copy pairs: {len(copy_pairs)}")

    # ---- Run per backbone ----
    backbones = [
        {
            "name": "dinov3",
            "cache": cache_dir / "dinov3_sr10_mf100.pt",
            "seq_key": "embeddings",       # (T, 1024) per-frame CLS
            "emb_key": "mean_emb",         # (1024,) pre-computed mean
        },
        {
            "name": "vjepa2",
            "cache": cache_dir / "vjepa2.pt",
            "seq_key": "temporal_residual", # (16, 1024)
            "emb_key": "mean_emb",          # (1024,)
        },
    ]

    all_results = {}

    for bb in backbones:
        print(f"\n{'='*70}")
        print(f"Backbone: {bb['name']}")
        print(f"{'='*70}")

        print(f"  Loading features from {bb['cache']}...")
        t0 = time.time()
        features = torch.load(bb["cache"], map_location="cpu", weights_only=False)
        keys = sorted(features.keys())
        print(f"  Loaded {len(keys)} videos ({time.time() - t0:.1f}s)")

        # Sample lengths
        seq_lens = []
        for k in keys:
            if bb["seq_key"] in features[k]:
                seq_lens.append(features[k][bb["seq_key"]].shape[0])
        if seq_lens:
            print(f"  Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
                  f"mean={np.mean(seq_lens):.1f}")

        # Sample pairs
        pairs, labels = sample_pairs(keys, copy_pairs)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        print(f"  Pairs: {len(pairs)} ({n_pos} pos, {n_neg} neg)")

        results: dict[str, dict] = {}

        # BoF / BoT (uses pre-computed mean_emb)
        results["bof"] = run_bof(features, bb["emb_key"], pairs, labels)

        # Chamfer
        results["chamfer"] = run_comparator(
            "chamfer", compare_chamfer, features, bb["seq_key"], pairs, labels,
        )

        # MaxSim
        results["maxsim"] = run_comparator(
            "maxsim", compare_maxsim, features, bb["seq_key"], pairs, labels,
        )

        # OrderedMaxSim soft (lambda sweep)
        for lam in LAMBDA_VALUES:
            name = f"oms_soft_lam={lam}"
            fn = lambda a, b, _lam=lam: compare_oms_soft(a, b, _lam)
            results[name] = run_comparator(
                name, fn, features, bb["seq_key"], pairs, labels,
            )

        # OrderedMaxSim hard
        results["oms_hard"] = run_comparator(
            "oms_hard", compare_oms_hard, features, bb["seq_key"], pairs, labels,
        )

        # DTW
        dtw_fn = lambda a, b: compare_dtw(a, b, device)
        results["dtw"] = run_comparator(
            "dtw", dtw_fn, features, bb["seq_key"], pairs, labels,
        )

        # Print summary
        print(f"\n  {'Comparator':<28s} {'AP':>6s} {'95% CI':>14s} {'AUC':>6s}")
        print("  " + "-" * 56)
        for name, r in results.items():
            print(f"  {name:<28s} {r['ap']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['auc']:.3f}")

        all_results[bb["name"]] = {
            "n_videos": len(keys),
            "n_pairs": len(pairs),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "results": results,
        }

    # ---- Cross-backbone summary ----
    print(f"\n{'='*70}")
    print("CROSS-BACKBONE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<28s} {'DINOv3 AP':>10s} {'V-JEPA 2 AP':>12s}")
    print("  " + "-" * 52)

    # Align methods
    if len(all_results) == 2:
        d = all_results["dinov3"]["results"]
        v = all_results["vjepa2"]["results"]
        for name in d:
            d_ap = d[name]["ap"]
            v_ap = v.get(name, {}).get("ap", float("nan"))
            print(f"  {name:<28s} {d_ap:>10.3f} {v_ap:>12.3f}")

    print(f"{'='*70}")

    # ---- Save ----
    out_path = vcdb_dir / "ordered_maxsim_ablation_results.json"
    output = {
        "experiment": "vcdb_ordered_maxsim_ablation",
        "description": (
            "OrderedMaxSim comparator suite on VCDB copy detection "
            "using cached DINOv3 and V-JEPA 2 features."
        ),
        **all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

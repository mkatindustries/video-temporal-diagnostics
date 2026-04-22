#!/usr/bin/env python3
"""Violation diagnostic on BOTH positive and negative VCDB pairs.

Compares median violations to determine if OMS penalty discriminates
between copy pairs and non-copy pairs.

Usage:
    python experiments/vcdb_violation_pos_neg.py \
        --vcdb-dir /checkpoint/dream/arjangt/video_retrieval/vcdb/core_dataset
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_vcdb_annotations(ann_dir: str) -> set[tuple[str, str]]:
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


def count_violations_directional(a: torch.Tensor, b: torch.Tensor) -> int:
    sim = a @ b.T
    assignments = sim.argmax(dim=1)
    violations = 0
    for i in range(1, len(assignments)):
        if assignments[i] < assignments[i - 1]:
            violations += 1
    return violations


def sample_negatives(keys, copy_pairs, n_target):
    """Sample negative pairs (1:1, seed=42) — matching eval_vcdb.py protocol."""
    n = len(keys)
    neg_pairs = []
    sampled = set()
    rng = np.random.RandomState(42)
    attempts = 0
    while len(neg_pairs) < n_target and attempts < n_target * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs and pair not in sampled:
            sampled.add(pair)
            neg_pairs.append(pair)
        attempts += 1
    return neg_pairs


def compute_violations(pairs, features, label):
    violations_sym = []
    violations_frac = []
    for a, b in tqdm(pairs, desc=f"  {label}"):
        fa = features.get(a)
        fb = features.get(b)
        if fa is None or fb is None:
            continue
        ea = F.normalize(fa["embeddings"], dim=-1)
        eb = F.normalize(fb["embeddings"], dim=-1)
        v_fwd = count_violations_directional(ea, eb)
        v_bwd = count_violations_directional(eb, ea)
        violations_sym.append(0.5 * (v_fwd + v_bwd))
        violations_frac.append(v_fwd / ea.shape[0])
    return np.array(violations_sym), np.array(violations_frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vcdb-dir", type=str,
        default="/checkpoint/dream/arjangt/video_retrieval/vcdb/core_dataset",
    )
    args = parser.parse_args()

    vcdb_dir = Path(args.vcdb_dir)
    cache_path = vcdb_dir / "feature_cache" / "dinov3_sr10_mf100.pt"

    print("Loading annotations...")
    copy_pairs = load_vcdb_annotations(str(vcdb_dir / "annotation"))

    print(f"Loading DINOv3 features from {cache_path}...")
    features = torch.load(cache_path, map_location="cpu", weights_only=False)
    keys = sorted(features.keys())
    key_set = set(keys)

    # Positive pairs
    pos_pairs = [(a, b) for a, b in copy_pairs if a in key_set and b in key_set]
    print(f"Positive pairs: {len(pos_pairs)}")

    # Negative pairs (same protocol as eval_vcdb.py)
    neg_pairs = sample_negatives(keys, copy_pairs, len(pos_pairs))
    print(f"Negative pairs: {len(neg_pairs)}")

    # Count violations
    print("\nCounting violations...")
    pos_v, pos_frac = compute_violations(pos_pairs, features, "positive")
    neg_v, neg_frac = compute_violations(neg_pairs, features, "negative")

    print(f"\n{'='*60}")
    print("VIOLATION DIAGNOSTIC: POSITIVE vs NEGATIVE PAIRS")
    print(f"{'='*60}")
    print(f"{'':>25s} {'Positive':>10s} {'Negative':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'median violations':<23s} {np.median(pos_v):>10.1f} {np.median(neg_v):>10.1f}")
    print(f"  {'mean violations':<23s} {np.mean(pos_v):>10.1f} {np.mean(neg_v):>10.1f}")
    print(f"  {'std violations':<23s} {np.std(pos_v):>10.1f} {np.std(neg_v):>10.1f}")
    print(f"  {'median v/length':<23s} {np.median(pos_frac):>10.3f} {np.median(neg_frac):>10.3f}")
    print(f"  {'mean v/length':<23s} {np.mean(pos_frac):>10.3f} {np.mean(neg_frac):>10.3f}")
    print()

    delta = abs(np.median(pos_v) - np.median(neg_v))
    print(f"  Median gap: {delta:.1f} violations")
    if delta <= 3:
        print(f"  → Gap ≤ 3: penalty cannot discriminate pos from neg.")
        print(f"    OMS-is-dead conclusion is airtight.")
    else:
        print(f"  → Gap > 3: penalty has discriminative power.")
        print(f"    Linear coefficient (lambda) may just be too small.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

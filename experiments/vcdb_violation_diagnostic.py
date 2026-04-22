#!/usr/bin/env python3
"""Quick diagnostic: count monotonicity violations per VCDB positive pair.

If median violations ≈ 0, the penalty is inactive and OMS=Chamfer by construction.
If median violations > 5 but OMS still matches Chamfer, the linear penalty is too gentle.

Usage:
    python experiments/vcdb_violation_diagnostic.py \
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
    """Count monotonicity violations in the greedy MaxSim assignment A->B."""
    sim = a @ b.T  # (Ka, Kb)
    assignments = sim.argmax(dim=1)  # (Ka,)
    violations = 0
    for i in range(1, len(assignments)):
        if assignments[i] < assignments[i - 1]:
            violations += 1
    return violations


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
    key_set = set(features.keys())

    # Positive pairs with features
    pos_pairs = [(a, b) for a, b in copy_pairs if a in key_set and b in key_set]
    print(f"Positive pairs with features: {len(pos_pairs)}")

    violations_fwd = []
    violations_bwd = []
    violations_sym = []
    seq_lens_a = []
    seq_lens_b = []

    for a, b in tqdm(pos_pairs, desc="Counting violations"):
        ea = F.normalize(features[a]["embeddings"], dim=-1)
        eb = F.normalize(features[b]["embeddings"], dim=-1)

        v_fwd = count_violations_directional(ea, eb)
        v_bwd = count_violations_directional(eb, ea)

        violations_fwd.append(v_fwd)
        violations_bwd.append(v_bwd)
        violations_sym.append(0.5 * (v_fwd + v_bwd))
        seq_lens_a.append(ea.shape[0])
        seq_lens_b.append(eb.shape[0])

    v = np.array(violations_sym)
    vf = np.array(violations_fwd)
    Ka = np.array(seq_lens_a)

    print(f"\n{'='*60}")
    print("MONOTONICITY VIOLATION DIAGNOSTIC (DINOv3, VCDB positive pairs)")
    print(f"{'='*60}")
    print(f"  Pairs:  {len(pos_pairs)}")
    print(f"  Seq lengths: mean={np.mean(Ka):.1f}, median={np.median(Ka):.0f}")
    print()
    print(f"  Violations (symmetrized):")
    print(f"    median:  {np.median(v):.1f}")
    print(f"    mean:    {np.mean(v):.1f}")
    print(f"    std:     {np.std(v):.1f}")
    print(f"    min:     {np.min(v):.0f}")
    print(f"    max:     {np.max(v):.0f}")
    print()

    # Violations as fraction of sequence length
    v_frac = vf / Ka
    print(f"  Violations / seq_length (forward only):")
    print(f"    median:  {np.median(v_frac):.3f}")
    print(f"    mean:    {np.mean(v_frac):.3f}")
    print()

    # Distribution buckets
    buckets = [0, 1, 2, 5, 10, 20, 50, 100]
    print(f"  Distribution (symmetrized):")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = np.sum((v >= lo) & (v < hi))
        print(f"    [{lo:>3d}, {hi:>3d}):  {count:>5d}  ({100*count/len(v):.1f}%)")
    count = np.sum(v >= buckets[-1])
    print(f"    [{buckets[-1]:>3d},  ∞):  {count:>5d}  ({100*count/len(v):.1f}%)")

    print()
    print(f"  Interpretation:")
    if np.median(v) <= 2:
        print(f"    Median ≤ 2 → penalty is effectively inactive.")
        print(f"    OMS = Chamfer because copy pairs are near-monotonic.")
    elif np.median(v) <= 5:
        print(f"    Median 2-5 → moderate violations. Penalty has some effect.")
    else:
        print(f"    Median > 5 → frequent violations. If OMS still = Chamfer,")
        print(f"    the linear penalty is too gentle. Try quadratic or hard gate.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

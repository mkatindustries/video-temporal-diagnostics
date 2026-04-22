#!/usr/bin/env python3
"""Item-3 ablation: Frozen DINOv3 + OrderedMaxSim on Honda HDD.

Tests whether an order-aware comparator on frozen frame features can close
the gap between DINOv3 Chamfer (AP~0.559) and DINOv3-sequence DTW — without
any training.

Comparators tested on identical DINOv3 ViT-L per-frame CLS tokens:
  1. mean_pool_cosine    — BoF baseline (bag-of-frames)
  2. chamfer             — bidirectional MaxSim (set matching, order-agnostic)
  3. maxsim              — unidirectional MaxSim (order-agnostic)
  4. ordered_maxsim_soft — MaxSim + soft monotonicity penalty (lambda sweep)
  5. ordered_maxsim_hard — greedy monotonic matching
  6. dtw                 — DTW reference (upper bound)

Each comparator also reports s_rev (reversal similarity) to verify whether
it is genuinely order-sensitive.

Usage:
    python experiments/eval_hdd_ordered_maxsim.py \
        --hdd-dir /checkpoint/dream/arjangt/video_retrieval/hdd
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_hdd_intersections import (
    MANEUVER_NAMES,
    ManeuverSegment,
    bootstrap_ap,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_clip,
    load_gps,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models.dinov3 import DINOv3Encoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EXTRACTION_FPS = 3.0
MAX_RESOLUTION = 518
LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_dinov3_features(
    encoder: DINOv3Encoder,
    segments: list[ManeuverSegment],
    cache_path: Path | None = None,
) -> dict[int, torch.Tensor]:
    """Extract per-frame DINOv3 CLS tokens for each segment.

    Returns:
        {segment_index: (T, D) tensor on CPU, L2-normalized}
    """
    features: dict[int, torch.Tensor] = {}

    if cache_path and cache_path.exists():
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
        features = ckpt["features"]
        print(f"  Resumed {len(features)}/{len(segments)} segments from cache")

    for i, seg in enumerate(tqdm(segments, desc="Extracting DINOv3 features")):
        if i in features:
            continue

        start_sec = seg.start_frame / 3.0
        end_sec = seg.end_frame / 3.0

        try:
            frames = load_clip(
                seg.video_path,
                start_sec,
                end_sec,
                target_fps=EXTRACTION_FPS,
                max_resolution=MAX_RESOLUTION,
            )
        except Exception:
            continue

        if len(frames) < 2:
            continue

        # (T, D) — L2-normalized CLS tokens
        embeddings = encoder.encode_frames(frames, batch_size=32, normalize=True)
        features[i] = embeddings.cpu()

        # Checkpoint every 100 segments
        if cache_path and (i + 1) % 100 == 0:
            torch.save({"features": features}, cache_path)

    if cache_path and len(features) > 0:
        torch.save({"features": features}, cache_path)

    return features


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------


def compare_mean_pool_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bag-of-frames: mean-pool -> cosine similarity."""
    a_mean = F.normalize(a.mean(dim=0, keepdim=True), dim=-1)
    b_mean = F.normalize(b.mean(dim=0, keepdim=True), dim=-1)
    return (a_mean @ b_mean.T).item()


def compare_chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bidirectional MaxSim (Chamfer similarity).

    For each frame in A find best match in B, and vice versa; average both.
    """
    sim = a @ b.T  # (Ka, Kb)
    forward = sim.max(dim=1).values.mean().item()
    backward = sim.max(dim=0).values.mean().item()
    return 0.5 * (forward + backward)


def compare_maxsim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Unidirectional MaxSim (ColBERT-style late interaction, order-agnostic)."""
    sim = a @ b.T  # (Ka, Kb)
    return sim.max(dim=1).values.mean().item()


def _ordered_maxsim_soft_directional(
    a: torch.Tensor, b: torch.Tensor, lam: float,
) -> float:
    """One-directional soft OrderedMaxSim: A -> B.

    Score = mean(max_j sim(a_i, b_j)) - lam * violations / Ka
    where violations = #{i : argmax_j(i) < argmax_j(i-1)}.
    """
    sim = a @ b.T  # (Ka, Kb)
    max_sims, assignments = sim.max(dim=1)  # (Ka,), (Ka,)
    base_score = max_sims.mean().item()

    violations = 0
    for i in range(1, len(assignments)):
        if assignments[i] < assignments[i - 1]:
            violations += 1

    return base_score - lam * violations / len(assignments)


def compare_ordered_maxsim_soft(
    a: torch.Tensor, b: torch.Tensor, lam: float = 0.1,
) -> float:
    """Symmetrized soft OrderedMaxSim: 0.5 * (A->B + B->A)."""
    forward = _ordered_maxsim_soft_directional(a, b, lam)
    backward = _ordered_maxsim_soft_directional(b, a, lam)
    return 0.5 * (forward + backward)


def _ordered_maxsim_hard_directional(
    a: torch.Tensor, b: torch.Tensor,
) -> float:
    """One-directional hard OrderedMaxSim: A -> B.

    Greedy left-to-right matching with non-strict monotonicity (j >= prev_j).
    """
    sim = a @ b.T  # (Ka, Kb)
    Ka, Kb = sim.shape

    total = 0.0
    prev_j = 0

    for i in range(Ka):
        if prev_j >= Kb:
            # Exhausted B tokens — fallback to unconstrained best
            total += sim[i].max().item()
        else:
            best_offset = sim[i, prev_j:].argmax().item()
            best_j = prev_j + best_offset
            total += sim[i, best_j].item()
            prev_j = best_j

    return total / Ka


def compare_ordered_maxsim_hard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetrized hard OrderedMaxSim: 0.5 * (A->B + B->A)."""
    forward = _ordered_maxsim_hard_directional(a, b)
    backward = _ordered_maxsim_hard_directional(b, a)
    return 0.5 * (forward + backward)


def compare_dtw(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> float:
    """DTW distance -> similarity via exp(-d)."""
    dist = dtw_distance_batch(
        [a.to(device)], [b.to(device)], normalize=True,
    )
    return torch.exp(-dist[0]).item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_comparator(
    name: str,
    compare_fn,
    features: dict[int, torch.Tensor],
    pairs: list[tuple[int, int]],
    labels: np.ndarray,
) -> dict:
    """Run a comparator on all pairs and compute AP + bootstrap CI."""
    scores = np.zeros(len(pairs))

    for k, (i, j) in enumerate(tqdm(pairs, desc=f"  {name}", leave=False)):
        if i not in features or j not in features:
            scores[k] = 0.0
            continue
        scores[k] = compare_fn(features[i], features[j])

    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels, n_resamples=2000)

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "ap": round(ap, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "auc": round(auc, 4),
        "mean_pos": round(float(scores[labels == 1].mean()), 4),
        "mean_neg": round(float(scores[labels == 0].mean()), 4),
        "gap": round(float(scores[labels == 1].mean() - scores[labels == 0].mean()), 4),
    }


def compute_s_rev(
    compare_fn,
    features: dict[int, torch.Tensor],
    n_samples: int = 100,
) -> float:
    """Mean reversal similarity: sim(original, reversed) for each segment."""
    indices = list(features.keys())[:n_samples]
    s_revs = []

    for idx in indices:
        feat = features[idx]
        feat_rev = feat.flip(0)
        s_fwd = compare_fn(feat, feat)
        s_rev_val = compare_fn(feat, feat_rev)
        # Normalize: s_rev / s_self to get a ratio in [0, 1]
        if abs(s_fwd) > 1e-8:
            s_revs.append(s_rev_val / s_fwd)
        else:
            s_revs.append(1.0)

    return round(float(np.mean(s_revs)), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Item-3 ablation: Frozen DINOv3 + OrderedMaxSim on HDD",
    )
    parser.add_argument(
        "--hdd-dir", type=str,
        default="/checkpoint/dream/arjangt/video_retrieval/hdd",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--n-rev-samples", type=int, default=100)
    args = parser.parse_args()

    hdd_dir = Path(args.hdd_dir)
    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parent.parent
    cache_path = project_root / "datasets" / "dinov3_hdd_frame_features.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Item-3 Ablation: Frozen DINOv3 + OrderedMaxSim (HDD)")
    print("=" * 70)
    print(f"  HDD dir:        {hdd_dir}")
    print(f"  Device:          {device}")
    print(f"  Feature cache:   {cache_path}")
    print(f"  Max clusters:    {args.max_clusters}")
    print(f"  Lambda sweep:    {LAMBDA_VALUES}")
    print()

    # ---- Step 1: Discover sessions ----
    t0 = time.time()
    print("Step 1: Discovering HDD sessions...")
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} sessions ({time.time() - t0:.1f}s)")

    # ---- Step 2: Extract segments & cluster ----
    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="  Sessions"):
        info = sessions[sid]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            sid, labels, gps_ts, gps_lats, gps_lngs,
            info["video_path"], info["video_start_unix"],
        )
        all_segments.extend(segs)
    print(f"  Total segments: {len(all_segments)}")

    print("  Clustering intersections...")
    clusters_raw = cluster_intersections(all_segments)
    clusters = filter_mixed_clusters(clusters_raw, max_clusters=args.max_clusters)

    # Build indexed structure
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, members in clusters.items():
        for seg in members:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    n_segs = len(eval_segments)
    print(f"  Mixed clusters: {len(clusters)}, segments: {n_segs}")

    # ---- Step 3: Load DINOv3 ----
    print("\nStep 3: Loading DINOv3 ViT-L...")
    t1 = time.time()

    import os
    dinov3_path = DINOV3_MODEL
    model_dir = os.environ.get("VTD_MODEL_DIR")
    if model_dir:
        local = Path(model_dir) / DINOV3_MODEL.split("/")[-1]
        if local.exists():
            dinov3_path = str(local)
            print(f"  Using local checkpoint: {dinov3_path}")

    encoder = DINOv3Encoder(model_name=dinov3_path, device=args.device)
    print(f"  Loaded ({time.time() - t1:.1f}s)")

    # ---- Step 4: Extract features ----
    print("\nStep 4: Extracting per-frame CLS tokens...")
    t2 = time.time()
    features = extract_dinov3_features(encoder, eval_segments, cache_path)
    n_extracted = len(features)
    print(f"  Extracted {n_extracted}/{n_segs} segments ({time.time() - t2:.1f}s)")

    # Sequence length stats
    lengths = [features[i].shape[0] for i in features]
    print(f"  Frames per segment: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")

    # Free encoder GPU memory
    del encoder
    torch.cuda.empty_cache()

    # ---- Step 5: Enumerate pairs ----
    pairs: list[tuple[int, int]] = []
    pair_labels: list[int] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                pair_labels.append(gt)

    labels = np.array(pair_labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\nStep 5: {len(pairs)} pairs ({n_pos} pos, {n_neg} neg)")

    # ---- Step 6: Run comparators ----
    print("\nStep 6: Running comparators...")
    results: dict[str, dict] = {}

    # 6a. Mean-pool cosine
    results["mean_pool_cosine"] = run_comparator(
        "mean_pool_cosine", compare_mean_pool_cosine, features, pairs, labels,
    )

    # 6b. Chamfer (bidirectional MaxSim — should reproduce ~0.559)
    results["chamfer"] = run_comparator(
        "chamfer", compare_chamfer, features, pairs, labels,
    )

    # 6c. MaxSim (unidirectional, order-agnostic)
    results["maxsim"] = run_comparator(
        "maxsim", compare_maxsim, features, pairs, labels,
    )

    # 6d. OrderedMaxSim soft (lambda sweep)
    for lam in LAMBDA_VALUES:
        name = f"ordered_maxsim_soft_lam={lam}"
        fn = lambda a, b, _lam=lam: compare_ordered_maxsim_soft(a, b, _lam)
        results[name] = run_comparator(name, fn, features, pairs, labels)

    # 6e. OrderedMaxSim hard
    results["ordered_maxsim_hard"] = run_comparator(
        "ordered_maxsim_hard", compare_ordered_maxsim_hard, features, pairs, labels,
    )

    # 6f. DTW (reference upper bound)
    dtw_fn = lambda a, b: compare_dtw(a, b, device)
    results["dtw"] = run_comparator("dtw", dtw_fn, features, pairs, labels)

    # ---- Step 7: Reversal similarity ----
    print("\nStep 7: Computing reversal similarity (s_rev)...")
    rev_results: dict[str, float] = {}
    rev_comparators = [
        ("mean_pool_cosine", compare_mean_pool_cosine),
        ("chamfer", compare_chamfer),
        ("maxsim", compare_maxsim),
        ("ordered_maxsim_soft_lam=0.1",
         lambda a, b: compare_ordered_maxsim_soft(a, b, 0.1)),
        ("ordered_maxsim_hard", compare_ordered_maxsim_hard),
        ("dtw", lambda a, b: compare_dtw(a, b, device)),
    ]
    for name, fn in rev_comparators:
        rev_results[name] = compute_s_rev(fn, features, args.n_rev_samples)

    # ---- Step 8: Print summary ----
    print("\n" + "=" * 70)
    print("RESULTS: Item-3 Ablation — Frozen DINOv3 ViT-L, No Training")
    print("=" * 70)

    header = f"  {'Comparator':<34s} {'AP':>6s} {'95% CI':>14s} {'AUC':>6s} {'Gap':>6s} {'s_rev':>6s}"
    print(header)
    print("  " + "-" * 68)

    for name, r in results.items():
        s_rev = rev_results.get(name, float("nan"))
        print(
            f"  {name:<34s} {r['ap']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
            f"{r['auc']:.3f} {r['gap']:.3f} {s_rev:.3f}"
        )

    # Gap closure analysis
    baseline_ap = results["chamfer"]["ap"]  # known reference: ~0.559
    dtw_ap = results["dtw"]["ap"]
    total_gap = dtw_ap - baseline_ap

    print()
    print("  " + "-" * 68)
    print(f"  Baseline (Chamfer):  AP = {baseline_ap:.3f}  (paper ref: ~0.559)")
    print(f"  Upper bound (DTW):   AP = {dtw_ap:.3f}")
    print(f"  Total gap:           {total_gap:.3f}")

    if total_gap > 0:
        print()
        for name, r in results.items():
            if name in ("mean_pool_cosine", "chamfer", "dtw"):
                continue
            closure = (r["ap"] - baseline_ap) / total_gap * 100
            marker = " <-- best" if name == max(
                (n for n in results if n not in ("mean_pool_cosine", "chamfer", "dtw")),
                key=lambda n: results[n]["ap"],
            ) else ""
            print(f"  {name:<34s} AP={r['ap']:.3f}  gap closure={closure:+.1f}%{marker}")

    print()
    print("  Decision thresholds:")
    print("    >= 70% gap closure  =>  OrderedMaxSim thesis validated")
    print("    30-70%              =>  partial; monotonicity helps but DTW DP matters more")
    print("    < 30%               =>  OrderedMaxSim insufficient; rethink approach")
    print("=" * 70)

    # ---- Step 9: Save results ----
    output = {
        "experiment": "item3_ordered_maxsim_ablation",
        "description": (
            "Frozen DINOv3 ViT-L + OrderedMaxSim on Honda HDD. "
            "Tests whether an order-aware comparator closes the gap between "
            "Chamfer (set matching) and DTW without any training."
        ),
        "config": {
            "model": DINOV3_MODEL,
            "extraction_fps": EXTRACTION_FPS,
            "max_resolution": MAX_RESOLUTION,
            "lambda_values": LAMBDA_VALUES,
            "max_clusters": args.max_clusters,
            "n_pairs": len(pairs),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_segments": n_extracted,
            "frame_lengths": {
                "min": int(min(lengths)),
                "max": int(max(lengths)),
                "mean": round(float(np.mean(lengths)), 1),
            },
        },
        "results": results,
        "reversal_similarity": rev_results,
        "gap_closure": {},
    }

    if total_gap > 0:
        for name, r in results.items():
            if name not in ("mean_pool_cosine", "chamfer", "dtw"):
                output["gap_closure"][name] = round(
                    (r["ap"] - baseline_ap) / total_gap * 100, 1,
                )

    out_path = hdd_dir / "ordered_maxsim_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Item-3b ablation: Frozen V-JEPA 2 encoder tokens + OrderedMaxSim on Honda HDD.

Follow-up to eval_hdd_ordered_maxsim.py (DINOv3, Item 3a). The DINOv3
experiment showed DTW score collapse and marginal OrderedMaxSim gains.
V-JEPA 2 encoder tokens are known to support DTW at AP=0.942 (from
eval_hdd_encoder_seq.py), so this is the real test of whether OrderedMaxSim
can capture that temporal structure at lower cost.

Comparators tested on identical V-JEPA 2 encoder-sequence tokens (32×1024):
  1. mean_pool_cosine    — BoT baseline (should reproduce ~0.825)
  2. chamfer             — bidirectional MaxSim (order-agnostic)
  3. maxsim              — unidirectional MaxSim (order-agnostic)
  4. ordered_maxsim_soft — MaxSim + soft monotonicity penalty (lambda sweep)
  5. ordered_maxsim_hard — greedy monotonic matching
  6. dtw                 — DTW reference (should reproduce ~0.942)

Usage:
    python experiments/eval_hdd_ordered_maxsim_vjepa2.py \
        --hdd-dir /path/to/hdd
"""

import argparse
import json
import os
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
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
    bootstrap_ap,
    build_temporal_masks,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_clip_vjepa2,
    load_gps,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5]


# ---------------------------------------------------------------------------
# Feature extraction (adapted from eval_hdd_encoder_seq.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_vjepa2_encoder_features(
    model: torch.nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
    cache_path: Path | None = None,
) -> dict[int, dict]:
    """Extract V-JEPA 2 encoder-sequence and mean embedding per segment.

    Returns:
        {segment_index: {encoder_seq: (T_PATCHES, D), mean_emb: (D,)}}
    """
    features: dict[int, dict] = {}

    if cache_path and cache_path.exists():
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
        features = ckpt["features"]
        print(f"  Resumed {len(features)}/{len(segments)} segments from cache")

    failed = 0
    for i, seg in enumerate(tqdm(segments, desc="V-JEPA 2 features")):
        if i in features:
            continue

        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            frames, stats = load_clip_vjepa2(seg.video_path, start_sec, end_sec)
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue

            # pyrefly: ignore [not-callable]
            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            enc_out = model(**inputs, skip_predictor=True)
            encoder_tokens = enc_out.last_hidden_state[0]  # (T*S, D)
            mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

            # Reshape to (T, S, D), spatial average -> (T, D)
            enc_reshaped = encoder_tokens.reshape(
                VJEPA2_T_PATCHES, VJEPA2_SPATIAL, -1
            )
            encoder_seq = enc_reshaped.mean(dim=1)  # (32, 1024)

            features[i] = {
                "encoder_seq": encoder_seq.cpu(),
                "mean_emb": mean_emb.cpu(),
            }
        except Exception:
            failed += 1
            continue

        # Checkpoint every 100 segments
        if cache_path and (i + 1) % 100 == 0:
            torch.save({"features": features}, cache_path)

    if cache_path and len(features) > 0:
        torch.save({"features": features}, cache_path)

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Comparators (same as eval_hdd_ordered_maxsim.py)
# ---------------------------------------------------------------------------


def compare_mean_pool_cosine(feat_a: dict, feat_b: dict) -> float:
    """BoT baseline: pre-computed mean embedding -> cosine."""
    return (feat_a["mean_emb"] @ feat_b["mean_emb"]).item()


def compare_chamfer(feat_a: dict, feat_b: dict) -> float:
    """Bidirectional MaxSim (Chamfer similarity) on encoder sequences."""
    a, b = feat_a["encoder_seq"], feat_b["encoder_seq"]
    # L2-normalize for cosine similarity
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = a @ b.T
    forward = sim.max(dim=1).values.mean().item()
    backward = sim.max(dim=0).values.mean().item()
    return 0.5 * (forward + backward)


def compare_maxsim(feat_a: dict, feat_b: dict) -> float:
    """Unidirectional MaxSim on encoder sequences."""
    a, b = feat_a["encoder_seq"], feat_b["encoder_seq"]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = a @ b.T
    return sim.max(dim=1).values.mean().item()


def _ordered_maxsim_soft_directional(
    a: torch.Tensor, b: torch.Tensor, lam: float,
) -> float:
    """One-directional soft OrderedMaxSim: A -> B."""
    sim = a @ b.T
    max_sims, assignments = sim.max(dim=1)
    base_score = max_sims.mean().item()

    violations = 0
    for i in range(1, len(assignments)):
        if assignments[i] < assignments[i - 1]:
            violations += 1

    return base_score - lam * violations / len(assignments)


def compare_ordered_maxsim_soft(feat_a: dict, feat_b: dict, lam: float = 0.1) -> float:
    """Symmetrized soft OrderedMaxSim on encoder sequences."""
    a, b = feat_a["encoder_seq"], feat_b["encoder_seq"]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    forward = _ordered_maxsim_soft_directional(a, b, lam)
    backward = _ordered_maxsim_soft_directional(b, a, lam)
    return 0.5 * (forward + backward)


def _ordered_maxsim_hard_directional(
    a: torch.Tensor, b: torch.Tensor,
) -> float:
    """One-directional hard OrderedMaxSim: A -> B."""
    sim = a @ b.T
    Ka, Kb = sim.shape

    total = 0.0
    prev_j = 0

    for i in range(Ka):
        if prev_j >= Kb:
            total += sim[i].max().item()
        else:
            best_offset = sim[i, prev_j:].argmax().item()
            best_j = prev_j + best_offset
            total += sim[i, best_j].item()
            prev_j = best_j

    return total / Ka


def compare_ordered_maxsim_hard(feat_a: dict, feat_b: dict) -> float:
    """Symmetrized hard OrderedMaxSim on encoder sequences."""
    a, b = feat_a["encoder_seq"], feat_b["encoder_seq"]
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    forward = _ordered_maxsim_hard_directional(a, b)
    backward = _ordered_maxsim_hard_directional(b, a)
    return 0.5 * (forward + backward)


def compare_dtw(feat_a: dict, feat_b: dict, device: torch.device) -> float:
    """DTW distance -> similarity via exp(-d)."""
    a = feat_a["encoder_seq"].to(device)
    b = feat_b["encoder_seq"].to(device)
    dist = dtw_distance_batch([a], [b], normalize=True)
    return torch.exp(-dist[0]).item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_comparator(
    name: str,
    compare_fn,
    features: dict[int, dict],
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
    features: dict[int, dict],
    n_samples: int = 100,
) -> float:
    """Mean reversal similarity on encoder sequences."""
    indices = list(features.keys())[:n_samples]
    s_revs = []

    for idx in indices:
        feat = features[idx]
        feat_rev = {
            "encoder_seq": feat["encoder_seq"].flip(0),
            "mean_emb": feat["mean_emb"],
        }
        s_fwd = compare_fn(feat, feat)
        s_rev_val = compare_fn(feat, feat_rev)
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
        description="Item-3b: Frozen V-JEPA 2 encoder + OrderedMaxSim on HDD",
    )
    parser.add_argument(
        "--hdd-dir", type=str,
        default=None,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-sec", type=float, default=3.0)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--n-rev-samples", type=int, default=100)
    args = parser.parse_args()

    hdd_dir = Path(args.hdd_dir)
    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parent.parent
    cache_path = project_root / "datasets" / "vjepa2_hdd_encoder_features.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Item-3b Ablation: Frozen V-JEPA 2 Encoder + OrderedMaxSim (HDD)")
    print("=" * 70)
    print(f"  HDD dir:        {hdd_dir}")
    print(f"  Device:          {device}")
    print(f"  Feature cache:   {cache_path}")
    print(f"  Max clusters:    {args.max_clusters}")
    print(f"  Context:         {args.context_sec}s")
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

    # ---- Step 3: Load V-JEPA 2 ----
    print("\nStep 3: Loading V-JEPA 2...")
    from transformers import AutoModel, AutoVideoProcessor

    vjepa2_path = VJEPA2_MODEL_NAME
    model_dir = os.environ.get("VTD_MODEL_DIR")
    if model_dir:
        local = Path(model_dir) / VJEPA2_MODEL_NAME.split("/")[-1]
        if local.exists():
            vjepa2_path = str(local)
            print(f"  Using local checkpoint: {vjepa2_path}")

    t1 = time.time()
    model = AutoModel.from_pretrained(
        vjepa2_path, trust_remote_code=True,
    ).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(
        vjepa2_path, trust_remote_code=True,
    )
    print(f"  Loaded ({time.time() - t1:.1f}s)")

    # ---- Step 4: Extract features ----
    print("\nStep 4: Extracting encoder-sequence tokens...")
    t2 = time.time()
    features = extract_vjepa2_encoder_features(
        model, processor, eval_segments, device,
        context_sec=args.context_sec, cache_path=cache_path,
    )
    n_extracted = len(features)
    print(f"  Total: {n_extracted}/{n_segs} ({time.time() - t2:.1f}s)")

    # Free model GPU memory
    del model, processor
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

    # 6a. BoT cosine (mean-pool — should reproduce ~0.825)
    results["mean_pool_cosine"] = run_comparator(
        "mean_pool_cosine", compare_mean_pool_cosine, features, pairs, labels,
    )

    # 6b. Chamfer
    results["chamfer"] = run_comparator(
        "chamfer", compare_chamfer, features, pairs, labels,
    )

    # 6c. MaxSim
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

    # 6f. DTW (should reproduce ~0.942)
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
    # Reference values from eval_hdd_encoder_seq.py
    ref_ap = {
        "mean_pool_cosine": 0.825,
        "dtw": 0.942,
    }

    print("\n" + "=" * 70)
    print("RESULTS: Item-3b — Frozen V-JEPA 2 Encoder, No Training")
    print("=" * 70)

    header = f"  {'Comparator':<34s} {'AP':>6s} {'95% CI':>14s} {'AUC':>6s} {'Gap':>6s} {'s_rev':>6s}"
    print(header)
    print("  " + "-" * 68)

    for name, r in results.items():
        s_rev = rev_results.get(name, float("nan"))
        ref = ref_ap.get(name)
        ref_str = f"  (ref={ref:.3f})" if ref else ""
        print(
            f"  {name:<34s} {r['ap']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
            f"{r['auc']:.3f} {r['gap']:.3f} {s_rev:.3f}{ref_str}"
        )

    # Gap closure analysis: BoT -> DTW
    bot_ap = results["mean_pool_cosine"]["ap"]
    dtw_ap = results["dtw"]["ap"]
    total_gap = dtw_ap - bot_ap

    print()
    print("  " + "-" * 68)
    print(f"  Baseline (BoT cosine):   AP = {bot_ap:.3f}  (ref: ~0.825)")
    print(f"  Upper bound (DTW):       AP = {dtw_ap:.3f}  (ref: ~0.942)")
    print(f"  Total gap:               {total_gap:.3f}")

    if total_gap > 0:
        print()
        for name, r in results.items():
            if name in ("mean_pool_cosine", "dtw"):
                continue
            closure = (r["ap"] - bot_ap) / total_gap * 100
            marker = " <-- best" if name == max(
                (n for n in results if n not in ("mean_pool_cosine", "dtw")),
                key=lambda n: results[n]["ap"],
            ) else ""
            print(f"  {name:<34s} AP={r['ap']:.3f}  gap closure={closure:+.1f}%{marker}")

    print()
    print("  Decision thresholds (gap closure from BoT to DTW):")
    print("    >= 70%  =>  OrderedMaxSim thesis validated")
    print("    30-70%  =>  partial; monotonicity helps but DTW DP matters more")
    print("    < 30%   =>  OrderedMaxSim insufficient on V-JEPA 2; thesis fails")
    print("=" * 70)

    # ---- Step 9: Save results ----
    output = {
        "experiment": "item3b_ordered_maxsim_vjepa2_ablation",
        "description": (
            "Frozen V-JEPA 2 encoder-sequence tokens + OrderedMaxSim on Honda HDD. "
            "Tests whether OrderedMaxSim captures the temporal structure that "
            "DTW exploits (AP=0.942) at lower O(K^2) cost."
        ),
        "config": {
            "model": VJEPA2_MODEL_NAME,
            "t_patches": VJEPA2_T_PATCHES,
            "spatial": VJEPA2_SPATIAL,
            "num_frames": VJEPA2_NUM_FRAMES,
            "context_sec": args.context_sec,
            "lambda_values": LAMBDA_VALUES,
            "max_clusters": args.max_clusters,
            "n_pairs": len(pairs),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_segments": n_extracted,
        },
        "results": results,
        "reversal_similarity": rev_results,
        "gap_closure": {},
    }

    if total_gap > 0:
        for name, r in results.items():
            if name not in ("mean_pool_cosine", "dtw"):
                output["gap_closure"][name] = round(
                    (r["ap"] - bot_ap) / total_gap * 100, 1,
                )

    out_path = hdd_dir / "ordered_maxsim_vjepa2_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

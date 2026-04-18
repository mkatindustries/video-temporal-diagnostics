#!/usr/bin/env python3
"""Honda HDD Comparator Ablation for V-JEPA 2 Temporal Residuals.

Answers the reviewer question: "Does the HDD AP=0.956 come from the residual
representation or from DTW alignment?"

Applies 5 different comparators to the SAME residual representation and the
SAME encoder representation, isolating representation vs alignment:

Comparators:
  a. DTW (baseline) — standard DTW on the sequence, exp(-d_DTW)
  b. Mean-pool + cosine — average the sequence to a single vector, cosine sim
  c. Temporal statistics — [mean, std, max, min] over time, cosine sim
  d. Last-frame cosine — use only the last temporal position, cosine sim
  e. Random temporal crop — sample 4 positions, pool, cosine (avg over 5 seeds)

Representations:
  1. Temporal residuals (predicted - ground_truth, spatially averaged)
  2. Encoder tokens (mean-pooled per timestep)

Expected result:
  - DTW on residuals >> mean-pool on residuals (temporal alignment matters)
  - DTW on encoder tokens ~ mean-pool on encoder tokens (DTW doesn't help
    without the right representation)
  - The COMBINATION of residuals + DTW achieves AP=0.956

Usage:
    python experiments/eval_hdd_residual_ablation.py \\
        --hdd-dir /path/to/hdd --device cuda
"""

import argparse
import json

# Re-use data loading, clustering, and clip extraction from the intersections
# experiment — import directly to avoid code duplication.
# Add experiments/ to sys.path so imports work from any working directory.
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from eval_hdd_intersections import (  # noqa: E402
    bootstrap_ap,
    build_temporal_masks,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_clip_vjepa2,
    load_gps,
    MANEUVER_NAMES,
    ManeuverSegment,
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from video_retrieval.fingerprints.dtw import dtw_distance_batch


# ---------------------------------------------------------------------------
# Feature extraction — residuals AND encoder token sequences
# ---------------------------------------------------------------------------


def extract_vjepa2_ablation_features(
    model: torch.nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for the comparator ablation.

    For each segment, computes:
    - temporal_residual: (n_target_steps, D) residual vectors
    - encoder_temporal: (T_PATCHES, D) encoder tokens averaged over spatial dim

    Args:
        model: V-JEPA 2 model.
        processor: V-JEPA 2 video processor.
        segments: Maneuver segments to process.
        device: Torch device.
        context_sec: Seconds of context before/after maneuver.

    Returns:
        Dict mapping segment index -> {
            'temporal_residual': (n_target, D),
            'encoder_temporal': (T_PATCHES, D),
        }
    """
    n_context_steps = VJEPA2_T_PATCHES // 2  # 16
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps  # 16
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting V-JEPA 2 features")):
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            frames, _stats = load_clip_vjepa2(seg.video_path, start_sec, end_sec)
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue

            # pyrefly: ignore [not-callable]
            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Encoder only: full token sequence
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]  # (T*S, D)
                # Reshape to (T_PATCHES, SPATIAL, D), mean over spatial
                encoder_temporal = encoder_tokens.reshape(
                    VJEPA2_T_PATCHES, VJEPA2_SPATIAL, -1
                ).mean(
                    dim=1
                )  # (T_PATCHES, D)

                # With predictor: temporal residuals
                pred_out = model(
                    **inputs,
                    context_mask=[context_mask],
                    target_mask=[target_mask],
                )
                predicted = pred_out.predictor_output.last_hidden_state[0]
                ground_truth = pred_out.predictor_output.target_hidden_state[0]

                # Reshape to (n_target_steps, SPATIAL, D), mean over spatial
                predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                residual = (predicted - ground_truth).mean(dim=1)  # (n_target, D)

            features[i] = {
                "temporal_residual": residual.cpu(),
                "encoder_temporal": encoder_temporal.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Comparator functions
# ---------------------------------------------------------------------------


def compare_dtw(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    device: torch.device,
    normalize: bool = True,
) -> list[float]:
    """DTW similarity: exp(-d_DTW)."""
    if not seqs_a:
        return []
    seqs_a_gpu = [s.to(device) for s in seqs_a]
    seqs_b_gpu = [s.to(device) for s in seqs_b]
    dists = dtw_distance_batch(seqs_a_gpu, seqs_b_gpu, normalize=normalize)
    return torch.exp(-dists).cpu().tolist()


def compare_mean_pool_cosine(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
) -> list[float]:
    """Mean-pool over time, then cosine similarity."""
    if not seqs_a:
        return []
    vecs_a = torch.stack([F.normalize(s.mean(dim=0), dim=0) for s in seqs_a])
    vecs_b = torch.stack([F.normalize(s.mean(dim=0), dim=0) for s in seqs_b])
    return (vecs_a * vecs_b).sum(dim=1).tolist()


def compare_temporal_stats_cosine(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
) -> list[float]:
    """Temporal statistics [mean, std, max, min] concatenated, cosine sim."""
    if not seqs_a:
        return []

    def _stats_vec(seq: torch.Tensor) -> torch.Tensor:
        # seq: (T, D)
        return torch.cat(
            [
                seq.mean(dim=0),
                seq.std(dim=0),
                seq.max(dim=0).values,
                seq.min(dim=0).values,
            ]
        )  # (4*D,)

    vecs_a = torch.stack([F.normalize(_stats_vec(s), dim=0) for s in seqs_a])
    vecs_b = torch.stack([F.normalize(_stats_vec(s), dim=0) for s in seqs_b])
    return (vecs_a * vecs_b).sum(dim=1).tolist()


def compare_last_frame_cosine(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
) -> list[float]:
    """Last temporal position, cosine similarity."""
    if not seqs_a:
        return []
    vecs_a = torch.stack([F.normalize(s[-1], dim=0) for s in seqs_a])
    vecs_b = torch.stack([F.normalize(s[-1], dim=0) for s in seqs_b])
    return (vecs_a * vecs_b).sum(dim=1).tolist()


def compare_random_crop_cosine(
    seqs_a: list[torch.Tensor],
    seqs_b: list[torch.Tensor],
    n_crops: int = 4,
    n_seeds: int = 5,
    base_seed: int = 42,
) -> list[float]:
    """Randomly sample n_crops temporal positions, mean-pool, cosine sim.

    Averaged over n_seeds random seeds for stability.
    """
    if not seqs_a:
        return []

    n_pairs = len(seqs_a)
    all_sims = torch.zeros(n_pairs)

    for seed_offset in range(n_seeds):
        rng = np.random.RandomState(base_seed + seed_offset)

        vecs_a = []
        vecs_b = []
        for sa, sb in zip(seqs_a, seqs_b):
            t_a = sa.shape[0]
            t_b = sb.shape[0]
            idx_a = rng.choice(t_a, size=min(n_crops, t_a), replace=False)
            idx_b = rng.choice(t_b, size=min(n_crops, t_b), replace=False)
            vecs_a.append(F.normalize(sa[idx_a].mean(dim=0), dim=0))
            vecs_b.append(F.normalize(sb[idx_b].mean(dim=0), dim=0))

        va = torch.stack(vecs_a)
        vb = torch.stack(vecs_b)
        sims = (va * vb).sum(dim=1)
        all_sims += sims

    all_sims /= n_seeds
    return all_sims.tolist()


# ---------------------------------------------------------------------------
# Pair enumeration
# ---------------------------------------------------------------------------


def enumerate_pairs(
    segments: list[ManeuverSegment],
    cluster_to_indices: dict[int, list[int]],
    features: dict[int, dict],
) -> tuple[list[int], list[int], list[int]]:
    """Enumerate all within-cluster pairs that have features.

    Returns:
        (pair_a_indices, pair_b_indices, pair_labels)
        where label=1 for same maneuver, 0 for different.
    """
    pair_a = []
    pair_b = []
    pair_gt = []

    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                ia, ib = indices[a_pos], indices[b_pos]
                pair_a.append(ia)
                pair_b.append(ib)
                gt = 1 if segments[ia].label == segments[ib].label else 0
                pair_gt.append(gt)

    return pair_a, pair_b, pair_gt


# ---------------------------------------------------------------------------
# Run all comparators on one representation
# ---------------------------------------------------------------------------


COMPARATOR_NAMES = [
    "dtw",
    "mean_pool_cosine",
    "temporal_stats_cosine",
    "last_frame_cosine",
    "random_crop_cosine",
]


def run_comparators(
    repr_name: str,
    repr_key: str,
    features: dict[int, dict],
    pair_a: list[int],
    pair_b: list[int],
    pair_gt: list[int],
    device: torch.device,
    n_bootstrap: int = 1000,
) -> dict[str, dict]:
    """Run all 5 comparators on a given representation.

    Args:
        repr_name: Human-readable name (e.g. "residual", "encoder").
        repr_key: Key in features dict (e.g. "temporal_residual", "encoder_temporal").
        features: Feature dict mapping segment index -> feature tensors.
        pair_a, pair_b, pair_gt: Pair indices and ground truth labels.
        device: Torch device.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Dict mapping comparator_name -> {ap, auc, ci_low, ci_high, ...}
    """
    seqs_a = [features[i][repr_key] for i in pair_a]
    seqs_b = [features[i][repr_key] for i in pair_b]

    results = {}

    for comp_name in COMPARATOR_NAMES:
        print(f"    {repr_name} x {comp_name}...")

        if comp_name == "dtw":
            sims = compare_dtw(seqs_a, seqs_b, device, normalize=True)
        elif comp_name == "mean_pool_cosine":
            sims = compare_mean_pool_cosine(seqs_a, seqs_b)
        elif comp_name == "temporal_stats_cosine":
            sims = compare_temporal_stats_cosine(seqs_a, seqs_b)
        elif comp_name == "last_frame_cosine":
            sims = compare_last_frame_cosine(seqs_a, seqs_b)
        elif comp_name == "random_crop_cosine":
            sims = compare_random_crop_cosine(seqs_a, seqs_b)
        else:
            raise ValueError(f"Unknown comparator: {comp_name}")

        scores = np.array(sims)
        labels = np.array(pair_gt)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[comp_name] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap_val, ci_low, ci_high = bootstrap_ap(
            scores, labels, n_resamples=n_bootstrap, seed=42
        )
        auc_val = float(roc_auc_score(labels, scores))

        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())

        results[comp_name] = {
            "ap": ap_val,
            "auc": auc_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": same_mean - diff_mean,
        }

        print(
            f"      AP={ap_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
            f"AUC={auc_val:.4f}  gap={same_mean - diff_mean:+.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Honda HDD Comparator Ablation for V-JEPA 2 Temporal Residuals"
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
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("HONDA HDD: COMPARATOR ABLATION FOR V-JEPA 2 TEMPORAL RESIDUALS")
    print("=" * 70)
    print(f"  Comparators: {COMPARATOR_NAMES}")
    print(f"  Representations: temporal_residual, encoder_temporal")
    print(f"  Bootstrap resamples: {args.n_bootstrap}")

    # ------------------------------------------------------------------
    # Step 1: Discover sessions
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    t0 = time.time()
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} valid sessions")
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
    print(f"  Mixed clusters (both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print("\nERROR: No mixed clusters found. Cannot evaluate.")
        return

    # Build flat segment list and cluster mapping
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
    # Step 5: Load V-JEPA 2 and extract features
    # ------------------------------------------------------------------
    print("\nStep 5: Loading V-JEPA 2 model...")
    from transformers import AutoModel, AutoVideoProcessor

    vjepa2_model = AutoModel.from_pretrained(VJEPA2_MODEL_NAME, trust_remote_code=True)
    vjepa2_model = vjepa2_model.to(args.device).eval()
    vjepa2_processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    print("  Extracting V-JEPA 2 features (residuals + encoder temporal)...")
    t_feat_start = time.time()
    features = extract_vjepa2_ablation_features(
        vjepa2_model,
        vjepa2_processor,
        eval_segments,
        device=device,
        context_sec=args.context_sec,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    # Free model memory
    del vjepa2_model, vjepa2_processor
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Enumerate pairs
    # ------------------------------------------------------------------
    print("\nStep 6: Enumerating within-cluster pairs...")
    pair_a, pair_b, pair_gt = enumerate_pairs(
        eval_segments, cluster_to_indices, features
    )
    n_pos = sum(pair_gt)
    n_neg = len(pair_gt) - n_pos
    print(f"  Total pairs: {len(pair_gt)} (pos={n_pos}, neg={n_neg})")

    if not pair_gt:
        print("\nERROR: No pairs to evaluate.")
        return

    # ------------------------------------------------------------------
    # Step 7: Run all comparators on both representations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING COMPARATOR ABLATION")
    print("=" * 70)

    all_results = {}

    # --- Temporal residuals ---
    print("\n  [Representation: Temporal Residuals]")
    residual_results = run_comparators(
        repr_name="residual",
        repr_key="temporal_residual",
        features=features,
        pair_a=pair_a,
        pair_b=pair_b,
        pair_gt=pair_gt,
        device=device,
        n_bootstrap=args.n_bootstrap,
    )
    for comp_name, res in residual_results.items():
        all_results[f"residual__{comp_name}"] = res

    # --- Encoder tokens ---
    print("\n  [Representation: Encoder Tokens]")
    encoder_results = run_comparators(
        repr_name="encoder",
        repr_key="encoder_temporal",
        features=features,
        pair_a=pair_a,
        pair_b=pair_b,
        pair_gt=pair_gt,
        device=device,
        n_bootstrap=args.n_bootstrap,
    )
    for comp_name, res in encoder_results.items():
        all_results[f"encoder__{comp_name}"] = res

    # ------------------------------------------------------------------
    # Step 8: Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"  {'Representation':<18s}  {'Comparator':<22s}  "
        f"{'AP':>7s}  {'95% CI':>15s}  {'AUC':>7s}  {'Gap':>7s}"
    )
    print("  " + "-" * 82)

    for repr_name in ["residual", "encoder"]:
        for comp_name in COMPARATOR_NAMES:
            key = f"{repr_name}__{comp_name}"
            r = all_results[key]
            ap = r["ap"]
            ci_lo = r["ci_low"]
            ci_hi = r["ci_high"]
            auc = r["auc"]
            gap = r.get("gap", float("nan"))
            print(
                f"  {repr_name:<18s}  {comp_name:<22s}  "
                f"{ap:7.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  "
                f"{auc:7.4f}  {gap:+7.4f}"
            )

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    out_path = hdd_dir / "residual_ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "hdd_residual_ablation",
        "description": (
            "Comparator ablation: 5 comparators x 2 representations "
            "(temporal residual vs encoder tokens) on Honda HDD"
        ),
        "config": {
            "max_clusters": args.max_clusters,
            "context_sec": args.context_sec,
            "n_bootstrap": args.n_bootstrap,
            "n_pairs": len(pair_gt),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_segments": len(eval_segments),
            "n_segments_with_features": len(features),
            "n_clusters": len(mixed),
        },
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Print interpretation
    # ------------------------------------------------------------------
    res_dtw = all_results.get("residual__dtw", {}).get("ap", float("nan"))
    res_mean = all_results.get("residual__mean_pool_cosine", {}).get("ap", float("nan"))
    enc_dtw = all_results.get("encoder__dtw", {}).get("ap", float("nan"))
    enc_mean = all_results.get("encoder__mean_pool_cosine", {}).get("ap", float("nan"))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  Residual + DTW:       AP={res_dtw:.4f}")
    print(f"  Residual + Mean-pool: AP={res_mean:.4f}")
    print(f"  Encoder  + DTW:       AP={enc_dtw:.4f}")
    print(f"  Encoder  + Mean-pool: AP={enc_mean:.4f}")
    if not (np.isnan(res_dtw) or np.isnan(res_mean)):
        dtw_lift = res_dtw - res_mean
        print(f"\n  DTW lift on residuals: {dtw_lift:+.4f}")
    if not (np.isnan(enc_dtw) or np.isnan(enc_mean)):
        dtw_lift_enc = enc_dtw - enc_mean
        print(f"  DTW lift on encoder:   {dtw_lift_enc:+.4f}")
    print(
        "\n  If DTW lift on residuals >> DTW lift on encoder, the result "
        "is driven by\n  the COMBINATION of residual representation + DTW "
        "alignment, not either alone."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

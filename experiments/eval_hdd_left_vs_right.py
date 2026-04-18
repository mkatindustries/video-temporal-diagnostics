#!/usr/bin/env python3
"""HDD Left-vs-Right Only Evaluation.

Filters the HDD maneuver discrimination to only left turn (label=2) vs
right turn (label=3) pairs, excluding intersection-passing (label=1).
This addresses the reviewer concern about passage inflation — label 1
segments at the same intersection are trivially similar (same straight-
through trajectory), which could inflate AP for methods that struggle
with the harder left/right discrimination.

Reuses all data loading, feature extraction, and similarity code from
eval_hdd_intersections.py. Operates on cached features to avoid re-
extraction (~1 min on CPU).

Usage:
    python experiments/eval_hdd_left_vs_right.py [--device cuda]
    python experiments/eval_hdd_left_vs_right.py --hdd-dir datasets/hdd
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# Reuse infrastructure from eval_hdd_intersections
from eval_hdd_intersections import (
    MANEUVER_NAMES,
    ManeuverSegment,
    bootstrap_ap,
    cluster_intersections,
    discover_sessions,
    extract_clip_features,
    extract_maneuver_segments,
    extract_vjepa2_features,
    filter_mixed_clusters,
    load_gps,
)
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"


def _resolve_model_path(model_name: str) -> str:
    """Resolve HuggingFace model name to local path if available.

    Checks $VTD_MODEL_DIR (if set) for a local directory matching the
    model's short name (e.g. 'vjepa2-vitl-fpc64-256' for
    'facebook/vjepa2-vitl-fpc64-256'). Falls back to the HuggingFace ID.
    """
    import os
    from pathlib import Path

    model_dir = os.environ.get("VTD_MODEL_DIR")
    if model_dir:
        local_name = model_name.split("/")[-1]
        local_path = Path(model_dir) / local_name
        if local_path.exists():
            return str(local_path)
    return model_name


def compute_left_right_similarities(
    segments: list[ManeuverSegment],
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    vjepa2_features: dict[int, dict] | None = None,
    device: torch.device | None = None,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within clusters, LEFT+RIGHT ONLY.

    Same as compute_all_similarities from eval_hdd_intersections but
    filters to pairs where BOTH segments have label in {2, 3}.

    Ground truth: same maneuver = positive (1), different = negative (0).
    So left-left and right-right are positive, left-right are negative.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()

    # Pre-compute fingerprints
    print("  Pre-computing fingerprints...")
    deriv_fps = {}
    traj_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])
        traj_fps[idx] = traj_fp.compute_fingerprint(features[idx]["centroids"])

    # Enumerate pairs — filter to left/right only
    print("  Enumerating left/right pairs...")
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []
    skipped = 0

    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                a_idx = indices[a_pos]
                b_idx = indices[b_pos]

                # Filter: both must be left (2) or right (3)
                if segments[a_idx].label not in (2, 3):
                    skipped += 1
                    continue
                if segments[b_idx].label not in (2, 3):
                    skipped += 1
                    continue

                pair_a_indices.append(a_idx)
                pair_b_indices.append(b_idx)
                gt = 1 if segments[a_idx].label == segments[b_idx].label else 0
                pair_gts.append(gt)

    total_pairs = len(pair_gts)
    n_pos = sum(pair_gts)
    n_neg = total_pairs - n_pos
    print(f"  Left/right pairs: {total_pairs} (pos={n_pos}, neg={n_neg}, skipped={skipped})")

    if total_pairs == 0:
        return {}

    # --- Bag-of-frames ---
    print("  Computing bag-of-frames similarities...")
    mean_embs_a = torch.stack([features[i]["mean_emb"] for i in pair_a_indices]).to(device)
    mean_embs_b = torch.stack([features[i]["mean_emb"] for i in pair_b_indices]).to(device)
    bof_sims = (mean_embs_a * mean_embs_b).sum(dim=1).cpu().tolist()

    # --- Chamfer ---
    print("  Computing Chamfer similarities...")
    chamfer_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        ea = features[a_idx]["embeddings"].to(device)
        eb = features[b_idx]["embeddings"].to(device)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims.append((max_ab + max_ba) / 2)

    # --- Temporal derivative DTW ---
    print("  Computing temporal derivative DTW (batched GPU)...")
    deriv_seqs_a = [deriv_fps[i].to(device) for i in pair_a_indices]
    deriv_seqs_b = [deriv_fps[i].to(device) for i in pair_b_indices]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).cpu().tolist()

    # --- Attention trajectory DTW ---
    print("  Computing attention trajectory DTW (batched GPU)...")
    traj_seqs_a = [traj_fps[i].to(device) for i in pair_a_indices]
    traj_seqs_b = [traj_fps[i].to(device) for i in pair_b_indices]
    traj_dists = dtw_distance_batch(traj_seqs_a, traj_seqs_b, normalize=True)
    traj_sims = torch.exp(-5.0 * traj_dists).cpu().tolist()

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": (bof_sims, list(pair_gts)),
        "chamfer": (chamfer_sims, list(pair_gts)),
        "temporal_derivative": (deriv_sims, list(pair_gts)),
        "attention_trajectory": (traj_sims, list(pair_gts)),
    }

    # --- V-JEPA 2 methods ---
    if vjepa2_features:
        vjepa2_mask = [
            (a_idx in vjepa2_features and b_idx in vjepa2_features)
            for a_idx, b_idx in zip(pair_a_indices, pair_b_indices)
        ]
        v_a_indices = [a for a, m in zip(pair_a_indices, vjepa2_mask) if m]
        v_b_indices = [b for b, m in zip(pair_b_indices, vjepa2_mask) if m]
        v_gts = [g for g, m in zip(pair_gts, vjepa2_mask) if m]

        if v_a_indices:
            print("  Computing V-JEPA 2 bag-of-tokens similarities...")
            v_mean_a = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_a_indices]
            ).to(device)
            v_mean_b = torch.stack(
                [vjepa2_features[i]["mean_emb"] for i in v_b_indices]
            ).to(device)
            bot_sims = (v_mean_a * v_mean_b).sum(dim=1).cpu().tolist()

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

    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="HDD Left-vs-Right Only Maneuver Discrimination"
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
        help="Maximum number of mixed clusters",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Seconds of context before/after maneuver",
    )
    parser.add_argument(
        "--use-cached-features",
        action="store_true",
        default=True,
        help="Use cached features if available (default: True)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir

    print("=" * 70)
    print("HDD: LEFT-vs-RIGHT ONLY EVALUATION")
    print("(Excludes intersection-passing segments to address inflation)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover sessions and extract segments
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} sessions")

    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []
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
            target_labels=(1, 2, 3),  # Need all for clustering
        )
        all_segments.extend(segs)

    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1

    print(f"  Total segments: {len(all_segments)}")
    for label_val, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(label_val, 0)}")

    # ------------------------------------------------------------------
    # Step 3: Cluster and filter — keep mixed clusters but note that
    # evaluation will only use left/right pairs within them
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
    print(f"  Mixed clusters: {len(mixed)}")

    # Build flat segment list with cluster mapping
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    # Count left/right in eval set
    lr_counts = defaultdict(int)
    for seg in eval_segments:
        if seg.label in (2, 3):
            lr_counts[seg.label] += 1

    print(f"  Segments in mixed clusters: {len(eval_segments)}")
    print(f"    left_turn (2): {lr_counts[2]}")
    print(f"    right_turn (3): {lr_counts[3]}")

    if lr_counts[2] == 0 or lr_counts[3] == 0:
        print("\nERROR: Need both left and right turns for evaluation.")
        return

    # ------------------------------------------------------------------
    # Step 4: Extract features (or load cached)
    # ------------------------------------------------------------------
    cache_dir = hdd_dir
    dinov3_cache = cache_dir / "left_right_dinov3_features.pt"
    vjepa2_cache = cache_dir / "left_right_vjepa2_features.pt"

    device = torch.device(args.device)

    if args.use_cached_features and dinov3_cache.exists():
        print(f"\nStep 4: Loading cached DINOv3 features from {dinov3_cache}")
        features = torch.load(dinov3_cache, weights_only=False)
    else:
        print("\nStep 4: Extracting DINOv3 features...")
        encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)
        features = extract_clip_features(
            encoder,
            eval_segments,
            context_sec=args.context_sec,
        )
        torch.save(
            {k: {fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv
                  for fk, fv in v.items()}
             for k, v in features.items()},
            dinov3_cache,
        )
        print(f"  Cached to {dinov3_cache}")
        del encoder
        torch.cuda.empty_cache()

    if args.use_cached_features and vjepa2_cache.exists():
        print(f"  Loading cached V-JEPA 2 features from {vjepa2_cache}")
        vjepa2_features = torch.load(vjepa2_cache, weights_only=False)
    else:
        print("  Extracting V-JEPA 2 features...")
        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_path = _resolve_model_path(VJEPA2_MODEL_NAME)
        vjepa2_model = AutoModel.from_pretrained(
            vjepa2_path, trust_remote_code=True
        ).to(device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            vjepa2_path, trust_remote_code=True
        )
        vjepa2_features, _ = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            eval_segments,
            device=device,
            context_sec=args.context_sec,
        )
        torch.save(
            {k: {fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv
                  for fk, fv in v.items()}
             for k, v in vjepa2_features.items()},
            vjepa2_cache,
        )
        print(f"  Cached to {vjepa2_cache}")
        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5: Compute left/right-only similarities
    # ------------------------------------------------------------------
    print("\nStep 5: Computing left/right-only pairwise similarities...")
    t0 = time.time()
    all_scores = compute_left_right_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        vjepa2_features=vjepa2_features,
        device=device,
    )
    print(f"  Similarity time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: LEFT-vs-RIGHT ONLY (no intersection-passing)")
    print("=" * 70)

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

    results = {}
    for method in method_order:
        if method not in all_scores:
            continue
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            print(f"  {method:<30s}  DEGENERATE (pos={n_pos}, neg={n_neg})")
            continue

        ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
        auc = roc_auc_score(labels, scores)

        results[method] = {
            "ap": ap,
            "ap_ci_low": ci_lo,
            "ap_ci_high": ci_hi,
            "auc": float(auc),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        print(
            f"  {method:<30s}  AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"AUC={auc:.4f}  (pos={n_pos}, neg={n_neg})"
        )

    # ------------------------------------------------------------------
    # Also show comparison with full eval (all labels) for context
    # ------------------------------------------------------------------
    full_pair_path = hdd_dir / "pair_scores.json"
    if full_pair_path.exists():
        print(f"\n{'=' * 70}")
        print("COMPARISON: Full eval (all labels) vs Left/Right only")
        print("=" * 70)
        with open(full_pair_path) as f:
            full_data = json.load(f)

        print(f"  {'Method':<30s}  {'Full AP':>8s}  {'L/R AP':>8s}  {'Delta':>8s}")
        print(f"  {'—' * 30}  {'—' * 8}  {'—' * 8}  {'—' * 8}")
        for method in method_order:
            if method not in results or method not in full_data:
                continue
            full_scores = np.array(full_data[method]["scores"])
            full_labels = np.array(full_data[method]["labels"])
            if full_labels.sum() == 0 or full_labels.sum() == len(full_labels):
                continue
            full_ap = average_precision_score(full_labels, full_scores)
            lr_ap = results[method]["ap"]
            delta = lr_ap - full_ap
            print(f"  {method:<30s}  {full_ap:>8.4f}  {lr_ap:>8.4f}  {delta:>+8.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "description": "Left-vs-right only evaluation (labels 2 and 3, excluding label 1)",
        "methods": results,
        "n_segments": len(eval_segments),
        "n_left": lr_counts[2],
        "n_right": lr_counts[3],
    }

    # Save pair-level scores
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(lbl) for lbl in labels_list],
        }
    # pyrefly: ignore [bad-typed-dict-key]
    output["pair_scores"] = pair_data

    out_path = hdd_dir / "left_right_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V-JEPA 2 Encoder-Sequence DTW Baseline on Honda HDD.

Evaluates V-JEPA 2 encoder patches spatially averaged per temporal position
and compared via DTW. Together with pooled cosine and residual DTW, this is
a controlled feature-by-comparator contrast, not a causal attribution.

Usage:
    python experiments/eval_hdd_encoder_seq.py --hdd-dir datasets/hdd
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from common import (
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
    ManeuverSegment,
    bootstrap_ap,
    build_temporal_masks,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_clip_vjepa2,
    load_gps,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from video_retrieval.fingerprints.dtw import (
    assignment_distance_batch,
    dtw_distance_batch,
    dtw_distance_batch_shuffled,
)


def extract_vjepa2_all_features(
    model: torch.nn.Module,
    processor: object,
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
) -> dict[int, dict]:
    """Extract V-JEPA 2 encoder sequence, mean embedding, and temporal residual.

    For each segment, computes:
    - encoder_seq: (T_PATCHES, D) spatially-averaged encoder output per timestep
    - mean_emb: L2-normalized mean-pooled encoder embedding (D,)
    - temporal_residual: (n_target, D) prediction residuals
    """
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="V-JEPA 2 features")):
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

            with torch.no_grad():
                # Encoder only
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]  # (T*S, D)
                mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

                # Reshape to (T, S, D), spatial average -> (T, D)
                enc_reshaped = encoder_tokens.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL, -1)
                encoder_seq = enc_reshaped.mean(dim=1)  # (32, 1024)

                # Predictor residuals
                pred_out = model(
                    **inputs,
                    context_mask=[context_mask],
                    target_mask=[target_mask],
                )
                predicted = pred_out.predictor_output.last_hidden_state[0]
                ground_truth = pred_out.predictor_output.target_hidden_state[0]
                predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                residual = (predicted - ground_truth).mean(dim=1)

            features[i] = {
                "encoder_seq": encoder_seq.cpu(),
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Encoder-Sequence DTW Baseline on HDD")
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-sec", type=float, default=3.0)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument(
        "--feature-cache",
        type=str,
        default="datasets/hdd/vjepa2_encoder_features.pt",
        help="Output cache used by the directed reranker.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("V-JEPA 2 ENCODER-SEQUENCE DTW BASELINE (HDD)")
    print("=" * 70)

    # Load data (same as eval_hdd_intersections.py)
    print("\nStep 1: Loading sessions and segments...")
    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="Loading sessions"):
        info = sessions[sid]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            sid,
            labels,
            gps_ts,
            gps_lats,
            gps_lngs,
            info["video_path"],
            info["video_start_unix"],
        )
        all_segments.extend(segs)

    clusters = cluster_intersections(all_segments)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    print(f"  {len(eval_segments)} segments in {len(mixed)} mixed clusters")

    # Resolve feature-cache path up front so we can reuse it across runs.
    feature_cache = Path(args.feature_cache)
    if not feature_cache.is_absolute():
        feature_cache = project_root / feature_cache

    if feature_cache.exists():
        # Reuse cached features (keyed by segment index) instead of re-extracting.
        # Extraction is deterministic for a fixed segment set, so this is exact,
        # and pair enumeration below already skips any index missing from features.
        print(f"\nStep 2: Loading cached V-JEPA 2 features from {feature_cache}")
        features = torch.load(feature_cache, weights_only=False)["features"]
        print(f"  Loaded {len(features)} cached features")
    else:
        print("\nStep 2: Loading V-JEPA 2...")
        # Resolve to local checkpoint if VTD_MODEL_DIR is set
        import os

        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_path = VJEPA2_MODEL_NAME
        model_dir = os.environ.get("VTD_MODEL_DIR")
        if model_dir:
            local = Path(model_dir) / VJEPA2_MODEL_NAME.split("/")[-1]
            if local.exists():
                vjepa2_path = str(local)

        model = AutoModel.from_pretrained(vjepa2_path, trust_remote_code=True).to(device).eval()
        processor = AutoVideoProcessor.from_pretrained(vjepa2_path, trust_remote_code=True)

        print("  Extracting features...")
        t0 = time.time()
        features = extract_vjepa2_all_features(
            model, processor, eval_segments, device, args.context_sec
        )
        print(f"  Extraction: {time.time() - t0:.1f}s")

        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features}, feature_cache)
        print(f"  Feature cache saved to {feature_cache}")

        del model, processor
        torch.cuda.empty_cache()

    # Enumerate pairs
    print("\nStep 3: Computing pairwise similarities...")
    pair_a = []
    pair_b = []
    pair_gt: list[int] = []
    pair_cluster_ids: list[int] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pair_a.append(indices[a])
                pair_b.append(indices[b])
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                pair_gt.append(gt)
                pair_cluster_ids.append(int(cid))  # numpy int64 -> python int for JSON

    labels = np.array(pair_gt)
    n_pairs = len(labels)
    print(f"  {n_pairs} pairs (pos={labels.sum()}, neg={n_pairs - labels.sum()})")

    # BoT cosine
    mean_a = torch.stack([features[i]["mean_emb"] for i in pair_a]).to(device)
    mean_b = torch.stack([features[i]["mean_emb"] for i in pair_b]).to(device)
    bot_scores = (mean_a * mean_b).sum(dim=1).cpu().numpy()
    del mean_a, mean_b

    # Encoder-sequence DTW (the new baseline)
    pair_indices = set(pair_a) | set(pair_b)
    enc_by_index = {
        index: features[index]["encoder_seq"].to(device) for index in pair_indices
    }
    enc_seqs_a = [enc_by_index[i] for i in pair_a]
    enc_seqs_b = [enc_by_index[i] for i in pair_b]
    enc_dists = dtw_distance_batch(enc_seqs_a, enc_seqs_b, normalize=True)
    enc_scores = torch.exp(-enc_dists).cpu().numpy()

    # Headline ordering control: the same encoder-sequence DTW after symmetrically
    # ablating temporal order, with stable endpoint IDs and ten paired draws.
    pair_ids = [
        (
            (eval_segments[a].session_id, eval_segments[a].start_frame),
            (eval_segments[b].session_id, eval_segments[b].start_frame),
        )
        for a, b in zip(pair_a, pair_b)
    ]
    enc_shuf_dists = dtw_distance_batch_shuffled(
        enc_seqs_a,
        enc_seqs_b,
        pair_ids=pair_ids,
        n_perms=10,
        base_seed=42,
        normalize=True,
    )
    enc_shuf_scores = torch.exp(-enc_shuf_dists).cpu().numpy()

    # Secondary rigid structural control. This removes monotonic ordering,
    # one-to-many warping, and endpoint anchoring jointly.
    enc_assignment_dists = assignment_distance_batch(
        enc_seqs_a, enc_seqs_b, normalize=True
    )
    enc_assignment_scores = torch.exp(-enc_assignment_dists).cpu().numpy()
    del enc_seqs_a, enc_seqs_b, enc_by_index
    torch.cuda.empty_cache()

    # Temporal residual DTW
    res_by_index = {
        index: features[index]["temporal_residual"].to(device) for index in pair_indices
    }
    res_seqs_a = [res_by_index[i] for i in pair_a]
    res_seqs_b = [res_by_index[i] for i in pair_b]
    res_dists = dtw_distance_batch(res_seqs_a, res_seqs_b, normalize=True)
    res_scores = torch.exp(-res_dists).cpu().numpy()
    res_shuf_dists = dtw_distance_batch_shuffled(
        res_seqs_a,
        res_seqs_b,
        pair_ids=pair_ids,
        n_perms=10,
        base_seed=42,
        normalize=True,
    )
    res_shuf_scores = torch.exp(-res_shuf_dists).cpu().numpy()
    res_assignment_dists = assignment_distance_batch(
        res_seqs_a, res_seqs_b, normalize=True
    )
    res_assignment_scores = torch.exp(-res_assignment_dists).cpu().numpy()

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, scores in [
        ("V-JEPA 2 BoT (cosine)", bot_scores),
        ("V-JEPA 2 Encoder-Seq DTW", enc_scores),
        ("V-JEPA 2 Encoder-Seq Shuffled DTW", enc_shuf_scores),
        ("V-JEPA 2 Encoder-Seq Assignment", enc_assignment_scores),
        ("V-JEPA 2 Temporal Residual DTW", res_scores),
        ("V-JEPA 2 Temporal Residual Shuffled DTW", res_shuf_scores),
        ("V-JEPA 2 Temporal Residual Assignment", res_assignment_scores),
    ]:
        ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
        auc = roc_auc_score(labels, scores)
        print(f"  {name:<35s}  AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  AUC={auc:.4f}")

    # Save
    output = {
        "bot_cosine": {"ap": float(average_precision_score(labels, bot_scores))},
        "encoder_seq_dtw": {"ap": float(average_precision_score(labels, enc_scores))},
        "encoder_seq_dtw_shuffled": {
            "ap": float(average_precision_score(labels, enc_shuf_scores))
        },
        "encoder_seq_assignment": {
            "ap": float(average_precision_score(labels, enc_assignment_scores))
        },
        "temporal_residual_dtw": {"ap": float(average_precision_score(labels, res_scores))},
        "temporal_residual_dtw_shuffled": {
            "ap": float(average_precision_score(labels, res_shuf_scores))
        },
        "temporal_residual_assignment": {
            "ap": float(average_precision_score(labels, res_assignment_scores))
        },
        "n_pairs": n_pairs,
        "n_segments": len(eval_segments),
    }
    out_path = hdd_dir / "encoder_seq_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    pair_output = {
        "bot_cosine": {
            "scores": bot_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "encoder_seq_dtw": {
            "scores": enc_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "encoder_seq_dtw_shuffled": {
            "scores": enc_shuf_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "encoder_seq_assignment": {
            "scores": enc_assignment_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "temporal_residual_dtw": {
            "scores": res_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "temporal_residual_dtw_shuffled": {
            "scores": res_shuf_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
        "temporal_residual_assignment": {
            "scores": res_assignment_scores.tolist(),
            "labels": labels.tolist(),
            "cluster_ids": pair_cluster_ids,
        },
    }
    pair_output_path = hdd_dir / "encoder_seq_pair_scores.json"
    with open(pair_output_path, "w") as f:
        json.dump(pair_output, f)
    print(f"Pair scores saved to {pair_output_path}")


if __name__ == "__main__":
    main()

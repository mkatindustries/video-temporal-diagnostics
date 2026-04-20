#!/usr/bin/env python3
"""PL-Stitch Evaluation on Honda HDD Maneuver Discrimination.

Evaluates PL-Stitch (ViT-Base with Plackett-Luce temporal ranking
pretraining) as a per-frame encoder on the HDD left-turn vs right-turn
task.

PL-Stitch produces per-frame 768-d CLS-token embeddings. We evaluate
four similarity methods to enable the feature-vs-comparator decomposition:

1. Bag-of-frames (BoF): cosine on mean-pooled embedding — tests encoder alone
2. Chamfer: bidirectional max frame similarity
3. Temporal derivative DTW: DTW on d(emb)/d(frame) — sequence-aware
4. DTW on raw embeddings: direct sequence comparison

If BoF fails but DTW succeeds, the 89% comparator finding extends to
temporally-trained encoders. Also computes s_rev for each method.

Usage:
    python experiments/eval_hdd_pl_stitch.py \\
        --hdd-dir datasets/hdd \\
        --weights ~/src/PL-Stitch/pl_lemon.pth

    python experiments/eval_hdd_pl_stitch.py \\
        --weights /checkpoint/dream/arjangt/video_retrieval/PL-Stitch/pl_lemon.pth
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# PL-Stitch config
PL_STITCH_EMBED_DIM = 768
PL_STITCH_IMG_SIZE = 224
PL_STITCH_TARGET_FPS = 3.0


# ---------------------------------------------------------------------------
# PL-Stitch model loading
# ---------------------------------------------------------------------------


def load_pl_stitch(weights_path: str, device: torch.device):
    """Load PL-Stitch ViT-Base model.

    Args:
        weights_path: Path to pl_lemon.pth checkpoint.
        device: Torch device.

    Returns:
        PL-Stitch model in eval mode.
    """
    pl_stitch_dir = os.path.expanduser("~/src/PL-Stitch")
    pl_stitch_pkg = os.path.join(pl_stitch_dir, "pl_stitch")
    sys.path.insert(0, pl_stitch_pkg)

    from build_model import vit  # pyrefly: ignore

    model = vit(
        model_size="vit_base",
        freeze_transformer=True,
        pretrained_weights=weights_path,
    )
    model = model.to(device).eval()

    sys.path.pop(0)
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def get_transform():
    """ImageNet-normalized transform for PL-Stitch (224x224)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (PL_STITCH_IMG_SIZE, PL_STITCH_IMG_SIZE), antialias=True
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_pl_stitch_features(
    model,
    video_path: str,
    start_sec: float,
    end_sec: float,
    device: torch.device,
    transform,
    target_fps: float = PL_STITCH_TARGET_FPS,
    max_resolution: int = 518,
) -> dict:
    """Extract per-frame PL-Stitch CLS-token embeddings for a video clip.

    Args:
        model: PL-Stitch model.
        video_path: Path to video file.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        device: Torch device.
        transform: Torchvision transform for preprocessing.
        target_fps: Target frame rate.
        max_resolution: Max height for raw frame extraction.

    Returns:
        Dict with:
            'embeddings': (T, 768) per-frame CLS tokens (L2-normalized)
            'mean_emb': (768,) L2-normalized mean-pooled embedding
    """
    # Use the shared load_clip function from eval_hdd_intersections
    from eval_hdd_intersections import load_clip  # pyrefly: ignore

    frames = load_clip(
        video_path,
        start_sec,
        end_sec,
        target_fps=target_fps,
        max_resolution=max_resolution,
    )

    if len(frames) < 3:
        raise ValueError(f"Too few frames ({len(frames)}) from {video_path}")

    # Preprocess and batch frames
    tensors = [transform(f) for f in frames]
    batch = torch.stack(tensors).to(device)  # (T, 3, 224, 224)

    # Forward pass — VisionTransformer returns CLS token: (T, 768)
    embeddings = model(batch)  # (T, 768)
    embeddings = F.normalize(embeddings, dim=-1)

    mean_emb = F.normalize(embeddings.mean(dim=0), dim=0)

    return {
        "embeddings": embeddings.cpu(),
        "mean_emb": mean_emb.cpu(),
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ap(scores, labels, n_resamples=1000, seed=42):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    ap = average_precision_score(labels, scores)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot.append(ap)
        else:
            boot.append(average_precision_score(l, s))
    boot = np.array(boot)
    return float(ap), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_similarities(
    segments,
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    device: torch.device,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each cluster using 4 methods.

    Methods:
        1. bag_of_frames: cosine on mean-pooled embedding
        2. chamfer: bidirectional max frame similarity
        3. temporal_derivative: DTW on d(emb)/d(frame)
        4. dtw_raw: DTW on raw per-frame embeddings

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    from video_retrieval.fingerprints import TemporalDerivativeFingerprint  # pyrefly: ignore
    from video_retrieval.fingerprints.dtw import dtw_distance_batch  # pyrefly: ignore

    deriv_fp = TemporalDerivativeFingerprint()

    # Pre-compute temporal derivative fingerprints
    print("  Pre-computing temporal derivative fingerprints...")
    deriv_fps = {}
    for idx in features:
        deriv_fps[idx] = deriv_fp.compute_fingerprint(features[idx]["embeddings"])

    # Enumerate all pairs
    print("  Enumerating pairs...")
    pair_a_indices = []
    pair_b_indices = []
    pair_gts = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                pair_a_indices.append(indices[a_pos])
                pair_b_indices.append(indices[b_pos])
                gt = (
                    1
                    if segments[indices[a_pos]].label == segments[indices[b_pos]].label
                    else 0
                )
                pair_gts.append(gt)

    total_pairs = len(pair_gts)
    print(f"  Total pairs: {total_pairs}")

    # --- Bag-of-frames (vectorized dot product) ---
    print("  Computing bag-of-frames similarities...")
    mean_embs_a = torch.stack([features[i]["mean_emb"] for i in pair_a_indices]).to(
        device
    )
    mean_embs_b = torch.stack([features[i]["mean_emb"] for i in pair_b_indices]).to(
        device
    )
    bof_sims = (mean_embs_a * mean_embs_b).sum(dim=1).cpu().tolist()

    # --- Chamfer (bidirectional max similarity) ---
    print("  Computing Chamfer similarities...")
    chamfer_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        ea = features[a_idx]["embeddings"].to(device)
        eb = features[b_idx]["embeddings"].to(device)
        sim_matrix = torch.mm(ea, eb.t())
        max_ab = sim_matrix.max(dim=1).values.mean().item()
        max_ba = sim_matrix.max(dim=0).values.mean().item()
        chamfer_sims.append((max_ab + max_ba) / 2)

    # --- Temporal derivative DTW (batched GPU) ---
    print("  Computing temporal derivative DTW (batched GPU)...")
    deriv_seqs_a = [deriv_fps[i].to(device) for i in pair_a_indices]
    deriv_seqs_b = [deriv_fps[i].to(device) for i in pair_b_indices]
    deriv_dists = dtw_distance_batch(deriv_seqs_a, deriv_seqs_b, normalize=False)
    deriv_sims = torch.exp(-deriv_dists).cpu().tolist()

    # --- DTW on raw embeddings (batched GPU) ---
    print("  Computing raw embedding DTW (batched GPU)...")
    raw_seqs_a = [features[i]["embeddings"].to(device) for i in pair_a_indices]
    raw_seqs_b = [features[i]["embeddings"].to(device) for i in pair_b_indices]
    raw_dists = dtw_distance_batch(raw_seqs_a, raw_seqs_b, normalize=True)
    raw_sims = torch.exp(-raw_dists).cpu().tolist()

    return {
        "bag_of_frames": (bof_sims, list(pair_gts)),
        "chamfer": (chamfer_sims, list(pair_gts)),
        "temporal_derivative": (deriv_sims, list(pair_gts)),
        "dtw_raw": (raw_sims, list(pair_gts)),
    }


# ---------------------------------------------------------------------------
# s_rev computation
# ---------------------------------------------------------------------------


def compute_s_rev(
    model,
    segments,
    features: dict[int, dict],
    device: torch.device,
    transform,
    context_sec: float,
    n_samples: int = 100,
) -> dict[str, dict]:
    """Compute s_rev (forward vs reversed) for each similarity method.

    Returns:
        Dict mapping method_name -> {'mean': float, 'std': float, 'n': int}.
    """
    from eval_hdd_intersections import load_clip  # pyrefly: ignore
    from video_retrieval.fingerprints import TemporalDerivativeFingerprint  # pyrefly: ignore
    from video_retrieval.fingerprints.dtw import dtw_distance  # pyrefly: ignore

    deriv_fp = TemporalDerivativeFingerprint()

    indices = list(features.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    sample_indices = indices[:n_samples]

    s_rev_bof = []
    s_rev_chamfer = []
    s_rev_deriv = []
    s_rev_dtw = []

    for i in tqdm(sample_indices, desc="PL-Stitch s_rev"):
        seg = segments[i]
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        try:
            frames = load_clip(seg.video_path, start_sec, end_sec)
            if len(frames) < 3:
                continue

            # Reversed frames
            rev_frames = frames[::-1]
            rev_tensors = [transform(f) for f in rev_frames]
            rev_batch = torch.stack(rev_tensors).to(device)

            with torch.no_grad():
                rev_emb_raw = model(rev_batch)
                rev_emb_raw = F.normalize(rev_emb_raw, dim=-1)

            rev_mean = F.normalize(rev_emb_raw.mean(dim=0), dim=0).cpu()
            rev_emb = rev_emb_raw.cpu()

            fwd_mean = features[i]["mean_emb"]
            fwd_emb = features[i]["embeddings"]

            # BoF
            s_rev_bof.append(float(torch.dot(fwd_mean, rev_mean).item()))

            # Chamfer
            sim_matrix = torch.mm(fwd_emb, rev_emb.t())
            max_ab = sim_matrix.max(dim=1).values.mean().item()
            max_ba = sim_matrix.max(dim=0).values.mean().item()
            s_rev_chamfer.append((max_ab + max_ba) / 2)

            # Temporal derivative DTW
            fwd_fp = deriv_fp.compute_fingerprint(fwd_emb)
            rev_fp = deriv_fp.compute_fingerprint(rev_emb)
            d = dtw_distance(fwd_fp, rev_fp, normalize=False)
            s_rev_deriv.append(math.exp(-d))

            # DTW raw
            d_raw = dtw_distance(fwd_emb, rev_emb, normalize=True)
            s_rev_dtw.append(math.exp(-d_raw))

        except Exception:
            continue

    def summarize(vals):
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n": len(vals),
        }

    return {
        "bag_of_frames": summarize(s_rev_bof),
        "chamfer": summarize(s_rev_chamfer),
        "temporal_derivative": summarize(s_rev_deriv),
        "dtw_raw": summarize(s_rev_dtw),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PL-Stitch evaluation on Honda HDD"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.expanduser("~/src/PL-Stitch/pl_lemon.pth"),
        help="Path to PL-Stitch checkpoint (pl_lemon.pth)",
    )
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Seconds of context before/after maneuver",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=PL_STITCH_TARGET_FPS,
        help="Target frame extraction rate",
    )
    parser.add_argument(
        "--n-rev-samples",
        type=int,
        default=100,
        help="Number of segments to sample for s_rev computation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = (
        Path(args.hdd_dir)
        if os.path.isabs(args.hdd_dir)
        else project_root / args.hdd_dir
    )
    device = torch.device(args.device)

    # Import HDD pipeline functions
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_hdd_intersections import (  # pyrefly: ignore
        ManeuverSegment,
        cluster_intersections,
        discover_sessions,
        extract_maneuver_segments,
        filter_mixed_clusters,
        load_gps,
    )

    print("=" * 70)
    print("PL-STITCH EVALUATION ON HONDA HDD")
    print("=" * 70)
    print(f"  Weights: {args.weights}")
    print(f"  HDD dir: {hdd_dir}")
    print(f"  Embedding: {PL_STITCH_EMBED_DIM}-d per frame, 4 methods")
    print(f"  FPS: {args.target_fps}, context: {args.context_sec}s")

    # ------------------------------------------------------------------
    # Step 1: Discover sessions and extract segments
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    t0 = time.time()
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} valid sessions")

    print("\nStep 2: Extracting maneuver segments...")
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="HDD sessions"):
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

    print(f"  Total segments: {len(all_segments)}")

    # ------------------------------------------------------------------
    # Step 3: Cluster and filter
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections...")
    clusters = cluster_intersections(all_segments)
    mixed = filter_mixed_clusters(clusters, max_clusters=50)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    print(f"  Eval segments: {len(eval_segments)} in {len(mixed)} mixed clusters")

    # ------------------------------------------------------------------
    # Step 4: Load PL-Stitch model
    # ------------------------------------------------------------------
    print("\nStep 4: Loading PL-Stitch model...")
    t0 = time.time()
    model = load_pl_stitch(args.weights, device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    transform = get_transform()

    # ------------------------------------------------------------------
    # Step 5: Extract per-frame features
    # ------------------------------------------------------------------
    print("\nStep 5: Extracting PL-Stitch features...")
    features: dict[int, dict] = {}
    failed = 0
    for i, seg in enumerate(tqdm(eval_segments, desc="PL-Stitch HDD")):
        start_sec = seg.start_frame / 3.0 - args.context_sec
        end_sec = seg.end_frame / 3.0 + args.context_sec
        start_sec = max(0.0, start_sec)
        try:
            feat = extract_pl_stitch_features(
                model,
                seg.video_path,
                start_sec,
                end_sec,
                device,
                transform,
                target_fps=args.target_fps,
            )
            features[i] = feat
        except Exception:
            failed += 1

    print(f"  Extracted: {len(features)}/{len(eval_segments)} ({failed} failed)")

    # ------------------------------------------------------------------
    # Step 6: Compute all similarities (4 methods)
    # ------------------------------------------------------------------
    print("\nStep 6: Computing pairwise similarities...")
    all_scores = compute_similarities(
        eval_segments, features, cluster_to_indices, device
    )

    # Compute AP/AUC for each method
    results = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
        auc = roc_auc_score(labels, scores)
        results[method_name] = {
            "ap": ap,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "auc": float(auc),
            "n_pairs": len(labels),
            "n_pos": int(labels.sum()),
        }
        print(
            f"  {method_name:25s}  AP={ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  AUC={auc:.4f}"
        )

    # ------------------------------------------------------------------
    # Step 7: Compute s_rev for each method
    # ------------------------------------------------------------------
    print("\nStep 7: Computing s_rev...")
    s_rev_results = compute_s_rev(
        model,
        eval_segments,
        features,
        device,
        transform,
        context_sec=args.context_sec,
        n_samples=args.n_rev_samples,
    )
    for method_name, stats in s_rev_results.items():
        print(
            f"  s_rev {method_name:25s}  {stats['mean']:.4f} +/- {stats['std']:.4f}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "model": "PL-Stitch",
        "architecture": "ViT-Base",
        "embedding_dim": PL_STITCH_EMBED_DIM,
        "target_fps": args.target_fps,
        "context_sec": args.context_sec,
        "n_segments": len(eval_segments),
        "n_clusters": len(mixed),
        "hdd": results,
        "s_rev": s_rev_results,
    }

    out_path = project_root / "datasets" / "pl_stitch_hdd_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — PL-Stitch on Honda HDD")
    print("=" * 70)
    for method_name in ["bag_of_frames", "chamfer", "temporal_derivative", "dtw_raw"]:
        r = results[method_name]
        s = s_rev_results[method_name]
        print(
            f"  {method_name:25s}  AP={r['ap']:.4f}  s_rev={s['mean']:.4f}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TARA Evaluation on Honda HDD Maneuver Discrimination.

Evaluates TARA (Tarsier-7B MLLM trained with chiral negatives) as a
single-vector video encoder on the HDD left-turn vs right-turn task.

TARA produces a single 4096-d embedding per video clip via an EOL prompt
strategy. If TARA fails on HDD despite being trained with reversed-video
negatives, this is strong evidence that the problem is in the comparator
(cosine on single vectors) rather than the encoder.

Also computes s_rev: forward vs reversed frame order similarity.

Usage:
    python experiments/eval_hdd_tara.py \\
        --hdd-dir datasets/hdd \\
        --model-path ~/src/TARA

    # With explicit checkpoint directory
    python experiments/eval_hdd_tara.py \\
        --model-path /checkpoint/dream/arjangt/video_retrieval/TARA
"""

import argparse
import json
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
from tqdm import tqdm

# TARA config
TARA_NUM_FRAMES = 16
TARA_EMBED_DIM = 4096


# ---------------------------------------------------------------------------
# TARA model loading
# ---------------------------------------------------------------------------


def load_tara(model_path: str):
    """Load TARA model from local directory.

    Args:
        model_path: Path to TARA model directory (with modeling_tara.py
            and model weights).

    Returns:
        TARA model in eval mode.
    """
    sys.path.insert(0, model_path)

    from modeling_tara import TARA  # pyrefly: ignore

    model = TARA.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    sys.path.pop(0)
    return model


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames_timed(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = TARA_NUM_FRAMES,
) -> torch.Tensor:
    """Extract n_frames uniformly from a time window, returning uint8 tensor.

    Uses PyAV for time-based seeking (efficient for long dashcam videos),
    then returns (T, C, H, W) uint8 tensor as expected by TARA.
    """
    import av  # pyrefly: ignore

    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base or 1 / 30)

    duration = end_sec - start_sec
    if duration <= 0:
        duration = 1.0
    target_times = [
        start_sec + i * duration / max(n_frames - 1, 1) for i in range(n_frames)
    ]

    seek_sec = max(0.0, start_sec - 1.0)
    container.seek(int(seek_sec / time_base), stream=stream)

    all_frames = []
    all_times = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        t = float(frame.pts) * time_base
        if t < start_sec - 0.1:
            continue
        if t > end_sec + 0.1:
            break
        img = frame.to_ndarray(format="rgb24")
        all_frames.append(img)
        all_times.append(t)

    container.close()

    if not all_frames:
        raise ValueError(f"No frames extracted from {video_path} [{start_sec:.1f}-{end_sec:.1f}]")

    # Pick closest to target times
    all_times_arr = np.array(all_times)
    selected = []
    for t in target_times:
        idx = int(np.argmin(np.abs(all_times_arr - t)))
        selected.append(all_frames[idx])

    # Convert to (T, C, H, W) uint8 tensor
    tensors = [torch.from_numpy(f).permute(2, 0, 1) for f in selected]
    return torch.stack(tensors)  # (T, C, H, W), uint8


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_video(model, frames: torch.Tensor) -> torch.Tensor:
    """Encode (T, C, H, W) uint8 tensor into a single L2-normalized 4096-d vector.

    Args:
        model: TARA model.
        frames: (T, C, H, W) uint8 tensor.

    Returns:
        (4096,) L2-normalized float32 embedding on CPU.
    """
    video = frames.unsqueeze(0)  # (1, T, C, H, W)
    video = video.to(model.model.device)
    emb = model.encode_vision(video)  # (1, 4096)
    emb = emb.squeeze(0).float()
    emb = F.normalize(emb, dim=-1)
    return emb.cpu()


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
# Main evaluation
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TARA evaluation on Honda HDD")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.expanduser("~/src/TARA"),
        help="Path to TARA model directory",
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
        "--n-frames",
        type=int,
        default=TARA_NUM_FRAMES,
        help="Number of frames to sample per clip",
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
    print("TARA EVALUATION ON HONDA HDD")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  HDD dir: {hdd_dir}")
    print(f"  Frames: {args.n_frames}")
    print(f"  Embedding: {TARA_EMBED_DIM}-d, cosine similarity")
    print(f"  Context: {args.context_sec}s before/after maneuver")

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
    # Step 3: Cluster intersections and filter mixed clusters
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
    # Step 4: Load TARA model
    # ------------------------------------------------------------------
    print("\nStep 4: Loading TARA model...")
    t0 = time.time()
    model = load_tara(args.model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Extract embeddings
    # ------------------------------------------------------------------
    # Checkpoint path for resuming
    ckpt_path = project_root / "datasets" / "tara_hdd_features.pt"

    print("\nStep 5: Extracting TARA embeddings...")
    features: dict[int, torch.Tensor] = {}
    failed = 0
    first_error_shown = False

    # Resume from checkpoint if available
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        features = ckpt["features"]
        print(f"  Resumed from checkpoint: {len(features)} features loaded")

    for i, seg in enumerate(tqdm(eval_segments, desc="TARA HDD")):
        if i in features:
            continue
        start_sec = seg.start_frame / 3.0 - args.context_sec
        end_sec = seg.end_frame / 3.0 + args.context_sec
        start_sec = max(0.0, start_sec)
        try:
            frames = extract_frames_timed(
                seg.video_path, start_sec, end_sec, n_frames=args.n_frames
            )
            features[i] = encode_video(model, frames)
        except Exception as e:
            failed += 1
            if not first_error_shown:
                import traceback
                print(f"\n  First failure (segment {i}): {e}")
                traceback.print_exc()
                first_error_shown = True

        # Checkpoint every 100 segments
        if (i + 1) % 100 == 0 and len(features) > 0:
            torch.save({"features": features}, ckpt_path)

    # Final checkpoint
    if len(features) > 0:
        torch.save({"features": features}, ckpt_path)

    print(f"  Extracted: {len(features)}/{len(eval_segments)} ({failed} failed)")

    if len(features) == 0:
        print("\n  ERROR: No features extracted. Cannot compute results.")
        del model
        torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------------
    # Step 6: Pairwise evaluation
    # ------------------------------------------------------------------
    print("\nStep 6: Computing pairwise similarities...")
    pair_scores = []
    pair_labels = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                sim = float(
                    torch.dot(features[indices[a]], features[indices[b]]).item()
                )
                gt = (
                    1
                    if eval_segments[indices[a]].label
                    == eval_segments[indices[b]].label
                    else 0
                )
                pair_scores.append(sim)
                pair_labels.append(gt)

    scores = np.array(pair_scores)
    labels = np.array(pair_labels)
    ap, ci_lo, ci_hi = bootstrap_ap(scores, labels)
    auc = roc_auc_score(labels, scores)

    print(f"\n  TARA HDD Maneuver Discrimination:")
    print(f"    AP  = {ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    AUC = {auc:.4f}")
    print(f"    Pairs: {len(labels)} ({int(labels.sum())} positive)")

    # ------------------------------------------------------------------
    # Step 7: Compute s_rev (forward vs reversed frame order)
    # ------------------------------------------------------------------
    print("\nStep 7: Computing s_rev...")
    rev_sims = []
    rev_indices = list(features.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(rev_indices)
    rev_sample = rev_indices[: args.n_rev_samples]

    for i in tqdm(rev_sample, desc="TARA s_rev"):
        seg = eval_segments[i]
        start_sec = seg.start_frame / 3.0 - args.context_sec
        end_sec = seg.end_frame / 3.0 + args.context_sec
        start_sec = max(0.0, start_sec)
        try:
            frames = extract_frames_timed(
                seg.video_path, start_sec, end_sec, n_frames=args.n_frames
            )
            fwd_emb = features[i]
            rev_frames = frames.flip(0)  # Reverse temporal order
            rev_emb = encode_video(model, rev_frames)
            s_rev = float(torch.dot(fwd_emb, rev_emb).item())
            rev_sims.append(s_rev)
        except Exception:
            continue

    mean_s_rev = float(np.mean(rev_sims)) if rev_sims else float("nan")
    std_s_rev = float(np.std(rev_sims)) if rev_sims else float("nan")
    print(f"  s_rev = {mean_s_rev:.4f} +/- {std_s_rev:.4f} ({len(rev_sims)} samples)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "model": "TARA",
        "embedding_dim": TARA_EMBED_DIM,
        "n_frames": args.n_frames,
        "context_sec": args.context_sec,
        "hdd": {
            "ap": ap,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "auc": float(auc),
            "n_pairs": len(labels),
            "n_pos": int(labels.sum()),
            "n_segments": len(eval_segments),
            "n_clusters": len(mixed),
        },
        "s_rev": {
            "mean": mean_s_rev,
            "std": std_s_rev,
            "n_samples": len(rev_sims),
        },
    }

    out_path = project_root / "datasets" / "tara_hdd_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  HDD Maneuver AP = {ap:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  HDD Maneuver AUC = {auc:.4f}")
    print(f"  s_rev = {mean_s_rev:.4f} +/- {std_s_rev:.4f}")


if __name__ == "__main__":
    main()

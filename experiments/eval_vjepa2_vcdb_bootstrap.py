#!/usr/bin/env python3
"""Bootstrap confidence intervals for V-JEPA 2 methods on VCDB.

Computes 95% bootstrap CIs for V-JEPA 2 Bag-of-Tokens and Temporal Residual
methods on the VCDB benchmark. These CIs are required for the paper's bootstrap
CI appendix table (reviewer R2 flagged them as missing).

Approach:
1. Load VCDB videos (528 videos)
2. Extract V-JEPA 2 features (BoT mean embeddings + temporal residuals)
3. Compute pairwise similarities for all positive pairs + 1:1 negative pairs
4. Run GPU-accelerated bootstrap (2,000 resamples, percentile method, seed=42)
5. Output formatted LaTeX table rows for the bootstrap CI table

Usage:
    python experiments/eval_vjepa2_vcdb_bootstrap.py [--device cuda] \
        [--vcdb-dir datasets/vcdb] [--n-resamples 2000]
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from video_retrieval.fingerprints.dtw import dtw_distance_batch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32
VJEPA2_SPATIAL = 256


# ---------------------------------------------------------------------------
# VCDB loading (same as eval_vcdb.py / eval_vcdb_scramble.py)
# ---------------------------------------------------------------------------


def load_vcdb_annotations(ann_dir: str, vid_base_dir: str) -> set[tuple[str, str]]:
    """Load all VCDB annotations as global (videoA_path, videoB_path) pairs."""
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
    # pyrefly: ignore [bad-return]
    return copy_pairs


def discover_videos(vid_base_dir: str) -> list[str]:
    """Discover all video files under the VCDB core_dataset directory."""
    videos = []
    for cat in sorted(os.listdir(vid_base_dir)):
        cat_path = os.path.join(vid_base_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for vf in sorted(os.listdir(cat_path)):
            if vf.endswith((".mp4", ".flv", ".webm", ".avi", ".mkv")):
                videos.append(os.path.join(cat, vf))
    return videos


# ---------------------------------------------------------------------------
# Feature caching (same pattern as eval_vcdb_scramble.py)
# ---------------------------------------------------------------------------


def save_feature_cache(features: dict, cache_path: Path) -> None:
    """Save extracted features to disk as a .pt file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_features = {}
    for k, v in features.items():
        cpu_features[k] = {
            fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv for fk, fv in v.items()
        }
    torch.save(cpu_features, cache_path)
    print(f"  Cache saved to {cache_path}")


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load cached features from disk, or return None if not found."""
    if not cache_path.exists():
        return None
    print(f"  Loading cache from {cache_path}")
    return torch.load(cache_path, weights_only=False)


# ---------------------------------------------------------------------------
# V-JEPA 2 feature extraction (same as eval_vcdb_scramble.py)
# ---------------------------------------------------------------------------


def build_temporal_masks(
    n_context_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context/target masks for V-JEPA 2 temporal prediction."""
    all_indices = torch.arange(VJEPA2_T_PATCHES * VJEPA2_SPATIAL, device=device)
    grid = all_indices.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL)
    context_indices = grid[:n_context_steps].reshape(-1)
    target_indices = grid[n_context_steps:].reshape(-1)
    return context_indices.unsqueeze(0), target_indices.unsqueeze(0)


def load_frames_for_vjepa2(
    video_path: str,
    max_resolution: int = 256,
) -> list[np.ndarray]:
    """Extract exactly VJEPA2_NUM_FRAMES frames for V-JEPA 2."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or 30)
    # pyrefly: ignore [unsupported-operation]
    duration = float(stream.duration * stream.time_base) if stream.duration else 60.0

    target_fps = VJEPA2_NUM_FRAMES / max(duration, 1.0)
    container.close()

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps_actual = float(stream.average_rate or 30)
    sample_interval = video_fps_actual / target_fps

    frames = []
    frame_count = 0
    next_sample = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None:
            frame_count += 1
            continue

        if frame_count >= next_sample:
            img = frame.to_ndarray(format="rgb24")

            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)
            next_sample += sample_interval

            if len(frames) >= VJEPA2_NUM_FRAMES + 10:
                break

        frame_count += 1

    container.close()

    if len(frames) == 0:
        raise ValueError("No frames extracted")

    while len(frames) < VJEPA2_NUM_FRAMES:
        frames.append(frames[-1])

    return frames[:VJEPA2_NUM_FRAMES]


def extract_vjepa2_features(
    model: torch.nn.Module,
    processor: object,
    vid_base_dir: str,
    video_relpaths: list[str],
    device: torch.device,
) -> dict[str, dict]:
    """Extract V-JEPA 2 features for all videos."""
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features = {}
    failed = 0

    for vp in tqdm(video_relpaths, desc="V-JEPA 2 features"):
        path = os.path.join(vid_base_dir, vp)
        try:
            frames = load_frames_for_vjepa2(path)

            # pyrefly: ignore [not-callable]
            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]
                mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

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

            features[vp] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1

    print(f"  V-JEPA 2: {len(features)}/{len(video_relpaths)} " f"({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# GPU-accelerated bootstrap (from bootstrap_cis.py)
# ---------------------------------------------------------------------------


def _ap_gpu_batch(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute average precision for a batch of bootstrap samples on GPU.

    Args:
        scores: (B, N) tensor of similarity scores.
        labels: (B, N) tensor of binary labels.

    Returns:
        (B,) tensor of AP values.
    """
    B, N = scores.shape

    # Sort each row by descending score
    sorted_indices = scores.argsort(dim=1, descending=True)
    sorted_labels = labels.gather(1, sorted_indices)  # (B, N)

    # Cumulative TP and total predictions
    tp_cumsum = sorted_labels.cumsum(dim=1)  # (B, N)
    positions = torch.arange(
        1, N + 1, device=scores.device, dtype=scores.dtype
    ).unsqueeze(
        0
    )  # (1, N)

    precision = tp_cumsum / positions  # (B, N)
    n_pos = labels.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
    recall_change = sorted_labels / n_pos  # (B, N)

    # AP = sum(precision * recall_change) per row
    ap = (precision * recall_change).sum(dim=1)  # (B,)
    return ap


def bootstrap_ap(scores, labels, n_resamples=2000, ci=0.95, seed=42, device=None):
    """Bootstrap 95% CI for average precision, GPU-accelerated when available.

    Uses percentile method with n_resamples resamples.
    """
    scores_np = np.asarray(scores)
    labels_np = np.asarray(labels)
    n = len(scores_np)

    # Point estimate (sklearn for reference accuracy)
    point_ap = average_precision_score(labels_np, scores_np)

    if device is not None and device.type == "cuda":
        # GPU path: generate all bootstrap indices and compute AP in batch
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        scores_t = torch.tensor(scores_np, device=device, dtype=torch.float32)
        labels_t = torch.tensor(labels_np, device=device, dtype=torch.float32)

        # Generate all bootstrap indices at once: (B, N)
        boot_indices = torch.randint(
            0, n, (n_resamples, n), device=device, generator=gen
        )

        boot_scores = scores_t[boot_indices]  # (B, N)
        boot_labels = labels_t[boot_indices]  # (B, N)

        # Check for degenerate samples
        pos_counts = boot_labels.sum(dim=1)
        degenerate = (pos_counts == 0) | (pos_counts == n)

        boot_aps = _ap_gpu_batch(boot_scores, boot_labels)
        boot_aps[degenerate] = point_ap

        boot_aps_np = boot_aps.cpu().numpy()
    else:
        # CPU path
        rng = np.random.RandomState(seed)
        boot_aps_list = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n, size=n)
            s_boot = scores_np[idx]
            l_boot = labels_np[idx]
            if l_boot.sum() == 0 or l_boot.sum() == n:
                boot_aps_list.append(point_ap)
                continue
            boot_aps_list.append(average_precision_score(l_boot, s_boot))
        boot_aps_np = np.array(boot_aps_list)

    alpha = 1 - ci
    ci_low = np.percentile(boot_aps_np, 100 * alpha / 2)
    ci_high = np.percentile(boot_aps_np, 100 * (1 - alpha / 2))
    return point_ap, ci_low, ci_high


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_vjepa2_similarities(
    features: dict[str, dict],
    pairs_to_compute: set[tuple[str, str]],
) -> dict[str, dict[tuple[str, str], float]]:
    """Compute V-JEPA 2 BoT and Temporal Residual similarities.

    Args:
        features: V-JEPA 2 features {relpath: {mean_emb, temporal_residual}}.
        pairs_to_compute: Set of (a, b) pairs.

    Returns:
        Dict mapping method name -> {(vidA, vidB): similarity}.
    """
    bot_sims = {}
    tr_sims = {}

    # Filter to valid pairs
    valid_pairs = [
        (a, b) for a, b in pairs_to_compute if a in features and b in features
    ]

    # --- BoT: cosine similarity of mean embeddings ---
    for a, b in tqdm(valid_pairs, desc="  BoT cosine"):
        sim = float(torch.dot(features[a]["mean_emb"], features[b]["mean_emb"]).item())
        bot_sims[(a, b)] = sim

    # --- Temporal Residual: DTW with exp(-d_DTW) ---
    dtw_pairs = []
    seqs_a = []
    seqs_b = []
    for a, b in valid_pairs:
        ra = features[a]["temporal_residual"]
        rb = features[b]["temporal_residual"]
        if ra.shape[0] > 0 and rb.shape[0] > 0:
            dtw_pairs.append((a, b))
            seqs_a.append(ra)
            seqs_b.append(rb)

    if dtw_pairs:
        print(f"  Computing batched DTW for {len(dtw_pairs)} pairs...")
        dists = dtw_distance_batch(seqs_a, seqs_b, normalize=True)
        sims = torch.exp(-dists)
        for idx, (a, b) in enumerate(dtw_pairs):
            tr_sims[(a, b)] = float(sims[idx].item())

    return {
        "vjepa2_bag_of_tokens": bot_sims,
        "vjepa2_temporal_residual": tr_sims,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for V-JEPA 2 methods on VCDB"
    )
    parser.add_argument(
        "--vcdb-dir",
        type=str,
        default="datasets/vcdb",
        help="Path to VCDB directory (contains annotation/ and core_dataset/)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction even if cached features exist",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vcdb_dir = (
        Path(args.vcdb_dir)
        if os.path.isabs(args.vcdb_dir)
        else project_root / args.vcdb_dir
    )
    ann_dir = vcdb_dir / "annotation"
    vid_dir = vcdb_dir / "core_dataset"

    # Validate paths
    if not ann_dir.exists():
        print(f"ERROR: Annotation dir not found: {ann_dir}")
        return
    if not vid_dir.exists():
        print(f"ERROR: Video dir not found: {vid_dir}")
        return

    # Auto-detect GPU for bootstrap
    if args.device == "cuda" and torch.cuda.is_available():
        bootstrap_device = torch.device("cuda")
    else:
        bootstrap_device = torch.device("cpu")

    print("=" * 70)
    print("V-JEPA 2 VCDB BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"  VCDB dir: {vcdb_dir}")
    print(f"  Device: {args.device}")
    print(f"  Bootstrap device: {bootstrap_device}")
    print(f"  Resamples: {args.n_resamples}")
    print(f"  Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Step 1: Discover videos + load annotations
    # ------------------------------------------------------------------
    print("\nStep 1: Loading dataset...")
    videos = discover_videos(str(vid_dir))
    copy_pairs = load_vcdb_annotations(str(ann_dir), str(vid_dir))
    print(f"  Videos: {len(videos)}")
    print(f"  Annotated copy pairs: {len(copy_pairs)}")

    # ------------------------------------------------------------------
    # Step 2: Extract V-JEPA 2 features (with caching)
    # ------------------------------------------------------------------
    print("\nStep 2: Loading V-JEPA 2 features...")
    cache_dir = vcdb_dir / ".cache"
    vjepa2_cache = cache_dir / "vjepa2_features.pt"

    vjepa2_features = load_feature_cache(vjepa2_cache) if not args.no_cache else None

    if vjepa2_features is None:
        print("  Loading V-JEPA 2 model...")
        from transformers import AutoModel, AutoVideoProcessor

        vjepa2_model = AutoModel.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )
        vjepa2_model = vjepa2_model.to(args.device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            VJEPA2_MODEL_NAME, trust_remote_code=True
        )

        print("  Extracting V-JEPA 2 features...")
        t0 = time.time()
        vjepa2_features = extract_vjepa2_features(
            vjepa2_model,
            vjepa2_processor,
            str(vid_dir),
            videos,
            torch.device(args.device),
        )
        print(f"  V-JEPA 2 extraction: {time.time() - t0:.1f}s")

        del vjepa2_model, vjepa2_processor
        torch.cuda.empty_cache()

        save_feature_cache(vjepa2_features, vjepa2_cache)
    else:
        print(f"  Loaded {len(vjepa2_features)} cached V-JEPA 2 features")

    keys = sorted(vjepa2_features.keys())
    print(f"  Videos with features: {len(keys)}")

    # ------------------------------------------------------------------
    # Step 3: Build pair set (positive + 1:1 negative, seed=42)
    # ------------------------------------------------------------------
    print("\nStep 3: Building pair set...")
    n = len(keys)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    pairs_to_compute = set()
    for a, b in copy_pairs:
        if a in key_to_idx and b in key_to_idx:
            pairs_to_compute.add((a, b))

    n_pos = len(pairs_to_compute)

    # Sample negative pairs (1:1 ratio, seed=42)
    n_neg_target = n_pos
    rng = np.random.RandomState(42)
    neg_count = 0
    neg_attempts = 0
    while neg_count < n_neg_target and neg_attempts < n_neg_target * 20:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            neg_attempts += 1
            continue
        pair = tuple(sorted([keys[i], keys[j]]))
        if pair not in copy_pairs and pair not in pairs_to_compute:
            # pyrefly: ignore [bad-argument-type]
            pairs_to_compute.add(pair)
            neg_count += 1
        neg_attempts += 1

    print(f"  Total pairs: {len(pairs_to_compute)} " f"({n_pos} pos + {neg_count} neg)")

    # ------------------------------------------------------------------
    # Step 4: Compute V-JEPA 2 similarities
    # ------------------------------------------------------------------
    print("\nStep 4: Computing V-JEPA 2 similarities...")
    t0 = time.time()
    all_sims = compute_vjepa2_similarities(vjepa2_features, pairs_to_compute)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # Verify point APs
    for method_name, sims_dict in all_sims.items():
        scores_list = []
        labels_list = []
        for pair, sim in sims_dict.items():
            scores_list.append(sim)
            labels_list.append(1 if pair in copy_pairs else 0)
        point_ap = average_precision_score(labels_list, scores_list)
        n_p = sum(labels_list)
        n_n = len(labels_list) - n_p
        print(f"  {method_name}: AP={point_ap:.4f} (pos={n_p}, neg={n_n})")

    # ------------------------------------------------------------------
    # Step 5: Bootstrap CIs
    # ------------------------------------------------------------------
    print(
        f"\nStep 5: Bootstrap CIs ({args.n_resamples} resamples, seed={args.seed})..."
    )
    t0 = time.time()

    results = {}
    method_display = {
        "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
        "vjepa2_temporal_residual": "V-JEPA 2 Temporal Res.",
    }

    for method_name, sims_dict in all_sims.items():
        scores_list = []
        labels_list = []
        for pair, sim in sims_dict.items():
            scores_list.append(sim)
            labels_list.append(1 if pair in copy_pairs else 0)

        scores_arr = np.array(scores_list)
        labels_arr = np.array(labels_list)

        ap, ap_lo, ap_hi = bootstrap_ap(
            scores_arr,
            labels_arr,
            n_resamples=args.n_resamples,
            seed=args.seed,
            device=bootstrap_device,
        )

        ci_width = ap_hi - ap_lo
        display = method_display.get(method_name, method_name)

        results[method_name] = {
            "display_name": display,
            "ap": float(ap),
            "ap_ci_low": float(ap_lo),
            "ap_ci_high": float(ap_hi),
            "ci_width": float(ci_width),
            "n_pos": int(sum(labels_list)),
            "n_neg": int(len(labels_list) - sum(labels_list)),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
        }

    elapsed = time.time() - t0
    print(f"  Bootstrap completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method_name in ["vjepa2_bag_of_tokens", "vjepa2_temporal_residual"]:
        r = results[method_name]
        display = r["display_name"]
        print(
            f"{display:<25s} AP = {r['ap']:.3f}  "
            f"95% CI = [{r['ap_ci_low']:.3f}, {r['ap_ci_high']:.3f}]  "
            f"Width = {r['ci_width']:.3f}"
        )

    print(f"\nLaTeX rows (paste into bootstrap CI table):")
    for method_name in ["vjepa2_bag_of_tokens", "vjepa2_temporal_residual"]:
        r = results[method_name]
        display_latex = {
            "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
            "vjepa2_temporal_residual": "V-JEPA 2 Temporal Res.",
        }
        name = display_latex[method_name]
        print(
            f" & {name:<25s} & {r['ap']:.3f} "
            f"& [{r['ap_ci_low']:.3f}, {r['ap_ci_high']:.3f}] "
            f"& {r['ci_width']:.3f} \\\\"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 7: Save results JSON
    # ------------------------------------------------------------------
    results_dir = project_root / "results" / "bootstrap"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "vjepa2_vcdb_bootstrap.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also dump pair-level scores for reproducibility / integration with
    # the main bootstrap_cis.py pipeline
    pair_data = {}
    for method_name, sims_dict in all_sims.items():
        scores_list = []
        labels_list = []
        for pair, sim in sims_dict.items():
            scores_list.append(float(sim))
            labels_list.append(1 if pair in copy_pairs else 0)
        pair_data[method_name] = {"scores": scores_list, "labels": labels_list}

    pair_path = results_dir / "vjepa2_vcdb_pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"Pair-level scores saved to {pair_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

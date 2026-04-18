#!/usr/bin/env python3
"""RAFT Optical Flow Baseline for Maneuver Discrimination.

Evaluates whether RAFT optical flow features can distinguish different
maneuvers (left turn vs right turn) at the same intersection, providing
a classical motion baseline for comparison with DINOv3 per-frame methods
(~0.50 AP) and V-JEPA 2 temporal residuals (0.956 AP).

Three optical flow representations are compared:
1. Flow magnitude histogram (BoF-style): 32-bin histogram of flow magnitudes,
   averaged across frame pairs, compared via cosine similarity.
2. Flow direction histogram (BoF-style): 8 angular bins x 4 magnitude bins
   = 32-bin joint histogram, averaged across frame pairs, cosine similarity.
3. Flow sequence DTW: Mean flow vector (dx, dy) per frame pair yields a
   (T-1, 2) trajectory, compared via DTW with exp(-alpha * d_DTW).

Pipeline:
1. Load HDD session data (labels, GPS, video paths)
2. Extract contiguous maneuver segments with GPS midpoints
3. Cluster intersections using DBSCAN on GPS coordinates
4. Extract 30 frames uniformly from +/-3s clips, compute RAFT optical flow
5. Build three flow representations per clip
6. Compare all pairs within each GPS-clustered intersection
7. Evaluate discrimination (AP, AUC) with bootstrap CIs

When --nuscenes is passed, runs on nuScenes instead (DBSCAN eps=30m on
ego_pose, CAN bus labels, same evaluation protocol).

Usage:
    python experiments/eval_hdd_optical_flow.py [--device cuda]
    python experiments/eval_hdd_optical_flow.py --nuscenes --nuscenes-dir ./data/nuscenes
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import HDD data loading utilities
from eval_hdd_intersections import (
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_clip,
    load_gps,
    ManeuverSegment,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm
from video_retrieval.fingerprints.dtw import dtw_distance_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAFT_INPUT_SIZE = (520, 960)  # (H, W) expected by RAFT Large
NUM_FRAMES = (
    30  # Frames per clip (matching DINOv3 frame count at 3fps * 2*3s + maneuver)
)
FLOW_MAG_BINS = 32
FLOW_DIR_ANGULAR_BINS = 8
FLOW_DIR_MAG_BINS = 4
FLOW_DIR_TOTAL_BINS = FLOW_DIR_ANGULAR_BINS * FLOW_DIR_MAG_BINS  # 32
DTW_ALPHA = 1.0

# Reference results from prior experiments for comparison table
REFERENCE_RESULTS = {
    "DINOv3 Bag-of-Frames": {"ap": 0.500, "auc": 0.500},
    "DINOv3 Attention Traj.": {"ap": 0.520, "auc": 0.530},
    "V-JEPA 2 Temporal Res.": {"ap": 0.956, "auc": 0.940},
}


# ---------------------------------------------------------------------------
# Bootstrap CI for AP
# ---------------------------------------------------------------------------


def bootstrap_ap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for Average Precision.

    Returns:
        (ap, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    ap = average_precision_score(labels, scores)

    boot_aps = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            boot_aps[i] = ap  # degenerate sample
        else:
            boot_aps[i] = average_precision_score(l, s)

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_aps, 100 * alpha))
    ci_high = float(np.percentile(boot_aps, 100 * (1 - alpha)))

    return float(ap), ci_low, ci_high


# ---------------------------------------------------------------------------
# RAFT optical flow computation
# ---------------------------------------------------------------------------


def preprocess_for_raft(
    frames: list[np.ndarray],
) -> torch.Tensor:
    """Preprocess RGB numpy frames for RAFT input.

    RAFT expects (N, 3, H, W) float tensors in [0, 255] range.
    Resizes to RAFT_INPUT_SIZE.

    Args:
        frames: List of (H, W, 3) uint8 RGB numpy arrays.

    Returns:
        (N, 3, H, W) float32 tensor on CPU.
    """
    tensors = []
    for frame in frames:
        # Resize to RAFT expected size
        resized = cv2.resize(frame, (RAFT_INPUT_SIZE[1], RAFT_INPUT_SIZE[0]))
        # HWC -> CHW, uint8 -> float32
        t = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensors.append(t)
    return torch.stack(tensors)


@torch.no_grad()
def compute_optical_flow(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    device: torch.device,
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """Compute RAFT optical flow between consecutive frame pairs.

    Args:
        model: RAFT model.
        frames: List of T RGB frames as numpy arrays.
        device: Torch device.
        batch_size: Number of frame pairs to process at once.

    Returns:
        List of T-1 flow fields, each (2, H, W) on CPU.
    """
    if len(frames) < 2:
        return []

    preprocessed = preprocess_for_raft(frames).to(device)
    flows = []

    # Process in batches of consecutive pairs
    for start in range(0, len(frames) - 1, batch_size):
        end = min(start + batch_size, len(frames) - 1)
        img1 = preprocessed[start:end]
        img2 = preprocessed[start + 1 : end + 1]

        # RAFT returns list of flow predictions at different iterations;
        # take the last (most refined) prediction
        flow_preds = model(img1, img2)
        final_flow = flow_preds[-1]  # (B, 2, H, W)

        for i in range(final_flow.shape[0]):
            flows.append(final_flow[i].cpu())

    return flows


# ---------------------------------------------------------------------------
# Flow feature representations
# ---------------------------------------------------------------------------


def flow_magnitude_histogram(
    flows: list[torch.Tensor],
    n_bins: int = FLOW_MAG_BINS,
    max_mag: float = 50.0,
) -> np.ndarray:
    """Compute average flow magnitude histogram across frame pairs.

    For each flow field, bins the per-pixel flow magnitudes into a histogram,
    then averages all histograms to produce a single descriptor per clip.

    Args:
        flows: List of (2, H, W) flow tensors.
        n_bins: Number of histogram bins.
        max_mag: Maximum flow magnitude for binning (pixels).

    Returns:
        (n_bins,) L1-normalized histogram vector.
    """
    if not flows:
        return np.zeros(n_bins)

    hist_sum = np.zeros(n_bins)
    for flow in flows:
        # flow: (2, H, W) -> magnitude: (H, W)
        mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2).numpy().flatten()
        hist, _ = np.histogram(mag, bins=n_bins, range=(0, max_mag))
        hist_sum += hist.astype(np.float64)

    hist_avg = hist_sum / len(flows)

    # L1 normalize for cosine similarity
    total = hist_avg.sum()
    if total > 0:
        hist_avg /= total

    return hist_avg.astype(np.float32)


def flow_direction_histogram(
    flows: list[torch.Tensor],
    n_angular_bins: int = FLOW_DIR_ANGULAR_BINS,
    n_mag_bins: int = FLOW_DIR_MAG_BINS,
    max_mag: float = 50.0,
) -> np.ndarray:
    """Compute average flow direction histogram (joint angle x magnitude).

    For each flow field, computes a joint histogram of flow angles (8 bins,
    covering [0, 2*pi]) and flow magnitudes (4 bins), then averages across
    all frame pairs.

    Args:
        flows: List of (2, H, W) flow tensors.
        n_angular_bins: Number of angular bins.
        n_mag_bins: Number of magnitude bins.
        max_mag: Maximum flow magnitude for binning.

    Returns:
        (n_angular_bins * n_mag_bins,) L1-normalized histogram vector.
    """
    n_total = n_angular_bins * n_mag_bins
    if not flows:
        return np.zeros(n_total)

    hist_sum = np.zeros(n_total)
    for flow in flows:
        dx = flow[0].numpy().flatten()
        dy = flow[1].numpy().flatten()

        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)  # range [-pi, pi]
        angle = (angle + 2 * np.pi) % (2 * np.pi)  # range [0, 2*pi]

        # Bin indices
        ang_bins = np.clip(
            (angle / (2 * np.pi) * n_angular_bins).astype(int),
            0,
            n_angular_bins - 1,
        )
        mag_bins = np.clip(
            (mag / max_mag * n_mag_bins).astype(int),
            0,
            n_mag_bins - 1,
        )

        # Flatten 2D bin index to 1D
        flat_bins = ang_bins * n_mag_bins + mag_bins
        hist, _ = np.histogram(flat_bins, bins=n_total, range=(0, n_total))
        hist_sum += hist.astype(np.float64)

    hist_avg = hist_sum / len(flows)

    # L1 normalize
    total = hist_avg.sum()
    if total > 0:
        hist_avg /= total

    return hist_avg.astype(np.float32)


def flow_mean_trajectory(
    flows: list[torch.Tensor],
) -> torch.Tensor:
    """Compute mean flow vector per frame pair -> (T-1, 2) trajectory.

    For each flow field, averages the (dx, dy) flow vectors across all pixels
    to produce a single 2D motion vector per timestep.

    Args:
        flows: List of (2, H, W) flow tensors.

    Returns:
        (T-1, 2) tensor of mean flow vectors.
    """
    if not flows:
        return torch.zeros(1, 2)

    vectors = []
    for flow in flows:
        # flow: (2, H, W) -> mean over spatial dims -> (2,)
        mean_flow = flow.mean(dim=(1, 2))  # (2,)
        vectors.append(mean_flow)

    return torch.stack(vectors)  # (T-1, 2)


# ---------------------------------------------------------------------------
# Feature extraction for all segments
# ---------------------------------------------------------------------------


def extract_optical_flow_features(
    model: torch.nn.Module,
    segments: list[ManeuverSegment],
    device: torch.device,
    context_sec: float = 3.0,
    num_frames: int = NUM_FRAMES,
) -> dict[int, dict]:
    """Extract optical flow features for all HDD maneuver segments.

    For each segment, extracts frames from the +/-context_sec clip,
    computes RAFT optical flow between consecutive frames, and builds
    three representations.

    Args:
        model: RAFT model.
        segments: Maneuver segments to process.
        device: Torch device.
        context_sec: Seconds of context before/after maneuver.
        num_frames: Number of frames to extract per clip.

    Returns:
        Dict mapping segment index -> {
            'mag_hist': (32,) np.ndarray,
            'dir_hist': (32,) np.ndarray,
            'flow_traj': (T-1, 2) torch.Tensor,
        }
    """
    features = {}
    failed = 0

    for i, seg in enumerate(tqdm(segments, desc="Extracting optical flow features")):
        start_sec = seg.start_frame / 3.0 - context_sec
        end_sec = seg.end_frame / 3.0 + context_sec
        start_sec = max(0.0, start_sec)

        # Compute target_fps to get exactly num_frames from the clip
        duration = end_sec - start_sec
        if duration <= 0:
            failed += 1
            continue
        target_fps = num_frames / duration

        try:
            frames = load_clip(
                seg.video_path,
                start_sec,
                end_sec,
                target_fps=target_fps,
                max_resolution=0,  # no downscale; RAFT preprocess handles resizing
            )
            if len(frames) < 4:
                failed += 1
                continue

            # Truncate to num_frames if we got more
            frames = frames[:num_frames]

            # Compute optical flow for consecutive pairs
            flows = compute_optical_flow(model, frames, device)

            if len(flows) < 3:
                failed += 1
                continue

            # Build three representations
            mag_hist = flow_magnitude_histogram(flows)
            dir_hist = flow_direction_histogram(flows)
            flow_traj = flow_mean_trajectory(flows)

            features[i] = {
                "mag_hist": mag_hist,
                "dir_hist": dir_hist,
                "flow_traj": flow_traj,
            }
        except Exception as e:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# nuScenes feature extraction
# ---------------------------------------------------------------------------


def extract_optical_flow_features_nuscenes(
    model: torch.nn.Module,
    segments: list,  # nuScenes ManeuverSegment
    scene_keyframes: dict,
    device: torch.device,
    num_frames: int = NUM_FRAMES,
) -> dict[int, dict]:
    """Extract optical flow features for nuScenes maneuver segments.

    Uses CAM_FRONT keyframes within each segment's time window.

    Args:
        model: RAFT model.
        segments: nuScenes ManeuverSegment objects with keyframe_paths.
        scene_keyframes: Dict of scene_name -> keyframe list (unused, paths on segments).
        device: Torch device.
        num_frames: Number of frames to extract per clip.

    Returns:
        Dict mapping segment index -> {mag_hist, dir_hist, flow_traj}.
    """
    features = {}
    failed = 0

    for i, seg in enumerate(
        tqdm(segments, desc="Extracting optical flow features (nuScenes)")
    ):
        if len(seg.keyframe_paths) < 4:
            failed += 1
            continue

        try:
            # Load keyframe images
            frames = []
            for path in seg.keyframe_paths[:num_frames]:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)

            if len(frames) < 4:
                failed += 1
                continue

            # Compute optical flow
            flows = compute_optical_flow(model, frames, device)
            if len(flows) < 3:
                failed += 1
                continue

            mag_hist = flow_magnitude_histogram(flows)
            dir_hist = flow_direction_histogram(flows)
            flow_traj = flow_mean_trajectory(flows)

            features[i] = {
                "mag_hist": mag_hist,
                "dir_hist": dir_hist,
                "flow_traj": flow_traj,
            }
        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(features)}/{len(segments)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_all_similarities(
    segments: list,
    features: dict[int, dict],
    cluster_to_indices: dict[int, list[int]],
    device: torch.device,
) -> dict[str, tuple[list[float], list[int]]]:
    """Compute pairwise similarities within each cluster for all flow methods.

    For all pairs within each cluster:
    - Ground truth: same-maneuver label = positive (1), different = negative (0)
    - Compute 3 optical flow similarity methods

    Returns:
        Dict mapping method_name -> (scores_list, labels_list).
    """
    # Enumerate all pairs across clusters
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
                seg_a = segments[indices[a_pos]]
                seg_b = segments[indices[b_pos]]
                gt = 1 if seg_a.label == seg_b.label else 0
                pair_gts.append(gt)

    total_pairs = len(pair_gts)
    print(f"  Total pairs to compute: {total_pairs}")

    if total_pairs == 0:
        return {
            "flow_magnitude_hist": ([], []),
            "flow_direction_hist": ([], []),
            "flow_sequence_dtw": ([], []),
        }

    # --- Flow magnitude histogram: cosine similarity ---
    print("  Computing flow magnitude histogram similarities...")
    mag_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        sim = compute_cosine_similarity(
            features[a_idx]["mag_hist"],
            features[b_idx]["mag_hist"],
        )
        mag_sims.append(sim)

    # --- Flow direction histogram: cosine similarity ---
    print("  Computing flow direction histogram similarities...")
    dir_sims = []
    for a_idx, b_idx in zip(pair_a_indices, pair_b_indices):
        sim = compute_cosine_similarity(
            features[a_idx]["dir_hist"],
            features[b_idx]["dir_hist"],
        )
        dir_sims.append(sim)

    # --- Flow sequence DTW: batched GPU DTW ---
    print("  Computing flow sequence DTW similarities (batched GPU)...")
    traj_seqs_a = [features[i]["flow_traj"].to(device) for i in pair_a_indices]
    traj_seqs_b = [features[i]["flow_traj"].to(device) for i in pair_b_indices]
    traj_dists = dtw_distance_batch(traj_seqs_a, traj_seqs_b, normalize=True)
    dtw_sims = torch.exp(-DTW_ALPHA * traj_dists).cpu().tolist()

    all_scores: dict[str, tuple[list[float], list[int]]] = {
        "flow_magnitude_hist": (mag_sims, list(pair_gts)),
        "flow_direction_hist": (dir_sims, list(pair_gts)),
        "flow_sequence_dtw": (dtw_sims, list(pair_gts)),
    }

    return all_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrimination(
    results: dict,
    fig_dir: Path,
    dataset_name: str = "HDD",
):
    """Generate AP/AUC bar chart for optical flow methods."""
    methods = ["flow_magnitude_hist", "flow_direction_hist", "flow_sequence_dtw"]
    labels = [
        "Flow Magnitude\nHistogram",
        "Flow Direction\nHistogram",
        "Flow Sequence\nDTW",
    ]
    colors = {
        "flow_magnitude_hist": "#e67e22",
        "flow_direction_hist": "#2980b9",
        "flow_sequence_dtw": "#27ae60",
    }

    aps = [results[m]["ap"] for m in methods]
    aucs = [results[m]["auc"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AP
    bars = ax1.bar(
        range(len(methods)),
        aps,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, aps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylabel("Average Precision", fontsize=12)
    ax1.set_title("RAFT Optical Flow: AP", fontsize=13)
    ax1.set_ylim(0, min(1.0, max(aps) * 1.15 + 0.05))
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax1.legend()

    # AUC
    bars = ax2.bar(
        range(len(methods)),
        aucs,
        color=[colors[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, aucs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("RAFT Optical Flow: AUC", fontsize=13)
    ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.legend()

    prefix = dataset_name.lower()
    fig.suptitle(
        f"{dataset_name}: RAFT Optical Flow Maneuver Discrimination",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    path = fig_dir / f"{prefix}_optical_flow_discrimination.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


def print_comparison_table(results: dict, dataset_name: str = "HDD"):
    """Print a comparison table with flow methods and reference results."""
    print("\n" + "=" * 75)
    print(f"COMPARISON TABLE: {dataset_name} Maneuver Discrimination")
    print("=" * 75)
    print(f"  {'Method':<35s}  {'AP':>8s}  {'95% CI':>18s}  {'AUC':>8s}")
    print("  " + "-" * 71)

    # Reference methods
    for name, ref in REFERENCE_RESULTS.items():
        print(
            f"  {name:<35s}  {ref['ap']:>8.3f}  {'(reference)':>18s}  {ref['auc']:>8.3f}"
        )

    print("  " + "-" * 71)

    # Optical flow methods
    method_names = {
        "flow_magnitude_hist": "RAFT Flow Mag. Histogram",
        "flow_direction_hist": "RAFT Flow Dir. Histogram",
        "flow_sequence_dtw": "RAFT Flow Sequence DTW",
    }
    for method_key, display_name in method_names.items():
        if method_key in results:
            r = results[method_key]
            ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            print(
                f"  {display_name:<35s}  {r['ap']:>8.3f}  {ci_str:>18s}  {r['auc']:>8.3f}"
            )

    print("=" * 75)


# ---------------------------------------------------------------------------
# HDD pipeline
# ---------------------------------------------------------------------------


def run_hdd(args):
    """Run optical flow evaluation on Honda HDD dataset."""
    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    output_dir = project_root / args.output_dir
    fig_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    print("=" * 70)
    print("HONDA HDD: RAFT OPTICAL FLOW MANEUVER DISCRIMINATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Discover sessions
    # ------------------------------------------------------------------
    print("\nStep 1: Discovering sessions...")
    t0 = time.time()
    sessions = discover_sessions(hdd_dir)
    print(f"  Found {len(sessions)} valid sessions (with labels, GPS, video)")
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

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections
    # ------------------------------------------------------------------
    print("\nStep 3: Clustering intersections (DBSCAN eps=0.0003)...")
    clusters = cluster_intersections(all_segments, eps=0.0003, min_samples=3)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters
    # ------------------------------------------------------------------
    mixed = filter_mixed_clusters(clusters, max_clusters=50)

    total_segs_in_mixed = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print("\nERROR: No mixed clusters found. Cannot evaluate.")
        return

    # Build flat list of segments in qualifying clusters
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

    # ------------------------------------------------------------------
    # Step 5: Load RAFT model and extract optical flow features
    # ------------------------------------------------------------------
    print("\nStep 5: Loading RAFT Large model...")
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    print("\nStep 6: Extracting optical flow features...")
    t_feat_start = time.time()
    features = extract_optical_flow_features(
        raft_model,
        eval_segments,
        device,
        context_sec=3.0,
        num_frames=NUM_FRAMES,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    del raft_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 7: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 7: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_all_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        device,
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: Evaluate with bootstrap CIs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: RAFT OPTICAL FLOW MANEUVER DISCRIMINATION (HDD)")
    print("=" * 70)

    results = {}
    method_order = [
        "flow_magnitude_hist",
        "flow_direction_hist",
        "flow_sequence_dtw",
    ]

    for method in method_order:
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap, ci_low, ci_high = bootstrap_ap_ci(scores, labels, n_resamples=2000)
        auc = roc_auc_score(labels, scores)

        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())
        gap = same_mean - diff_mean

        results[method] = {
            "ap": float(ap),
            "auc": float(auc),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": gap,
        }

        print(
            f"  {method:<25s}  AP={ap:.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
            f"AUC={auc:.4f}  gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 9: Save results and generate figures
    # ------------------------------------------------------------------
    results_path = output_dir / "hdd_optical_flow_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save pair-level scores
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }
    pair_path = output_dir / "hdd_optical_flow_pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir, dataset_name="HDD")

    # Print comparison table
    print_comparison_table(results, dataset_name="HDD")

    # Summary
    n_total_pairs = len(all_scores["flow_magnitude_hist"][0])
    print("\nSummary:")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Total maneuver segments: {len(all_segments)}")
    print(f"  Mixed intersection clusters: {len(mixed)}")
    print(f"  Evaluation segments: {len(eval_segments)}")
    print(f"  Total pairs evaluated: {n_total_pairs}")
    print(f"  Feature extraction time: {t_feat:.1f}s")
    print("\nDone.")


# ---------------------------------------------------------------------------
# nuScenes pipeline
# ---------------------------------------------------------------------------


def run_nuscenes(args):
    """Run optical flow evaluation on nuScenes dataset."""
    from eval_nuscenes_intersections import (
        assign_keyframes_to_segments,
        cluster_intersections as nuscenes_cluster_intersections,
        filter_mixed_clusters as nuscenes_filter_mixed_clusters,
        get_scene_keyframes,
        load_can_bus,
        load_nuscenes_metadata,
        ManeuverSegment as NuScenesManeuverSegment,
        segment_maneuvers,
    )

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.nuscenes_dir)
    output_dir = project_root / args.output_dir
    fig_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    version = args.nuscenes_version

    print("=" * 70)
    print("NUSCENES: RAFT OPTICAL FLOW MANEUVER DISCRIMINATION")
    print("=" * 70)
    print(f"  Version: {version}")

    # ------------------------------------------------------------------
    # Step 1: Load nuScenes metadata
    # ------------------------------------------------------------------
    print("\nStep 1: Loading nuScenes metadata...")
    metadata = load_nuscenes_metadata(data_dir, version)

    # ------------------------------------------------------------------
    # Step 2: Segment maneuvers from CAN bus
    # ------------------------------------------------------------------
    print("\nStep 2: Segmenting maneuvers from CAN bus data...")
    can_dir = data_dir / "can_bus" / "can_bus"

    all_segments: list[NuScenesManeuverSegment] = []
    scene_keyframes: dict[str, list[tuple[float, str, float, float]]] = {}
    scenes_with_segments = 0

    for scene in tqdm(metadata.scenes, desc="Processing scenes"):
        scene_name = scene["name"]

        can_data = load_can_bus(can_dir, scene_name)
        if can_data is None:
            continue

        timestamps, steering, yaw_rates, positions, speeds = can_data

        segs = segment_maneuvers(
            timestamps,
            steering,
            yaw_rates,
            positions,
            speeds,
            scene_name,
        )

        if segs:
            scenes_with_segments += 1
        all_segments.extend(segs)

        kfs = get_scene_keyframes(scene, metadata, data_dir)
        if kfs:
            scene_keyframes[scene_name] = kfs

    label_counts: dict[int, int] = defaultdict(int)
    for seg in all_segments:
        label_counts[seg.label] += 1

    print(
        f"  Total segments: {len(all_segments)} "
        f"from {scenes_with_segments}/{len(metadata.scenes)} scenes"
    )

    if not all_segments:
        print("\n  No maneuver segments found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 3: Cluster intersections
    # ------------------------------------------------------------------
    print(f"\nStep 3: Clustering intersections (DBSCAN eps=30m)...")
    clusters = nuscenes_cluster_intersections(all_segments, eps=30.0, min_samples=2)
    print(f"  Total clusters: {len(clusters)}")

    # ------------------------------------------------------------------
    # Step 4: Filter for mixed clusters
    # ------------------------------------------------------------------
    mixed = nuscenes_filter_mixed_clusters(clusters, max_clusters=50)

    total_segs_in_mixed = sum(len(segs) for segs in mixed.values())
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print("\n  No mixed clusters found. Cannot evaluate.")
        if version == "v1.0-mini":
            print("  Use --nuscenes-version v1.0-trainval for real results.")
        return

    # Build flat list
    eval_segments: list[NuScenesManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)

    # Assign keyframes
    assign_keyframes_to_segments(eval_segments, scene_keyframes)

    print(f"\n  Segments for evaluation: {len(eval_segments)}")

    # ------------------------------------------------------------------
    # Step 5: Load RAFT and extract features
    # ------------------------------------------------------------------
    print("\nStep 5: Loading RAFT Large model...")
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    print("\nStep 6: Extracting optical flow features...")
    t_feat_start = time.time()
    features = extract_optical_flow_features_nuscenes(
        raft_model,
        eval_segments,
        scene_keyframes,
        device,
        num_frames=NUM_FRAMES,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature extraction time: {t_feat:.1f}s")

    del raft_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 7: Compute similarities
    # ------------------------------------------------------------------
    print("\nStep 7: Computing pairwise similarities...")
    t_sim_start = time.time()
    all_scores = compute_all_similarities(
        eval_segments,
        features,
        cluster_to_indices,
        device,
    )
    t_sim = time.time() - t_sim_start
    print(f"  Similarity computation time: {t_sim:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: Evaluate with bootstrap CIs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: RAFT OPTICAL FLOW MANEUVER DISCRIMINATION (nuScenes)")
    print("=" * 70)

    results = {}
    method_order = [
        "flow_magnitude_hist",
        "flow_direction_hist",
        "flow_sequence_dtw",
    ]

    for method in method_order:
        scores_list, labels_list = all_scores[method]
        scores = np.array(scores_list)
        labels = np.array(labels_list)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            results[method] = {
                "ap": float("nan"),
                "auc": float("nan"),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            continue

        ap, ci_low, ci_high = bootstrap_ap_ci(scores, labels, n_resamples=2000)
        auc = roc_auc_score(labels, scores)

        same_mean = float(scores[labels == 1].mean())
        diff_mean = float(scores[labels == 0].mean())
        gap = same_mean - diff_mean

        results[method] = {
            "ap": float(ap),
            "auc": float(auc),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "same_mean": same_mean,
            "diff_mean": diff_mean,
            "gap": gap,
        }

        print(
            f"  {method:<25s}  AP={ap:.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
            f"AUC={auc:.4f}  gap={gap:+.4f}  (pos={n_pos}, neg={n_neg})"
        )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 9: Save results and generate figures
    # ------------------------------------------------------------------
    results_path = output_dir / "nuscenes_optical_flow_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save pair-level scores
    pair_data = {}
    for method_name, (scores_list, labels_list) in all_scores.items():
        pair_data[method_name] = {
            "scores": [float(s) for s in scores_list],
            "labels": [int(l) for l in labels_list],
        }
    pair_path = output_dir / "nuscenes_optical_flow_pair_scores.json"
    with open(pair_path, "w") as f:
        json.dump(pair_data, f)
    print(f"  Pair-level scores saved to {pair_path}")

    print("\nGenerating figures...")
    plot_discrimination(results, fig_dir, dataset_name="nuScenes")

    # Print comparison table
    print_comparison_table(results, dataset_name="nuScenes")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RAFT Optical Flow Baseline for Maneuver Discrimination"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--hdd-dir",
        type=str,
        default="datasets/hdd",
        help="Path to HDD dataset directory",
    )
    parser.add_argument(
        "--nuscenes",
        action="store_true",
        help="Run on nuScenes instead of HDD",
    )
    parser.add_argument(
        "--nuscenes-dir",
        type=str,
        default=None,
        help="Path to nuScenes data root (required with --nuscenes)",
    )
    parser.add_argument(
        "--nuscenes-version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/hdd",
        help="Output directory for results JSON",
    )
    args = parser.parse_args()

    if args.nuscenes:
        if args.nuscenes_dir is None:
            parser.error("--nuscenes-dir is required when --nuscenes is set")
        run_nuscenes(args)
    else:
        run_hdd(args)


if __name__ == "__main__":
    main()

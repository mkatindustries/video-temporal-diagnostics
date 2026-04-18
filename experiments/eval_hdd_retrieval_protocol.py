#!/usr/bin/env python3
"""HDD Maneuver Discrimination — Standard Retrieval Protocol (mAP@K, R@K).

Complements the binary same/different AP evaluation in eval_hdd_intersections.py
with a standard retrieval protocol that captures ranking quality at various
depths. A reviewer noted that binary AP does not reflect whether the top-K
retrieved segments are truly same-maneuver hits.

For each query segment q in a cluster, all other segments in the same cluster
are ranked by similarity (descending). A "relevant" result is one with the
same maneuver label as q (both left or both right). Metrics:

- mAP@K:  mean Average Precision at K = {1, 3, 5, 10}
- R@K:    mean Recall at K = {1, 3, 5, 10}
- NDCG@K: Normalized Discounted Cumulative Gain at K = {1, 5, 10}
- MRR:    Mean Reciprocal Rank (rank of first relevant result)

All 6 core methods + 2 optional VLM methods:
  1. DINOv3 BoF (cosine of mean embeddings)
  2. DINOv3 Chamfer (Chamfer distance)
  3. DINOv3 Attn. Trajectory (DTW on attention centroids)
  4. DINOv3 Temporal Derivative (DTW on temporal derivatives)
  5. V-JEPA 2 BoT (cosine of mean encoder tokens)
  6. V-JEPA 2 Temporal Residual (DTW on residuals)
  7. SigLIP pooled (cosine, optional with --include-vlm)
  8. CLIP pooled (cosine, optional with --include-vlm)

Protocol: 128 sessions, 1,687 maneuvers, 50 mixed-direction clusters,
DBSCAN eps=0.0003, min_samples=3, +/-3 s context, 1,000 bootstrap
resamples for CIs, seed=42.

Usage:
    # Core 6 methods
    python experiments/eval_hdd_retrieval_protocol.py \\
        --hdd-dir datasets/hdd

    # Include VLM methods
    python experiments/eval_hdd_retrieval_protocol.py \\
        --hdd-dir datasets/hdd --include-vlm --vlm-family gemma4

    # Skip V-JEPA 2 (faster, DINOv3-only)
    python experiments/eval_hdd_retrieval_protocol.py \\
        --hdd-dir datasets/hdd --skip-vjepa2

    # Smoke test (2 clusters)
    python experiments/eval_hdd_retrieval_protocol.py \\
        --hdd-dir datasets/hdd --max-clusters 2
"""

import argparse
import json

# Re-use HDD data loading utilities from eval_hdd_vlm_bridge (canonical source).
# Add experiments/ to sys.path so imports work from any working directory.
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_hdd_vlm_bridge import (  # noqa: E402
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    extract_vlm_vision_features,
    filter_mixed_clusters,
    load_gps,
    MANEUVER_NAMES,
    ManeuverSegment,
    VLM_ADAPTERS,
    VLM_DEFAULT_PATHS,
    VLM_DISPLAY_NAMES,
)
from tqdm import tqdm
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder

# V-JEPA 2 constants (same as eval_hdd_intersections.py)
DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VJEPA2_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_NUM_FRAMES = 64
VJEPA2_T_PATCHES = 32  # 64 frames / tubelet_size 2
VJEPA2_SPATIAL = 256  # 16h x 16w

# Retrieval depths to evaluate
K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def average_precision_at_k(relevant: np.ndarray, k: int) -> float:
    """Compute Average Precision at K for a single query.

    Args:
        relevant: Binary array (1=relevant, 0=irrelevant) sorted by
            descending similarity. Shape (n_gallery,).
        k: Depth cutoff.

    Returns:
        AP@K value.
    """
    relevant_k = relevant[:k]
    n_rel_at_k = 0
    sum_precision = 0.0
    for i in range(len(relevant_k)):
        if relevant_k[i] == 1:
            n_rel_at_k += 1
            sum_precision += n_rel_at_k / (i + 1)
    if n_rel_at_k == 0:
        return 0.0
    return sum_precision / n_rel_at_k


def recall_at_k(relevant: np.ndarray, k: int, total_relevant: int) -> float:
    """Compute Recall at K for a single query.

    Args:
        relevant: Binary array (1=relevant, 0=irrelevant) sorted by
            descending similarity.
        k: Depth cutoff.
        total_relevant: Total number of relevant items in the gallery.

    Returns:
        Recall@K value.
    """
    if total_relevant == 0:
        return 0.0
    return float(relevant[:k].sum()) / total_relevant


def ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    """Compute NDCG at K for a single query.

    Uses binary relevance: gain = 1 for relevant, 0 for irrelevant.
    DCG  = sum_{i=1}^{K} rel_i / log2(i+1)
    IDCG = sum_{i=1}^{min(K, n_rel)} 1 / log2(i+1)

    Args:
        relevant: Binary array (1=relevant, 0=irrelevant) sorted by
            descending similarity.
        k: Depth cutoff.

    Returns:
        NDCG@K value.
    """
    relevant_k = relevant[:k].astype(float)
    discounts = np.log2(np.arange(1, len(relevant_k) + 1) + 1)
    dcg = float((relevant_k / discounts).sum())

    # Ideal: all relevant items at the top
    n_rel = int(relevant.sum())
    ideal_k = min(k, n_rel)
    if ideal_k == 0:
        return 0.0
    ideal_discounts = np.log2(np.arange(1, ideal_k + 1) + 1)
    idcg = float((1.0 / ideal_discounts).sum())

    return dcg / idcg


def reciprocal_rank(relevant: np.ndarray) -> float:
    """Compute Reciprocal Rank for a single query.

    Args:
        relevant: Binary array (1=relevant, 0=irrelevant) sorted by
            descending similarity.

    Returns:
        1 / rank_of_first_relevant, or 0.0 if no relevant items.
    """
    hits = np.where(relevant == 1)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / (hits[0] + 1)


# ---------------------------------------------------------------------------
# Per-query retrieval evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval_per_query(
    relevant: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query.

    Args:
        relevant: Binary array (1=relevant, 0=irrelevant) sorted by
            descending similarity (query excluded).
        k_values: List of K depths to evaluate.

    Returns:
        Dict with keys like 'ap@1', 'ap@5', 'recall@1', 'ndcg@5', 'mrr'.
    """
    total_relevant = int(relevant.sum())
    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"ap@{k}"] = average_precision_at_k(relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(relevant, k, total_relevant)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevant, k)

    metrics["mrr"] = reciprocal_rank(relevant)

    return metrics


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_metric(
    per_query_values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for a per-query metric (mAP@K, R@K, NDCG@K, MRR).

    Resamples over queries. Returns (mean, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    n = len(per_query_values)
    mean_val = float(per_query_values.mean())

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = per_query_values[idx].mean()

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return mean_val, ci_low, ci_high


# ---------------------------------------------------------------------------
# Similarity computation helpers
# ---------------------------------------------------------------------------


def compute_pairwise_cosine(
    embeddings: dict[int, torch.Tensor],
    indices: list[int],
    device: torch.device,
) -> np.ndarray:
    """Compute all-pairs cosine similarity for a set of segment indices.

    Args:
        embeddings: Dict mapping segment index -> L2-normed embedding (D,).
        indices: Segment indices in this cluster that have features.
        device: Torch device.

    Returns:
        Similarity matrix of shape (N, N) where N = len(indices).
    """
    n = len(indices)
    emb_stack = torch.stack([embeddings[i] for i in indices]).to(device)  # (N, D)
    sim = torch.mm(emb_stack, emb_stack.t()).cpu().numpy()  # (N, N)
    return sim


def compute_pairwise_chamfer(
    per_frame_embs: dict[int, torch.Tensor],
    indices: list[int],
    device: torch.device,
) -> np.ndarray:
    """Compute all-pairs Chamfer similarity for a set of segment indices.

    Args:
        per_frame_embs: Dict mapping segment index -> per-frame embeddings (T, D).
        indices: Segment indices in this cluster that have features.
        device: Torch device.

    Returns:
        Similarity matrix of shape (N, N).
    """
    n = len(indices)
    sim = np.zeros((n, n), dtype=np.float64)
    for a_pos in range(n):
        ea = per_frame_embs[indices[a_pos]].to(device)
        for b_pos in range(a_pos, n):
            eb = per_frame_embs[indices[b_pos]].to(device)
            sim_matrix = torch.mm(ea, eb.t())
            max_ab = sim_matrix.max(dim=1).values.mean().item()
            max_ba = sim_matrix.max(dim=0).values.mean().item()
            s = (max_ab + max_ba) / 2
            sim[a_pos, b_pos] = s
            sim[b_pos, a_pos] = s
    return sim


def compute_pairwise_dtw(
    fingerprints: dict[int, torch.Tensor],
    indices: list[int],
    device: torch.device,
    normalize: bool,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute all-pairs DTW similarity for a set of segment indices.

    similarity = exp(-scale * dtw_distance)

    Args:
        fingerprints: Dict mapping segment index -> fingerprint sequence (T, D).
        indices: Segment indices in this cluster that have features.
        device: Torch device.
        normalize: Whether to normalize DTW distances by path length.
        scale: Multiplier for the exponent (e.g. 5.0 for attention trajectory).

    Returns:
        Similarity matrix of shape (N, N).
    """
    n = len(indices)
    sim = np.eye(n, dtype=np.float64)  # self-similarity = 1

    # Build list of all upper-triangle pairs
    pair_a = []
    pair_b = []
    pair_positions = []  # (a_pos, b_pos) for filling the matrix
    for a_pos in range(n):
        for b_pos in range(a_pos + 1, n):
            pair_a.append(fingerprints[indices[a_pos]].to(device))
            pair_b.append(fingerprints[indices[b_pos]].to(device))
            pair_positions.append((a_pos, b_pos))

    if not pair_a:
        return sim

    dists = dtw_distance_batch(pair_a, pair_b, normalize=normalize)
    sims = torch.exp(-scale * dists).cpu().numpy()

    for (a_pos, b_pos), s in zip(pair_positions, sims):
        sim[a_pos, b_pos] = s
        sim[b_pos, a_pos] = s

    return sim


# ---------------------------------------------------------------------------
# Retrieval evaluation driver
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    method_name: str,
    sim_matrices: dict[int, np.ndarray],
    cluster_labels: dict[int, np.ndarray],
    cluster_valid_indices: dict[int, list[int]],
    k_values: list[int],
) -> dict[str, object]:
    """Run retrieval evaluation for one method across all clusters.

    For each query segment, ranks all other segments in the same cluster
    by similarity (descending), then computes per-query metrics.

    Args:
        method_name: Display name for the method.
        sim_matrices: Dict mapping cluster_id -> (N, N) similarity matrix.
        cluster_labels: Dict mapping cluster_id -> (N,) maneuver labels.
        cluster_valid_indices: Dict mapping cluster_id -> list of valid
            positions within the cluster (for methods with missing features).
        k_values: List of K depths.

    Returns:
        Dict with aggregated metrics and per-query values.
    """
    all_query_metrics: dict[str, list[float]] = defaultdict(list)

    total_queries = 0
    for cid in sorted(sim_matrices.keys()):
        sim = sim_matrices[cid]
        labels = cluster_labels[cid]
        n = sim.shape[0]

        if n < 2:
            continue

        for q in range(n):
            # Gallery: all indices except the query
            gallery_mask = np.ones(n, dtype=bool)
            gallery_mask[q] = False
            gallery_sims = sim[q, gallery_mask]
            gallery_labels = labels[gallery_mask]

            # Relevance: same maneuver label as query
            query_label = labels[q]
            relevant_gallery = (gallery_labels == query_label).astype(int)

            # Skip if no relevant items or all items are relevant
            total_rel = int(relevant_gallery.sum())
            if total_rel == 0 or total_rel == len(relevant_gallery):
                continue

            # Sort gallery by similarity (descending)
            sort_idx = np.argsort(-gallery_sims)
            relevant_sorted = relevant_gallery[sort_idx]

            # Compute metrics
            query_metrics = evaluate_retrieval_per_query(relevant_sorted, k_values)
            for metric_name, val in query_metrics.items():
                all_query_metrics[metric_name].append(val)

            total_queries += 1

    # Aggregate with bootstrap CIs
    results: dict[str, object] = {"n_queries": total_queries}
    for metric_name, values in sorted(all_query_metrics.items()):
        arr = np.array(values)
        mean_val, ci_lo, ci_hi = bootstrap_metric(arr)
        results[metric_name] = {
            "mean": mean_val,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }

    return results


# ---------------------------------------------------------------------------
# Feature extraction wrappers
# ---------------------------------------------------------------------------


def extract_dinov3_features(
    segments: list[ManeuverSegment],
    device: str,
    context_sec: float,
) -> dict[int, dict]:
    """Extract DINOv3 features for all segments.

    Returns dict mapping segment index -> {
        'embeddings': (T, 1024),
        'centroids': (T, 2),
        'mean_emb': (1024,),
    }
    """
    from eval_hdd_intersections import extract_clip_features

    encoder = DINOv3Encoder(device=device, model_name=DINOV3_MODEL_NAME)
    features = extract_clip_features(
        encoder,
        segments,
        context_sec=context_sec,
        target_fps=3.0,
        max_resolution=518,
    )
    del encoder
    torch.cuda.empty_cache()
    return features


def extract_vjepa2_features_wrapper(
    segments: list[ManeuverSegment],
    device: str,
    context_sec: float,
) -> dict[int, dict]:
    """Extract V-JEPA 2 features for all segments.

    Returns dict mapping segment index -> {
        'mean_emb': (1024,),
        'temporal_residual': (n_target_steps, 1024),
    }
    """
    from eval_hdd_intersections import extract_vjepa2_features
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(VJEPA2_MODEL_NAME, trust_remote_code=True)
    model = model.to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(
        VJEPA2_MODEL_NAME, trust_remote_code=True
    )

    features, _ = extract_vjepa2_features(
        model,
        processor,
        segments,
        device=torch.device(device),
        context_sec=context_sec,
    )

    del model, processor
    torch.cuda.empty_cache()
    return features


# ---------------------------------------------------------------------------
# Build similarity matrices per method
# ---------------------------------------------------------------------------


def build_similarity_matrices(
    method_key: str,
    dino_features: dict[int, dict] | None,
    vjepa2_features: dict[int, dict] | None,
    vlm_features: dict[int, dict] | None,
    cluster_to_indices: dict[int, list[int]],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, list[int]]]:
    """Build per-cluster similarity matrices for a single method.

    Returns:
        (sim_matrices, cluster_labels, cluster_valid_indices)
        Each is a dict keyed by cluster_id.
    """
    sim_matrices: dict[int, np.ndarray] = {}
    cluster_labels_out: dict[int, np.ndarray] = {}
    cluster_valid: dict[int, list[int]] = {}

    # Determine which feature set to use
    if method_key in (
        "bag_of_frames",
        "chamfer",
        "temporal_derivative",
        "attention_trajectory",
    ):
        feat_set = dino_features
    elif method_key in ("vjepa2_bag_of_tokens", "vjepa2_temporal_residual"):
        feat_set = vjepa2_features
    elif method_key in ("vlm_pooled",):
        feat_set = vlm_features
    else:
        return sim_matrices, cluster_labels_out, cluster_valid

    if feat_set is None:
        return sim_matrices, cluster_labels_out, cluster_valid

    # Pre-compute fingerprints if needed (once across all clusters)
    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()
    fingerprints: dict[int, torch.Tensor] = {}

    if method_key == "temporal_derivative":
        for idx in feat_set:
            fingerprints[idx] = deriv_fp.compute_fingerprint(
                feat_set[idx]["embeddings"]
            )
    elif method_key == "attention_trajectory":
        for idx in feat_set:
            fingerprints[idx] = traj_fp.compute_fingerprint(feat_set[idx]["centroids"])
    elif method_key == "vjepa2_temporal_residual":
        # Temporal residual sequences are used directly
        for idx in feat_set:
            fingerprints[idx] = feat_set[idx]["temporal_residual"]

    for cid in sorted(cluster_to_indices.keys()):
        # Filter to indices with valid features
        raw_indices = cluster_to_indices[cid]
        valid_indices = [i for i in raw_indices if i in feat_set]
        if len(valid_indices) < 2:
            continue

        cluster_valid[cid] = valid_indices

        # Build label array for this cluster (will be imported from segments)
        # We store indices and compute labels externally
        if method_key == "bag_of_frames":
            mean_embs = {i: feat_set[i]["mean_emb"] for i in valid_indices}
            sim = compute_pairwise_cosine(mean_embs, valid_indices, device)
        elif method_key == "chamfer":
            per_frame = {i: feat_set[i]["embeddings"] for i in valid_indices}
            sim = compute_pairwise_chamfer(per_frame, valid_indices, device)
        elif method_key == "temporal_derivative":
            sim = compute_pairwise_dtw(
                fingerprints,
                valid_indices,
                device,
                normalize=False,
                scale=1.0,
            )
        elif method_key == "attention_trajectory":
            sim = compute_pairwise_dtw(
                fingerprints,
                valid_indices,
                device,
                normalize=True,
                scale=5.0,
            )
        elif method_key == "vjepa2_bag_of_tokens":
            mean_embs = {i: feat_set[i]["mean_emb"] for i in valid_indices}
            sim = compute_pairwise_cosine(mean_embs, valid_indices, device)
        elif method_key == "vjepa2_temporal_residual":
            sim = compute_pairwise_dtw(
                fingerprints,
                valid_indices,
                device,
                normalize=True,
                scale=1.0,
            )
        elif method_key == "vlm_pooled":
            mean_embs = {i: feat_set[i]["vision_repr"] for i in valid_indices}
            sim = compute_pairwise_cosine(mean_embs, valid_indices, device)
        else:
            continue

        sim_matrices[cid] = sim

    return sim_matrices, cluster_labels_out, cluster_valid


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


METHOD_DISPLAY_NAMES = {
    "bag_of_frames": "DINOv3 BoF",
    "chamfer": "DINOv3 Chamfer",
    "attention_trajectory": "DINOv3 Attn. Traj.",
    "temporal_derivative": "DINOv3 Temp. Deriv.",
    "vjepa2_bag_of_tokens": "V-JEPA 2 BoT",
    "vjepa2_temporal_residual": "V-JEPA 2 Temp. Res.",
}


def print_summary_table(
    all_results: dict[str, dict],
    method_order: list[str],
    display_names: dict[str, str],
) -> None:
    """Print a formatted summary table of retrieval metrics."""
    header = (
        f"{'Method':<25s} | {'mAP@1':>7s} | {'mAP@5':>7s} | "
        f"{'R@1':>7s} | {'R@5':>7s} | {'NDCG@5':>7s} | {'MRR':>7s}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for method_key in method_order:
        if method_key not in all_results:
            continue
        r = all_results[method_key]
        name = display_names.get(method_key, method_key)

        def _get(metric: str) -> str:
            if metric in r and isinstance(r[metric], dict):
                return f"{r[metric]['mean']:.3f}"
            return "  -  "

        print(
            f"{name:<25s} | {_get('ap@1'):>7s} | {_get('ap@5'):>7s} | "
            f"{_get('recall@1'):>7s} | {_get('recall@5'):>7s} | "
            f"{_get('ndcg@5'):>7s} | {_get('mrr'):>7s}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="HDD Retrieval Protocol: mAP@K, Recall@K, NDCG@K, MRR"
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
        "--skip-vjepa2",
        action="store_true",
        help="Skip V-JEPA 2 methods (faster, DINOv3-only)",
    )
    parser.add_argument(
        "--include-vlm",
        action="store_true",
        help="Include VLM vision tower methods (SigLIP or CLIP)",
    )
    parser.add_argument(
        "--vlm-family",
        type=str,
        default="gemma4",
        choices=["gemma4", "llava-video"],
        help="VLM family to evaluate (used with --include-vlm)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("HDD RETRIEVAL PROTOCOL: mAP@K, Recall@K, NDCG@K, MRR")
    print("=" * 70)
    print(f"  K values: {K_VALUES}")
    print(f"  Context: +/-{args.context_sec}s")
    print(f"  Max clusters: {args.max_clusters}")
    print(f"  Skip V-JEPA 2: {args.skip_vjepa2}")
    print(f"  Include VLM: {args.include_vlm}")

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
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {label_counts.get(lv, 0)}")

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
    print(f"  Mixed clusters (contain both left+right turns): {len(mixed)}")
    print(f"  Total segments in mixed clusters: {total_segs_in_mixed}")

    if not mixed:
        print(
            "\nERROR: No mixed clusters found. "
            "Cannot evaluate maneuver discrimination."
        )
        return

    # Build flat list of segments in qualifying clusters, with cluster mapping
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
    for lv, name in sorted(MANEUVER_NAMES.items()):
        print(f"    {name}: {eval_label_counts.get(lv, 0)}")

    # ------------------------------------------------------------------
    # Step 5: Extract features
    # ------------------------------------------------------------------

    # 5a: DINOv3
    print("\nStep 5a: Extracting DINOv3 features...")
    t_dino_start = time.time()
    dino_features = extract_dinov3_features(
        eval_segments, args.device, args.context_sec
    )
    print(f"  DINOv3 extraction time: {time.time() - t_dino_start:.1f}s")

    # 5b: V-JEPA 2
    vjepa2_features: dict[int, dict] | None = None
    if not args.skip_vjepa2:
        print("\nStep 5b: Extracting V-JEPA 2 features...")
        t_vjepa_start = time.time()
        vjepa2_features = extract_vjepa2_features_wrapper(
            eval_segments, args.device, args.context_sec
        )
        print(f"  V-JEPA 2 extraction time: {time.time() - t_vjepa_start:.1f}s")

    # 5c: VLM (optional)
    vlm_features: dict[int, dict] | None = None
    vlm_display_name: str | None = None
    if args.include_vlm:
        vlm_family = args.vlm_family
        model_path = VLM_DEFAULT_PATHS[vlm_family]
        vlm_display_name = VLM_DISPLAY_NAMES[vlm_family]
        print(f"\nStep 5c: Extracting {vlm_display_name} vision features...")
        t_vlm_start = time.time()

        adapter = VLM_ADAPTERS[vlm_family]()
        model, processor = adapter.load(model_path, device)

        cache_dir = hdd_dir / "vlm_bridge_cache"
        cache_path = cache_dir / f"{vlm_family}_vision_features.pt"

        vlm_features = extract_vlm_vision_features(
            adapter,
            model,
            processor,
            eval_segments,
            context_sec=args.context_sec,
            n_frames=16,
            device=device,
            cache_path=cache_path,
        )
        print(f"  VLM extraction time: {time.time() - t_vlm_start:.1f}s")

        del model, processor
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 6: Build per-cluster label arrays
    # ------------------------------------------------------------------
    cluster_labels: dict[int, np.ndarray] = {}
    for cid, indices in cluster_to_indices.items():
        cluster_labels[cid] = np.array([eval_segments[i].label for i in indices])

    # ------------------------------------------------------------------
    # Step 7: Evaluate each method under the retrieval protocol
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RETRIEVAL PROTOCOL EVALUATION")
    print("=" * 70)

    method_order = [
        "bag_of_frames",
        "chamfer",
        "attention_trajectory",
        "temporal_derivative",
    ]
    if not args.skip_vjepa2:
        method_order.extend(["vjepa2_bag_of_tokens", "vjepa2_temporal_residual"])

    display_names = dict(METHOD_DISPLAY_NAMES)

    if args.include_vlm and vlm_features is not None:
        vlm_key = "vlm_pooled"
        method_order.append(vlm_key)
        display_names[vlm_key] = f"{vlm_display_name} pooled"

    all_results: dict[str, dict] = {}

    for method_key in method_order:
        method_display = display_names.get(method_key, method_key)
        print(f"\n  Evaluating: {method_display}")

        # Build similarity matrices
        sim_matrices, _, cluster_valid = build_similarity_matrices(
            method_key,
            dino_features,
            vjepa2_features,
            vlm_features,
            cluster_to_indices,
            device,
        )

        if not sim_matrices:
            print(f"    WARNING: No valid clusters for {method_display}")
            continue

        # Build cluster label arrays aligned to valid indices
        method_cluster_labels: dict[int, np.ndarray] = {}
        for cid, valid_indices in cluster_valid.items():
            method_cluster_labels[cid] = np.array(
                [eval_segments[i].label for i in valid_indices]
            )

        # Run retrieval evaluation
        results = evaluate_retrieval(
            method_display,
            sim_matrices,
            method_cluster_labels,
            cluster_valid,
            K_VALUES,
        )
        all_results[method_key] = results

        # Print per-method summary
        n_q = results["n_queries"]
        mrr_info = results.get("mrr", {})
        mrr_val = (
            mrr_info.get("mean", float("nan"))
            if isinstance(mrr_info, dict)
            else float("nan")
        )
        print(f"    Queries: {n_q}  |  MRR: {mrr_val:.3f}")
        for k in K_VALUES:
            ap_info = results.get(f"ap@{k}", {})
            r_info = results.get(f"recall@{k}", {})
            ndcg_info = results.get(f"ndcg@{k}", {})
            ap_val = (
                ap_info.get("mean", float("nan"))
                if isinstance(ap_info, dict)
                else float("nan")
            )
            r_val = (
                r_info.get("mean", float("nan"))
                if isinstance(r_info, dict)
                else float("nan")
            )
            ndcg_val = (
                ndcg_info.get("mean", float("nan"))
                if isinstance(ndcg_info, dict)
                else float("nan")
            )
            print(
                f"    K={k:<3d}  mAP@K={ap_val:.3f}  "
                f"R@K={r_val:.3f}  NDCG@K={ndcg_val:.3f}"
            )

    # ------------------------------------------------------------------
    # Step 8: Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print_summary_table(all_results, method_order, display_names)

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    out_dir = hdd_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_protocol_results.json"

    # Convert numpy types for JSON serialization
    def _to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "protocol": "retrieval",
        "k_values": K_VALUES,
        "context_sec": args.context_sec,
        "max_clusters": args.max_clusters,
        "n_segments": len(eval_segments),
        "n_clusters": len(mixed),
        "skip_vjepa2": args.skip_vjepa2,
        "include_vlm": args.include_vlm,
        "vlm_family": args.vlm_family if args.include_vlm else None,
        "methods": {},
    }

    for method_key, results in all_results.items():
        method_out: dict[str, object] = {
            "display_name": display_names.get(method_key, method_key),
            "n_queries": _to_serializable(results["n_queries"]),
        }
        for metric_key, val in results.items():
            if metric_key == "n_queries":
                continue
            if isinstance(val, dict):
                method_out[metric_key] = {
                    k: _to_serializable(v) for k, v in val.items()
                }
            else:
                method_out[metric_key] = _to_serializable(val)
        output["methods"][method_key] = method_out

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BoF (BoT) -> Encoder-Sequence DTW reranker on Honda HDD.

Prescriptive follow-up to the feature-vs-comparator decomposition:
if 89 percent of the BoT-to-residual gap lives in the matching stage,
a two-stage pipeline (cheap pre-filter + expensive rerank) should
recover most of the full-DTW AP at a fraction of the compute.

Pipeline per query q over the full corpus of segments:
    1. Rank all other segments by BoT cosine similarity.
    2. Keep the top-k candidates (the "rerank set").
    3. Score rerank survivors with DTW; push non-survivors to the bottom.
    4. Compute AP, AUC, and recall@k = fraction of positive pairs where
       at least one end survives the top-k filter.

Two scoring strategies:
    - Survivor-only: survivors get DTW score, non-survivors get a sentinel
      below BoT's minimum (honest two-stage retriever evaluation).
    - RRF (Reciprocal Rank Fusion): rank-based fusion of BoT and DTW
      rankings, sidestepping the score-scale mismatch.

Usage (assumes cached features from eval_hdd_ordered_maxsim_vjepa2.py):
    python experiments/eval_hdd_bof_dtw_rerank.py --hdd-dir datasets/hdd

Output: datasets/hdd/bof_dtw_rerank_results.json
"""

from __future__ import annotations

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

from eval_hdd_intersections import (
    ManeuverSegment,
    bootstrap_ap,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_gps,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch


# ---------------------------------------------------------------------------
# Candidate set construction (global BoT top-k per query)
# ---------------------------------------------------------------------------


def build_bot_topk(
    mean_embs: torch.Tensor,
    k_values: list[int],
) -> dict[int, dict[int, set[int]]]:
    """For each k, compute the top-k BoT neighbor set for each segment.

    mean_embs: (N, D) L2-normalized mean embeddings for all eval segments.
    Returns: {k: {query_idx: set(candidate_idx, ...)}}, size <= k
    (self-matches excluded).
    """
    n = mean_embs.shape[0]
    # Full pairwise cosine similarity; L2-normalized already.
    sim = mean_embs @ mean_embs.T  # (N, N)
    # Mask self.
    sim.fill_diagonal_(-float("inf"))

    out: dict[int, dict[int, set[int]]] = {}
    max_k = max(k_values)
    max_k = min(max_k, n - 1)
    # One topk at the largest k; slice down for smaller k.
    topk_vals, topk_idx = torch.topk(sim, k=max_k, dim=1)  # (N, max_k)
    topk_idx = topk_idx.cpu().numpy()

    for k in k_values:
        k_eff = min(k, n - 1)
        neighbors: dict[int, set[int]] = {}
        for q in range(n):
            neighbors[q] = set(topk_idx[q, :k_eff].tolist())
        out[k] = neighbors
    return out


# ---------------------------------------------------------------------------
# Scoring strategies
# ---------------------------------------------------------------------------


def survivor_scores(
    pair_a: list[int],
    pair_b: list[int],
    bot_scores: np.ndarray,
    dtw_scores: np.ndarray,
    topk_neighbors: dict[int, set[int]],
) -> np.ndarray:
    """Survivors get DTW score; non-survivors get a sentinel below all DTW values.

    This is the honest two-stage retriever: non-survivors are strictly ranked
    last, so AP is capped by recall@k.
    """
    sentinel = float(dtw_scores.min()) - 1.0
    scores = np.full_like(bot_scores, sentinel)
    for i, (a, b) in enumerate(zip(pair_a, pair_b)):
        if b in topk_neighbors.get(a, set()) or a in topk_neighbors.get(b, set()):
            scores[i] = dtw_scores[i]
    return scores


def rrf_scores(
    pair_a: list[int],
    pair_b: list[int],
    bot_scores: np.ndarray,
    dtw_scores: np.ndarray,
    topk_neighbors: dict[int, set[int]],
    rrf_k: int = 60,
) -> np.ndarray:
    """Reciprocal Rank Fusion of BoT and DTW rankings.

    All pairs get a BoT rank contribution.  Survivor pairs additionally get a
    DTW rank contribution; non-survivors get only the BoT term.
    """
    n = len(bot_scores)
    # Ranks: 0 = highest score.
    bot_rank = np.empty(n, dtype=np.float64)
    bot_rank[np.argsort(-bot_scores)] = np.arange(n)

    dtw_rank = np.empty(n, dtype=np.float64)
    dtw_rank[np.argsort(-dtw_scores)] = np.arange(n)

    scores = 1.0 / (rrf_k + bot_rank)
    for i, (a, b) in enumerate(zip(pair_a, pair_b)):
        if b in topk_neighbors.get(a, set()) or a in topk_neighbors.get(b, set()):
            scores[i] += 1.0 / (rrf_k + dtw_rank[i])
    return scores


def recall_at_k(
    pair_a: list[int],
    pair_b: list[int],
    labels: np.ndarray,
    topk_neighbors: dict[int, set[int]],
) -> float:
    """Fraction of positive pairs where at least one end survives the top-k."""
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) == 0:
        return float("nan")
    hit = 0
    for i in pos_idx:
        a, b = pair_a[i], pair_b[i]
        if b in topk_neighbors.get(a, set()) or a in topk_neighbors.get(b, set()):
            hit += 1
    return hit / len(pos_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="BoF->DTW reranker on HDD (prescriptive follow-up to §3.2)."
    )
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-sec", type=float, default=3.0)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument(
        "--k-sweep",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 250, 500, 1000],
        help="Top-k candidate budgets to sweep.",
    )
    parser.add_argument(
        "--feature-cache",
        type=str,
        default="datasets/vjepa2_hdd_encoder_features.pt",
        help="Pre-extracted V-JEPA 2 encoder features (encoder_seq + mean_emb).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("BoF -> Encoder-Seq DTW RERANKER (HDD)")
    print("=" * 70)

    # -- Step 1: Rebuild the eval protocol from eval_hdd_encoder_seq.py --
    print("\nStep 1: Loading sessions and segments...")
    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for sid in tqdm(sorted(sessions.keys()), desc="Loading sessions"):
        info = sessions[sid]
        labels_arr = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        segs = extract_maneuver_segments(
            sid, labels_arr, gps_ts, gps_lats, gps_lngs,
            info["video_path"], info["video_start_unix"],
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
    n_segments = len(eval_segments)
    print(f"  {n_segments} segments in {len(mixed)} mixed clusters")

    # -- Step 2: Load cached V-JEPA 2 features --
    cache_path = project_root / args.feature_cache
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {cache_path}. Run "
            "eval_hdd_ordered_maxsim_vjepa2.py first to populate it."
        )
    print(f"\nStep 2: Loading cached features from {cache_path.name}...")
    ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
    features: dict[int, dict] = ckpt["features"]
    missing = [i for i in range(n_segments) if i not in features]
    if missing:
        print(
            f"  WARNING: {len(missing)} segments missing from cache "
            f"(first few: {missing[:5]}); they will be dropped."
        )
    valid_idx = [i for i in range(n_segments) if i in features]
    print(f"  Using {len(valid_idx)}/{n_segments} cached segments.")

    # -- Step 3: Enumerate in-cluster pairs (same protocol as §3.2) --
    print("\nStep 3: Enumerating in-cluster pairs...")
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_gt: list[int] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = [i for i in cluster_to_indices[cid] if i in features]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pair_a.append(indices[a])
                pair_b.append(indices[b])
                gt = 1 if eval_segments[indices[a]].label == eval_segments[indices[b]].label else 0
                pair_gt.append(gt)
    labels = np.array(pair_gt)
    n_pairs = len(labels)
    n_pos = int(labels.sum())
    print(f"  {n_pairs} pairs (pos={n_pos}, neg={n_pairs - n_pos})")

    # -- Step 4: Compute baseline scores (BoT cosine, full DTW) --
    print("\nStep 4: Computing baseline pairwise scores...")
    # BoT cosine (already L2-normalized in cache).
    mean_a = torch.stack([features[i]["mean_emb"] for i in pair_a]).to(device)
    mean_b = torch.stack([features[i]["mean_emb"] for i in pair_b]).to(device)
    bot_scores = (mean_a * mean_b).sum(dim=1).cpu().numpy()

    # Encoder-sequence DTW on all pairs (this is the full-DTW ceiling).
    enc_seqs_a = [features[i]["encoder_seq"].to(device) for i in pair_a]
    enc_seqs_b = [features[i]["encoder_seq"].to(device) for i in pair_b]
    t0 = time.time()
    enc_dists = dtw_distance_batch(enc_seqs_a, enc_seqs_b, normalize=True)
    dtw_scores = torch.exp(-enc_dists).cpu().numpy()
    print(f"  Full DTW over {n_pairs} pairs: {time.time() - t0:.1f}s")

    # -- Step 5: Build corpus-wide BoT top-k neighbor sets --
    print("\nStep 5: Building BoT top-k candidate sets over the full corpus...")
    mean_embs_corpus = torch.stack(
        [features[i]["mean_emb"] for i in range(n_segments) if i in features]
    )
    # Map: dense corpus row -> original segment index.
    corpus_idx = [i for i in range(n_segments) if i in features]
    corpus_to_dense = {seg_idx: dense for dense, seg_idx in enumerate(corpus_idx)}

    mean_embs_corpus = mean_embs_corpus.to(device)
    # Normalize in case cache wasn't strict (defensive).
    mean_embs_corpus = F.normalize(mean_embs_corpus, dim=-1)
    topk_dense = build_bot_topk(mean_embs_corpus, args.k_sweep)

    # Translate dense-index neighbor sets back to original segment indices.
    topk_by_k: dict[int, dict[int, set[int]]] = {}
    for k, neighbors_dense in topk_dense.items():
        neighbors_seg: dict[int, set[int]] = {}
        for q_dense, cand_set in neighbors_dense.items():
            q_seg = corpus_idx[q_dense]
            neighbors_seg[q_seg] = {corpus_idx[c] for c in cand_set}
        topk_by_k[k] = neighbors_seg

    # -- Step 6: Sweep k: composite AP / AUC / recall@k --
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Baselines first, for the table.
    bot_ap, bot_lo, bot_hi = bootstrap_ap(bot_scores, labels)
    bot_auc = roc_auc_score(labels, bot_scores)
    dtw_ap, dtw_lo, dtw_hi = bootstrap_ap(dtw_scores, labels)
    dtw_auc = roc_auc_score(labels, dtw_scores)
    print(
        f"  {'BoT only (k=1 floor)':<32s}"
        f"  AP={bot_ap:.4f} [{bot_lo:.4f},{bot_hi:.4f}]  AUC={bot_auc:.4f}"
    )
    print(
        f"  {'Full DTW (k=N ceiling)':<32s}"
        f"  AP={dtw_ap:.4f} [{dtw_lo:.4f},{dtw_hi:.4f}]  AUC={dtw_auc:.4f}"
    )
    print()

    results: dict[str, object] = {
        "n_pairs": int(n_pairs),
        "n_segments": int(len(valid_idx)),
        "n_positives": int(n_pos),
        "baselines": {
            "bot_only": {
                "ap": float(bot_ap),
                "ap_ci": [float(bot_lo), float(bot_hi)],
                "auc": float(bot_auc),
            },
            "full_encoder_seq_dtw": {
                "ap": float(dtw_ap),
                "ap_ci": [float(dtw_lo), float(dtw_hi)],
                "auc": float(dtw_auc),
            },
        },
        "rerank_sweep": [],
    }

    for k in args.k_sweep:
        neighbors = topk_by_k[k]
        r_at_k = recall_at_k(pair_a, pair_b, labels, neighbors)
        pos_idx = np.where(labels == 1)[0]
        pos_overrides = sum(
            1 for i in pos_idx
            if pair_b[i] in neighbors.get(pair_a[i], set())
            or pair_a[i] in neighbors.get(pair_b[i], set())
        )

        # Survivor-only scoring.
        surv = survivor_scores(pair_a, pair_b, bot_scores, dtw_scores, neighbors)
        surv_ap, surv_lo, surv_hi = bootstrap_ap(surv, labels)
        surv_auc = roc_auc_score(labels, surv)

        # RRF scoring.
        rrf = rrf_scores(pair_a, pair_b, bot_scores, dtw_scores, neighbors)
        rrf_ap, rrf_lo, rrf_hi = bootstrap_ap(rrf, labels)
        rrf_auc = roc_auc_score(labels, rrf)

        label = f"k={k:<4d}"
        print(
            f"  {label}  recall={r_at_k:.3f}  "
            f"survivor AP={surv_ap:.4f} [{surv_lo:.4f},{surv_hi:.4f}]  "
            f"RRF AP={rrf_ap:.4f} [{rrf_lo:.4f},{rrf_hi:.4f}]  "
            f"(pos_rerank={pos_overrides}/{n_pos})"
        )
        results["rerank_sweep"].append({
            "k": int(k),
            "recall_at_k": float(r_at_k),
            "positives_reranked": int(pos_overrides),
            "survivor": {
                "ap": float(surv_ap),
                "ap_ci": [float(surv_lo), float(surv_hi)],
                "auc": float(surv_auc),
            },
            "rrf": {
                "ap": float(rrf_ap),
                "ap_ci": [float(rrf_lo), float(rrf_hi)],
                "auc": float(rrf_auc),
            },
        })

    # -- Step 7: Save --
    out_path = hdd_dir / "bof_dtw_rerank_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

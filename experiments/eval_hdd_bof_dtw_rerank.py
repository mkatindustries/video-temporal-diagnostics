#!/usr/bin/env python3
"""Directed BoT-to-DTW retrieval evaluation on Honda HDD.

For each query segment, the full corpus is the gallery. A candidate is relevant
only when it belongs to the same GPS intersection cluster and has the same
maneuver label. BoT retrieves a directional top-k candidate set; encoder-token
DTW reranks only those candidates. Metrics are macro-averaged per query and
cluster-bootstrap confidence intervals account for queries that share an
intersection.

This supersedes the earlier unordered-pair protocol, which used an OR over the
two retrieval directions and omitted out-of-cluster gallery negatives.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from common import (
    ManeuverSegment,
    cluster_intersections,
    discover_sessions,
    extract_maneuver_segments,
    filter_mixed_clusters,
    load_gps,
)
from tqdm import tqdm

from video_retrieval.diagnostics.statistics import truncated_average_precision
from video_retrieval.fingerprints.dtw import dtw_distance_batch


def reciprocal_rank(relevant: np.ndarray) -> float:
    hits = np.flatnonzero(relevant)
    return 0.0 if len(hits) == 0 else float(1.0 / (hits[0] + 1))


def cluster_bootstrap_mean(
    values: np.ndarray,
    query_clusters: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Macro mean and percentile CI from intersection-cluster resampling."""
    point = float(np.mean(values))
    clusters = np.unique(query_clusters)
    by_cluster = {cluster: values[query_clusters == cluster] for cluster in clusters}
    rng = np.random.RandomState(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for sample_idx in range(n_resamples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        samples[sample_idx] = np.mean(np.concatenate([by_cluster[c] for c in selected]))
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(
    values: list[float],
    query_clusters: list[int],
    n_resamples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    point, low, high = cluster_bootstrap_mean(
        np.asarray(values), np.asarray(query_clusters), n_resamples, seed
    )
    return {"mean": point, "cluster_ci": [low, high]}


def load_evaluation_segments(
    hdd_dir: Path,
    max_clusters: int,
) -> tuple[list[ManeuverSegment], dict[int, list[int]], dict[int, int]]:
    sessions = discover_sessions(hdd_dir)
    all_segments: list[ManeuverSegment] = []
    for session_id in tqdm(sorted(sessions), desc="Loading HDD metadata"):
        info = sessions[session_id]
        labels = np.load(info["label_path"])
        try:
            gps_ts, gps_lats, gps_lngs = load_gps(info["gps_path"])
        except Exception:
            continue
        all_segments.extend(
            extract_maneuver_segments(
                session_id,
                labels,
                gps_ts,
                gps_lats,
                gps_lngs,
                info["video_path"],
                info["video_start_unix"],
            )
        )

    mixed = filter_mixed_clusters(cluster_intersections(all_segments), max_clusters=max_clusters)
    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    segment_to_cluster: dict[int, int] = {}
    for cluster_id, segments in mixed.items():
        for segment in segments:
            index = len(eval_segments)
            eval_segments.append(segment)
            cluster_to_indices[cluster_id].append(index)
            segment_to_cluster[index] = cluster_id
    return eval_segments, dict(cluster_to_indices), segment_to_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdd-dir", default="datasets/hdd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--k-sweep", type=int, nargs="+", default=[10, 25, 50, 100, 250])
    parser.add_argument(
        "--feature-cache",
        default="datasets/hdd/vjepa2_encoder_features.pt",
        help="Cache containing per-segment mean_emb and encoder_seq tensors.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-dtw",
        action="store_true",
        help="Also rank the full gallery by encoder-sequence DTW.",
    )
    parser.add_argument(
        "--dtw-batch-size",
        type=int,
        default=256,
        help="Candidate batch size for full-gallery DTW.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    cache_path = project_root / args.feature_cache
    device = torch.device(args.device)

    eval_segments, cluster_to_indices, segment_to_cluster = load_evaluation_segments(
        hdd_dir, args.max_clusters
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {cache_path}; run eval_hdd_encoder_seq.py first"
        )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    features: dict[int, dict] = cache["features"]
    corpus_indices = [idx for idx in range(len(eval_segments)) if idx in features]
    if len(corpus_indices) < 2:
        raise ValueError("feature cache contains fewer than two evaluation segments")

    dense_to_segment = corpus_indices
    segment_to_dense = {segment: dense for dense, segment in enumerate(dense_to_segment)}
    mean_embeddings = F.normalize(
        torch.stack([features[idx]["mean_emb"] for idx in dense_to_segment]).to(device),
        dim=-1,
    )
    bot_similarity = mean_embeddings @ mean_embeddings.T
    bot_similarity.fill_diagonal_(-float("inf"))

    max_k = min(max(args.k_sweep), len(corpus_indices) - 1)
    top_scores, top_dense = torch.topk(bot_similarity, k=max_k, dim=1)
    top_scores = top_scores.cpu().numpy()
    top_dense = top_dense.cpu().numpy()

    metric_values: dict[str, dict[int, dict[str, list[float]]]] = {
        "bot": defaultdict(lambda: defaultdict(list)),
        "dtw_rerank": defaultdict(lambda: defaultdict(list)),
    }
    query_clusters: list[int] = []
    full_bot_ap: list[float] = []
    full_bot_mrr: list[float] = []
    full_dtw_ap: list[float] = []
    full_dtw_mrr: list[float] = []
    evaluated_queries: list[int] = []

    for query_dense, query_segment in enumerate(tqdm(dense_to_segment, desc="Queries")):
        query_cluster = segment_to_cluster[query_segment]
        query_label = eval_segments[query_segment].label
        relevant_segments = {
            candidate
            for candidate in cluster_to_indices[query_cluster]
            if candidate != query_segment
            and candidate in segment_to_dense
            and eval_segments[candidate].label == query_label
        }
        if not relevant_segments:
            continue

        candidate_dense = top_dense[query_dense]
        candidate_segments = np.asarray([dense_to_segment[idx] for idx in candidate_dense])
        candidate_relevance = np.isin(candidate_segments, list(relevant_segments)).astype(np.int64)

        query_sequence = features[query_segment]["encoder_seq"].to(device)
        candidate_sequences = [
            features[idx]["encoder_seq"].to(device) for idx in candidate_segments
        ]
        distances = (
            dtw_distance_batch(
                [query_sequence] * len(candidate_sequences),
                candidate_sequences,
                normalize=True,
            )
            .cpu()
            .numpy()
        )

        all_order = torch.argsort(bot_similarity[query_dense], descending=True).cpu().numpy()
        all_order = all_order[all_order != query_dense]
        all_segments = [dense_to_segment[idx] for idx in all_order]
        all_relevance = np.isin(all_segments, list(relevant_segments)).astype(np.int64)
        full_bot_ap.append(truncated_average_precision(all_relevance, len(relevant_segments)))
        full_bot_mrr.append(reciprocal_rank(all_relevance))

        if args.full_dtw:
            full_distances = []
            for start in range(0, len(all_segments), args.dtw_batch_size):
                batch_segments = all_segments[start : start + args.dtw_batch_size]
                batch_sequences = [
                    features[idx]["encoder_seq"].to(device) for idx in batch_segments
                ]
                full_distances.append(
                    dtw_distance_batch(
                        [query_sequence] * len(batch_sequences),
                        batch_sequences,
                        normalize=True,
                    ).cpu()
                )
            full_dtw_order = torch.argsort(torch.cat(full_distances)).numpy()
            full_dtw_relevance = all_relevance[full_dtw_order]
            full_dtw_ap.append(
                truncated_average_precision(full_dtw_relevance, len(relevant_segments))
            )
            full_dtw_mrr.append(reciprocal_rank(full_dtw_relevance))

        for k_requested in sorted(set(args.k_sweep)):
            k = min(k_requested, max_k)
            bot_relevance = candidate_relevance[:k]
            rerank_order = np.argsort(distances[:k])
            reranked_relevance = bot_relevance[rerank_order]

            for method, relevance in (("bot", bot_relevance), ("dtw_rerank", reranked_relevance)):
                metric_values[method][k_requested]["ap"].append(
                    truncated_average_precision(relevance, len(relevant_segments))
                )
                metric_values[method][k_requested]["recall"].append(
                    float(relevance.sum() / len(relevant_segments))
                )
                metric_values[method][k_requested]["mrr"].append(reciprocal_rank(relevance))

        query_clusters.append(query_cluster)
        evaluated_queries.append(query_segment)

    if not evaluated_queries:
        raise ValueError("no query has a relevant gallery item")

    results: dict[str, object] = {
        "protocol": {
            "directional": True,
            "gallery": "all cached HDD evaluation segments except the query",
            "relevance": "same GPS cluster and same maneuver label",
            "uncertainty": "intersection-cluster bootstrap over query metrics",
        },
        "n_segments": len(corpus_indices),
        "n_queries": len(evaluated_queries),
        "n_clusters": len(set(query_clusters)),
        "bot_full_gallery": {
            "map": summarize(full_bot_ap, query_clusters, args.n_bootstrap, args.seed),
            "mrr": summarize(full_bot_mrr, query_clusters, args.n_bootstrap, args.seed),
        },
        "k_sweep": {},
    }

    if args.full_dtw:
        results["encoder_seq_dtw_full_gallery"] = {
            "map": summarize(full_dtw_ap, query_clusters, args.n_bootstrap, args.seed),
            "mrr": summarize(full_dtw_mrr, query_clusters, args.n_bootstrap, args.seed),
        }

    for k in sorted(set(args.k_sweep)):
        k_result: dict[str, object] = {"dtw_ops_per_query": min(k, max_k)}
        for method in ("bot", "dtw_rerank"):
            k_result[method] = {
                metric: summarize(values, query_clusters, args.n_bootstrap, args.seed)
                for metric, values in metric_values[method][k].items()
            }
        results["k_sweep"][str(k)] = k_result  # type: ignore[index]

    output_path = hdd_dir / "bof_dtw_directed_rerank_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

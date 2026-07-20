#!/usr/bin/env python3
"""Directed full-gallery retrieval, cascade, and held-out fusion on nuScenes.

Applies the Honda HDD protocol (eval_hdd_bof_dtw_rerank.py + eval_hdd_fusion.py)
to nuScenes to test whether HDD's conditional-versus-global reversal generalizes:

- relevance = same intersection cluster and same maneuver label;
- full-evaluation-gallery BoT and encoder-sequence DTW (query-macro mAP/MRR);
- BoT -> DTW cascade sweep (truncated AP@k / recall@k / MRR);
- leakage-safe leave-one-cluster-out score fusion, held-out clusters excluded
  from tuning queries and galleries, evaluated against the full evaluation gallery.

Segments/clusters/labels are rebuilt with the exact parameters used to build the
cached V-JEPA 2 features (version, max-clusters, min-segment-duration, DBSCAN
eps=30 m / min_samples=2) so positional feature-cache indices stay aligned. No
GPU feature extraction is performed; only full-gallery DTW is computed on GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from eval_hdd_bof_dtw_rerank import summarize
from eval_hdd_fusion import (
    ScoreCache,
    build_score_cache,
    cache_metadata_matches,
    label_ids,
    resolve_path,
)
from eval_nuscenes_intersections import (
    ManeuverSegment,
    cluster_intersections,
    filter_mixed_clusters,
    load_can_bus,
    load_nuscenes_metadata,
    segment_maneuvers,
)
from tqdm import tqdm

from video_retrieval.diagnostics.fusion import (
    OUTCOME_CATEGORIES,
    bot_dtw_cascade,
    evaluate_queries,
    leave_one_cluster_out_alpha,
    paired_cluster_bootstrap_mean_difference,
    ranked_outcome_composition,
)

# DBSCAN parameters are fixed to match eval_nuscenes_intersections' feature build.
DBSCAN_EPS_M = 30.0
DBSCAN_MIN_SAMPLES = 2


def build_eval_segments(
    data_dir: Path,
    version: str,
    max_clusters: int,
    min_segment_duration: float,
) -> tuple[list[ManeuverSegment], dict[int, list[int]]]:
    """Rebuild the nuScenes evaluation segments in feature-cache index order.

    Mirrors eval_nuscenes_intersections.main() steps 1-4 exactly (minus keyframe
    assignment, which only affects feature extraction) so segment index i matches
    the cached feature key i.
    """
    metadata = load_nuscenes_metadata(data_dir, version)
    can_dir = data_dir / "can_bus" / "can_bus"

    all_segments: list[ManeuverSegment] = []
    for scene in tqdm(metadata.scenes, desc="Rebuilding nuScenes segments"):
        scene_name = scene["name"]
        can_data = load_can_bus(can_dir, scene_name)
        if can_data is None:
            continue
        timestamps, steering, yaw_rates, positions, speeds = can_data
        all_segments.extend(
            segment_maneuvers(
                timestamps,
                steering,
                yaw_rates,
                positions,
                speeds,
                scene_name,
                min_duration=min_segment_duration,
            )
        )

    clusters = cluster_intersections(
        all_segments, eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES
    )
    mixed = filter_mixed_clusters(clusters, max_clusters=max_clusters)

    eval_segments: list[ManeuverSegment] = []
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for cid, segs in mixed.items():
        for seg in segs:
            idx = len(eval_segments)
            eval_segments.append(seg)
            cluster_to_indices[cid].append(idx)
    return eval_segments, cluster_to_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval",
                        choices=["v1.0-mini", "v1.0-trainval"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--min-segment-duration", type=float, default=2.0)
    parser.add_argument(
        "--feature-cache",
        default=None,
        help="V-JEPA 2 cache (default: <nuscenes-dir>/feature_cache/nuscenes_vjepa2_<version>.pt)",
    )
    parser.add_argument(
        "--dist-cache",
        default=None,
        help="Reusable BoT/DTW score-matrix cache (default under <nuscenes-dir>/feature_cache).",
    )
    parser.add_argument("--rebuild-dist-cache", action="store_true")
    parser.add_argument("--k-sweep", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--composition-k", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--dtw-batch-size", type=int, default=256)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.dtw_batch_size <= 0:
        parser.error("--dtw-batch-size must be positive")

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.nuscenes_dir).expanduser()
    device = torch.device(args.device)
    feature_cache_path = (
        resolve_path(project_root, args.feature_cache)
        if args.feature_cache
        else data_dir / "feature_cache" / f"nuscenes_vjepa2_{args.version}.pt"
    )
    dist_cache_path = (
        resolve_path(project_root, args.dist_cache)
        if args.dist_cache
        else data_dir / "feature_cache" / f"nuscenes_fusion_score_cache_{args.version}.pt"
    )
    output_path = (
        resolve_path(project_root, args.output)
        if args.output
        else data_dir / "fusion_results.json"
    )

    # 1. Rebuild segments in feature-cache index order.
    eval_segments, cluster_to_indices = build_eval_segments(
        data_dir, args.version, args.max_clusters, args.min_segment_duration
    )
    if not eval_segments:
        raise ValueError("no mixed-cluster evaluation segments were rebuilt")
    segment_to_cluster: dict[int, int] = {
        idx: cid for cid, indices in cluster_to_indices.items() for idx in indices
    }

    # 2. Load cached V-JEPA 2 features (index-keyed: {idx: {mean_emb, encoder_seq, ...}}).
    if not feature_cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {feature_cache_path}; "
            "run eval_nuscenes_intersections.py (with V-JEPA 2) first"
        )
    features: dict[int, dict] = torch.load(
        feature_cache_path, map_location="cpu", weights_only=False
    )
    if any("encoder_seq" not in v or "mean_emb" not in v for v in features.values()):
        raise ValueError("feature cache lacks mean_emb/encoder_seq required for BoT+DTW")
    # filter_mixed_clusters sorts by size and takes the top max_clusters, so a smaller
    # max_clusters yields a deterministic prefix whose indices still address the cache
    # built at the full max_clusters. A rebuild larger than the cache means the params
    # do not match the cache build.
    if len(eval_segments) > len(features):
        raise ValueError(
            f"rebuilt {len(eval_segments)} segments but the feature cache has only "
            f"{len(features)}; segment build is misaligned with the cache "
            "(check --version/--max-clusters/--min-segment-duration)"
        )

    dense_to_segment = [idx for idx in range(len(eval_segments)) if idx in features]
    if len(dense_to_segment) < 2:
        raise ValueError("feature cache contains fewer than two evaluation segments")
    if len(dense_to_segment) != len(eval_segments):
        print(
            f"Note: {len(eval_segments) - len(dense_to_segment)} rebuilt segment(s) "
            "absent from the feature cache (extraction gaps); excluded."
        )
    print(
        f"Rebuilt {len(eval_segments)} segments; {len(dense_to_segment)} present in cache "
        f"({len(features)} cached)."
    )

    # 3. Relevance = same cluster AND same maneuver label (excluding self).
    clusters = np.asarray(
        [segment_to_cluster[segment] for segment in dense_to_segment], dtype=np.int64
    )
    labels = label_ids([eval_segments[segment].label for segment in dense_to_segment])
    n_dense = len(dense_to_segment)
    relevance = (
        (clusters[:, None] == clusters[None, :])
        & (labels[:, None] == labels[None, :])
        & ~np.eye(n_dense, dtype=bool)
    )
    query_indices = np.flatnonzero(relevance.any(axis=1)).astype(np.int64)
    if len(np.unique(clusters[query_indices])) < 2:
        raise ValueError("fusion evaluation requires at least two eligible clusters")

    # 4. Build or reuse the BoT/DTW score-matrix cache (only DTW needs GPU).
    # Features are rekeyed to contiguous dense positions [0..n_dense-1], so the
    # score cache is validated against those positions, not original segment ids.
    dense_positions = list(range(n_dense))
    score_cache = None
    if dist_cache_path.exists() and not args.rebuild_dist_cache:
        raw = torch.load(dist_cache_path, map_location="cpu", weights_only=True)
        required_keys = set(ScoreCache.__required_keys__)
        candidate = cast(ScoreCache, raw)
        if (
            isinstance(raw, dict)
            and required_keys.issubset(raw)
            and cache_metadata_matches(
                candidate, dense_positions, clusters, labels, feature_cache_path
            )
        ):
            score_cache = candidate
            print(f"Loaded validated score cache: {dist_cache_path}")
        else:
            print(f"Ignoring stale/incompatible score cache: {dist_cache_path}")
    if score_cache is None:
        dense_features = {
            dense: features[segment] for dense, segment in enumerate(dense_to_segment)
        }
        score_cache = build_score_cache(
            dense_features,
            dense_positions,
            np.asarray(dense_positions, dtype=np.int64),
            clusters,
            labels,
            device,
            args.dtw_batch_size,
            feature_cache_path,
        )
        dist_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(score_cache, dist_cache_path)
        print(f"Saved score cache: {dist_cache_path}")

    bot_similarity = score_cache["bot_similarity"].numpy().astype(np.float64)
    dtw_distance = score_cache["dtw_distance"].numpy().astype(np.float64)
    for query_idx in query_indices:
        valid = np.arange(n_dense) != query_idx
        if not np.all(np.isfinite(dtw_distance[query_idx, valid])):
            raise ValueError(f"score cache has an incomplete DTW row for query {query_idx}")

    # 5. Full-gallery baselines, cascade sweep, and leakage-safe LOCO fusion.
    bot_ap, bot_mrr = evaluate_queries(
        bot_similarity, dtw_distance, relevance, clusters, query_indices, alpha=1.0
    )
    dtw_ap, dtw_mrr = evaluate_queries(
        bot_similarity, dtw_distance, relevance, clusters, query_indices, alpha=0.0
    )
    cascade = bot_dtw_cascade(
        bot_similarity, dtw_distance, relevance, query_indices, args.k_sweep
    )
    composition_k = sorted({int(k) for k in args.composition_k})
    outcome_by_method = {
        "bot": ranked_outcome_composition(
            bot_similarity, clusters, labels, query_indices, composition_k
        ),
        "encoder_seq_dtw": ranked_outcome_composition(
            -dtw_distance, clusters, labels, query_indices, composition_k
        ),
    }
    alpha_grid = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    loco = leave_one_cluster_out_alpha(
        bot_similarity, dtw_distance, relevance, clusters, query_indices, alpha_grid
    )
    fused_ap = np.asarray(loco["ap"])
    fused_mrr = np.asarray(loco["mrr"])
    query_clusters = np.asarray(loco["query_clusters"])
    qc_list = query_clusters.tolist()
    folds = loco["folds"]
    selected = np.asarray([fold["alpha"] for fold in folds], dtype=np.float64)

    def cluster_ci(values: np.ndarray) -> dict:
        return summarize(values.tolist(), qc_list, args.n_bootstrap, args.seed)

    def summarize_outcomes() -> dict[str, object]:
        methods: dict[str, object] = {}
        for method, by_k in outcome_by_method.items():
            methods[method] = {
                str(k): {
                    category: cluster_ci(by_k[k][category])
                    for category in OUTCOME_CATEGORIES
                }
                for k in composition_k
            }
        paired = {
            str(k): {
                category: paired_cluster_bootstrap_mean_difference(
                    outcome_by_method["encoder_seq_dtw"][k][category],
                    outcome_by_method["bot"][k][category],
                    query_clusters,
                    args.n_bootstrap,
                    args.seed,
                )
                for category in OUTCOME_CATEGORIES
            }
            for k in composition_k
        }
        return {
            "categories": {
                "relevant": "same intersection cluster and same maneuver label",
                "same_cluster_wrong_label": (
                    "same intersection cluster and different maneuver label"
                ),
                "wrong_cluster": "different intersection cluster",
            },
            "methods": methods,
            "paired_encoder_seq_dtw_minus_bot": paired,
        }

    k_sweep_out: dict[str, object] = {}
    for k in sorted({int(k) for k in args.k_sweep}):
        k_sweep_out[str(k)] = {
            method: {
                metric: cluster_ci(cascade[k][method][metric])
                for metric in ("ap", "recall", "mrr")
            }
            for method in ("bot", "dtw_rerank")
        }

    results: dict[str, object] = {
        "protocol": {
            "dataset": "nuScenes",
            "version": args.version,
            "directional": True,
            "gallery": "all cached nuScenes evaluation segments except the query",
            "relevance": "same intersection cluster and same maneuver label",
            "outcome_composition": (
                "top-k fractions split into relevant, same-cluster wrong-label, "
                "and wrong-cluster candidates"
            ),
            "cascade": (
                "BoT top-k then encoder-sequence-DTW rerank; AP@k truncated by total relevant"
            ),
            "fusion": "alpha*z(BoT cosine) + (1-alpha)*z(-DTW distance), per query",
            "selection": (
                "leave-one-intersection-out; held-out cluster excluded from tuning "
                "queries and tuning galleries; evaluated on the full evaluation gallery"
            ),
            "uncertainty": (
                "intersection-cluster bootstrap over fixed out-of-fold query metrics"
            ),
        },
        "n_segments": n_dense,
        "n_queries": len(query_indices),
        "n_clusters": int(len(np.unique(query_clusters))),
        "alpha_grid": alpha_grid.tolist(),
        "bot_full_gallery": {"map": cluster_ci(bot_ap), "mrr": cluster_ci(bot_mrr)},
        "encoder_seq_dtw_full_gallery": {
            "map": cluster_ci(dtw_ap),
            "mrr": cluster_ci(dtw_mrr),
        },
        "k_sweep": k_sweep_out,
        "fused_loco": {"map": cluster_ci(fused_ap), "mrr": cluster_ci(fused_mrr)},
        "paired_map_differences": {
            "fused_minus_bot": paired_cluster_bootstrap_mean_difference(
                fused_ap, bot_ap, query_clusters, args.n_bootstrap, args.seed
            ),
            "fused_minus_encoder_seq_dtw": paired_cluster_bootstrap_mean_difference(
                fused_ap, dtw_ap, query_clusters, args.n_bootstrap, args.seed
            ),
            "encoder_seq_dtw_minus_bot": paired_cluster_bootstrap_mean_difference(
                dtw_ap, bot_ap, query_clusters, args.n_bootstrap, args.seed
            ),
        },
        "ranked_outcome_composition": summarize_outcomes(),
        "alpha_selected": {
            "folds": folds,
            "median": float(np.median(selected)),
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
            "counts": {
                f"{alpha:.2f}": int(np.sum(np.isclose(selected, alpha)))
                for alpha in alpha_grid
                if np.any(np.isclose(selected, alpha))
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

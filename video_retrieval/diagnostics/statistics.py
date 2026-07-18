"""Statistical helpers for retrieval diagnostics."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.metrics import average_precision_score


class PairedBootstrapResult(TypedDict):
    ap_a: float
    ap_b: float
    difference_a_minus_b: float
    ci: list[float]
    bootstrap_probability_a_gt_b: float
    n_clusters: int


def truncated_average_precision(relevant: np.ndarray, total_relevant: int) -> float:
    """Compute AP for a finite ranking, counting unreturned relevant items as misses."""
    if total_relevant <= 0:
        return float("nan")
    hit_positions = np.flatnonzero(relevant)
    if len(hit_positions) == 0:
        return 0.0
    precisions = np.arange(1, len(hit_positions) + 1) / (hit_positions + 1)
    return float(precisions.sum() / total_relevant)


def cluster_bootstrap_ap(
    cluster_scores: dict[int, tuple[np.ndarray, np.ndarray]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Block-bootstrap pooled AP by cluster."""
    all_scores = np.concatenate([scores for scores, _ in cluster_scores.values()])
    all_labels = np.concatenate([labels for _, labels in cluster_scores.values()])
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return float("nan"), float("nan"), float("nan")

    point_ap = float(average_precision_score(all_labels, all_scores))
    cluster_ids = list(cluster_scores)
    rng = np.random.RandomState(seed)
    boot_aps: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        boot_scores = np.concatenate([cluster_scores[c][0] for c in sampled])
        boot_labels = np.concatenate([cluster_scores[c][1] for c in sampled])
        if boot_labels.sum() == 0 or boot_labels.sum() == len(boot_labels):
            boot_aps.append(point_ap)
        else:
            boot_aps.append(float(average_precision_score(boot_labels, boot_scores)))

    return (
        point_ap,
        float(np.percentile(boot_aps, 2.5)),
        float(np.percentile(boot_aps, 97.5)),
    )


def paired_cluster_bootstrap_ap_difference(
    scores_a: dict[int, tuple[np.ndarray, np.ndarray]],
    scores_b: dict[int, tuple[np.ndarray, np.ndarray]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Bootstrap AP(A) minus AP(B) by resampling shared clusters."""
    cluster_ids = sorted(set(scores_a) & set(scores_b))
    if not cluster_ids:
        raise ValueError("Methods have no shared clusters")

    for cluster_id in cluster_ids:
        a_scores, a_labels = scores_a[cluster_id]
        b_scores, b_labels = scores_b[cluster_id]
        if len(a_scores) != len(b_scores) or not np.array_equal(a_labels, b_labels):
            raise ValueError(f"Pair mismatch in cluster {cluster_id}")

    def pooled_ap(
        method_scores: dict[int, tuple[np.ndarray, np.ndarray]],
        sampled_ids: list[int] | np.ndarray,
    ) -> float:
        scores = np.concatenate([method_scores[c][0] for c in sampled_ids])
        labels = np.concatenate([method_scores[c][1] for c in sampled_ids])
        return float(average_precision_score(labels, scores))

    point_a = pooled_ap(scores_a, cluster_ids)
    point_b = pooled_ap(scores_b, cluster_ids)
    rng = np.random.RandomState(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        differences[index] = pooled_ap(scores_a, sampled) - pooled_ap(scores_b, sampled)

    return {
        "ap_a": point_a,
        "ap_b": point_b,
        "difference_a_minus_b": point_a - point_b,
        "ci": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "bootstrap_probability_a_gt_b": float(np.mean(differences > 0)),
        "n_clusters": len(cluster_ids),
    }

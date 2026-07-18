"""Feature-vs-comparator factorial for pairwise retrieval evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from sklearn.metrics import average_precision_score
from torch import Tensor


def feature_comparator_decomposition(
    feature_sets: Mapping[str, Mapping[str, Tensor]],
    comparators: Mapping[str, Callable[[Tensor, Tensor], float]],
    pairs: list[tuple[str, str, int]],
) -> dict:
    """Evaluate every feature/comparator combination on one shared pair set.

    Pairs missing from any feature set are removed before scoring so all cells
    are directly comparable. The returned AP matrix supports controlled rows
    (fixed features, varied comparator) and columns (fixed comparator, varied
    features); causal attribution still requires an appropriate experimental
    design and uncertainty analysis.
    """
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not comparators:
        raise ValueError("comparators must not be empty")

    shared_pairs = [
        pair
        for pair in pairs
        if all(pair[0] in features and pair[1] in features for features in feature_sets.values())
    ]
    if not shared_pairs:
        raise ValueError("no pairs are present in every feature set")

    labels = [label for _, _, label in shared_pairs]
    if len(set(labels)) < 2:
        raise ValueError("average precision requires both positive and negative pairs")

    ap_matrix: dict[str, dict[str, float]] = {}
    for feature_name, features in feature_sets.items():
        ap_matrix[feature_name] = {}
        for comparator_name, comparator in comparators.items():
            scores = [comparator(features[id_a], features[id_b]) for id_a, id_b, _ in shared_pairs]
            ap_matrix[feature_name][comparator_name] = float(
                average_precision_score(labels, scores)
            )

    return {
        "n_pairs": len(shared_pairs),
        "n_positive": int(sum(labels)),
        "feature_sets": list(feature_sets),
        "comparators": list(comparators),
        "ap_matrix": ap_matrix,
    }

"""Regression test for location-aware nuScenes intersection clustering.

nuScenes ego-pose (x, y) is a per-map local frame: boston-seaport and the
singapore-* maps do not share a coordinate system, and their local ranges
can numerically overlap. cluster_intersections(..., locations=...) must
never let DBSCAN merge segments from two different locations, even when
their raw coordinates fall within eps of each other.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from eval_nuscenes_intersections import (  # noqa: E402
    ManeuverSegment,
    cluster_intersections,
    flatten_cluster_segments,
    load_feature_cache,
    save_feature_cache,
    segment_order_fingerprint,
)


def _segment(scene_name: str, label: int, x: float, y: float) -> ManeuverSegment:
    return ManeuverSegment(
        scene_name=scene_name,
        label=label,
        start_ts=0.0,
        end_ts=1.0,
        midpoint_x=x,
        midpoint_y=y,
    )


def test_naive_clustering_can_merge_different_locations():
    """Documents the bug: without `locations`, coincidentally-close
    coordinates from two different maps land in one cluster."""
    segments = [
        _segment("scene-boston-1", 2, 100.0, 100.0),
        _segment("scene-boston-2", 3, 100.5, 100.5),
        _segment("scene-singapore-1", 2, 100.2, 100.2),
        _segment("scene-singapore-2", 3, 100.7, 100.7),
    ]
    clusters = cluster_intersections(segments, eps=30.0, min_samples=2)
    assert len(clusters) == 1
    (only_cluster,) = clusters.values()
    assert len(only_cluster) == 4


def test_location_aware_clustering_never_mixes_locations():
    segments = [
        _segment("scene-boston-1", 2, 100.0, 100.0),
        _segment("scene-boston-2", 3, 100.5, 100.5),
        _segment("scene-singapore-1", 2, 100.2, 100.2),
        _segment("scene-singapore-2", 3, 100.7, 100.7),
    ]
    locations = ["boston-seaport", "boston-seaport", "singapore-onenorth", "singapore-onenorth"]
    clusters = cluster_intersections(segments, eps=30.0, min_samples=2, locations=locations)

    assert len(clusters) == 2
    for segs in clusters.values():
        scene_names = {seg.scene_name for seg in segs}
        assert scene_names <= {"scene-boston-1", "scene-boston-2"} or scene_names <= {
            "scene-singapore-1",
            "scene-singapore-2",
        }


def test_location_aware_clustering_still_separates_far_apart_segments():
    segments = [
        _segment("scene-a", 2, 0.0, 0.0),
        _segment("scene-b", 3, 5.0, 5.0),
        _segment("scene-c", 2, 5000.0, 5000.0),
        _segment("scene-d", 3, 5005.0, 5005.0),
    ]
    locations = ["boston-seaport"] * 4
    clusters = cluster_intersections(segments, eps=30.0, min_samples=2, locations=locations)
    assert len(clusters) == 2


def test_locations_length_mismatch_raises():
    segments = [_segment("scene-a", 2, 0.0, 0.0)]
    try:
        cluster_intersections(segments, locations=["boston-seaport", "extra"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched locations length")


def test_flattened_segment_order_and_fingerprint_are_deterministic():
    first = _segment("scene-a", 2, 0.0, 0.0)
    second = _segment("scene-b", 3, 1.0, 1.0)
    third = _segment("scene-c", 2, 2.0, 2.0)
    segments, cluster_to_indices = flatten_cluster_segments(
        {7: [first, second], 3: [third]}
    )

    assert segments == [first, second, third]
    assert cluster_to_indices == {7: [0, 1], 3: [2]}
    assert segment_order_fingerprint(segments) == segment_order_fingerprint(
        [first, second, third]
    )
    assert segment_order_fingerprint(segments) != segment_order_fingerprint(
        [second, first, third]
    )


def test_feature_cache_rejects_stale_segment_order(tmp_path):
    segments = [
        _segment("scene-a", 2, 0.0, 0.0),
        _segment("scene-b", 3, 1.0, 1.0),
    ]
    fingerprint = segment_order_fingerprint(segments)
    cache_path = tmp_path / "features.pt"
    features = {0: {"mean_emb": torch.tensor([1.0])}}

    save_feature_cache(features, cache_path, fingerprint)

    loaded = load_feature_cache(cache_path, fingerprint)
    assert loaded is not None
    assert torch.equal(loaded[0]["mean_emb"], features[0]["mean_emb"])
    assert load_feature_cache(
        cache_path,
        segment_order_fingerprint(list(reversed(segments))),
    ) is None


def test_feature_cache_rejects_legacy_positional_payload(tmp_path):
    cache_path = tmp_path / "legacy.pt"
    torch.save({0: {"mean_emb": torch.tensor([1.0])}}, cache_path)

    assert load_feature_cache(cache_path, "expected-fingerprint") is None

#!/usr/bin/env python3
"""Verify the location-aware DBSCAN fix in eval_nuscenes_intersections.py.

scripts/audit_nuscenes_cluster_locations.py proved the *old* clustering (no
location awareness) mixes cities in 10/50 retained clusters. This script runs
the *fixed* clustering (cluster_intersections(..., locations=...)) over the
same segments and reports the resulting cluster/segment counts plus a purity
check, so you can see the new scale before committing to a full GPU rerun.

No feature extraction, no GPU.

Usage:
    python scripts/verify_nuscenes_cluster_fix.py \
        --nuscenes-dir /path/to/nuscenes --version v1.0-trainval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from eval_nuscenes_intersections import (  # noqa: E402
    cluster_intersections,
    filter_mixed_clusters,
    load_can_bus,
    load_nuscenes_metadata,
    load_scene_locations,
    segment_maneuvers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-dir", type=Path, required=True)
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        choices=["v1.0-mini", "v1.0-trainval"],
    )
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--min-segment-duration", type=float, default=2.0)
    args = parser.parse_args()

    data_dir = args.nuscenes_dir
    metadata = load_nuscenes_metadata(data_dir, args.version)
    scene_location = load_scene_locations(data_dir, args.version, metadata.scenes)

    can_dir = data_dir / "can_bus" / "can_bus"
    all_segments = []
    for scene in metadata.scenes:
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
                min_duration=args.min_segment_duration,
            )
        )

    if not all_segments:
        print("VERDICT: ERROR -- no maneuver segments found; check the data and version.")
        return 2

    segment_locations = [scene_location[s.scene_name] for s in all_segments]

    old_clusters = cluster_intersections(all_segments, eps=30.0, min_samples=2)
    old_mixed = filter_mixed_clusters(old_clusters, max_clusters=args.max_clusters)
    old_n_segs = sum(len(v) for v in old_mixed.values())

    new_clusters = cluster_intersections(
        all_segments, eps=30.0, min_samples=2, locations=segment_locations
    )
    new_mixed = filter_mixed_clusters(new_clusters, max_clusters=args.max_clusters)
    new_n_segs = sum(len(v) for v in new_mixed.values())

    n_bad = 0
    for segs in new_mixed.values():
        if len({scene_location[s.scene_name] for s in segs}) > 1:
            n_bad += 1

    print(f"OLD (no location awareness): {len(old_mixed)} clusters, {old_n_segs} segments")
    print(f"NEW (location-aware):        {len(new_mixed)} clusters, {new_n_segs} segments")
    print(f"NEW clusters that still mix locations: {n_bad}")

    if n_bad:
        print("VERDICT: FAIL -- the fix did not eliminate cross-location clusters.")
        return 1

    print("VERDICT: PASS -- every fixed cluster is confined to a single map/location.")
    if new_n_segs < old_n_segs:
        pct = 100.0 * (old_n_segs - new_n_segs) / old_n_segs
        print(
            f"Note: {pct:.0f}% fewer evaluation segments than the flawed run "
            f"({old_n_segs} -> {new_n_segs}) -- expect the paper's nuScenes numbers "
            "to shift after a full rerun."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

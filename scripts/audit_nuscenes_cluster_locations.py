#!/usr/bin/env python3
"""Metadata-only audit: does any nuScenes intersection cluster mix scenes from
different maps (cities)?

experiments/eval_nuscenes_intersections.py clusters maneuver segments by raw
ego-pose (x, y) via DBSCAN, without ever loading log.json's `location` field.
nuScenes ego-pose coordinates are in a per-map local frame (boston-seaport vs.
the three singapore-* maps), so numerically close (x, y) values do not imply
the same real-world intersection across maps. This script reuses the exact
same segmentation/clustering call as the real run (same eps, min_samples,
min_duration) so it checks the same clusters behind the Video4Real headline
results, then reports how many scene locations appear in each cluster.

No feature extraction, no GPU: only scene.json, log.json, sample.json,
ego_pose.json, and can_bus/ are needed.

Usage:
    python scripts/audit_nuscenes_cluster_locations.py \
        --nuscenes-dir /path/to/nuscenes --version v1.0-trainval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from eval_nuscenes_intersections import (  # noqa: E402
    cluster_intersections,
    filter_mixed_clusters,
    load_can_bus,
    load_nuscenes_metadata,
    segment_maneuvers,
)


def main() -> None:
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

    with open(data_dir / args.version / "log.json") as f:
        logs = json.load(f)
    log_location = {log["token"]: log["location"] for log in logs}
    scene_location = {
        scene["name"]: log_location.get(scene["log_token"], "UNKNOWN")
        for scene in metadata.scenes
    }

    can_dir = data_dir / "can_bus" / "can_bus"
    all_segments = []
    for scene in metadata.scenes:
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
            min_duration=args.min_segment_duration,
        )
        all_segments.extend(segs)

    if not all_segments:
        print("No maneuver segments found; check --nuscenes-dir/--version.")
        return

    # Same clustering call as experiments/eval_nuscenes_intersections.py main().
    clusters = cluster_intersections(all_segments, eps=30.0, min_samples=2)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)

    print(f"{len(mixed)} mixed (left+right) clusters, top {args.max_clusters} by size\n")
    n_bad = 0
    for cid, segs in sorted(mixed.items(), key=lambda kv: -len(kv[1])):
        locs = sorted({scene_location.get(s.scene_name, "UNKNOWN") for s in segs})
        flag = "  <-- MIXES CITIES/MAPS" if len(locs) > 1 else ""
        print(f"cluster {cid:>4}  n_segments={len(segs):<4}  locations={locs}{flag}")
        if len(locs) > 1:
            n_bad += 1

    print(f"\n{n_bad}/{len(mixed)} clusters mix more than one map/location.")
    if n_bad:
        print(
            "VERDICT: FAIL -- the reported nuScenes clusters conflate different "
            "real-world locations. 'Same cluster' does not mean 'same intersection' "
            "until re-clustered per-location."
        )
    else:
        print(
            "VERDICT: PASS -- every evaluated cluster is confined to a single "
            "map/location."
        )


if __name__ == "__main__":
    main()

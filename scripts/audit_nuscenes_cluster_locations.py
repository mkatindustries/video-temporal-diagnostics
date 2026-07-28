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

No feature extraction, no GPU: the nuScenes metadata tables loaded by the
evaluator (scene, log, sample, sample_data, ego_pose, sensor, and
calibrated_sensor) plus can_bus/ are needed.

Usage:
    python scripts/audit_nuscenes_cluster_locations.py \
        --nuscenes-dir /path/to/nuscenes --version v1.0-trainval
"""

from __future__ import annotations

import argparse
import json
import math
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

PAPER_VERSION = "v1.0-trainval"
PAPER_MAX_CLUSTERS = 50
PAPER_MIN_SEGMENT_DURATION = 2.0
PAPER_N_SEGMENTS = 264


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
    if args.max_clusters < 1:
        parser.error("--max-clusters must be positive")
    if not math.isfinite(args.min_segment_duration) or args.min_segment_duration <= 0:
        parser.error("--min-segment-duration must be positive")

    data_dir = args.nuscenes_dir
    metadata = load_nuscenes_metadata(data_dir, args.version)

    with open(data_dir / args.version / "log.json") as f:
        logs = json.load(f)
    log_location = {log["token"]: log["location"] for log in logs}
    scene_location = {
        scene["name"]: log_location.get(scene["log_token"])
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
        print("VERDICT: ERROR -- no maneuver segments found; check the data and version.")
        return 2

    # Same clustering call as experiments/eval_nuscenes_intersections.py main().
    clusters = cluster_intersections(all_segments, eps=30.0, min_samples=2)
    mixed = filter_mixed_clusters(clusters, max_clusters=args.max_clusters)
    if not mixed:
        print("VERDICT: ERROR -- reconstruction produced no mixed clusters.")
        return 2

    selected_segments = [segment for segments in mixed.values() for segment in segments]
    selected_scene_names = sorted({segment.scene_name for segment in selected_segments})
    selected_scene_locations: dict[str, str] = {}
    missing_locations = []
    for scene_name in selected_scene_names:
        location = scene_location.get(scene_name)
        if not isinstance(location, str) or not location.strip():
            missing_locations.append(scene_name)
        else:
            selected_scene_locations[scene_name] = location
    if missing_locations:
        print(
            "VERDICT: ERROR -- missing log.location for evaluated scenes: "
            + ", ".join(missing_locations)
        )
        return 2

    paper_protocol = (
        args.version == PAPER_VERSION
        and args.max_clusters == PAPER_MAX_CLUSTERS
        and math.isclose(
            args.min_segment_duration,
            PAPER_MIN_SEGMENT_DURATION,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if paper_protocol and (
        len(mixed) != PAPER_MAX_CLUSTERS
        or len(selected_segments) != PAPER_N_SEGMENTS
    ):
        print(
            "VERDICT: ERROR -- reconstruction does not match the paper run: "
            f"expected {PAPER_MAX_CLUSTERS} clusters/{PAPER_N_SEGMENTS} segments, "
            f"got {len(mixed)} clusters/{len(selected_segments)} segments."
        )
        return 2

    print(
        f"Reconstructed {len(mixed)} mixed clusters with "
        f"{len(selected_segments)} segments (top {args.max_clusters} by size).\n"
    )
    n_bad = 0
    for cid, segs in sorted(mixed.items(), key=lambda kv: -len(kv[1])):
        locs = sorted({selected_scene_locations[s.scene_name] for s in segs})
        flag = "  <-- MIXES MAPS" if len(locs) > 1 else ""
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
        return 1
    else:
        print(
            "VERDICT: PASS -- every evaluated cluster is confined to a single "
            "map/location."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

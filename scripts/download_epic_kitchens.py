#!/usr/bin/env python3
"""Download and prepare EPIC-Kitchens-100 data for temporal order experiments.

Downloads EPIC_100_train.csv annotations from the official GitHub repo,
analyzes narrations to build temporal order sequences (2-3 consecutive
narrations, 6-15s total duration), and downloads video subsets.

EPIC-Kitchens-100 videos are hosted at the University of Bristol and
require an academic license. This script handles annotation download
and sequence selection; video download requires manual steps or the
official epic_downloader tool.

Target directory: ./data/epic_kitchens/

Usage:
    # Step 1: Download annotations + build sequence manifest (no videos)
    python scripts/download_epic_kitchens.py --epic-dir datasets/epic_kitchens --setup

    # Step 2: Verify manifest + print stats
    python scripts/download_epic_kitchens.py --epic-dir datasets/epic_kitchens --verify

    # Step 3: Download videos (explicit selection required)
    python scripts/download_epic_kitchens.py --epic-dir datasets/epic_kitchens \\
        --download-videos --participants P01 P22 --max-videos 30

    # Or: download top-K participants by sequence count
    python scripts/download_epic_kitchens.py --epic-dir datasets/epic_kitchens \\
        --download-videos --top-k-participants 5
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EPIC-Kitchens-100 annotations on GitHub
ANNOTATIONS_BASE = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations"
    "/master"
)
TRAIN_CSV_URL = f"{ANNOTATIONS_BASE}/EPIC_100_train.csv"

# Sequence selection parameters
MIN_SEQUENCE_DURATION = 6.0  # seconds
MAX_SEQUENCE_DURATION = 15.0  # seconds
MIN_NARRATIONS = 2
MAX_NARRATIONS = 3

# Manifest versioning
MANIFEST_VERSION = "v1"
MANIFEST_SEED = 42

# Rough estimate: EPIC untrimmed video ~1.5 GB average
ESTIMATED_GB_PER_VIDEO = 1.5


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)


def _time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.ss to seconds."""
    parts = time_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def _manifest_filename() -> str:
    """Stable, versioned manifest filename."""
    return (
        f"temporal_order_sequences_{MANIFEST_VERSION}"
        f"_len{int(MIN_SEQUENCE_DURATION)}-{int(MAX_SEQUENCE_DURATION)}"
        f"_narr{MIN_NARRATIONS}-{MAX_NARRATIONS}"
        f"_seed{MANIFEST_SEED}.json"
    )


def download_annotations(epic_dir: Path) -> Path:
    """Download EPIC_100_train.csv from GitHub."""
    ann_dir = epic_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ann_dir / "EPIC_100_train.csv"
    if csv_path.exists():
        print(f"  Annotations already at {csv_path}")
        return csv_path

    print(f"  Downloading EPIC_100_train.csv...")
    urlretrieve(TRAIN_CSV_URL, csv_path, reporthook=_download_progress)
    print()
    print(f"  Saved to {csv_path}")
    return csv_path


def parse_annotations(csv_path: Path) -> list[dict]:
    """Parse EPIC_100_train.csv into list of narration dicts."""
    narrations = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            narrations.append(
                {
                    "narration_id": row["narration_id"],
                    "participant_id": row["participant_id"],
                    "video_id": row["video_id"],
                    "start_sec": _time_to_seconds(row["start_timestamp"]),
                    "stop_sec": _time_to_seconds(row["stop_timestamp"]),
                    "narration": row["narration"],
                    "verb_class": int(row["verb_class"]),
                    "noun_class": int(row["noun_class"]),
                }
            )
    return narrations


def build_sequences(narrations: list[dict]) -> list[dict]:
    """Build temporal order sequences from consecutive narrations.

    Groups narrations by (participant_id, video_id), sorts by start time,
    and finds windows of 2-3 consecutive narrations with 6-15s total duration.

    Each entry includes all fields needed for the eval script:
    participant_id, video_id, t0, t1, narration_ids, duration_sec,
    recommended_frame_count, split.
    """
    # Group by (participant, video)
    groups = defaultdict(list)
    for n in narrations:
        key = (n["participant_id"], n["video_id"])
        groups[key].append(n)

    # Sort each group by start time
    for key in groups:
        groups[key].sort(key=lambda x: x["start_sec"])

    sequences = []
    for (pid, vid), group_narrations in groups.items():
        # Sliding window over consecutive narrations
        for win_size in [MIN_NARRATIONS, MAX_NARRATIONS]:
            for i in range(len(group_narrations) - win_size + 1):
                window = group_narrations[i : i + win_size]
                start = window[0]["start_sec"]
                end = window[-1]["stop_sec"]
                duration = end - start

                if MIN_SEQUENCE_DURATION <= duration <= MAX_SEQUENCE_DURATION:
                    # Recommended frame count at 3fps
                    rec_frames = int(round(duration * 3.0))

                    sequences.append(
                        {
                            "sequence_id": f"{vid}_{i}_{win_size}",
                            "participant_id": pid,
                            "video_id": vid,
                            "start_sec": round(start, 3),
                            "stop_sec": round(end, 3),
                            "duration_sec": round(duration, 3),
                            "n_narrations": win_size,
                            "narrations": [n["narration"] for n in window],
                            "narration_ids": [n["narration_id"] for n in window],
                            "recommended_frame_count": rec_frames,
                            "split": "train",  # All from EPIC_100_train.csv
                        }
                    )

    # Deduplicate: a 2-narration window might overlap with a 3-narration window
    # Keep unique by (video_id, start_sec) — prefer longer windows
    seen = {}
    for seq in sorted(sequences, key=lambda s: -s["n_narrations"]):
        key = (seq["video_id"], seq["start_sec"])
        if key not in seen:
            seen[key] = seq
    sequences = list(seen.values())

    return sorted(
        sequences, key=lambda s: (s["participant_id"], s["video_id"], s["start_sec"])
    )


def print_stats(sequences: list[dict]) -> None:
    """Print detailed dry-run stats (no downloads triggered)."""
    print(f"\n  {'=' * 50}")
    print(f"  SEQUENCE STATS (dry-run)")
    print(f"  {'=' * 50}")
    print(f"  Total sequences: {len(sequences)}")

    # Per-participant breakdown
    part_counts = defaultdict(int)
    part_videos = defaultdict(set)
    for seq in sequences:
        part_counts[seq["participant_id"]] += 1
        part_videos[seq["participant_id"]].add(seq["video_id"])

    print(f"\n  Sequences per participant:")
    for pid in sorted(part_counts, key=lambda p: -part_counts[p]):
        n_seq = part_counts[pid]
        n_vid = len(part_videos[pid])
        est_gb = n_vid * ESTIMATED_GB_PER_VIDEO
        print(
            f"    {pid}: {n_seq:3d} sequences across {n_vid:2d} videos "
            f"(~{est_gb:.1f} GB)"
        )

    # Duration histogram (text-based)
    durations = [s["duration_sec"] for s in sequences]
    bins = [6, 8, 10, 12, 15]
    print(f"\n  Duration histogram (6-15s):")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = sum(1 for d in durations if lo <= d < hi)
        bar = "#" * (count // 2)
        print(f"    {lo:2d}-{hi:2d}s: {count:4d} {bar}")

    avg_dur = sum(durations) / len(durations)
    print(
        f"    mean={avg_dur:.1f}s, min={min(durations):.1f}s, max={max(durations):.1f}s"
    )

    # Top-K participant estimates
    ranked = sorted(part_counts.items(), key=lambda x: -x[1])
    print(f"\n  Estimated disk usage for top-K participants:")
    cumul_videos = set()
    cumul_seqs = 0
    for k, (pid, n_seq) in enumerate(ranked[:10], 1):
        cumul_videos |= part_videos[pid]
        cumul_seqs += n_seq
        est_gb = len(cumul_videos) * ESTIMATED_GB_PER_VIDEO
        print(
            f"    top-{k}: {cumul_seqs:4d} sequences, "
            f"{len(cumul_videos):3d} videos, ~{est_gb:.0f} GB"
        )

    # All unique videos
    all_videos = set(s["video_id"] for s in sequences)
    print(
        f"\n  All participants: {len(all_videos)} unique videos, "
        f"~{len(all_videos) * ESTIMATED_GB_PER_VIDEO:.0f} GB"
    )


def setup(epic_dir: Path) -> None:
    """Download annotations + build sequence manifest (no video download)."""
    print("\n=== Setup ===")

    # Download annotations
    csv_path = download_annotations(epic_dir)

    # Parse
    print("  Parsing annotations...")
    narrations = parse_annotations(csv_path)
    print(f"  Total narrations: {len(narrations)}")

    participants = set(n["participant_id"] for n in narrations)
    videos = set(n["video_id"] for n in narrations)
    print(f"  Participants: {len(participants)}, Videos: {len(videos)}")

    # Build ALL sequences (manifest covers everything; filtering at download/eval time)
    print("  Building temporal order sequences...")
    sequences = build_sequences(narrations)
    print(f"  Total sequences (6-15s, 2-3 narrations): {len(sequences)}")

    # Print stats
    print_stats(sequences)

    # Save versioned manifest (all sequences, not pre-filtered)
    manifest = {
        "metadata": {
            "source": "EPIC-Kitchens-100",
            "version": MANIFEST_VERSION,
            "seed": MANIFEST_SEED,
            "min_duration_sec": MIN_SEQUENCE_DURATION,
            "max_duration_sec": MAX_SEQUENCE_DURATION,
            "min_narrations": MIN_NARRATIONS,
            "max_narrations": MAX_NARRATIONS,
            "n_sequences": len(sequences),
            "n_videos": len(set(s["video_id"] for s in sequences)),
            "n_participants": len(set(s["participant_id"] for s in sequences)),
        },
        "sequences": sequences,
    }

    manifest_path = epic_dir / "annotations" / _manifest_filename()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to {manifest_path}")

    # Also save as a stable symlink for the eval script
    latest_link = epic_dir / "annotations" / "temporal_order_sequences.json"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(manifest_path.name)
    print(f"  Symlinked: temporal_order_sequences.json -> {manifest_path.name}")

    print("\n  Next: --verify, then --download-videos with explicit selection")


def _resolve_participants(
    manifest: dict,
    participants: list[str] | None,
    top_k: int | None,
) -> list[str]:
    """Resolve which participants to download."""
    sequences = manifest["sequences"]

    # Count per participant
    part_counts = defaultdict(int)
    for seq in sequences:
        part_counts[seq["participant_id"]] += 1

    if participants:
        # Explicit list
        unknown = [p for p in participants if p not in part_counts]
        if unknown:
            print(f"  WARNING: Unknown participants (no sequences): {unknown}")
        return [p for p in participants if p in part_counts]

    if top_k:
        ranked = sorted(part_counts.items(), key=lambda x: -x[1])
        return [pid for pid, _ in ranked[:top_k]]

    # Default: error
    print("  ERROR: --download-videos requires explicit selection.")
    print("  Use --participants P01 P22 ... or --top-k-participants 5")
    print("  Run --setup first to see participant stats.")
    sys.exit(1)


def download_videos(
    epic_dir: Path,
    participants: list[str] | None,
    top_k: int | None,
    max_videos: int | None,
) -> None:
    """Download videos for explicitly selected participants.

    EPIC-Kitchens videos are large (~1-2GB each, untrimmed kitchen recordings).
    Requires explicit participant selection to prevent accidental bulk downloads.
    """
    print("\n=== Download Videos ===")

    manifest_path = epic_dir / "annotations" / "temporal_order_sequences.json"
    if not manifest_path.exists():
        print("  ERROR: Run --setup first to generate sequence manifest")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    selected_pids = _resolve_participants(manifest, participants, top_k)
    print(f"  Selected participants: {', '.join(selected_pids)}")

    # Get videos for selected participants
    needed_videos = sorted(
        set(
            s["video_id"]
            for s in manifest["sequences"]
            if s["participant_id"] in set(selected_pids)
        )
    )

    if max_videos and len(needed_videos) > max_videos:
        print(f"  Capping at --max-videos={max_videos} (from {len(needed_videos)})")
        needed_videos = needed_videos[:max_videos]

    video_dir = epic_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    # Check which already exist
    existing = set()
    for vid in needed_videos:
        for ext in [".MP4", ".mp4"]:
            if (video_dir / f"{vid}{ext}").exists():
                existing.add(vid)
                break

    remaining = [v for v in needed_videos if v not in existing]
    est_gb = len(remaining) * ESTIMATED_GB_PER_VIDEO

    print(f"  Need {len(needed_videos)} videos, {len(existing)} already downloaded")
    print(f"  Remaining: {len(remaining)} videos (~{est_gb:.0f} GB)")

    if not remaining:
        print("  All videos already downloaded!")
        return

    # Try epic_downloader
    try:
        subprocess.run(["epic_downloader", "--help"], capture_output=True, check=True)
        has_downloader = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        has_downloader = False

    if has_downloader:
        parts = set(v.split("_")[0] for v in remaining)
        for pid in sorted(parts):
            pid_videos = [v for v in remaining if v.startswith(pid)]
            print(f"\n  Downloading {len(pid_videos)} videos for {pid}...")
            cmd = [
                "epic_downloader",
                "--output-path",
                str(video_dir),
                "--participants",
                pid,
                "--extension",
                "MP4",
                "--videos",
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  WARNING: Download failed for {pid}: {e}")
    else:
        print("\n  epic_downloader not found. Install with:")
        print("    pip install epic-kitchens-downloader")
        print("\n  Or download manually:")
        print(f"    Target directory: {video_dir}")
        for vid in remaining[:10]:
            pid = vid.split("_")[0]
            print(f"    - {vid}.MP4 (participant {pid})")
        if len(remaining) > 10:
            print(f"    ... and {len(remaining) - 10} more")
        print(f"\n  Full list of needed videos:")
        for vid in remaining:
            print(f"    {vid}.MP4")
        print(
            f"\n  Download from: https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m"
        )


def verify(epic_dir: Path) -> None:
    """Verify data completeness for temporal order experiments."""
    print("\n=== Verification ===")

    # Check annotations
    csv_path = epic_dir / "annotations" / "EPIC_100_train.csv"
    if csv_path.exists():
        narrations = parse_annotations(csv_path)
        print(f"  Annotations: {len(narrations)} narrations")
    else:
        print("  FAIL: EPIC_100_train.csv not found (run --setup)")
        return

    # Check manifest (versioned)
    manifest_path = epic_dir / "annotations" / "temporal_order_sequences.json"
    versioned_path = epic_dir / "annotations" / _manifest_filename()

    if versioned_path.exists():
        print(f"  Versioned manifest: {versioned_path.name}")
    elif manifest_path.exists():
        print(f"  Manifest: {manifest_path.name} (not versioned)")
    else:
        print("  FAIL: Sequence manifest not found (run --setup)")
        return

    # Use whichever exists
    load_path = versioned_path if versioned_path.exists() else manifest_path
    with open(load_path) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]
    meta = manifest["metadata"]
    n_seq = meta["n_sequences"]
    n_vid = meta.get("n_videos", len(set(s["video_id"] for s in sequences)))
    n_part = meta.get(
        "n_participants", len(set(s["participant_id"] for s in sequences))
    )
    print(f"  Sequences: {n_seq} across {n_vid} videos, {n_part} participants")

    if n_seq >= 150:
        print("  OK: Sufficient sequences (>=150)")
    else:
        print(f"  WARNING: Only {n_seq} sequences (target: >=150)")

    # Check videos
    video_dir = epic_dir / "videos"
    all_needed = sorted(set(s["video_id"] for s in sequences))
    found = 0
    missing = []
    for vid in all_needed:
        for ext in [".MP4", ".mp4"]:
            if (video_dir / f"{vid}{ext}").exists():
                found += 1
                break
        else:
            missing.append(vid)

    print(f"  Videos: {found}/{len(all_needed)} available")
    if missing:
        # Group missing by participant
        missing_by_part = defaultdict(list)
        for vid in missing:
            pid = vid.split("_")[0]
            missing_by_part[pid].append(vid)
        for pid in sorted(missing_by_part):
            vids = missing_by_part[pid]
            print(
                f"    {pid}: {len(vids)} missing ({', '.join(vids[:5])}"
                f"{'...' if len(vids) > 5 else ''})"
            )
    else:
        print("  OK: All videos available")

    # Count usable sequences
    available_videos = set(all_needed) - set(missing)
    usable = [s for s in sequences if s["video_id"] in available_videos]
    usable_parts = set(s["participant_id"] for s in usable)
    print(
        f"  Usable sequences (with video): {len(usable)}/{n_seq} "
        f"({len(usable_parts)} participants)"
    )

    # Print stats for usable set
    if usable:
        durations = [s["duration_sec"] for s in usable]
        print(
            f"  Duration: mean={sum(durations)/len(durations):.1f}s, "
            f"min={min(durations):.1f}s, max={max(durations):.1f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare EPIC-Kitchens-100 data"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default=None,
        required=True,
        help="Path to EPIC-Kitchens data directory",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download annotations + build sequence manifest",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download videos (requires --participants or --top-k-participants)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify data completeness"
    )

    # Video download selection
    parser.add_argument(
        "--participants",
        nargs="+",
        type=str,
        default=None,
        help="Explicit participant IDs to download (e.g. P01 P22)",
    )
    parser.add_argument(
        "--top-k-participants",
        type=int,
        default=None,
        help="Download top K participants by sequence count",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap total number of videos to download",
    )

    args = parser.parse_args()

    epic_dir = Path(args.epic_dir)
    epic_dir.mkdir(parents=True, exist_ok=True)

    if not (args.setup or args.download_videos or args.verify):
        print("No action specified. Use --setup, --download-videos, or --verify")
        parser.print_help()
        sys.exit(1)

    if args.setup:
        setup(epic_dir)
    if args.download_videos:
        download_videos(
            epic_dir, args.participants, args.top_k_participants, args.max_videos
        )
    if args.verify:
        verify(epic_dir)


if __name__ == "__main__":
    main()

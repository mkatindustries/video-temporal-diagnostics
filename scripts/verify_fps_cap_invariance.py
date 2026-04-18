#!/usr/bin/env python3
"""Verify that FPS caps ≥10 are no-ops for HDD V-JEPA 2 frame selection.

For each HDD maneuver segment, computes:
  - duration = (end_frame/3.0 + context_sec) - (start_frame/3.0 - context_sec)
  - target_fps = 64 / duration
  - Whether each fps_cap would actually reduce the sampling rate

No GPU or model loading needed — pure arithmetic on segment metadata.

Usage:
    python scripts/verify_fps_cap_invariance.py --hdd-dir /path/to/hdd
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Segment metadata (copied from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------

VJEPA2_NUM_FRAMES = 64
LABEL_FPS = 3.0  # HDD labels are at 3 fps


@dataclass
class ManeuverSegment:
    session_id: str
    label: int
    start_frame: int
    end_frame: int
    lat: float
    lng: float
    video_path: str
    video_start_unix: float


LABEL_NAMES = {1: "intersection_passing", 2: "left_turn", 3: "right_turn"}


# ---------------------------------------------------------------------------
# Session discovery (simplified from eval_hdd_intersections.py)
# ---------------------------------------------------------------------------


def discover_sessions(hdd_dir: Path) -> dict[str, dict]:
    """Find HDD sessions with labels and video.

    Labels are in hdd_dir/labels/target/{session_id}.npy (NOT inside session
    directories — the label dir is separate from the release dir).
    Videos are in hdd_dir/release_2019_07_08/{date}/{session}/camera/center/*.mp4.
    """
    import zoneinfo
    from datetime import datetime

    label_dir = hdd_dir / "labels" / "target"
    release_dir = hdd_dir / "release_2019_07_08"
    if not release_dir.exists():
        raise FileNotFoundError(f"HDD release dir not found: {release_dir}")

    # Index all label files by session ID
    label_files = {}
    for f in sorted(label_dir.glob("*.npy")):
        label_files[f.stem] = f

    sessions = {}
    for date_dir in sorted(release_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            if session_id not in label_files:
                continue

            # Find video
            video_dir = session_dir / "camera" / "center"
            if not video_dir.exists():
                continue
            video_files = list(video_dir.glob("*.mp4"))
            if not video_files:
                continue
            video_path = video_files[0]

            # Parse video start time from filename
            m = re.match(
                r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",
                video_path.name,
            )
            if not m:
                continue

            year, month, day, hour, minute, second = (int(g) for g in m.groups())
            tz = zoneinfo.ZoneInfo("America/Los_Angeles")
            dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
            video_start_unix = dt.timestamp()

            sessions[session_id] = {
                "label_path": str(label_files[session_id]),
                "video_path": str(video_path),
                "video_start_unix": video_start_unix,
            }

    return sessions


def extract_maneuver_segments(
    session_id: str,
    labels: np.ndarray,
    video_path: str,
    video_start_unix: float,
    target_labels: tuple[int, ...] = (1, 2, 3),
) -> list[ManeuverSegment]:
    """Extract contiguous maneuver segments from frame-level labels."""
    segments = []
    current_label: int | None = None
    start_frame = 0

    for i, lbl in enumerate(labels):
        lbl_int = int(lbl)
        if lbl_int in target_labels:
            if current_label != lbl_int:
                if current_label is not None and current_label in target_labels:
                    segments.append(
                        ManeuverSegment(
                            session_id=session_id,
                            label=current_label,
                            start_frame=start_frame,
                            end_frame=i,
                            lat=0.0,
                            lng=0.0,
                            video_path=video_path,
                            video_start_unix=video_start_unix,
                        )
                    )
                current_label = lbl_int
                start_frame = i
        else:
            if current_label is not None and current_label in target_labels:
                segments.append(
                    ManeuverSegment(
                        session_id=session_id,
                        label=current_label,
                        start_frame=start_frame,
                        end_frame=i,
                        lat=0.0,
                        lng=0.0,
                        video_path=video_path,
                        video_start_unix=video_start_unix,
                    )
                )
            current_label = lbl_int
            start_frame = i

    # Final segment
    if current_label is not None and current_label in target_labels:
        segments.append(
            ManeuverSegment(
                session_id=session_id,
                label=current_label,
                start_frame=start_frame,
                end_frame=len(labels),
                lat=0.0,
                lng=0.0,
                video_path=video_path,
                video_start_unix=video_start_unix,
            )
        )

    return segments


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify FPS cap invariance for HDD V-JEPA 2 frame selection"
    )
    parser.add_argument(
        "--hdd-dir",
        type=Path,
        default=None,
        required=True,
        help="Path to HDD dataset root",
    )
    parser.add_argument(
        "--context-sec",
        type=float,
        default=3.0,
        help="Context window added to each side of segment (seconds)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Max segments to analyze (0 = all)",
    )
    args = parser.parse_args()

    print(f"HDD directory: {args.hdd_dir}")
    print(f"Context window: ±{args.context_sec}s")
    print()

    # Discover sessions and extract segments
    sessions = discover_sessions(args.hdd_dir)
    print(f"Found {len(sessions)} sessions with labels + video")

    all_segments: list[ManeuverSegment] = []
    for session_id in sorted(sessions.keys()):
        info = sessions[session_id]
        labels = np.load(info["label_path"])
        segs = extract_maneuver_segments(
            session_id,
            labels,
            info["video_path"],
            info["video_start_unix"],
        )
        all_segments.extend(segs)

    print(f"Total maneuver segments: {len(all_segments)}")

    if args.max_segments > 0:
        all_segments = all_segments[: args.max_segments]
        print(f"Analyzing first {len(all_segments)} segments")
    print()

    # Compute target_fps for each segment
    fps_caps = [2.0, 5.0, 10.0, 15.0, 30.0]
    target_fps_values = []
    durations = []

    for seg in all_segments:
        start_sec = max(0.0, seg.start_frame / LABEL_FPS - args.context_sec)
        end_sec = seg.end_frame / LABEL_FPS + args.context_sec
        duration = end_sec - start_sec
        target_fps = VJEPA2_NUM_FRAMES / duration

        target_fps_values.append(target_fps)
        durations.append(duration)

    target_fps_arr = np.array(target_fps_values)
    durations_arr = np.array(durations)

    # --- Distribution summary ---
    print("=" * 70)
    print("SEGMENT DURATION & TARGET FPS DISTRIBUTION")
    print("=" * 70)
    print(f"  Segments analyzed: {len(target_fps_arr)}")
    print()
    print("  Duration (seconds):")
    print(f"    min    = {durations_arr.min():.2f}")
    print(f"    median = {np.median(durations_arr):.2f}")
    print(f"    mean   = {durations_arr.mean():.2f}")
    print(f"    max    = {durations_arr.max():.2f}")
    print()
    print("  Target FPS (= 64 / duration):")
    print(f"    min    = {target_fps_arr.min():.2f}")
    print(f"    median = {np.median(target_fps_arr):.2f}")
    print(f"    mean   = {target_fps_arr.mean():.2f}")
    print(f"    max    = {target_fps_arr.max():.2f}")
    print()

    # --- FPS cap activation analysis ---
    print("=" * 70)
    print("FPS CAP ACTIVATION ANALYSIS")
    print("=" * 70)
    print()
    print("A cap is 'active' when target_fps > cap (i.e., the cap actually")
    print("reduces the sampling rate). When inactive, the capped run produces")
    print("identical frame selection to the uncapped baseline.")
    print()
    print(f"  {'FPS Cap':>10}  {'% Active':>10}  {'# Active':>10}  {'# Inactive':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    for cap in fps_caps:
        active = np.sum(target_fps_arr > cap)
        inactive = len(target_fps_arr) - active
        pct = 100.0 * active / len(target_fps_arr)
        print(f"  {cap:>10.1f}  {pct:>9.1f}%  {active:>10d}  {inactive:>12d}")

    print()

    # --- Per-cap effective frame counts ---
    print("=" * 70)
    print("EFFECTIVE UNIQUE FRAMES PER CAP")
    print("=" * 70)
    print()
    print("Unique frames = min(64, floor(effective_fps × duration))")
    print(f"Padding frames = 64 - unique_frames")
    print()
    print(
        f"  {'FPS Cap':>10}  {'Mean Unique':>12}  {'Mean Dup%':>10}  {'Min Unique':>11}  {'Max Unique':>11}"
    )
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*11}  {'-'*11}")

    for cap in fps_caps + [None]:
        unique_counts = []
        for fps, dur in zip(target_fps_arr, durations_arr):
            eff_fps = min(fps, cap) if cap is not None else fps
            unique = min(VJEPA2_NUM_FRAMES, int(eff_fps * dur))
            unique = max(1, unique)
            unique_counts.append(unique)
        unique_arr = np.array(unique_counts)
        dup_pct = 100.0 * (1.0 - unique_arr / VJEPA2_NUM_FRAMES)
        cap_str = f"{cap:.1f}" if cap is not None else "None"
        print(
            f"  {cap_str:>10}  {unique_arr.mean():>12.1f}  {dup_pct.mean():>9.1f}%  {unique_arr.min():>11d}  {unique_arr.max():>11d}"
        )

    print()

    # --- Conclusion ---
    pct_below_10 = 100.0 * np.sum(target_fps_arr < 10.0) / len(target_fps_arr)
    pct_below_15 = 100.0 * np.sum(target_fps_arr < 15.0) / len(target_fps_arr)
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print(f"  {pct_below_10:.1f}% of segments have target_fps < 10.0")
    print(f"  {pct_below_15:.1f}% of segments have target_fps < 15.0")
    print()
    if pct_below_10 > 99.0:
        print("  ✓ CONFIRMED: FPS caps ≥10 are no-ops for virtually all segments.")
        print("    The identical AP values at caps 10/15/30/baseline are explained")
        print("    by frame selection invariance — the sampler never requests >10 FPS.")
    elif pct_below_10 > 90.0:
        print(
            f"  ~ MOSTLY CONFIRMED: {pct_below_10:.1f}% of segments unaffected by cap=10."
        )
        print("    A small fraction of short segments may be affected.")
    else:
        print(
            f"  ✗ NOT CONFIRMED: Only {pct_below_10:.1f}% of segments have target_fps < 10."
        )
        print("    The FPS cap plateau requires a different explanation.")
    print()


if __name__ == "__main__":
    main()

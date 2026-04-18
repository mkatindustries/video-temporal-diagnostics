#!/usr/bin/env python3
"""Quick HDD dataset stats: video durations and navigation event counts.

Usage:
    python scripts/hdd_quick_stats.py              # all sessions
    python scripts/hdd_quick_stats.py -n 10        # first 10
    python scripts/hdd_quick_stats.py -n 20 --sort duration
"""

import argparse
from pathlib import Path

import numpy as np

LABEL_NAMES = {
    0: "background",
    1: "intersection_pass",
    2: "left_turn",
    3: "right_turn",
    4: "left_lane_change",
    5: "right_lane_change",
    6: "left_lane_branch",
    7: "right_lane_branch",
    8: "crosswalk_pass",
    9: "railroad_pass",
    10: "merge",
    11: "u_turn",
}

# Navigation events = everything except background (0)
NAV_LABELS = {k: v for k, v in LABEL_NAMES.items() if k != 0}


def count_events(labels: np.ndarray) -> dict[int, int]:
    """Count contiguous segments (events) for each label value."""
    counts: dict[int, int] = {}
    prev = -1
    for lbl in labels:
        lbl = int(lbl)
        if lbl != prev and lbl in NAV_LABELS:
            counts[lbl] = counts.get(lbl, 0) + 1
        prev = lbl
    return counts


def main():
    parser = argparse.ArgumentParser(description="Quick HDD dataset stats")
    parser.add_argument("-n", type=int, default=None,
                        help="Number of sessions to show (default: all)")
    parser.add_argument("--hdd-dir", type=str, default="datasets/hdd")
    parser.add_argument("--sort", type=str, default="session",
                        choices=["session", "duration", "events"],
                        help="Sort order (default: session)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    hdd_dir = project_root / args.hdd_dir
    label_dir = hdd_dir / "labels" / "target"

    if not label_dir.exists():
        print(f"Label directory not found: {label_dir}")
        return

    label_files = sorted(label_dir.glob("*.npy"))
    print(f"Total sessions with labels: {len(label_files)}\n")

    rows = []
    totals: dict[int, int] = {}

    for f in label_files:
        labels = np.load(f)
        duration_sec = len(labels) / 3.0  # labels at 3 fps
        events = count_events(labels)
        total_nav = sum(events.values())

        rows.append({
            "session": f.stem,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "events": events,
            "total_nav": total_nav,
        })

        for k, v in events.items():
            totals[k] = totals.get(k, 0) + v

    # Sort
    if args.sort == "duration":
        rows.sort(key=lambda r: r["duration_sec"], reverse=True)
    elif args.sort == "events":
        rows.sort(key=lambda r: r["total_nav"], reverse=True)
    # else: already sorted by session name

    # Limit
    if args.n is not None:
        rows = rows[:args.n]

    # Compact event columns: only show labels that actually appear
    # pyrefly: ignore [not-iterable]
    active_labels = sorted({k for r in rows for k in r["events"]})

    # Header
    hdr = f"{'Session':<16s} {'Dur(min)':>8s}"
    for lbl in active_labels:
        col = NAV_LABELS[lbl][:10]
        hdr += f" {col:>10s}"
    hdr += f" {'TOTAL':>6s}"
    print(hdr)
    print("-" * len(hdr))

    # Rows
    total_dur = 0.0
    total_events_shown = 0
    # pyrefly: ignore [bad-assignment]
    for r in rows:
        line = f"{r['session']:<16s} {r['duration_min']:8.1f}"
        for lbl in active_labels:
            # pyrefly: ignore [missing-attribute]
            cnt = r["events"].get(lbl, 0)
            line += f" {cnt:10d}"
        line += f" {r['total_nav']:6d}"
        print(line)
        # pyrefly: ignore [unsupported-operation]
        total_dur += r["duration_min"]
        # pyrefly: ignore [unsupported-operation]
        total_events_shown += r["total_nav"]

    print("-" * len(hdr))

    # Totals row
    tot_line = f"{'TOTAL':<16s} {total_dur:8.1f}"
    for lbl in active_labels:
        tot_line += f" {totals.get(lbl, 0):10d}"
    tot_line += f" {total_events_shown:6d}"
    print(tot_line)

    # Summary
    print(f"\nShowing {len(rows)}/{len(label_files)} sessions")
    total_dur_all = sum(len(np.load(f)) / 3.0 / 60.0 for f in label_files)
    total_events_all = sum(totals.values())
    print(f"Dataset total: {total_dur_all:.1f} min across {len(label_files)} sessions, "
          f"{total_events_all} navigation events")


if __name__ == "__main__":
    main()

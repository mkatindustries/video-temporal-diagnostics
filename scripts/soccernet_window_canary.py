#!/usr/bin/env python3
"""Train-only window-policy canary for the SoccerNet replay-grounding positives.

Decides the frozen pre/post window used to extract the LIVE positive clip around
each event anchor. Predeclared candidate windows are extracted for a STRATIFIED
TRAIN-ONLY sample (across leagues x action-class x aired-live camera state) and
rendered as per-event montages for VISUAL coverage scoring. The smallest
class-agnostic window that meets the coverage threshold is then frozen into the
manifest's ``window_policy`` by hand after review.

Guardrails (locked with reviewer 2026-07-23):
  * Train split only — never inspect valid/test here.
  * No retrieval metrics inform this choice; it is a pure coverage decision.
  * "aired-live" = replay-type of the camera interval CONTAINING link.position
    (the corrected calc: real-time interval => action was on a live camera).

This does CPU video decode only (PyAV) — no features, no GPU.

Usage:
    python scripts/soccernet_window_canary.py --max-samples 48 --out-dir results/soccernet/canary
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_SN_DIR = Path(
    "/checkpoint/dream/arjangt/video_retrieval/datasets/soccernet_v2_replay_grounding"
)
# Predeclared candidate windows around the anchor, (pre_s, post_s). Frozen set.
CANDIDATE_WINDOWS = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (-2.0, 4.0)]
N_FRAMES = 8  # frames rendered per window (matches the SONAR2-PE 8-frame window)
THUMB_W = 160
ACTION_BUCKETS = {
    "Goal": "goal", "Foul": "foul", "Shots on target": "shot", "Shots off target": "shot",
    "Ball out of play": "ballout", "Offside": "offside",
}


def bucket(action: str) -> str:
    return ACTION_BUCKETS.get(action, "other")


def half_of(gt: str) -> str:
    return gt.split("-")[0].strip()


def aired_live_state(sn_dir: Path, game: str, half: str, anchor_ms: int) -> str:
    """Replay-type of the camera interval [prev_boundary, row) containing anchor_ms."""
    cam = json.loads((sn_dir / game / "Labels-cameras.json").read_text())["annotations"]
    rows = sorted(
        ((int(a.get("position", 0)), a) for a in cam if half_of(a["gameTime"]) == half),
        key=lambda x: x[0],
    )
    for j, (pos, a) in enumerate(rows):
        prev = rows[j - 1][0] if j > 0 else 0
        if prev <= anchor_ms < pos:
            return a.get("replay", "unknown")
    return "no-interval"


def extract_frames(video_path: Path, t0: float, t1: float, n: int) -> list[np.ndarray]:
    """Decode RGB frames in [t0, t1] and return n uniformly-spaced ones."""
    import av

    t0 = max(0.0, t0)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(t0 / stream.time_base), stream=stream, any_frame=False, backward=True)
        frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base)
            if t < t0:
                continue
            if t > t1:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [frames[i] for i in idx]


def montage(rows_of_frames: list[list[np.ndarray]], labels: list[str], out_path: Path) -> bool:
    """Grid: one row per candidate window, N_FRAMES columns. Returns True if written."""
    from PIL import Image, ImageDraw

    if not any(rows_of_frames):
        return False
    thumbs = []
    for row in rows_of_frames:
        trow = []
        for f in row:
            im = Image.fromarray(f)
            h = int(im.height * THUMB_W / im.width)
            trow.append(im.resize((THUMB_W, h)))
        thumbs.append(trow)
    cell_h = max((t[0].height for t in thumbs if t), default=90)
    pad, label_w = 3, 90
    grid = Image.new("RGB", (label_w + N_FRAMES * (THUMB_W + pad),
                             len(thumbs) * (cell_h + pad)), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for r, (trow, lab) in enumerate(zip(thumbs, labels)):
        y = r * (cell_h + pad)
        draw.text((4, y + cell_h // 2), lab, fill=(230, 230, 230))
        for c, im in enumerate(trow):
            grid.paste(im, (label_w + c * (THUMB_W + pad), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return True


def stratified_sample(manifest: dict, sn_dir: Path, max_samples: int, seed: int) -> list[dict]:
    """Balanced train-only sample over (league, action-bucket); records aired-live."""
    rng = random.Random(seed)
    ev_by_id = {e["event_id"]: e for e in manifest["events"] if e["split"] == "train"}
    # one representative primary query per event (for the replay-half; positives are live anyway)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in ev_by_id.values():
        league = e["game"].split("/")[0]
        strata[(league, bucket(e["action_label"]))].append(e)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    picked: list[dict] = []
    i = 0
    while len(picked) < max_samples and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            e = strata[k].pop()
            e = dict(e)
            e["aired_live"] = aired_live_state(sn_dir, e["game"], e["half"], int(e["anchor_ms"]))
            e["league"] = k[0]
            e["action_bucket"] = k[1]
            picked.append(e)
        i += 1
    return picked


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soccernet-dir", type=Path, default=DEFAULT_SN_DIR)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results/soccernet/canary"))
    p.add_argument("--max-samples", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest_path = args.manifest or (
        args.soccernet_dir / "replay_event_manifest_v1_seed42.json")
    manifest = json.loads(Path(manifest_path).read_text())

    samples = stratified_sample(manifest, args.soccernet_dir, args.max_samples, args.seed)
    print(f"sampled {len(samples)} TRAIN events across "
          f"{len({(s['league'], s['action_bucket']) for s in samples})} strata")
    records = []
    for n, e in enumerate(samples):
        video = args.soccernet_dir / e["game"] / f"{e['half']}_224p.mkv"
        anchor = int(e["anchor_ms"]) / 1000.0
        rows, labels = [], []
        for pre, post in CANDIDATE_WINDOWS:
            frames = extract_frames(video, anchor + pre, anchor + post, N_FRAMES)
            rows.append(frames)
            labels.append(f"[{pre:+.0f},{post:+.0f}]")
        stem = f"{n:03d}_{e['action_bucket']}_{e['aired_live'].replace('-', '')}"
        ok = montage(rows, labels, args.out_dir / f"{stem}.png")
        records.append({
            "event_id": e["event_id"], "league": e["league"],
            "action_label": e["action_label"], "action_bucket": e["action_bucket"],
            "aired_live": e["aired_live"], "anchor_s": anchor,
            "montage": f"{stem}.png" if ok else None,
        })
        if not ok:
            print(f"  WARN no frames decoded for {e['event_id']} ({video.name})")
    (args.out_dir / "canary_index.json").write_text(json.dumps({
        "candidate_windows_s": CANDIDATE_WINDOWS, "n_frames": N_FRAMES,
        "split": "train", "seed": args.seed, "samples": records,
    }, indent=2))
    print(f"wrote {len(records)} montages + index to {args.out_dir}")
    print("REVIEW: score per-window event coverage visually; freeze the smallest "
          "class-agnostic window meeting threshold into manifest.window_policy. "
          "Do NOT inspect valid/test or use retrieval metrics for this choice.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the immutable SoccerNet-v2 replay-grounding *event-retrieval* manifest.

The task (label-only here; features come later): a replay clip must be matched to
the live-action moment it depicts, against **same-match negatives** — other live
events in the identical broadcast (same pitch, teams, camera style). This is the
"smearing" test: can a representation localise the *event* rather than the scene.

Conventions locked with the reviewer (2026-07-23):

  * Replay query span = ``[previous camera-boundary, this replay row's position)``
    within the replay's own half; the linked row marks the replay END. The live
    ``link.position`` is a single ANCHOR (the pre/post window around it is a
    separately-frozen policy selected by the train-only canary; left null here).
  * Live event identity = exact ``(link.half, link.position)`` join. Multiple
    replay angles share the identical anchor, so they collapse to one event and
    become MULTI-POSITIVE siblings (never negatives). Near-but-not-equal anchors
    are reported as a diagnostic, not merged.
  * Visibility: only ``not shown`` is excluded. ``not applicable`` / ``default``
    are kept as a ``review`` cohort, ``visible`` is the ``primary`` cohort.
  * Same-half queries are the primary set; ``link.half != replay half`` (cross-half,
    e.g. half-time montages) is a separately-counted extension.
  * Bootstrap/group unit = match (``game``); positives within a query = its event.

Usage:
    python scripts/setup_soccernet.py                 # build manifest
    python scripts/setup_soccernet.py --verify-only   # integrity check only

Layout under --soccernet-dir (SoccerNet-v2 replay grounding download):
    download_manifest.json / _SUCCESS
    <league>/<season>/<match>/{1_224p.mkv,2_224p.mkv,Labels-cameras.json,...}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SN_DIR = Path("/path/to/soccernet_v2_replay_grounding")
MANIFEST_SCHEMA = "soccernet_replay_event_retrieval_v1"
EXCLUDE_VISIBILITY = {"not shown"}
REVIEW_VISIBILITY = {"not applicable", "default"}
NEAR_DUP_MS = 1000  # diagnostic only: anchors closer than this (same half, differing) are flagged


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def half_of(game_time: str) -> str:
    """'1 - 01:03' -> '1'."""
    return game_time.split("-")[0].strip()


def load_download_manifest(sn_dir: Path) -> tuple[dict, dict]:
    mp = sn_dir / "download_manifest.json"
    sp = sn_dir / "_SUCCESS"
    if not mp.exists():
        raise SystemExit(f"missing {mp}; is --soccernet-dir correct?")
    if not sp.exists():
        raise SystemExit(f"missing {sp}; download incomplete?")
    return json.loads(mp.read_text()), json.loads(sp.read_text())


def verify(sn_dir: Path) -> bool:
    """Re-check download integrity: manifest sha vs _SUCCESS + every payload
    present at its recorded size. Prints a report; returns True iff intact."""
    manifest, success = load_download_manifest(sn_dir)
    actual = sha256_file(sn_dir / "download_manifest.json")
    expected = success.get("manifest_sha256")
    man_ok = actual == expected
    files = manifest["files"]
    missing, size_mismatch, total = [], [], 0
    for f in files:
        p = sn_dir / f["relative_path"]
        if not p.exists():
            missing.append(f["relative_path"])
            continue
        sz = p.stat().st_size
        total += sz
        if f.get("bytes") is not None and sz != f["bytes"]:
            size_mismatch.append((f["relative_path"], f["bytes"], sz))
    print(f"schema:          {success.get('schema')}")
    print(f"download_id:     {success.get('download_id')}")
    print(f"manifest sha256: {'OK' if man_ok else 'MISMATCH'} ({actual})")
    print(f"payloads:        {len(files) - len(missing)}/{len(files)} present, "
          f"{len(missing)} missing, {len(size_mismatch)} size-mismatch")
    print(f"payload bytes:   {total / 1e9:.1f} GB")
    for rp in missing[:5]:
        print(f"  MISSING: {rp}")
    for rp, e, a in size_mismatch[:5]:
        print(f"  SIZE {rp}: expected {e} got {a}")
    ok = man_ok and not missing and not size_mismatch
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


def build_manifest(sn_dir: Path, seed: int) -> dict:
    download, success = load_download_manifest(sn_dir)
    # game -> split, and (game, half) -> {bytes, sha256, duration_s} from the download manifest
    game_split: dict[str, str] = {}
    half_info: dict[tuple[str, str], dict] = {}
    for f in download["files"]:
        g = f.get("game")
        if g and "split" in f:
            game_split[g] = f["split"]
        if g and f.get("name", "").endswith(".mkv"):
            half_info[(g, f["name"][0])] = {
                "video": f["name"],
                "bytes": f.get("bytes"),
                "sha256": f.get("sha256"),
                "duration_s": f.get("duration_seconds"),
            }

    matches, queries = [], []
    event_meta: dict[str, dict] = {}
    event_angles: Counter = Counter()
    event_primary: set[str] = set()
    diag = Counter()
    label_dist = Counter()
    per_split = defaultdict(lambda: Counter())

    for cam_path in sorted(glob.glob(str(sn_dir / "*" / "*" / "*" / "Labels-cameras.json"))):
        game = str(Path(cam_path).parent.relative_to(sn_dir))
        split = game_split.get(game, "unknown")
        anns = json.loads(Path(cam_path).read_text())["annotations"]

        # Sort each half's rows by position so shot spans [prev, cur) are well-defined.
        by_half: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for a in anns:
            try:
                pos = int(a["position"])
            except (KeyError, ValueError):
                pos = 0
            by_half[half_of(a["gameTime"])].append((pos, a))
        for h in by_half:
            by_half[h].sort(key=lambda x: x[0])

        match_events: set[tuple[str, int]] = set()
        n_replay = 0
        for h, rows in by_half.items():
            for i, (pos, a) in enumerate(rows):
                if a.get("replay") != "replay" or "link" not in a:
                    continue
                n_replay += 1
                link = a["link"]
                vis = link.get("visibility", "default")
                if vis in EXCLUDE_VISIBILITY:
                    per_split[split]["excluded_not_shown"] += 1
                    continue
                cohort = "review" if vis in REVIEW_VISIBILITY else "primary"
                lh = str(link.get("half", h))
                try:
                    lp = int(link.get("position", "0"))
                except ValueError:
                    lp = 0
                prev = rows[i - 1][0] if i > 0 else 0
                cross = lh != h
                event_id = f"{game}|{lh}|{lp}"
                action = link.get("label", "unknown")
                label_dist[action] += 1
                if event_id not in event_meta:
                    event_meta[event_id] = {
                        "event_id": event_id, "game": game, "split": split,
                        "half": lh, "anchor_ms": lp, "action_label": action,
                        "team": link.get("team", "unknown"),
                    }
                event_angles[event_id] += 1
                if cohort == "primary":
                    event_primary.add(event_id)
                match_events.add((lh, lp))
                queries.append({
                    "query_id": f"{game}|{h}|{pos}",
                    "game": game, "split": split, "event_id": event_id,
                    "replay_half": h, "replay_span_ms": [prev, pos],
                    "visibility": vis, "cohort": cohort, "cross_half": cross,
                })
                per_split[split]["usable_queries"] += 1
                per_split[split][f"cohort_{cohort}"] += 1
                if cross:
                    per_split[split]["cross_half"] += 1
                else:
                    per_split[split]["same_half_primary" if cohort == "primary"
                                     else "same_half_review"] += 1

        # near-duplicate anchor diagnostic (same half, distinct, < NEAR_DUP_MS apart)
        anchors_by_half = defaultdict(list)
        for (hh, pp) in match_events:
            anchors_by_half[hh].append(pp)
        for hh, ps in anchors_by_half.items():
            ps.sort()
            for a1, a2 in zip(ps, ps[1:]):
                if 0 < a2 - a1 < NEAR_DUP_MS:
                    diag["near_dup_anchor_pairs"] += 1

        matches.append({
            "game": game, "split": split,
            "league": game.split("/")[0], "season": game.split("/")[1] if "/" in game else "",
            "halves": {h: half_info.get((game, h), {}) for h in ("1", "2")},
            "labels_cameras": "Labels-cameras.json", "labels_v2": "Labels-v2.json",
            "n_replays_linked": n_replay, "n_events": len(match_events),
        })

    events = sorted(
        ({**event_meta[e], "n_angles": event_angles[e], "has_primary": e in event_primary}
         for e in event_meta),
        key=lambda e: (e["game"], e["half"], e["anchor_ms"]),
    )
    counts = {
        "matches": len(matches),
        "events": len(events),
        "usable_queries": len(queries),
        "per_split": {sp: dict(c) for sp, c in sorted(per_split.items())},
        "action_label_dist": dict(label_dist.most_common()),
        "diagnostics": dict(diag),
    }
    return {
        "metadata": {
            "schema": MANIFEST_SCHEMA,
            "dataset": "soccernet_v2_replay_grounding",
            "builder_git_commit": git_commit(),
            "seed": seed,
            "download_id": success.get("download_id"),
            "download_manifest_sha256": success.get("manifest_sha256"),
            "source": download.get("source"),
            "span_convention": ("replay=[prev_boundary,row_pos); "
                                "event=exact (link.half,link.position)"),
            "visibility_policy": {
                "excluded": sorted(EXCLUDE_VISIBILITY),
                "review_cohort": sorted(REVIEW_VISIBILITY),
                "primary_cohort": ["visible"],
            },
            "group_unit": "game", "positive_key": "event_id",
            "window_policy": None,  # frozen later by the train-only canary
            "counts": counts,
        },
        "matches": matches,
        "events": events,
        "queries": queries,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soccernet-dir", type=Path, default=DEFAULT_SN_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="manifest output (default: "
                        "<soccernet-dir>/replay_event_manifest_v1_seed<seed>.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-only", action="store_true",
                   help="run download integrity check and exit (no manifest build)")
    p.add_argument("--set-window-policy", action="store_true",
                   help="freeze the canary-selected window policy into an existing manifest")
    p.add_argument("--pre", type=float, help="live-clip window start rel. to anchor (s), e.g. -2")
    p.add_argument("--post", type=float, help="live-clip window end rel. to anchor (s), e.g. 2")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--approve", action="store_true",
                   help="mark the window policy approved (locked)")
    p.add_argument("--canary-ref", type=str, default=None,
                   help="canary_index.json path justifying the frozen window")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting an already-approved window policy (re-freeze)")
    args = p.parse_args()

    if not args.soccernet_dir.exists():
        raise SystemExit(f"missing --soccernet-dir {args.soccernet_dir}")

    manifest_path = (args.out
                     or args.soccernet_dir / f"replay_event_manifest_v1_seed{args.seed}.json")

    if args.set_window_policy:
        if args.pre is None or args.post is None:
            raise SystemExit("--set-window-policy requires --pre and --post")
        if not manifest_path.exists():
            raise SystemExit(f"manifest {manifest_path} not found; build it first")
        if args.approve and not args.canary_ref:
            raise SystemExit("--approve requires --canary-ref (the scored canary audit artifact)")
        canary_sha = None
        if args.canary_ref:
            if not Path(args.canary_ref).exists():
                raise SystemExit(f"--canary-ref {args.canary_ref} not found")
            canary_sha = sha256_file(Path(args.canary_ref))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        from soccernet_plan import make_window_policy, require_scored_canary
        if args.approve:  # bind a SCORED decision artifact, not the raw index
            require_scored_canary(Path(args.canary_ref), args.pre, args.post)
        manifest = json.loads(manifest_path.read_text())
        existing = manifest.get("metadata", {}).get("window_policy")
        if existing and existing.get("approved") and not args.force:
            raise SystemExit(
                f"window policy already frozen at "
                f"[{existing.get('pre_s')},{existing.get('post_s')}]; re-freezing needs --force")
        # Bind the SOURCE manifest identity (pre-lock content hash) into the policy.
        manifest_sha = sha256_file(manifest_path)
        wp = make_window_policy(args.pre, args.post, args.n_frames, approved=args.approve,
                                canary_ref=args.canary_ref, commit=git_commit(),
                                canary_sha256=canary_sha, manifest_sha256=manifest_sha)
        manifest["metadata"]["window_policy"] = wp
        manifest_path.write_text(json.dumps(manifest, indent=1))
        state = "APPROVED/LOCKED" if args.approve else "PROPOSED (not approved)"
        print(f"window policy {state}: {wp}")
        print(f"  source manifest sha256 (bound): {manifest_sha}")
        print(f"  canary artifact sha256 (bound): {canary_sha}")
        print(f"  manifest sha256 after write:    {sha256_file(manifest_path)}")
        sys.exit(0)

    if args.verify_only:
        sys.exit(0 if verify(args.soccernet_dir) else 1)

    if not verify(args.soccernet_dir):
        raise SystemExit("integrity check failed; refusing to build manifest")

    out = manifest_path
    manifest = build_manifest(args.soccernet_dir, args.seed)
    out.write_text(json.dumps(manifest, indent=1))
    c = manifest["metadata"]["counts"]
    print(f"\nwrote {out}")
    print(f"  sha256: {sha256_file(out)}")
    print(f"  matches={c['matches']} events={c['events']} usable_queries={c['usable_queries']}")
    for sp, cc in c["per_split"].items():
        print(f"  [{sp}] {dict(cc)}")
    print(f"  diagnostics: {c['diagnostics']}")


if __name__ == "__main__":
    main()

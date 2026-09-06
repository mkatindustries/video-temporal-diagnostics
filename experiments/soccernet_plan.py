"""Window-policy gate + immutable extraction plans for SoccerNet replay grounding.

The window policy (pre/post seconds + frames) is a MANIFEST INPUT, frozen once by
the train-only canary and approved via ``setup_soccernet.py --set-window-policy``.
Nothing downstream hardcodes it.

An *extraction plan* binds everything that determines the produced features into a
single content-addressed record so a feature cache is reproducible and verifiable:

  * model: path + full sha256 + preprocessing recipe + dtype + frames/window
  * the approved window policy read from the manifest
  * deterministic row order (the exact clip order features are written in)
  * per-clip SOURCE SPANS: ``[t0_ms, t1_ms]`` in the named half video

This module only PLANS. It never decodes video, loads a model, or trains. Extraction,
eval, and SLURM must call :func:`require_locked_window_policy` and refuse to run
against an unlocked/unapproved policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

WINDOW_POLICY_SCHEMA = "soccernet_window_policy_v1"
PLAN_SCHEMA = "soccernet_extraction_plan_v1"

# Frozen public model spec. The checkpoint hash is filled by fingerprinting the
# local weights at plan-build time (recorded=None -> hash the weights).
MODEL_SPECS = {
    "vjepa2_encoder_seq": {
        "name": "facebook/vjepa2-vitl-fpc64-256",
        "path": None,  # resolved from env/HF cache by the extractor; hashed at build
        "recorded_sha256": None,
        "preprocessing": "AutoVideoProcessor(vjepa2-vitl-fpc64-256); 64 frames @256",
        "dtype": "float32",
        "n_frames": 64,
        "produces": "encoder_seq[32,1024] + mean_emb[1024] + temporal_residual[16,1024]",
        "windowing": None,
    },
}

# Frozen comparator configuration, bound into the plan so ``plan_sha256`` covers
# the full experiment spec (extraction + comparison), not just the features.
# DTW config matches the driving-dataset evaluators for cross-domain comparability.
# V-JEPA 2 emits fixed-length sequences (encoder 32 / residual 16 steps), so there
# are no padding masks, no length normalization, and no coverage penalty.
COMPARATORS_SCHEMA = "soccernet_comparators_v1"
COMPARATORS = {
    "bot": {"feature": "mean_emb", "kind": "cosine"},
    "encoder_seq_dtw": {"feature": "encoder_seq", "kind": "dtw", "cost": "l2",
                        "normalize": "per_feature_minmax_over_time",
                        "path_norm": "T1+T2", "warping": "unconstrained"},
    "temporal_residual_dtw": {"feature": "temporal_residual", "kind": "dtw", "cost": "l2",
                              "normalize": "per_feature_minmax_over_time",
                              "path_norm": "T1+T2", "warping": "unconstrained"},
    # HEADLINE ordering control: the SAME DTW (same cost, warp tolerance, endpoint anchoring,
    # normalization) run on the query vs the event with its TIME AXIS randomly permuted, averaged
    # over K independent permutations. min-max normalization is a per-dim over-time statistic and
    # COMMUTES with the shuffle, so the ONLY thing that differs from encoder_seq_dtw is the
    # temporal arrangement the monotonic path sees -> encoder_seq_dtw - encoder_seq_dtw_shuffled
    # isolates ordering with warp/anchoring/metric/normalization all held constant. Permutations
    # are drawn independently per (query, event, k) from a stable hash seed: reproducible, and
    # never one fixed per-event shuffle reused across queries (which would correlate errors).
    "encoder_seq_dtw_shuffled": {"feature": "encoder_seq", "kind": "dtw_shuffled", "cost": "l2",
                                 "normalize": "per_feature_minmax_over_time",
                                 "path_norm": "T1+T2", "warping": "unconstrained",
                                 "n_permutations": 10, "seed": 42,
                                 "shuffle": "event_time_axis_per_query_event_perm"},
    # Secondary STRUCTURAL control: min-cost ONE-TO-ONE assignment over the SAME normalized
    # Euclidean cost matrix DTW builds. This is a rigid bijective match; vs encoder_seq_dtw it
    # removes ordering AND DTW's one-to-many warp tolerance AND endpoint anchoring JOINTLY -- NOT
    # ordering alone (use encoder_seq_dtw_shuffled for that). Reported as a "does any rigid
    # alignment structure help" comparator, not an ordering isolator.
    "encoder_seq_assignment": {"feature": "encoder_seq", "kind": "assignment", "cost": "l2",
                               "normalize": "per_feature_minmax_over_time",
                               "matching": "min_cost_one_to_one"},
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj: object) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def make_window_policy(pre_s: float, post_s: float, n_frames: int, *, approved: bool,
                       canary_ref: str | None, commit: str,
                       canary_sha256: str | None = None,
                       manifest_sha256: str | None = None) -> dict:
    """Frozen window policy. An approved lock binds the exact scored canary artifact
    (``canary_sha256``) and the source manifest it was frozen against
    (``manifest_sha256``), so approval is auditable and tied to that evidence."""
    if post_s <= pre_s:
        raise ValueError(f"window post_s ({post_s}) must exceed pre_s ({pre_s})")
    return {
        "schema": WINDOW_POLICY_SCHEMA,
        "pre_s": float(pre_s), "post_s": float(post_s), "n_frames": int(n_frames),
        "approved": bool(approved), "canary_ref": canary_ref,
        "canary_sha256": canary_sha256, "manifest_sha256": manifest_sha256,
        "frozen_at_commit": commit,
    }


def recompute_selected_window(canary_data: dict) -> tuple[float, float] | None:
    """Recompute the frozen window directly from the per-sample scores, using the
    predeclared rubric in ``scripts/soccernet_window_canary.py``: the SMALLEST candidate
    window whose *median coverage* >= 2 and *median contamination* <= 1 across the train
    sample; equal-width ties broken by candidate order. Returns ``(pre_s, post_s)`` or
    ``None`` if no window qualifies. Self-contained — reads only the canary artifact."""
    windows = canary_data.get("candidate_windows_s", [])
    samples = canary_data.get("samples", [])
    labels = [f"[{float(w[0]):+.0f},{float(w[1]):+.0f}]" for w in windows]
    if len(set(labels)) != len(labels):
        raise SystemExit("candidate window labels collide after rounding; score lookup ambiguous")
    eligible: list[tuple[float, int, float, float]] = []
    for idx, win in enumerate(windows):
        pre, post = float(win[0]), float(win[1])
        label = f"[{pre:+.0f},{post:+.0f}]"
        cov = [s["scores"][label]["coverage"] for s in samples
               if s.get("scores", {}).get(label, {}).get("coverage") is not None]
        con = [s["scores"][label]["contamination"] for s in samples
               if s.get("scores", {}).get(label, {}).get("contamination") is not None]
        if not cov or not con:
            continue  # this window was not scored across the sample
        if statistics.median(cov) >= 2 and statistics.median(con) <= 1:
            eligible.append((post - pre, idx, pre, post))
    if not eligible:
        return None
    eligible.sort(key=lambda e: (e[0], e[1]))  # smallest width, then candidate order
    return (eligible[0][2], eligible[0][3])


def require_scored_canary(canary_ref: Path, pre_s: float, post_s: float) -> dict:
    """Return the SCORED canary decision or raise. Enforces that an approved lock binds a
    *scored decision* artifact (not a raw index), that the frozen window matches the
    recorded decision, AND that the decision matches the rubric recomputed from the actual
    per-window scores — so a typo / copy-paste / hand-edited ``selected_window_s`` cannot
    pass silently. A deliberate override is allowed only with a non-empty
    ``decision.rationale`` (logged loudly)."""
    data = json.loads(Path(canary_ref).read_text())
    sel = data.get("decision", {}).get("selected_window_s")
    if sel is None:
        raise SystemExit(
            f"{canary_ref}: unscored canary (decision.selected_window_s is null). Fill in the "
            "coverage/contamination scores and the selected window before approving a lock.")
    if [float(sel[0]), float(sel[1])] != [float(pre_s), float(post_s)]:
        raise SystemExit(f"frozen window [{pre_s}, {post_s}] != canary decision {sel}; must match")
    recomputed = recompute_selected_window(data)
    if recomputed != (float(sel[0]), float(sel[1])):
        rationale = data.get("decision", {}).get("rationale")
        if not (isinstance(rationale, str) and rationale.strip()):
            raise SystemExit(
                f"{canary_ref}: decision.selected_window_s {sel} != rubric recompute "
                f"{list(recomputed) if recomputed else None} from the scores. Fix the scores/"
                "decision, or set a non-empty decision.rationale to justify a deliberate override.")
        print(f"[canary override] decision {sel} != rubric recompute "
              f"{list(recomputed) if recomputed else None}; proceeding on rationale: {rationale}")
    return data["decision"]


def require_locked_window_policy(manifest: dict, repo_root: Path | None = None) -> dict:
    """Return the approved window policy or raise SystemExit. The single gate that
    extraction / eval / SLURM must pass before touching real data.

    Beyond checking ``approved``, this RE-GROUNDS the lock against live content: the
    bound canary artifact must still exist and hash to ``canary_sha256``, and its scored
    decision must still recompute-validate against the frozen window (so a post-freeze
    edit to the evidence or the decision cannot slip through). ``repo_root`` resolves the
    (repo-relative) ``canary_ref``; defaults to this file's repo root."""
    wp = manifest.get("metadata", {}).get("window_policy")
    if not wp or not wp.get("approved"):
        raise SystemExit(
            "SoccerNet window policy is not locked/approved. Run the train-only canary "
            "(scripts/soccernet_window_canary.py), then freeze it with "
            "`setup_soccernet.py --set-window-policy --pre P --post Q --n-frames N --approve`.")
    canary_ref, bound = wp.get("canary_ref"), wp.get("canary_sha256")
    if canary_ref and bound:
        root = repo_root or Path(__file__).resolve().parent.parent
        p = Path(canary_ref) if Path(canary_ref).is_absolute() else root / canary_ref
        if not p.exists():
            raise SystemExit(f"locked canary artifact {p} is missing; cannot re-ground the lock")
        live = sha256_file(p)
        if live != bound:
            raise SystemExit(f"canary artifact {p} changed since freeze (sha {live[:12]} != bound "
                             f"{bound[:12]}); the window lock no longer grounds its evidence")
        require_scored_canary(p, wp["pre_s"], wp["post_s"])  # re-validate scores -> window
    return wp


def _resolve_model_dir(spec: dict) -> Path:
    """Resolve the dir holding the model's *.safetensors, mirroring how the extractor
    loads it: explicit spec['path'], then $VTD_MODEL_DIR/<name-tail>, then the HF hub
    cache snapshot for spec['name']. Raises if none is found (so a full plan cannot be
    built without a real, hashable checkpoint)."""
    if spec.get("path") and Path(spec["path"]).exists():
        return Path(spec["path"])
    name = spec.get("name", "")
    tail = name.split("/")[-1]
    md = os.environ.get("VTD_MODEL_DIR")
    if md and (Path(md) / tail).exists():
        return Path(md) / tail
    bases = []
    if os.environ.get("HF_HOME"):
        bases.append(Path(os.environ["HF_HOME"]) / "hub")
    bases.append(Path.home() / ".cache" / "huggingface" / "hub")
    for base in bases:
        snaps = base / f"models--{name.replace('/', '--')}" / "snapshots"
        if snaps.exists():
            for snap in sorted(snaps.glob("*")):
                if any(snap.glob("*.safetensors")):
                    return snap
    raise FileNotFoundError(
        f"cannot resolve weights for {name!r}: set spec['path'] or $VTD_MODEL_DIR, or "
        "ensure the HuggingFace hub cache holds it")


def model_fingerprint(spec: dict) -> str:
    """Full sha256 for the model: recorded hash if present, else resolve + hash the
    *.safetensors (name-prefixed so ordering/renames are covered)."""
    if spec.get("recorded_sha256"):
        return spec["recorded_sha256"]
    d = _resolve_model_dir(spec)
    weights = sorted(d.glob("*.safetensors")) if d.is_dir() else [d]
    if not weights:
        raise FileNotFoundError(f"no *.safetensors under {d} to fingerprint")
    h = hashlib.sha256()
    for w in weights:
        h.update(w.name.encode())
        h.update(sha256_file(w).encode())
    return h.hexdigest()


def build_extraction_plan(manifest_path: Path, arm: str, split: str = "test",
                          *, fingerprint: bool = True) -> dict:
    """Build the immutable plan for one arm/split. Refuses if the window policy is
    not locked. ``fingerprint=False`` skips (slow) weight hashing for tests."""
    if arm not in MODEL_SPECS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(MODEL_SPECS)}")
    manifest = json.loads(Path(manifest_path).read_text())
    wp = require_locked_window_policy(manifest)
    spec = dict(MODEL_SPECS[arm])
    if fingerprint:
        spec["sha256"] = model_fingerprint(spec)
    else:
        spec["sha256"] = spec.get("recorded_sha256") or "UNRESOLVED"

    def video_rel(game: str, half: str) -> str:
        return f"{game}/{half}_224p.mkv"

    # Per-half durations (ms), to clamp/validate event windows at the boundaries.
    half_dur_ms: dict[tuple[str, str], int] = {}
    for mt in manifest.get("matches", []):
        for h, info in (mt.get("halves") or {}).items():
            if info.get("duration_s") is not None:
                half_dur_ms[(mt["game"], h)] = int(info["duration_s"] * 1000)

    pre_ms = int(round(wp["pre_s"] * 1000))
    post_ms = int(round(wp["post_s"] * 1000))
    width = post_ms - pre_ms  # frozen window width, preserved when clamping at a boundary

    clips: list[dict] = []
    for q in manifest["queries"]:
        if q["split"] != split or q["cohort"] != "primary":
            continue
        clips.append({
            "clip_id": q["query_id"], "kind": "query", "game": q["game"],
            "half": q["replay_half"], "video": video_rel(q["game"], q["replay_half"]),
            "span_ms": [int(q["replay_span_ms"][0]), int(q["replay_span_ms"][1])],
        })
    for e in manifest["events"]:
        if e["split"] != split:
            continue
        anchor = int(e["anchor_ms"])
        t0, t1 = anchor + pre_ms, anchor + post_ms
        if t0 < 0:  # shift (not shorten) to preserve the frozen width at the start boundary
            t0, t1 = 0, width
        dur = half_dur_ms.get((e["game"], e["half"]))
        if dur is None or dur <= 0:  # NewM3: no valid half length -> can't validate end boundary
            raise SystemExit(
                f"event {e['event_id']}: half {e['game']}|{e['half']} has no valid duration_s "
                f"(got {dur}); cannot clamp the end-boundary window (fail-closed)")
        if t1 > dur:  # shift to preserve width at the end boundary
            t1, t0 = dur, max(0, dur - width)
        clips.append({
            "clip_id": e["event_id"], "kind": "event", "game": e["game"],
            "half": e["half"], "video": video_rel(e["game"], e["half"]),
            "span_ms": [t0, t1],
        })
    # Deterministic row order: (kind, clip_id) — manifest lists are already sorted,
    # but sort explicitly so the plan is order-stable regardless of manifest order.
    clips.sort(key=lambda c: (c["kind"], c["clip_id"]))
    row_order = [c["clip_id"] for c in clips]

    body = {
        "schema": PLAN_SCHEMA, "arm": arm, "split": split,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "window_policy": wp, "model": spec,
        "comparators_schema": COMPARATORS_SCHEMA, "comparators": COMPARATORS,
        "counts": {"clips": len(clips),
                   "queries": sum(c["kind"] == "query" for c in clips),
                   "events": sum(c["kind"] == "event" for c in clips)},
        "row_order": row_order, "clips": clips,
    }
    body["plan_sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return body


def write_plan(plan: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"plan_{plan['arm']}_{plan['split']}_{plan['plan_sha256'][:12]}.json"
    out.write_text(json.dumps(plan, indent=1))
    return out


# --------------------------------------------------------------------------- #
# Canonical feature-cache contract — the SINGLE source of truth shared by the
# extractor (writer) and the evaluator (reader), so the two cannot drift.
# --------------------------------------------------------------------------- #
FEATURE_CACHE_SCHEMA = "soccernet_feature_cache_v1"


def _success_marker(cache_path: Path) -> Path:
    return cache_path.parent / (cache_path.name + "._SUCCESS")


MAX_DROP_RATE = 0.02  # NewM1: full run fails closed if more than this fraction of rows drop


def write_feature_cache(cache_path: Path, plan: dict, features: dict,
                        dropped: dict | None = None, *, canary: bool, limit: int = 0) -> dict:
    """Write an extractor cache bound to ``plan``. Every plan row must be *accounted for* —
    either extracted (in ``features``) or explicitly recorded in ``dropped`` ({clip_id: reason},
    e.g. short/non-finite/degenerate-static clips, NewM1). A run is ``complete`` only if it is
    NOT a canary, every row is accounted for, and the drop-rate is within the module constant
    ``MAX_DROP_RATE``; only then is a sibling ``_SUCCESS`` marker written. A full run whose
    drop-rate EXCEEDS the guard fails closed (hard error), never a silently-degraded cache.
    The guard is the code constant (not a caller/cache-supplied knob), so a cache cannot
    self-relax it. ``--limit`` (canary) outputs are tagged, never complete, unpublishable."""
    import torch

    dropped = dict(dropped or {})
    expected = plan["row_order"]
    accounted = set(features) | set(dropped)
    drop_rate = (len(dropped) / len(expected)) if expected else 0.0
    fully_accounted = len(expected) > 0 and set(expected) <= accounted
    if (not canary) and fully_accounted and drop_rate > MAX_DROP_RATE:
        raise SystemExit(
            f"{cache_path}: drop-rate {drop_rate:.3f} > {MAX_DROP_RATE} "
            f"({len(dropped)}/{len(expected)} rows dropped); fail-closed (bad extraction)")
    complete = (not canary) and fully_accounted and drop_rate <= MAX_DROP_RATE
    blob = {
        "schema": FEATURE_CACHE_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "arm": plan["arm"], "split": plan["split"],
        "model_sha256": plan["model"]["sha256"],
        "preprocessing": plan["model"]["preprocessing"],
        "comparators_schema": plan.get("comparators_schema"),
        "canary": bool(canary), "limit": int(limit), "complete": complete,
        "dropped": dropped, "max_drop_rate": float(MAX_DROP_RATE),  # recorded for audit only
        "n_features": len(features), "n_dropped": len(dropped),
        "n_expected": len(expected), "features": features,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache_path)
    marker = _success_marker(cache_path)
    if complete:
        marker.write_text(json.dumps(
            {"plan_sha256": plan["plan_sha256"], "n_features": len(features),
             "n_dropped": len(dropped)}))
    elif marker.exists():
        marker.unlink()  # never leave a stale success marker beside a partial cache
    return blob


def load_feature_cache(cache_path: Path, plan: dict) -> dict:
    """Load an extractor cache FAIL-CLOSED against ``plan``: refuse a cache that is a
    canary, not marked complete, missing its ``_SUCCESS`` marker, bound to a
    different plan, or missing any required row. The single canonical reader."""
    import torch

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("schema") != FEATURE_CACHE_SCHEMA:
        raise SystemExit(f"{cache_path}: not a {FEATURE_CACHE_SCHEMA} cache; fail-closed")
    if cache.get("canary"):
        raise SystemExit(f"{cache_path}: CANARY cache (--limit); refusing to publish as full")
    if not cache.get("complete") or not _success_marker(cache_path).exists():
        raise SystemExit(f"{cache_path}: not complete / no _SUCCESS marker; fail-closed")
    if cache.get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit(f"{cache_path}: plan_sha256 {cache.get('plan_sha256')} != current "
                         f"plan {plan['plan_sha256']}; fail-closed")
    # Every plan row must be accounted for: extracted OR explicitly recorded as dropped (NewM1).
    dropped = cache.get("dropped", {})
    accounted = set(cache["features"]) | set(dropped)
    unaccounted = [c for c in plan["row_order"] if c not in accounted]
    if unaccounted:
        raise SystemExit(
            f"{cache_path}: {len(unaccounted)}/{len(plan['row_order'])} rows neither extracted "
            f"nor recorded-dropped (first {unaccounted[0]}); fail-closed")
    drop_rate = (len(dropped) / len(plan["row_order"])) if plan["row_order"] else 0.0
    # Enforce the CODE constant, not the cache-recorded value: a cache cannot self-relax the guard.
    stored = float(cache.get("max_drop_rate", MAX_DROP_RATE))
    if stored > MAX_DROP_RATE:
        raise SystemExit(
            f"{cache_path}: recorded max_drop_rate {stored} exceeds ceiling {MAX_DROP_RATE}; "
            "fail-closed")
    if drop_rate > MAX_DROP_RATE:
        raise SystemExit(
            f"{cache_path}: drop-rate {drop_rate:.3f} exceeds guard {MAX_DROP_RATE}; fail-closed")
    # Content validation over PRESENT features only: a FINITE tensor of sane rank.
    # (Non-finite values would otherwise invert into a best rank — a silent metric inflation.)
    for cid, fd in cache["features"].items():
        for k in ("mean_emb", "encoder_seq", "temporal_residual"):
            t = fd.get(k)
            if not torch.is_tensor(t):
                raise SystemExit(
                    f"{cache_path}: clip {cid} feature {k!r} missing/invalid; fail-closed")
            if not bool(torch.isfinite(t).all()):
                raise SystemExit(
                    f"{cache_path}: clip {cid} feature {k!r} non-finite; fail-closed")
        ndims = (fd["mean_emb"].ndim, fd["encoder_seq"].ndim, fd["temporal_residual"].ndim)
        if ndims != (1, 2, 2):
            raise SystemExit(
                f"{cache_path}: clip {cid} feature rank {ndims} != (1,2,2); fail-closed")
    return cache

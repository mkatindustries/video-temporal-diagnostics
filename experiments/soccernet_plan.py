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
from pathlib import Path

WINDOW_POLICY_SCHEMA = "soccernet_window_policy_v1"
PLAN_SCHEMA = "soccernet_extraction_plan_v1"

# Frozen model specs. sha256 for SONAR2-PE is the confirmed VIDEO checkpoint hash
# (from its plan.json); vjepa2 sha is filled by fingerprinting the local checkpoint
# at plan-build time (recorded=None -> hash the weights).
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
    "sonar2pe": {
        "name": "sonar2-pe",
        "path": "/checkpoint/dream/arjangt/sonar2-pe",
        "recorded_sha256": "90d02aa2188b70743a4f75efdb90afaa102633fa9d5a0769cd5f03232fe353e8",
        "preprocessing": "SonarOmniPEImageProcessor._transform (resize 448, Normalize(0.5,0.5))",
        "dtype": "float16",
        "n_frames": 8,
        "produces": "ordered window sequence [T,1024] (BoT = mean+renorm)",
        "windowing": {"window_s": 2, "stride_s": 1, "frames_per_window": 8},
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
    # Order-agnostic control: symmetric-max (Chamfer) over the SAME per-position
    # encoder_seq vectors. Isolates whether DTW's monotonic alignment (order)
    # helps beyond having per-position features.
    "encoder_seq_unordered": {"feature": "encoder_seq", "kind": "chamfer"},
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


def require_scored_canary(canary_ref: Path, pre_s: float, post_s: float) -> dict:
    """Return the SCORED canary decision or raise. Enforces that an approved lock binds
    a *scored decision* artifact — the reviewer filled ``decision.selected_window_s``
    after coverage/contamination scoring — not the raw unscored index, and that the
    frozen window matches the recorded decision."""
    data = json.loads(Path(canary_ref).read_text())
    sel = data.get("decision", {}).get("selected_window_s")
    if sel is None:
        raise SystemExit(
            f"{canary_ref}: unscored canary (decision.selected_window_s is null). Fill in the "
            "coverage/contamination scores and the selected window before approving a lock.")
    if [float(sel[0]), float(sel[1])] != [float(pre_s), float(post_s)]:
        raise SystemExit(f"frozen window [{pre_s}, {post_s}] != canary decision {sel}; must match")
    return data["decision"]


def require_locked_window_policy(manifest: dict) -> dict:
    """Return the approved window policy or raise SystemExit. The single gate that
    extraction / eval / SLURM must pass before touching real data."""
    wp = manifest.get("metadata", {}).get("window_policy")
    if not wp or not wp.get("approved"):
        raise SystemExit(
            "SoccerNet window policy is not locked/approved. Run the train-only canary "
            "(scripts/soccernet_window_canary.py), then freeze it with "
            "`setup_soccernet.py --set-window-policy --pre P --post Q --n-frames N --approve`.")
    return wp


def model_fingerprint(spec: dict) -> str:
    """Full sha256 for the model: recorded hash if present, else hash the weights."""
    if spec.get("recorded_sha256"):
        return spec["recorded_sha256"]
    path = spec.get("path")
    if not path:
        raise ValueError(f"model spec {spec.get('name')!r} has neither recorded_sha256 nor path")
    p = Path(path)
    weights = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not weights:
        raise FileNotFoundError(f"no *.safetensors under {p} to fingerprint")
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
        t0 = anchor + int(round(wp["pre_s"] * 1000))
        t1 = anchor + int(round(wp["post_s"] * 1000))
        clips.append({
            "clip_id": e["event_id"], "kind": "event", "game": e["game"],
            "half": e["half"], "video": video_rel(e["game"], e["half"]),
            "span_ms": [max(0, t0), t1],
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


def write_feature_cache(cache_path: Path, plan: dict, features: dict, *,
                        canary: bool, limit: int = 0) -> dict:
    """Write an extractor cache bound to ``plan``. A run is ``complete`` only if it
    is NOT a canary and every plan row is present; only then is a sibling
    ``_SUCCESS`` marker written. ``--limit`` (canary) outputs are tagged and never
    marked complete, so they can never be published as full results."""
    import torch

    expected = plan["row_order"]
    complete = (not canary) and len(expected) > 0 and all(c in features for c in expected)
    blob = {
        "schema": FEATURE_CACHE_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "arm": plan["arm"], "split": plan["split"],
        "model_sha256": plan["model"]["sha256"],
        "preprocessing": plan["model"]["preprocessing"],
        "comparators_schema": plan.get("comparators_schema"),
        "canary": bool(canary), "limit": int(limit), "complete": complete,
        "n_features": len(features), "n_expected": len(expected), "features": features,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache_path)
    marker = _success_marker(cache_path)
    if complete:
        marker.write_text(json.dumps(
            {"plan_sha256": plan["plan_sha256"], "n_features": len(features)}))
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
    missing = [c for c in plan["row_order"] if c not in cache["features"]]
    if missing:
        raise SystemExit(f"{cache_path}: missing {len(missing)}/{len(plan['row_order'])} rows "
                         f"(first {missing[0]}); fail-closed")
    return cache

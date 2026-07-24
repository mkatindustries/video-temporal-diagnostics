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
                       canary_ref: str | None, commit: str) -> dict:
    if post_s <= pre_s:
        raise ValueError(f"window post_s ({post_s}) must exceed pre_s ({pre_s})")
    return {
        "schema": WINDOW_POLICY_SCHEMA,
        "pre_s": float(pre_s), "post_s": float(post_s), "n_frames": int(n_frames),
        "approved": bool(approved), "canary_ref": canary_ref, "frozen_at_commit": commit,
    }


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

"""Synthetic tests for the SoccerNet window-policy gate + immutable plan builder.

No video, no model, no launches — pure planning/determinism/gating logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from soccernet_plan import (  # noqa: E402
    MODEL_SPECS,
    build_extraction_plan,
    make_window_policy,
    model_fingerprint,
    require_locked_window_policy,
    require_scored_canary,
)

LOCKED = make_window_policy(-2, 2, 8, approved=True, canary_ref=None, commit="test")


def _manifest(window_policy) -> dict:
    return {
        "metadata": {"window_policy": window_policy},
        "events": [
            {"event_id": "A|1|5000", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 5000, "action_label": "Goal"},
            {"event_id": "A|1|9000", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 9000, "action_label": "Foul"},
        ],
        "queries": [
            {"query_id": "A|1|6000", "split": "test", "game": "A", "event_id": "A|1|5000",
             "replay_half": "1", "replay_span_ms": [5800, 6300], "cohort": "primary",
             "cross_half": False},
            # review-cohort query is excluded from the plan:
            {"query_id": "A|1|6500", "split": "test", "game": "A", "event_id": "A|1|5000",
             "replay_half": "1", "replay_span_ms": [6300, 6900], "cohort": "review",
             "cross_half": False},
        ],
    }


def _write(tmp_path: Path, window_policy) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(_manifest(window_policy)))
    return p


def test_gate_requires_approved_policy():
    with pytest.raises(SystemExit):
        require_locked_window_policy(_manifest(None))
    proposed = make_window_policy(-2, 2, 8, approved=False, canary_ref=None, commit="x")
    with pytest.raises(SystemExit):
        require_locked_window_policy(_manifest(proposed))
    assert require_locked_window_policy(_manifest(LOCKED))["pre_s"] == -2.0


def test_build_refuses_when_unlocked(tmp_path):
    with pytest.raises(SystemExit):
        build_extraction_plan(_write(tmp_path, None), "sonar2pe", fingerprint=False)


def test_plan_spans_order_and_counts(tmp_path):
    plan = build_extraction_plan(_write(tmp_path, LOCKED), "sonar2pe", fingerprint=False)
    assert plan["counts"] == {"clips": 3, "queries": 1, "events": 2}
    spans = {c["clip_id"]: c["span_ms"] for c in plan["clips"]}
    # event window = anchor + [pre_s, post_s] * 1000
    assert spans["A|1|5000"] == [3000, 7000]
    assert spans["A|1|9000"] == [7000, 11000]
    assert spans["A|1|6000"] == [5800, 6300]  # query = replay span, untouched
    # deterministic order: events before queries (kind sort), then by clip_id
    assert plan["row_order"] == ["A|1|5000", "A|1|9000", "A|1|6000"]


def test_plan_hash_deterministic_and_window_sensitive(tmp_path):
    m = _write(tmp_path, LOCKED)
    h1 = build_extraction_plan(m, "sonar2pe", fingerprint=False)["plan_sha256"]
    h2 = build_extraction_plan(m, "sonar2pe", fingerprint=False)["plan_sha256"]
    assert h1 == h2  # same manifest + policy -> identical plan
    wider = make_window_policy(-3, 3, 8, approved=True, canary_ref=None, commit="test")
    p3 = build_extraction_plan(_write(tmp_path, wider), "sonar2pe", fingerprint=False)
    h3 = p3["plan_sha256"]
    assert h3 != h1  # a different window yields a different plan


def test_sonar2pe_uses_recorded_video_hash():
    fp = model_fingerprint(MODEL_SPECS["sonar2pe"])
    assert fp == "90d02aa2188b70743a4f75efdb90afaa102633fa9d5a0769cd5f03232fe353e8"


def test_sonar2pe_plan_binds_model_and_window(tmp_path):
    plan = build_extraction_plan(_write(tmp_path, LOCKED), "sonar2pe", fingerprint=True)
    assert plan["model"]["sha256"] == MODEL_SPECS["sonar2pe"]["recorded_sha256"]
    assert plan["model"]["windowing"] == {"window_s": 2, "stride_s": 1, "frames_per_window": 8}
    assert plan["window_policy"]["approved"] is True
    assert plan["manifest_sha256"]  # manifest is content-addressed into the plan


def test_plan_binds_comparator_spec(tmp_path):
    plan = build_extraction_plan(_write(tmp_path, LOCKED), "vjepa2_encoder_seq", fingerprint=False)
    comps = plan["comparators"]
    assert set(comps) == {
        "bot", "encoder_seq_dtw", "temporal_residual_dtw", "encoder_seq_unordered"}
    assert comps["encoder_seq_dtw"] == {
        "feature": "encoder_seq", "kind": "dtw", "cost": "l2",
        "normalize": "per_feature_minmax_over_time", "path_norm": "T1+T2",
        "warping": "unconstrained"}
    assert comps["encoder_seq_unordered"]["kind"] == "chamfer"  # order-agnostic control


def test_require_scored_canary(tmp_path):
    unscored = tmp_path / "u.json"
    unscored.write_text(json.dumps({"decision": {"selected_window_s": None, "rationale": None}}))
    with pytest.raises(SystemExit):  # raw/unscored index cannot back an approval
        require_scored_canary(unscored, -2, 2)
    scored = tmp_path / "s.json"
    scored.write_text(json.dumps(
        {"decision": {"selected_window_s": [-2, 2], "rationale": "full coverage, no bleed"}}))
    assert require_scored_canary(scored, -2, 2)["rationale"] == "full coverage, no bleed"
    with pytest.raises(SystemExit):  # frozen window must match the scored decision
        require_scored_canary(scored, -1, 1)

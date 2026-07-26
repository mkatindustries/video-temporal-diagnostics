"""Tests for the SoccerNet frozen-feature comparator scoring + fail-closed cache.

Uses tiny torch tensors on CPU; no GPU, no real features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from eval_soccernet_replay import (  # noqa: E402
    _assignment_distance,
    build_gallery,
    evaluate_method,
    score_comparator,
)
from soccernet_plan import (  # noqa: E402
    build_extraction_plan,
    load_feature_cache,
    make_window_policy,
    write_feature_cache,
)


def _mini_manifest() -> dict:
    return {
        "events": [
            {"event_id": "A|1|1", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 1, "action_label": "Goal"},
            {"event_id": "A|1|2", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 2, "action_label": "Foul"},
        ],
        "queries": [
            {"query_id": "qA", "split": "test", "game": "A", "event_id": "A|1|1",
             "replay_half": "1", "replay_span_ms": [0, 1], "visibility": "visible",
             "cohort": "primary", "cross_half": False},
        ],
    }


def test_assignment_distance_order_invariant_and_symmetric():
    torch.manual_seed(0)
    a = torch.randn(5, 8)
    b = a[torch.randperm(5)]  # same row-set, shuffled order
    # self-distance is zero (the min-cost matching is the identity)
    assert _assignment_distance(a, a) == pytest.approx(0.0, abs=1e-5)
    # order-agnostic: a row-permutation is still a perfect matching -> zero cost (control's point).
    # Normalization is per-feature min-max OVER TIME, invariant to row order, so this holds under
    # the SAME normalization DTW uses — the difference vs DTW is ONLY the ordering constraint.
    assert _assignment_distance(a, b) == pytest.approx(0.0, abs=1e-5)
    assert _assignment_distance(a, b) == pytest.approx(_assignment_distance(b, a), abs=1e-5)


def test_score_comparator_cosine_dtw_assignment():
    g = build_gallery(_mini_manifest(), "test")
    eye = torch.eye(3)
    feats = {
        "qA":    {"mean_emb": torch.tensor([1.0, 0, 0]), "encoder_seq": eye.clone()},
        "A|1|1": {"mean_emb": torch.tensor([1.0, 0, 0]), "encoder_seq": eye.clone()},  # identical
        "A|1|2": {"mean_emb": torch.tensor([0.0, 1, 0]),
                  "encoder_seq": torch.tensor([[1.0, 1, 0], [0, 1, 1], [1, 0, 1]])},   # different
    }
    cos = score_comparator(feats, g, {"feature": "mean_emb", "kind": "cosine"})
    assert cos["qA"]["A|1|1"] == pytest.approx(1.0, abs=1e-5)
    assert cos["qA"]["A|1|2"] == pytest.approx(0.0, abs=1e-5)

    dtw = score_comparator(feats, g, {"feature": "encoder_seq", "kind": "dtw"})
    # identical sequence -> distance 0 -> score 0; different -> negative
    assert dtw["qA"]["A|1|1"] == pytest.approx(0.0, abs=1e-6)
    assert dtw["qA"]["A|1|1"] > dtw["qA"]["A|1|2"]

    asg = score_comparator(feats, g, {"feature": "encoder_seq", "kind": "assignment"})
    # identical sequence -> assignment cost 0 -> score 0; different -> negative
    assert asg["qA"]["A|1|1"] == pytest.approx(0.0, abs=1e-6)
    assert asg["qA"]["A|1|1"] > asg["qA"]["A|1|2"]


def test_score_comparator_query_subset():
    g = build_gallery(_mini_manifest(), "test")
    feats = {k: {"mean_emb": torch.tensor([1.0, 0, 0])} for k in ("qA", "A|1|1", "A|1|2")}
    assert score_comparator(feats, g, {"feature": "mean_emb", "kind": "cosine"},
                            query_ids=set()) == {}  # empty subset -> no scores


def _feat() -> dict:
    return {"mean_emb": torch.randn(4), "encoder_seq": torch.randn(3, 4),
            "temporal_residual": torch.randn(2, 4)}


def _fake_plan(sha: str = "plan-abc", rows=("c1", "c2")) -> dict:
    return {"plan_sha256": sha, "arm": "vjepa2_encoder_seq", "split": "test",
            "row_order": list(rows), "comparators_schema": "x",
            "model": {"sha256": "m", "preprocessing": "p"}}


def test_feature_cache_roundtrip_and_fail_closed(tmp_path):
    plan = _fake_plan()
    ok = tmp_path / "ok.pt"
    blob = write_feature_cache(ok, plan, {"c1": _feat(), "c2": _feat()}, canary=False)
    assert blob["complete"] is True and (tmp_path / "ok.pt._SUCCESS").exists()
    assert load_feature_cache(ok, plan)["complete"] is True  # round-trips
    # canary -> tagged, no _SUCCESS, refused
    can = tmp_path / "can.pt"
    b2 = write_feature_cache(can, plan, {"c1": _feat()}, canary=True, limit=1)
    assert b2["complete"] is False and not (tmp_path / "can.pt._SUCCESS").exists()
    with pytest.raises(SystemExit):
        load_feature_cache(can, plan)
    with pytest.raises(SystemExit):  # plan mismatch
        load_feature_cache(ok, _fake_plan(sha="different"))
    with pytest.raises(SystemExit):  # complete + marker but a required row unaccounted
        load_feature_cache(ok, _fake_plan(rows=("c1", "c2", "c3")))
    # M2: a "complete" cache with non-finite features is still refused
    nan = tmp_path / "nan.pt"
    bad = _feat()
    bad["encoder_seq"][0, 0] = float("nan")
    write_feature_cache(nan, plan, {"c1": _feat(), "c2": bad}, canary=False)
    with pytest.raises(SystemExit):
        load_feature_cache(nan, plan)


def test_feature_cache_dropped_accounting_and_drop_rate(tmp_path):
    # NewM1: a recorded drop within the guard -> row accounted -> complete -> round-trips,
    # and the eval sees the drop set (so it can restrict the gallery to present clips).
    plan3 = _fake_plan(rows=("c1", "c2", "c3"))
    drp = tmp_path / "drop.pt"
    bd = write_feature_cache(drp, plan3, {"c1": _feat(), "c2": _feat()},
                             {"c3": "degenerate_static"}, canary=False, max_drop_rate=0.5)
    assert bd["complete"] is True and bd["n_dropped"] == 1
    loaded = load_feature_cache(drp, plan3)
    assert loaded["dropped"] == {"c3": "degenerate_static"}
    # exceeding the drop-rate guard fails closed at write (not a silently-degraded cache)
    with pytest.raises(SystemExit):
        write_feature_cache(tmp_path / "toomany.pt", plan3, {"c1": _feat()},
                            {"c2": "short_clip", "c3": "short_clip"}, canary=False)


def _e2e_manifest() -> dict:
    wp = make_window_policy(-2, 2, 8, approved=True, canary_ref=None, commit="test")
    return {
        "metadata": {"window_policy": wp},
        "matches": [{"game": "A", "halves": {"1": {"duration_s": 2700}}}],
        "events": [
            {"event_id": "A|1|1000", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 1000, "action_label": "Goal"},
            {"event_id": "A|1|5000", "split": "test", "game": "A", "half": "1",
             "anchor_ms": 5000, "action_label": "Foul"},
        ],
        "queries": [
            {"query_id": "A|1|2000", "split": "test", "game": "A", "event_id": "A|1|1000",
             "replay_half": "1", "replay_span_ms": [1500, 2000], "visibility": "visible",
             "cohort": "primary", "cross_half": False},
        ],
    }


def test_extractor_eval_contract_end_to_end(tmp_path):
    """Extractor cache-write -> canonical loader -> eval scoring, end-to-end (no GPU)."""
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(_e2e_manifest()))
    plan = build_extraction_plan(mpath, "vjepa2_encoder_seq", fingerprint=False)
    torch.manual_seed(0)
    feats = {cid: {"mean_emb": torch.randn(4), "encoder_seq": torch.randn(3, 4),
                   "temporal_residual": torch.randn(2, 4)} for cid in plan["row_order"]}
    cache_path = tmp_path / "cache.pt"
    write_feature_cache(cache_path, plan, feats, canary=False)
    cache = load_feature_cache(cache_path, plan)  # extractor->loader contract must hold
    gallery = build_gallery(json.loads(mpath.read_text()), "test")
    for name, cfg in plan["comparators"].items():
        scores = score_comparator(cache["features"], gallery, cfg)
        res = evaluate_method(scores, gallery)
        assert res["aggregate"]["n_queries"] == 1, name

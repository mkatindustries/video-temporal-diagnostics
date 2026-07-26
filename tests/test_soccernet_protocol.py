"""Unit tests for the SoccerNet replay event-retrieval protocol core.

Synthetic 2-match manifest with hand-crafted score maps so every metric has a
closed-form expected value. No features / GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from eval_soccernet_replay import (  # noqa: E402
    build_gallery,
    evaluate_method,
    paired_cluster_bootstrap_mean_difference,
    per_query_metrics,
    restrict_gallery,
    smearing_composition,
)


def _ev(eid: str, g: str, ms: int, act: str) -> dict:
    return {"event_id": eid, "split": "test", "game": g, "half": "1",
            "anchor_ms": ms, "action_label": act}


def _qr(qid: str, g: str, eid: str) -> dict:
    return {"query_id": qid, "split": "test", "game": g, "event_id": eid,
            "replay_half": "1", "replay_span_ms": [0, 1], "visibility": "visible",
            "cohort": "primary", "cross_half": False}


def _manifest() -> dict:
    # Game A: 3 linked events + 1 distractor event (eA4, far, same-action Foul).
    events = [
        _ev("A|1|1000", "A", 1000, "Goal"),
        _ev("A|1|5000", "A", 5000, "Foul"),
        _ev("A|1|9000", "A", 9000, "Goal"),
        _ev("A|1|60000", "A", 60000, "Foul"),
        _ev("B|1|2000", "B", 2000, "Corner"),
        _ev("B|1|6000", "B", 6000, "Foul"),
    ]
    queries = [
        _qr("qA1", "A", "A|1|1000"), _qr("qA2", "A", "A|1|5000"),
        _qr("qA3", "A", "A|1|9000"), _qr("qB1", "B", "B|1|2000"),
    ]
    return {"events": events, "queries": queries}


# Scores designed so: qA1 rank2 (top1 = temporal neighbour eA2),
# qA2 rank2 (top1 = same-action far eA4), qA3 rank2 (top1 = other eA4),
# qB1 rank1 (correct).
SCORES = {
    "qA1": {"A|1|1000": 0.3, "A|1|5000": 0.9, "A|1|9000": 0.1, "A|1|60000": 0.1},
    "qA2": {"A|1|1000": 0.1, "A|1|5000": 0.4, "A|1|9000": 0.1, "A|1|60000": 0.9},
    "qA3": {"A|1|1000": 0.1, "A|1|5000": 0.1, "A|1|9000": 0.2, "A|1|60000": 0.9},
    "qB1": {"B|1|2000": 0.9, "B|1|6000": 0.1},
}


def test_ranks_and_macros():
    g = build_gallery(_manifest(), "test")
    pq = per_query_metrics(SCORES, g)
    assert {q: pq[q]["rank"] for q in pq} == {"qA1": 2, "qA2": 2, "qA3": 2, "qB1": 1}

    res = evaluate_method(SCORES, g)
    agg = res["aggregate"]
    # Match A mean rr = 0.5 (all rank 2); Match B = 1.0. Match-macro = 0.75.
    assert agg["match_macro"]["rr"] == pytest.approx(0.75)
    assert agg["match_macro"]["r@1"] == pytest.approx(0.5)   # A:0, B:1
    assert agg["match_macro"]["r@5"] == pytest.approx(1.0)
    # Query-micro rr = mean(.5,.5,.5,1) = 0.625.
    assert agg["query_micro"]["rr"] == pytest.approx(0.625)
    # Action-macro rr: Goal .5, Foul .5, Corner 1 -> 0.6667.
    assert agg["action_macro"]["rr"] == pytest.approx(2.0 / 3.0)
    # Event-macro rr: eA1 .5, eA2 .5, eA3 .5, eB1 1 -> 0.625.
    assert agg["event_macro"]["rr"] == pytest.approx(0.625)


def test_smearing_categories():
    g = build_gallery(_manifest(), "test")
    pq = per_query_metrics(SCORES, g)
    sm = smearing_composition(SCORES, pq, g)
    assert sm["n_wrong_top1"] == 3
    assert sm["counts"] == {"temporal_neighbour": 1, "same_action": 1, "other": 1}


def test_smearing_tie_excludes_positive():
    # Tie between the positive (Goal@1000) and a far, different-action distractor (Foul@60000):
    # query_rank counts it WRONG (pessimistic ties), and the wrong top-1 must be the distractor
    # -> "other", NOT the positive misattributed as "temporal_neighbour".
    g = build_gallery(_manifest(), "test")
    scores = {"qA1": {"A|1|1000": 0.5, "A|1|60000": 0.5, "A|1|5000": 0.1, "A|1|9000": 0.1}}
    pq = per_query_metrics(scores, g, query_ids={"qA1"})
    assert pq["qA1"]["rank"] == 2  # tie counts against the positive
    sm = smearing_composition(scores, pq, g)
    assert sm["counts"] == {"temporal_neighbour": 0, "same_action": 0, "other": 1}


def test_bootstrap_ci_contains_point():
    g = build_gallery(_manifest(), "test")
    res = evaluate_method(SCORES, g)
    ci = res["match_macro_ci"]["rr"]
    assert ci["mean"] == pytest.approx(0.75)
    lo, hi = ci["cluster_ci"]
    assert lo <= ci["mean"] <= hi


def test_paired_difference_sign():
    # Method B is perfect (all rank 1) -> should beat SCORES on match-macro rr.
    perfect = {q: {e: (1.0 if e == g_pos else 0.0) for e in cand}
               for q, (g_pos, cand) in {
                   "qA1": ("A|1|1000", ["A|1|1000", "A|1|5000", "A|1|9000", "A|1|60000"]),
                   "qA2": ("A|1|5000", ["A|1|1000", "A|1|5000", "A|1|9000", "A|1|60000"]),
                   "qA3": ("A|1|9000", ["A|1|1000", "A|1|5000", "A|1|9000", "A|1|60000"]),
                   "qB1": ("B|1|2000", ["B|1|2000", "B|1|6000"]),
               }.items()}
    g = build_gallery(_manifest(), "test")
    a = evaluate_method(perfect, g)["_per_match_rr"]
    b = evaluate_method(SCORES, g)["_per_match_rr"]
    d = paired_cluster_bootstrap_mean_difference(a, b)
    assert d["difference_a_minus_b"] == pytest.approx(0.25)  # A perfect (1.0) - SCORES (0.75)
    # Only 2 matches; match B has zero A-B difference, so resampling {B,B} gives
    # diff 0 (not >0) -> P(A>B) < 1 even though A never loses. Just assert A wins.
    assert d["bootstrap_probability_a_gt_b"] > 0.5
    assert d["n_matches"] == 2


def test_restrict_gallery_keeps_only_present():
    # NewM1: dropped clips (static/short/non-finite) are excluded from queries + galleries.
    g = build_gallery(_manifest(), "test")
    all_ids = set(g.game_of_query) | {e for evs in g.events_of_game.values() for e in evs}
    assert set(restrict_gallery(g, all_ids).game_of_query) == set(g.game_of_query)  # no-op
    # a dropped distractor candidate is filtered from its game's gallery, queries survive
    r = restrict_gallery(g, all_ids - {"A|1|60000"})
    assert "A|1|60000" not in r.events_of_game["A"]
    assert set(r.game_of_query) == set(g.game_of_query)
    # a query whose POSITIVE event was dropped is itself removed (never silently unrankable)
    r2 = restrict_gallery(g, all_ids - {"A|1|1000"})  # qA1's positive
    assert "qA1" not in r2.game_of_query


def test_gallery_needs_two_candidates():
    # A game with a single event yields no rankable queries.
    m = {"events": [{"event_id": "C|1|1", "split": "test", "game": "C", "half": "1",
                     "anchor_ms": 1, "action_label": "Goal"}],
         "queries": [{"query_id": "qC", "split": "test", "game": "C", "event_id": "C|1|1",
                      "replay_half": "1", "replay_span_ms": [0, 1], "visibility": "visible",
                      "cohort": "primary", "cross_half": False}]}
    g = build_gallery(m, "test")
    assert per_query_metrics({"qC": {"C|1|1": 0.9}}, g) == {}

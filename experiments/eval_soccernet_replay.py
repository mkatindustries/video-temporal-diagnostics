#!/usr/bin/env python3
"""SoccerNet-v2 replay-grounding as within-match event retrieval.

Query = a replay clip. Gallery = the distinct live events of the SAME match
(same-match negatives = the smearing test). Positive = the query's linked event
(exact ``(link.half, link.position)`` join). Each query has exactly ONE positive
event, so AP == reciprocal rank; we therefore report R@1 / R@5 / MRR (not mAP).

This module is the method-agnostic protocol core: it consumes the manifest built
by ``scripts/setup_soccernet.py`` and, per method, a score map
``scores[query_id][event_id] -> float`` (higher = more similar, over the query's
own-match gallery). Method scoring (BoT cosine, encoder-seq DTW, SDM aggregator,
SONAR2-PE) is produced by the extraction/eval driver and passed in here, so the
metric logic is unit-testable without any features.

Reporting (locked with reviewer 2026-07-23):
  * Primary: match-macro R@1, R@5, MRR + paired match-clustered 95% CIs.
  * Secondary: event-macro, action-class-macro, query-micro.
  * Predeclared smearing-error composition on top-1 wrong retrievals.
Resampling unit = match (``game``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RANKS = (1, 5)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
# Predeclared smearing window: a wrong top-1 in the same half within this many ms
# of the true anchor is a "temporal-neighbour" error (vs same-action / other).
SMEARING_ADJ_MS = 15_000


@dataclass
class Gallery:
    """Per-split retrieval structure derived from the manifest."""
    game_of_query: dict[str, str]
    pos_event_of_query: dict[str, str]
    events_of_game: dict[str, list[str]]          # game -> ordered candidate event_ids
    event_anchor: dict[str, tuple[str, int]]      # event_id -> (half, anchor_ms)
    event_action: dict[str, str]                  # event_id -> action label
    query_cohort: dict[str, str]                  # query_id -> primary|review
    query_cross_half: dict[str, bool]


def build_gallery(manifest: dict, split: str, cohort: str = "primary") -> Gallery:
    """Build the within-match gallery for a split.

    cohort="primary": keep only primary (visibility=visible) queries; "all": also
    include the review cohort. Gallery events are ALL events of the game (the live
    negatives), independent of the query cohort.
    """
    events = [e for e in manifest["events"] if e["split"] == split]
    events_of_game: dict[str, list[str]] = defaultdict(list)
    event_anchor, event_action = {}, {}
    for e in events:
        events_of_game[e["game"]].append(e["event_id"])
        event_anchor[e["event_id"]] = (e["half"], int(e["anchor_ms"]))
        event_action[e["event_id"]] = e["action_label"]

    game_of_q, pos_of_q, cohort_of_q, cross_of_q = {}, {}, {}, {}
    for q in manifest["queries"]:
        if q["split"] != split:
            continue
        if cohort == "primary" and q["cohort"] != "primary":
            continue
        qid = q["query_id"]
        game_of_q[qid] = q["game"]
        pos_of_q[qid] = q["event_id"]
        cohort_of_q[qid] = q["cohort"]
        cross_of_q[qid] = bool(q["cross_half"])
    return Gallery(game_of_q, pos_of_q, dict(events_of_game), event_anchor,
                   event_action, cohort_of_q, cross_of_q)


def query_rank(scores_q: dict[str, float], candidates: list[str], positive: str) -> int:
    """1-based rank of the positive event among the game's candidates.

    Ties are broken so the positive gets the WORST rank among equal scores (a
    conservative choice: no credit for accidental ties).
    """
    pos_score = scores_q[positive]
    better = sum(1 for e in candidates if scores_q[e] > pos_score)
    equal = sum(1 for e in candidates if scores_q[e] == pos_score and e != positive)
    return better + equal + 1


def per_query_metrics(scores: dict[str, dict[str, float]], gallery: Gallery) -> dict[str, dict]:
    """Rank/RR/hit@k for every query with a valid same-match gallery (>=2 events)."""
    out: dict[str, dict] = {}
    for qid, pos in gallery.pos_event_of_query.items():
        game = gallery.game_of_query[qid]
        cands = gallery.events_of_game.get(game, [])
        if len(cands) < 2 or qid not in scores or pos not in scores[qid]:
            continue
        r = query_rank(scores[qid], cands, pos)
        out[qid] = {
            "game": game, "rank": r, "rr": 1.0 / r,
            **{f"r@{k}": float(r <= k) for k in RANKS},
            "gallery_size": len(cands),
        }
    return out


def _macro(values_by_group: dict[str, list[float]]) -> float:
    per_group = [float(np.mean(v)) for v in values_by_group.values() if v]
    return float(np.mean(per_group)) if per_group else float("nan")


def aggregate(pq: dict[str, dict], gallery: Gallery) -> dict:
    """Match-macro (primary), event-macro, action-class-macro, query-micro."""
    metric_keys = ["rr"] + [f"r@{k}" for k in RANKS]
    by_match: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_event: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_action: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    micro: dict[str, list[float]] = defaultdict(list)
    for qid, m in pq.items():
        pos = gallery.pos_event_of_query[qid]
        act = gallery.event_action.get(pos, "unknown")
        for key in metric_keys:
            by_match[m["game"]][key].append(m[key])
            by_event[pos][key].append(m[key])
            by_action[act][key].append(m[key])
            micro[key].append(m[key])
    return {
        "n_queries": len(pq),
        "n_matches": len(by_match),
        "match_macro": {k: _macro({g: d[k] for g, d in by_match.items()}) for k in metric_keys},
        "event_macro": {k: _macro({e: d[k] for e, d in by_event.items()}) for k in metric_keys},
        "action_macro": {k: _macro({a: d[k] for a, d in by_action.items()}) for k in metric_keys},
        "query_micro": {k: (float(np.mean(v)) if v else float("nan")) for k, v in micro.items()},
        "_per_match_rr": {g: float(np.mean(d["rr"])) for g, d in by_match.items()},
    }


def cluster_bootstrap_mean(per_match: dict[str, float], seed: int = BOOTSTRAP_SEED,
                           n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Match-clustered CI of a per-match mean (resampling unit = match)."""
    games = list(per_match)
    vals = np.array([per_match[g] for g in games], dtype=np.float64)
    point = float(vals.mean())
    rng = np.random.RandomState(seed)
    idx = np.arange(len(games))
    boot = np.array([vals[rng.choice(idx, len(idx), replace=True)].mean()
                     for _ in range(n_resamples)])
    return {"mean": point, "cluster_ci": [float(np.percentile(boot, 2.5)),
                                          float(np.percentile(boot, 97.5))]}


def paired_cluster_bootstrap_mean_difference(per_match_a: dict[str, float],
                                             per_match_b: dict[str, float],
                                             seed: int = BOOTSTRAP_SEED,
                                             n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Paired match-clustered CI of mean(A) - mean(B) over shared matches."""
    games = sorted(set(per_match_a) & set(per_match_b))
    if not games:
        raise ValueError("methods share no matches")
    a = np.array([per_match_a[g] for g in games])
    b = np.array([per_match_b[g] for g in games])
    rng = np.random.RandomState(seed)
    idx = np.arange(len(games))
    samples = (rng.choice(idx, len(idx), replace=True) for _ in range(n_resamples))
    diffs = np.array([(a[s].mean() - b[s].mean()) for s in samples])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "difference_a_minus_b": float(a.mean() - b.mean()),
            "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "bootstrap_probability_a_gt_b": float(np.mean(diffs > 0)), "n_matches": len(games)}


def smearing_composition(scores: dict[str, dict[str, float]], pq: dict[str, dict],
                         gallery: Gallery, adj_ms: int = SMEARING_ADJ_MS) -> dict:
    """Categorise top-1 WRONG retrievals: temporal-neighbour / same-action / other."""
    cats = {"temporal_neighbour": 0, "same_action": 0, "other": 0}
    n_wrong = 0
    for qid, m in pq.items():
        if m["rank"] == 1:
            continue
        n_wrong += 1
        pos = gallery.pos_event_of_query[qid]
        cands = gallery.events_of_game[gallery.game_of_query[qid]]
        top1 = max(cands, key=lambda e: scores[qid][e])
        ph, pa = gallery.event_anchor[pos]
        th, ta = gallery.event_anchor[top1]
        if ph == th and abs(ta - pa) <= adj_ms:
            cats["temporal_neighbour"] += 1
        elif gallery.event_action.get(top1) == gallery.event_action.get(pos):
            cats["same_action"] += 1
        else:
            cats["other"] += 1
    frac = {k: (v / n_wrong if n_wrong else 0.0) for k, v in cats.items()}
    return {"n_wrong_top1": n_wrong, "counts": cats, "fractions": frac,
            "adj_ms": adj_ms}


def evaluate_method(scores: dict[str, dict[str, float]], gallery: Gallery) -> dict:
    """Full metric bundle for one method's score map."""
    pq = per_query_metrics(scores, gallery)
    agg = aggregate(pq, gallery)
    per_match_rr = agg.pop("_per_match_rr")
    boots = {k: cluster_bootstrap_mean(
        {g: float(np.mean([pq[q][k] for q in pq if pq[q]["game"] == g]))
         for g in {pq[q]["game"] for q in pq}})
        for k in (["rr"] + [f"r@{k}" for k in RANKS])}
    return {"aggregate": agg, "match_macro_ci": boots,
            "smearing": smearing_composition(scores, pq, gallery),
            "_per_match_rr": per_match_rr}


# --------------------------------------------------------------------------- #
# CLI: load manifest + feature cache, score each method, write results JSON.
# Feature caches are produced by the (gated) extraction job; this errors clearly
# if they are absent so the protocol core stays runnable/testable on its own.
# --------------------------------------------------------------------------- #
def _cosine_scores(query_vecs, event_vecs, gallery: Gallery) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for qid, game in gallery.game_of_query.items():
        if qid not in query_vecs:
            continue
        qv = query_vecs[qid]
        scores[qid] = {e: float(np.dot(qv, event_vecs[e]))
                       for e in gallery.events_of_game.get(game, []) if e in event_vecs}
    return scores


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True,
                   help="per-clip embeddings {'queries':{qid:vec},'events':{eid:vec}} per method "
                        "(built by the extraction job)")
    p.add_argument("--split", default="test")
    p.add_argument("--cohort", default="primary", choices=["primary", "all"])
    p.add_argument("--output", type=Path, default=Path("results/soccernet/replay_results.json"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    gallery = build_gallery(manifest, args.split, args.cohort)
    if not args.feature_cache.exists():
        raise SystemExit(
            f"feature cache {args.feature_cache} not found; run the (gated) SoccerNet "
            "extraction job first. The protocol core in this module is importable/testable "
            "without features (see tests/test_soccernet_protocol.py).")

    import torch
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    results = {"protocol": {
        "dataset": "soccernet_v2_replay_grounding", "task": "within-match event retrieval",
        "split": args.split, "cohort": args.cohort,
        "gallery": "distinct live events of the same match (same-match negatives)",
        "positive": "exact (link.half, link.position) event; single positive per query -> AP==RR",
        "metrics": ("match-macro R@1/R@5/MRR + paired match-clustered CIs; "
                    "secondary event/action/query-micro; smearing composition"),
        "resampling_unit": "match",
    }, "methods": {}}
    for method, blob in cache.items():
        scores = _cosine_scores(blob["queries"], blob["events"], gallery)
        results["methods"][method] = {k: v for k, v in evaluate_method(scores, gallery).items()
                                      if not k.startswith("_")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

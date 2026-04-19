"""CLI entry point for the temporal-diag diagnostic toolkit.

Usage::

    temporal-diag scramble-gradient \\
        --embeddings-a features_a.pt --embeddings-b features_b.pt \\
        --pairs pairs.csv --similarity cosine --k-values 1 4 16 \\
        --output report.json

    temporal-diag s-rev \\
        --embeddings features.pt --similarity cosine --output report.json

    temporal-diag report \\
        --embeddings-a features_a.pt --embeddings-b features_b.pt \\
        --pairs pairs.csv --similarity cosine --output report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from video_retrieval.fingerprints.dtw import dtw_distance

from .report import temporal_report
from .reversal import compute_s_rev
from .scramble import scramble_gradient


# ---- Built-in similarity functions ----------------------------------------


def _cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between mean-pooled embeddings."""
    va = F.normalize(a.mean(dim=0), dim=0)
    vb = F.normalize(b.mean(dim=0), dim=0)
    return float(torch.dot(va, vb).item())


def _dtw_similarity(a: Tensor, b: Tensor) -> float:
    """DTW-based similarity: exp(-dtw_distance)."""
    return float(torch.exp(torch.tensor(-dtw_distance(a, b))).item())


SIMILARITY_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], float]] = {
    "cosine": _cosine_similarity,
    "dtw": _dtw_similarity,
}


# ---- Helpers ---------------------------------------------------------------


def _load_embeddings(path: str) -> dict[str, Tensor]:
    """Load a .pt file mapping video IDs to (T, D) tensors."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def _load_pairs(path: str) -> list[tuple[str, str, int]]:
    """Load pairs CSV with columns: id_a, id_b, label."""
    pairs = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Skip header if present
        if header and header[0].strip().lower() in ("id_a", "video_a", "query"):
            pass
        else:
            # No header — treat first row as data
            if header:
                pairs.append((header[0].strip(), header[1].strip(), int(header[2].strip())))
        for row in reader:
            if len(row) >= 3:
                pairs.append((row[0].strip(), row[1].strip(), int(row[2].strip())))
    return pairs


def _get_similarity(name: str) -> Callable[[Tensor, Tensor], float]:
    if name not in SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"Unknown similarity '{name}'. Choose from: {list(SIMILARITY_FUNCTIONS)}"
        )
    return SIMILARITY_FUNCTIONS[name]


def _write_output(result: dict, path: str | None) -> None:
    text = json.dumps(result, indent=2)
    if path:
        Path(path).write_text(text)
        print(f"Report written to {path}")
    else:
        print(text)


# ---- Subcommands -----------------------------------------------------------


def cmd_scramble_gradient(args: argparse.Namespace) -> None:
    emb_a = _load_embeddings(args.embeddings_a)
    emb_b = _load_embeddings(args.embeddings_b)
    pairs = _load_pairs(args.pairs)
    sim_fn = _get_similarity(args.similarity)

    result = scramble_gradient(
        emb_a, emb_b, pairs, sim_fn, k_values=args.k_values, seed=args.seed
    )
    _write_output(result, args.output)


def cmd_s_rev(args: argparse.Namespace) -> None:
    emb = _load_embeddings(args.embeddings)
    sim_fn = _get_similarity(args.similarity)

    result = compute_s_rev(emb, sim_fn)
    # Drop per-video detail for CLI output (can be large)
    summary = {"mean": result["mean"], "std": result["std"], "n_videos": len(result["per_video"])}
    _write_output(summary, args.output)


def cmd_report(args: argparse.Namespace) -> None:
    emb_a = _load_embeddings(args.embeddings_a)
    emb_b = _load_embeddings(args.embeddings_b)
    pairs = _load_pairs(args.pairs)
    sim_fn = _get_similarity(args.similarity)

    result = temporal_report(
        emb_a, emb_b, pairs, sim_fn, k_values=args.k_values, seed=args.seed
    )
    _write_output(result, args.output)


# ---- Argument parser -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temporal-diag",
        description="Diagnostic toolkit for temporal sensitivity in video retrieval embeddings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- scramble-gradient ---------------------------------------------------
    sg = sub.add_parser(
        "scramble-gradient",
        help="Run the temporal scramble gradient test.",
    )
    sg.add_argument("--embeddings-a", required=True, help="Path to reference embeddings (.pt)")
    sg.add_argument("--embeddings-b", required=True, help="Path to query embeddings (.pt)")
    sg.add_argument("--pairs", required=True, help="CSV with columns: id_a, id_b, label")
    sg.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_FUNCTIONS))
    sg.add_argument(
        "--k-values", nargs="+", type=int, default=[1, 4, 16],
        help="Chunk counts to sweep (default: 1 4 16)",
    )
    sg.add_argument("--seed", type=int, default=0)
    sg.add_argument("--output", help="Output JSON path (prints to stdout if omitted)")
    sg.set_defaults(func=cmd_scramble_gradient)

    # -- s-rev ---------------------------------------------------------------
    sr = sub.add_parser(
        "s-rev",
        help="Compute reversal sensitivity (s_rev) per video.",
    )
    sr.add_argument("--embeddings", required=True, help="Path to embeddings (.pt)")
    sr.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_FUNCTIONS))
    sr.add_argument("--output", help="Output JSON path (prints to stdout if omitted)")
    sr.set_defaults(func=cmd_s_rev)

    # -- report --------------------------------------------------------------
    rp = sub.add_parser(
        "report",
        help="Run full diagnostic report (scramble gradient + s_rev).",
    )
    rp.add_argument("--embeddings-a", required=True, help="Path to reference embeddings (.pt)")
    rp.add_argument("--embeddings-b", required=True, help="Path to query embeddings (.pt)")
    rp.add_argument("--pairs", required=True, help="CSV with columns: id_a, id_b, label")
    rp.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_FUNCTIONS))
    rp.add_argument(
        "--k-values", nargs="+", type=int, default=[1, 4, 16],
        help="Chunk counts to sweep (default: 1 4 16)",
    )
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--output", help="Output JSON path (prints to stdout if omitted)")
    rp.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

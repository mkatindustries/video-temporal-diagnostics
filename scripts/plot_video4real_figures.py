#!/usr/bin/env python3
"""Generate Video4Real figures from tracked result JSONs.

No GPU; reads results/{hdd,nuscenes}/*.json and writes figures/v4r_*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def cascade_figure() -> None:
    hdd = load("results/hdd/bof_dtw_directed_rerank_results.json")
    nus = load("results/nuscenes/fusion_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for ax, data, title in (
        (axes[0], hdd, "Honda HDD"),
        (axes[1], nus, "nuScenes"),
    ):
        ks = sorted((int(k) for k in data["k_sweep"]))
        bot = [data["k_sweep"][str(k)]["bot"]["ap"]["mean"] for k in ks]
        dtw = [data["k_sweep"][str(k)]["dtw_rerank"]["ap"]["mean"] for k in ks]
        rec = [data["k_sweep"][str(k)]["bot"]["recall"]["mean"] for k in ks]
        ax.plot(ks, bot, "o-", color="#1f77b4", label="BoT AP@k")
        ax.plot(ks, dtw, "s--", color="#d62728", label="BoT$\\to$DTW rerank AP@k")
        ax.plot(ks, rec, "^:", color="#7f7f7f", label="recall@k (shared)")
        # Keep the numeric axis linear, but omit the second early label where the
        # tightly spaced low-k ticks would collide at the paper's rendered width.
        displayed_ticks = [ks[0], *ks[2:]]
        ax.set_xticks(displayed_ticks)
        ax.set_xticklabels([str(k) for k in displayed_ticks])
        ax.set_xlabel("k (candidates reranked)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("AP@k / recall@k")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG / "v4r_cascade.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def loco_figure() -> None:
    hdd = load("results/hdd/fusion_results.json")
    nus = load("results/nuscenes/fusion_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for ax, data, title, star in (
        (axes[0], hdd, "Honda HDD", 0.95),
        (axes[1], nus, "nuScenes", 1.0),
    ):
        alphas = data["alpha_grid"]
        folds = data["alpha_selected"]["folds"]
        curves = [f["training_map_by_alpha"] for f in folds]
        for c in curves:
            ax.plot(alphas, c, color="#1f77b4", alpha=0.12, linewidth=0.6)
        mean = [sum(col) / len(col) for col in zip(*curves)]
        ax.plot(alphas, mean, color="#1f77b4", linewidth=2.0, label="mean over folds")
        ax.axvline(star, color="#d62728", linestyle="--", linewidth=1.2,
                   label="selected $\\alpha^\\star$")
        ax.set_xlabel("$\\alpha$ (BoT weight)")
        ax.set_title(f"{title} ({len(folds)} folds)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("training mAP")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = FIG / "v4r_loco_alpha.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def error_composition_figure() -> None:
    """Plot whether top-1 failures confuse maneuvers or intersection locations."""
    import numpy as np

    hdd = load("results/hdd/fusion_results.json")
    nus = load("results/nuscenes/fusion_results.json")
    key = "ranked_outcome_composition"
    if key not in hdd or key not in nus:
        print(
            "skipped figures/v4r_error_composition.png: rerun both fusion evaluators "
            "to add ranked_outcome_composition"
        )
        return

    categories = [
        ("relevant", "Relevant", "#2ca02c"),
        ("same_cluster_wrong_label", "Right location, wrong maneuver", "#ffbf00"),
        ("wrong_cluster", "Wrong location", "#7f7f7f"),
    ]
    methods = [("bot", "BoT"), ("encoder_seq_dtw", "Encoder-seq\nDTW")]
    if all(
        "temporal_residual_dtw" in data[key]["methods"] for data in (hdd, nus)
    ):
        methods.append(("temporal_residual_dtw", "Temporal-residual\nDTW"))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), sharey=True)
    for ax, data, title in (
        (axes[0], hdd, "Honda HDD"),
        (axes[1], nus, "nuScenes"),
    ):
        composition = data[key]["methods"]
        x = np.arange(len(methods))
        bottom = np.zeros(len(methods))
        for category, label, color in categories:
            values = np.asarray(
                [composition[method]["1"][category]["mean"] for method, _ in methods]
            )
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.66)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in methods])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction of top-1 outcomes")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    out = FIG / "v4r_error_composition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def vlm_figure() -> None:
    """MLLM interface failure: prompt dependence and pooled-vs-sequence readout s_rev.

    Numbers are transcribed from the tracked main paper (paper/paper.tex): EPIC
    forward/reverse balanced accuracy (direct vs. integrity prompt) and Table
    tab:summary_vlm EPIC s_rev for pooled (cosine) vs. per-frame-sequence (DTW) readouts.
    """
    import numpy as np

    models = ["Gemma-4", "LLaVA-Video", "Qwen3-VL"]
    x = np.arange(len(models))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))

    # Panel A: direct direction prompt vs integrity prompt (balanced accuracy).
    direct = [0.538, 0.503, 0.508]
    integrity = [0.845, float("nan"), 0.879]  # LLaVA integrity is degenerate/constant
    axes[0].bar(x - w / 2, direct, w, color="#7f7f7f", label="direct direction prompt")
    axes[0].bar(x + w / 2, integrity, w, color="#2ca02c", label="integrity prompt")
    axes[0].axhline(0.5, color="k", linestyle=":", linewidth=0.8, label="chance")
    axes[0].text(1 + w / 2, 0.06, "degenerate", fontsize=7, rotation=90,
                 ha="center", va="bottom", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=8)
    axes[0].set_ylabel("EPIC fwd/rev balanced acc.")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Prompt dependence")
    axes[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.14),
                   ncol=2, columnspacing=0.8, handlelength=1.6)

    # Panel B: pooled (cosine) vs per-frame sequence (DTW) reversal score s_rev.
    pooled = [1.000, 1.000, 0.998]
    sequence = [0.492, 0.495, 0.542]
    axes[1].bar(x - w / 2, pooled, w, color="#1f77b4", label="pooled (cosine)")
    axes[1].bar(x + w / 2, sequence, w, color="#d62728", label="per-frame seq. (DTW)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=8)
    axes[1].set_ylabel("$s_\\mathrm{rev}$ (comparator-specific)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Readout dependence")
    axes[1].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.14),
                   ncol=2, columnspacing=0.8, handlelength=1.6)

    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = FIG / "v4r_vlm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    functions = {
        "cascade": cascade_figure,
        "loco": loco_figure,
        "vlm": vlm_figure,
        "error-composition": error_composition_figure,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures",
        nargs="*",
        choices=functions,
        help="Figures to generate (default: all).",
    )
    args = parser.parse_args()
    selected = args.figures or list(functions)
    for name in selected:
        functions[name]()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Video4Real cascade and LOCO-alpha figures from tracked result JSONs.

No GPU; reads results/{hdd,nuscenes}/*.json and writes figures/v4r_*.png.
"""
from __future__ import annotations

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
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
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
                   label=f"selected $\\alpha^\\star$={star:g}")
        ax.set_xlabel("$\\alpha$ (BoT weight)")
        ax.set_title(f"{title} ({len(folds)} folds)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("training mAP")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = FIG / "v4r_loco_alpha.png"
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
    axes[0].legend(fontsize=7, loc="upper left")

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
    axes[1].legend(fontsize=7, loc="center right")

    fig.tight_layout()
    out = FIG / "v4r_vlm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    cascade_figure()
    loco_figure()
    vlm_figure()

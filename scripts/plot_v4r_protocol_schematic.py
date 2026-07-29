"""Regenerate the Video4Real conditional-vs-global protocol schematic.

Illustrative icons (not real detections) set up the protocol; the bar values are
real, read from the tracked result JSONs rather than hardcoded, so a future rerun
can't leave the figure silently stale. Left panel: within-intersection query-macro
mAP (conditional). Right panel: full-gallery query-macro mAP (global). Both use the
same query and relevance rule -- only the gallery expands.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

project_root = Path(__file__).parent.parent
fig_dir = project_root / "figures"
results_dir = project_root / "results"

GREEN = "#2ca02c"
AMBER = "#e6a817"
GRAY = "#b0b0b0"
BOT_COLOR = "#4c72b0"
DTW_COLOR = "#8e44ad"


def _load_conditional(dataset: str) -> dict:
    d = json.loads((results_dir / dataset / "conditional_querywise_results.json").read_text())
    m = d["methods"]
    return {
        "BoT": m["bot_cosine"]["query_macro_ap"]["mean"],
        "DTW": m["encoder_seq_dtw"]["query_macro_ap"]["mean"],
    }


def _load_global(dataset: str) -> dict:
    d = json.loads((results_dir / dataset / "fusion_results.json").read_text())
    return {
        "BoT": d["bot_full_gallery"]["map"]["mean"],
        "DTW": d["encoder_seq_dtw_full_gallery"]["map"]["mean"],
    }


COND_HDD = _load_conditional("hdd")
COND_NUS = _load_conditional("nuscenes")
GLOB_HDD = _load_global("hdd")
GLOB_NUS = _load_global("nuscenes")


def clip(ax, x, y, color, w=0.5, h=0.34, z=3):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=color, edgecolor="none", zorder=z)
    ax.add_patch(box)
    ax.plot([x - 0.07], [y], marker=(3, 0, -90), markersize=5.5, color="white", zorder=z + 1)


def gallery_scope(ax, cx, cy, include_others):
    offsets = [(-0.32, 0.2), (0.32, 0.2), (-0.32, -0.2), (0.32, -0.2)]
    colors = [GREEN, GREEN, AMBER, AMBER]
    for (dx, dy), c in zip(offsets, colors):
        clip(ax, cx + dx, cy + dy, c)
    ax.add_patch(Rectangle((cx - 0.62, cy - 0.42), 1.24, 0.84, fill=False,
                            linestyle=(0, (3, 3)), edgecolor="#888888", linewidth=1.1, zorder=1))
    if include_others:
        for ocx, ocy in [(-1.55, 0.55), (1.55, 0.5), (-1.65, -0.6), (1.6, -0.65)]:
            for dx, dy in [(-0.16, 0.1), (0.16, 0.1), (0.0, -0.13)]:
                clip(ax, ocx + dx, ocy + dy, GRAY, w=0.36, h=0.24)


def bar_group(ax, hdd, nus, winner_label, winner_color):
    scale = 2.0
    groups = [("HDD", hdd), ("nuScenes", nus)]
    x = 0.0
    centers = []
    for _gname, vals in groups:
        for m, c in [("BoT", BOT_COLOR), ("DTW", DTW_COLOR)]:
            v = vals[m]
            ax.add_patch(Rectangle((x - 0.16, 0), 0.32, v * scale, facecolor=c, zorder=3))
            ax.text(x, v * scale + 0.05, f"{v:.3f}", ha="center", fontsize=8.3,
                    fontweight="bold", color=c)
            x += 0.38
        centers.append(x - 0.38 - 0.16)
        x += 0.22
    ax.text(centers[0] + 0.03, -0.13, "HDD", ha="center", fontsize=9, color="#333333")
    ax.text(centers[1] + 0.03, -0.13, "nuScenes", ha="center", fontsize=9, color="#333333")
    ax.text((centers[0] + centers[1]) / 2 + 0.03, 2.35, winner_label, ha="center",
            fontsize=10, fontweight="bold", color=winner_color)
    ax.set_xlim(-0.3, x + 0.1)
    ax.set_ylim(-0.35, 2.65)


def main() -> None:
    fig = plt.figure(figsize=(10.5, 4.5))
    fig.suptitle("Same query, same relevance rule, same two comparators — only the gallery expands.",
                 fontsize=12.5, fontweight="bold", y=0.99)

    legend_ax = fig.add_axes((0.03, 0.88, 0.94, 0.07))
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    lx = 0.12
    for c, label in [(GREEN, "relevant"), (AMBER, "same intersection, wrong maneuver"),
                      (GRAY, "wrong intersection"), (BOT_COLOR, "BoT"), (DTW_COLOR, "Enc-seq DTW")]:
        legend_ax.add_patch(Rectangle((lx, 0.2), 0.012, 0.6, facecolor=c, transform=legend_ax.transAxes))
        legend_ax.text(lx + 0.018, 0.5, label, fontsize=9, va="center", transform=legend_ax.transAxes)
        lx += 0.018 + 0.0095 * len(label) + 0.03

    axA = fig.add_axes((0.03, 0.48, 0.44, 0.32))
    axB = fig.add_axes((0.53, 0.48, 0.44, 0.32))
    for ax, title, others in [(axA, "Conditional: query's intersection only", False),
                              (axB, "Global: every retained intersection", True)]:
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-0.85, 0.85)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=2)
        gallery_scope(ax, 0.0, 0.0, others)

    barA = fig.add_axes((0.10, 0.01, 0.32, 0.48))
    barB = fig.add_axes((0.60, 0.01, 0.32, 0.48))
    barA.axis("off")
    barB.axis("off")
    bar_group(barA, COND_HDD, COND_NUS, "DTW ranks ahead of BoT", DTW_COLOR)
    bar_group(barB, GLOB_HDD, GLOB_NUS, "BoT ranks ahead of DTW", BOT_COLOR)
    barA.text((barA.get_xlim()[0] + barA.get_xlim()[1]) / 2, -0.32, "within-intersection mAP",
              ha="center", fontsize=9.5, color="#333333")
    barB.text((barB.get_xlim()[0] + barB.get_xlim()[1]) / 2, -0.32, "full-gallery mAP",
              ha="center", fontsize=9.5, color="#333333")

    out = fig_dir / "v4r_protocol_schematic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

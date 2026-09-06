"""Render the Video4Real poster charts at print resolution.

The main scorecard reads ``poster/production_scorecard.json``, a public-safe
snapshot of the immutable 13-by-5 production scorecard. The matched
conditional/global diagnostic reads tracked JSONs under ``results/`` and is
kept visually separate because it uses a different protocol.

Figures are sized in real inches to match their slot on the 1400x1000 mm poster,
so matplotlib point sizes below ARE the printed point sizes.

    conda activate video_retrieval
    python poster/charts.py            # writes poster/build/*.png at 300 dpi
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

import tokens as T  # noqa: E402

# Figure heights in mm. The figures leave room for the evidence band at the
# foot of the page.
H = {
    "scorecard": 235,
    "diagnostic_pair": 205,
    "reversal_compact": 185,
    "schematic": 108,
    "reversal": 360,
    "errors": 315,
    "cascade": 238,
    "soccernet": 120,
    "copy_hdd": 300,
    "soccernet_evidence": 190,
    "aria_order": 150,
    "evidence_map": 142,
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "build"
DPI = 300
MM = 1.0 / 25.4  # mm -> inch

# Type scale, in printed points. Figures are placed on the poster at 1:1, so a
# point here is a point on paper. 20 pt is the floor for anything a reader is
# meant to decode -- below that the chart type falls under the 21-25 pt body
# copy beside it, which reads as an accident at poster viewing distance.
# Captions stay at 18 pt so each remains a single line inside the figure width.
TITLE_PT = 21
LABEL_PT = 21  # axis labels
TICK_PT = 20
LEGEND_PT = 20
ANNOT_PT = 20  # value labels and in-chart annotation
CAPTION_PT = 18


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text())


def load_scorecard() -> dict:
    return json.loads(
        (Path(__file__).resolve().parent / "production_scorecard.json").read_text()
    )


# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": T.SURFACE,
            "axes.facecolor": T.SURFACE,
            "savefig.facecolor": T.SURFACE,
            "axes.edgecolor": T.BASELINE,
            "axes.linewidth": 1.1,
            "axes.grid": False,
            "text.color": T.INK,
            "axes.labelcolor": T.INK_2,
            # Tick labels are text a reader decodes, so they take INK_2 (7.9:1 on
            # white). MUTED is 3.6:1 and fails WCAG AA -- it is a hairline/rule
            # colour only, never type.
            "xtick.color": T.INK_2,
            "ytick.color": T.INK_2,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def hgrid(ax, ticks) -> None:
    """Recessive hairline gridlines, drawn under the marks."""
    for y in ticks:
        ax.axhline(y, color=T.GRID, lw=1.0, zorder=0, solid_capstyle="butt")


def capped_bar(ax, x, w, h, color, radius_frac=0.16, zorder=3):
    """Bar with a rounded data-end and a square baseline (mark spec)."""
    if h <= 0:
        return
    r = min(w * radius_frac, h * 0.9)
    verts = [
        (x, 0.0),
        (x, h - r),
        (x, h),
        (x + r, h),
        (x + w - r, h),
        (x + w, h),
        (x + w, h - r),
        (x + w, 0.0),
        (x, 0.0),
    ]
    codes = [
        MPath.MOVETO,
        MPath.LINETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.LINETO,
        MPath.CURVE3,
        MPath.CURVE3,
        MPath.LINETO,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder)
    )


def legend_row(fig, entries, x0=0.015, y=0.985, size=LEGEND_PT):
    """Swatch + text-token label, laid out by matplotlib so labels cannot collide.

    Text never wears the data color -- identity comes from the swatch beside it.
    """
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for c, lab in entries]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x0, y),
        ncol=len(entries),
        frameon=False,
        fontsize=size,
        labelcolor=T.INK_2,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.55,
        columnspacing=2.2,
        borderpad=0.0,
    )


# ----------------------------------------------------------------------
# 1. protocol schematic
# ----------------------------------------------------------------------
def clipbox(ax, x, y, color, w=0.50, h=0.34, z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=color,
            edgecolor="none",
            zorder=z,
        )
    )
    ax.plot(
        [x],
        [y],
        marker=(3, 0, -90),
        markersize=max(4.0, 22.0 * h),
        color="white",
        zorder=z + 1,
    )


def gallery(ax, include_others: bool) -> None:
    """Four in-cluster candidates; optionally the rest of the world around them."""
    for (dx, dy), c in zip(
        [(-0.30, 0.19), (0.30, 0.19), (-0.30, -0.19), (0.30, -0.19)],
        [T.OUT_RELEVANT, T.OUT_RELEVANT, T.OUT_WRONG_MANEUVER, T.OUT_WRONG_MANEUVER],
    ):
        clipbox(ax, dx, dy, c, w=0.44, h=0.30)
    ax.add_patch(
        Rectangle(
            (-0.60, -0.41),
            1.20,
            0.82,
            fill=False,
            linestyle=(0, (4, 4)),
            edgecolor=T.MUTED,
            linewidth=1.6,
            zorder=1,
        )
    )
    ax.text(
        0.0, 0.50, "query's intersection", ha="center", va="bottom",
        fontsize=ANNOT_PT, color=T.INK_2, zorder=4,
    )
    if include_others:
        for ocx, ocy in [(-1.14, 0.26), (1.14, 0.26), (-1.14, -0.30), (1.14, -0.30)]:
            for dx, dy in [(-0.17, 0.13), (0.17, 0.13), (0.0, -0.13)]:
                clipbox(ax, ocx + dx, ocy + dy, T.OUT_WRONG_PLACE, w=0.27, h=0.19)
        ax.text(
            0.0, -0.60, "+ every other retained intersection", ha="center", va="top",
            fontsize=ANNOT_PT, color=T.INK_2, zorder=4,
        )


def fig_schematic() -> Path:
    # Panels use aspect="equal", so the axes boxes and the data ranges are sized
    # to the same ratio -- otherwise the drawing letterboxes inside its box.
    fig = plt.figure(figsize=(300 * MM, H["schematic"] * MM))

    legend_row(
        fig,
        [
            (T.OUT_RELEVANT, "relevant"),
            (T.OUT_WRONG_MANEUVER, "wrong maneuver"),
            (T.OUT_WRONG_PLACE, "wrong intersection"),
        ],
        x0=0.020,
        y=0.995,
        size=LEGEND_PT,
    )

    box_w, box_h = 0.462, 0.660          # axes fraction
    ratio = (box_w * 300.0) / (box_h * (H["schematic"] - 20.0))
    y_lo, y_hi = -0.80, 0.68
    x_half = (y_hi - y_lo) * ratio / 2.0

    for i, (title, others) in enumerate(
        [("Conditional: one intersection", False), ("Global: all intersections", True)]
    ):
        ax = fig.add_axes((0.018 + i * 0.502, 0.035, box_w, box_h))
        ax.set_xlim(-x_half, x_half)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold", color=T.INK, pad=6)
        gallery(ax, others)

    out = OUT / "p_schematic.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 2. the reversal
# ----------------------------------------------------------------------
def fig_reversal() -> Path:
    hc = load("hdd/conditional_querywise_results.json")
    nc = load("nuscenes/conditional_querywise_results.json")
    hf = load("hdd/fusion_results.json")
    nf = load("nuscenes/fusion_results.json")

    order = ["bot_cosine", "encoder_seq_dtw", "temporal_residual_dtw"]
    gkey = {
        "bot_cosine": "bot_full_gallery",
        "encoder_seq_dtw": "encoder_seq_dtw_full_gallery",
        "temporal_residual_dtw": "temporal_residual_dtw_full_gallery",
    }

    def cond(d, m):
        v = d["methods"][m]["query_macro_ap"]
        return v["mean"], v["cluster_ci"]

    def glob(d, m):
        v = d[gkey[m]]["map"]
        return v["mean"], v["cluster_ci"]

    panels = [
        ("Conditional retrieval — within the query's intersection", cond, (hc, nc)),
        ("Global retrieval — full evaluation gallery", glob, (hf, nf)),
    ]

    fig = plt.figure(figsize=(300 * MM, H["reversal"] * MM))
    legend_row(
        fig,
        [(T.METHOD_COLOR[m], T.METHOD_LABEL[m]) for m in order],
        x0=0.105,
        y=0.995,
        size=LEGEND_PT,
    )

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for pi, (title, getter, data) in enumerate(panels):
        ax = fig.add_axes((0.115, 0.545 - pi * 0.485, 0.865, 0.365))
        hgrid(ax, ticks)
        bw, gap_in, gap_between = 0.105, 0.022, 0.40
        centers = []
        for di, d in enumerate(data):
            base = di * (3 * bw + 2 * gap_in + gap_between)
            for mi, m in enumerate(order):
                mean, ci = getter(d, m)
                x = base + mi * (bw + gap_in)
                capped_bar(ax, x, bw, mean, T.METHOD_COLOR[m])
                ax.plot(
                    [x + bw / 2, x + bw / 2],
                    ci,
                    color=T.INK_2,
                    lw=1.8,
                    zorder=5,
                    solid_capstyle="butt",
                )
                for yy in ci:
                    ax.plot(
                        [x + bw * 0.30, x + bw * 0.70],
                        [yy, yy],
                        color=T.INK_2,
                        lw=1.8,
                        zorder=5,
                    )
                # Three ANNOT_PT labels will not sit side by side over three bars.
                # Stagger all three explicitly; this also keeps HDD's 0.256/0.177
                # global labels apart when their marginal CIs pull them together.
                label_offset = (0.030, 0.075, 0.025)[mi]
                ax.text(
                    x + bw / 2,
                    max(ci[1], mean) + label_offset,
                    f"{mean:.3f}",
                    ha="center",
                    fontsize=ANNOT_PT,
                    fontweight="bold",
                    color=T.INK,
                    zorder=6,
                )
            centers.append(base + (3 * bw + 2 * gap_in) / 2)

        ax.set_xticks(centers)
        ax.set_xticklabels(["Honda HDD", "nuScenes"], fontsize=TICK_PT, color=T.INK)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=TICK_PT)
        ax.set_ylim(0, 1.12)  # headroom for the lifted middle value label
        half = (3 * bw + 2 * gap_in) / 2
        ax.set_xlim(-half - 0.10, centers[-1] + half + 0.10)
        ax.set_ylabel("query-macro mAP", fontsize=LABEL_PT, color=T.INK_2, labelpad=10)
        ax.set_title(
            title, fontsize=TITLE_PT, fontweight="bold", color=T.INK, pad=14, loc="left"
        )
        despine(ax)
        ax.spines["left"].set_color(T.BASELINE)
        ax.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.115,
        0.012,
        "Bars: cluster-bootstrap mean.  Whiskers: marginal 95% intersection-cluster CI.",
        fontsize=CAPTION_PT,
        color=T.INK_2,
    )

    out = OUT / "p_reversal.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Poster hero: production scorecard and compact matched diagnostic
# ----------------------------------------------------------------------
def _scorecard_value(dataset: dict, value: float) -> str:
    """Format scorecard values exactly as rates, without mixing task units."""
    return f"{value:.6f}"


def fig_scorecard() -> Path:
    """Six public-safe recipes from the complete production scorecard.

    Ranks are deliberately recomputed within the six displayed recipes. This
    keeps excluded recipes out of the public view, and the denominator is
    printed in every cell so the restricted scope cannot be mistaken for the
    complete 13-recipe ordering.
    """
    d = load_scorecard()
    datasets = d["datasets"]
    recipes = d["recipes"]

    fig = plt.figure(figsize=(1320 * MM, H["scorecard"] * MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left = 0.205
    right = 0.995
    header_y = 0.910
    table_top = 0.790
    table_bottom = 0.185
    col_w = (right - left) / len(datasets)
    row_h = (table_top - table_bottom) / len(recipes)

    headings = {
        "synthetic_copies": ("Internal synthetic copies", "best F1  ↑"),
        "vcdb": ("VCDB", "copy AP  ↑"),
        "soccernet": ("SoccerNet-v2", "match-macro MRR  ↑"),
        "hdd": ("Honda HDD", "global AP  ↑"),
        "aria": ("Project Aria", "false-merge rate  ↓"),
    }
    for j, dataset in enumerate(datasets):
        x = left + (j + 0.5) * col_w
        title, metric = headings[dataset["key"]]
        ax.text(
            x,
            header_y,
            title,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=T.INK,
        )
        ax.text(
            x,
            header_y - 0.065,
            metric,
            ha="center",
            va="center",
            fontsize=17,
            color=T.INK_2,
        )

    label_lines = {
        "internvideo_next_l": "InternVideo-Next L",
        "sam3_perception_encoder": "SAM3 Perception\nEncoder",
        "internal_copy_detector": "Internal copy\ndetector",
        "dinov3": "DINOv3",
        "vjepa2": "V-JEPA 2",
        "levjepa": "LeVJEPA*",
    }
    leader_fill = "#eeeafe"
    for i, recipe in enumerate(recipes):
        y_top = table_top - i * row_h
        y_bottom = y_top - row_h
        y_mid = (y_top + y_bottom) / 2
        base_fill = T.PLANE if i % 2 == 0 else T.SURFACE
        ax.add_patch(
            Rectangle(
                (0.002, y_bottom),
                right - 0.002,
                row_h,
                facecolor=base_fill,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.text(
            0.012,
            y_mid,
            label_lines[recipe["key"]],
            ha="left",
            va="center",
            fontsize=20,
            fontweight="bold" if recipe["key"] in ("vjepa2", "levjepa") else "normal",
            color=T.INK,
            linespacing=1.12,
        )

        for j, dataset in enumerate(datasets):
            key = dataset["key"]
            rank = recipe["ranks_among_shown"][key]
            x_left = left + j * col_w
            x_mid = x_left + col_w / 2
            if rank == 1:
                ax.add_patch(
                    FancyBboxPatch(
                        (x_left + 0.006, y_bottom + 0.010),
                        col_w - 0.012,
                        row_h - 0.020,
                        boxstyle="round,pad=0.002,rounding_size=0.008",
                        facecolor=leader_fill,
                        edgecolor=T.ACCENT,
                        linewidth=1.3,
                        zorder=1,
                    )
                )
            ax.text(
                x_mid,
                y_mid + 0.017,
                _scorecard_value(dataset, recipe["values"][key]),
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold" if rank == 1 else "normal",
                color=T.ACCENT if rank == 1 else T.INK,
                zorder=2,
            )
            ax.text(
                x_mid,
                y_mid - 0.027,
                f"rank {rank}/6 shown",
                ha="center",
                va="center",
                fontsize=15,
                color=T.INK_2,
                zorder=2,
            )

    for j in range(len(datasets) + 1):
        x = left + j * col_w
        ax.plot([x, x], [table_bottom, table_top], color=T.RULE, lw=0.9, zorder=0)
    ax.plot([0.002, right], [table_top, table_top], color=T.BASELINE, lw=1.2)
    ax.plot([0.002, right], [table_bottom, table_bottom], color=T.BASELINE, lw=1.2)

    ax.text(
        0.012,
        0.115,
        "Each row is a complete model + frame schedule + preprocessing + pooling recipe. "
        "Ranks are descriptive within the six shown; no paired intervals. Compare values only "
        "within a task.",
        fontsize=18,
        color=T.INK_2,
        style="italic",
        va="center",
    )
    ax.text(
        0.012,
        0.045,
        "VCDB: 527 videos, 5,585 + 5,585 matched pairs  ·  SoccerNet-v2: clip cosine, "
        "6,376 + 28 queries  ·  HDD: 97,731 global pairs, chance .4469  ·  "
        "Aria: 78,324 negatives, transferred P99.  * Supplemental recipe.",
        fontsize=16,
        color=T.INK_2,
        style="italic",
        va="center",
    )

    out = OUT / "p_scorecard.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def fig_reversal_compact() -> Path:
    """Paired conditional-to-global differences for the separate diagnostic."""
    hc = load("hdd/conditional_querywise_results.json")
    nc = load("nuscenes/conditional_querywise_results.json")
    hf = load("hdd/fusion_results.json")
    nf = load("nuscenes/fusion_results.json")

    def conditional(d):
        return d["paired_ap_differences"]["encoder_seq_dtw_minus_bot_cosine"]

    def global_result(d):
        return d["paired_map_differences"]["encoder_seq_dtw_minus_bot"]

    rows = [
        ("Honda HDD  ·  conditional", conditional(hc)),
        ("Honda HDD  ·  global", global_result(hf)),
        ("nuScenes  ·  conditional", conditional(nc)),
        ("nuScenes  ·  global", global_result(nf)),
    ]

    fig = plt.figure(figsize=(640 * MM, H["reversal_compact"] * MM))
    ax = fig.add_axes((0.245, 0.235, 0.515, 0.665))
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color=T.ACCENT, lw=2.0, zorder=1)
    for i, ((label, result), y) in enumerate(zip(rows, ys)):
        diff = result["difference_a_minus_b"]
        lo, hi = result["ci"]
        color = T.OUT_RELEVANT if "conditional" in label else T.OUT_WRONG_PLACE
        ax.plot([lo, hi], [y, y], color=color, lw=4.0, solid_capstyle="round", zorder=3)
        ax.plot(
            [diff],
            [y],
            "o",
            markersize=12,
            color=color,
            markeredgecolor=T.SURFACE,
            markeredgewidth=1.8,
            zorder=4,
        )
        if i == 1:
            ax.axhline(y - 0.5, color=T.RULE, lw=1.1, zorder=0)
        fig.text(
            0.982,
            0.235 + 0.665 * ((y + 0.06) / 3.55),
            f"{diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
            ha="right",
            va="center",
            fontsize=21,
            fontweight="bold",
            color=T.INK,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=23, color=T.INK)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlim(-0.26, 0.115)
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1])
    ax.set_xticklabels(["−0.2", "−0.1", "0", "+0.1"], fontsize=22)
    ax.set_xlabel(
        "encoder-sequence DTW − pooled cosine  (mAP)",
        fontsize=24,
        color=T.INK_2,
        labelpad=8,
    )
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.015,
        0.035,
        "Dots: paired mean difference. Lines: 95% intersection-cluster intervals; "
        "2,000 resamples.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
    )
    out = OUT / "p_reversal_compact.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def fig_diagnostic_pair() -> Path:
    """Show the matched sign reversal and its global top-1 error composition."""
    hc = load("hdd/conditional_querywise_results.json")
    nc = load("nuscenes/conditional_querywise_results.json")
    hf = load("hdd/fusion_results.json")
    nf = load("nuscenes/fusion_results.json")

    def conditional(d):
        return d["paired_ap_differences"]["encoder_seq_dtw_minus_bot_cosine"]

    def global_result(d):
        return d["paired_map_differences"]["encoder_seq_dtw_minus_bot"]

    forest_rows = [
        ("HDD · conditional", conditional(hc)),
        ("HDD · global", global_result(hf)),
        ("nuScenes · conditional", conditional(nc)),
        ("nuScenes · global", global_result(nf)),
    ]

    fig = plt.figure(figsize=(900 * MM, H["diagnostic_pair"] * MM))

    # Left: the matched difference that changes sign when the gallery expands.
    ax = fig.add_axes((0.135, 0.235, 0.255, 0.600))
    ys = np.arange(len(forest_rows))[::-1]
    ax.axvline(0, color=T.ACCENT, lw=2.0, zorder=1)
    for (label, result), y in zip(forest_rows, ys):
        diff = result["difference_a_minus_b"]
        lo, hi = result["ci"]
        color = T.OUT_RELEVANT if "conditional" in label else T.OUT_WRONG_PLACE
        ax.plot([lo, hi], [y, y], color=color, lw=4.0, solid_capstyle="round", zorder=3)
        ax.plot(
            [diff], [y], "o", markersize=11, color=color,
            markeredgecolor=T.SURFACE, markeredgewidth=1.6, zorder=4,
        )
        fig.text(
            0.475,
            0.235 + 0.600 * ((y + 0.08) / 3.58),
            f"{diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
            ha="right",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=T.INK,
        )
    ax.axhline(1.5, color=T.RULE, lw=1.1, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in forest_rows], fontsize=18, color=T.INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlim(-0.26, 0.115)
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1])
    ax.set_xticklabels(["−0.2", "−0.1", "0", "+0.1"], fontsize=17)
    ax.set_xlabel("DTW − pooled cosine  (mAP)", fontsize=19, color=T.INK_2, labelpad=6)
    ax.set_title("That it reverses", fontsize=22, fontweight="bold", color=T.INK,
                 pad=10, loc="left")
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_color(T.BASELINE)

    # Right: all tracked global top-1 outcomes. The tiny amber portion remains
    # visible as a real category even when it is too narrow to label directly.
    methods = [
        ("BoT", "bot"),
        ("Encoder-seq DTW", "encoder_seq_dtw"),
        ("Residual DTW", "temporal_residual_dtw"),
    ]
    error_rows = []
    for dataset, result in (("HDD", hf), ("nuScenes", nf)):
        for method_label, key in methods:
            cell = result["ranked_outcome_composition"]["methods"][key]["1"]
            error_rows.append(
                (
                    f"{dataset} · {method_label}",
                    cell["relevant"]["mean"],
                    cell["same_cluster_wrong_label"]["mean"],
                    cell["wrong_cluster"]["mean"],
                )
            )

    ax2 = fig.add_axes((0.665, 0.235, 0.325, 0.600))
    ys2 = np.arange(len(error_rows))[::-1]
    for (label, relevant, wrong_maneuver, wrong_place), y in zip(error_rows, ys2):
        left = 0.0
        for value, color in (
            (relevant, T.OUT_RELEVANT),
            (wrong_maneuver, T.OUT_WRONG_MANEUVER),
            (wrong_place, T.OUT_WRONG_PLACE),
        ):
            ax2.barh(y, value, left=left, height=0.62, color=color, edgecolor="white",
                     linewidth=0.8, zorder=3)
            left += value
        if relevant >= 0.15:
            ax2.text(relevant / 2, y, f"{relevant * 100:.1f}%", ha="center", va="center",
                     fontsize=18, fontweight="bold", color="white", zorder=4)
        if wrong_place >= 0.25:
            ax2.text(1 - wrong_place / 2, y, f"{wrong_place * 100:.1f}%", ha="center",
                     va="center", fontsize=18, fontweight="bold", color="white", zorder=4)

    ax2.axhline(2.5, color=T.RULE, lw=1.1, zorder=0)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([r[0] for r in error_rows], fontsize=15, color=T.INK)
    ax2.tick_params(axis="y", length=0, pad=7)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(["0%", "50%", "100%"], fontsize=17)
    ax2.set_xlabel("share of top-1 retrievals", fontsize=19, color=T.INK_2, labelpad=6)
    ax2.set_title(
        "Where the global top-1 lands\ngreen = relevant  ·  red = wrong place",
        fontsize=20,
        fontweight="bold",
        color=T.INK,
        pad=8,
        loc="left",
    )
    despine(ax2, keep=("bottom",))
    ax2.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.012,
        0.040,
        "Left: paired mean differences with 95% intersection-cluster intervals. Right: global "
        "top-1 outcomes; ≥98.9% of errors are wrong-place, while right-place/wrong-maneuver "
        "is ≤0.6% of all retrievals.",
        fontsize=16,
        style="italic",
        color=T.INK_2,
        va="bottom",
    )

    out = OUT / "p_diagnostic_pair.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 3. top-1 error composition
# ----------------------------------------------------------------------
def fig_errors() -> Path:
    hf = load("hdd/fusion_results.json")
    nf = load("nuscenes/fusion_results.json")
    order = ["bot_cosine", "encoder_seq_dtw", "temporal_residual_dtw"]
    jkey = {
        "bot_cosine": "bot",
        "encoder_seq_dtw": "encoder_seq_dtw",
        "temporal_residual_dtw": "temporal_residual_dtw",
    }
    cats = [
        ("relevant", T.OUT_RELEVANT),
        ("same_cluster_wrong_label", T.OUT_WRONG_MANEUVER),
        ("wrong_cluster", T.OUT_WRONG_PLACE),
    ]

    fig = plt.figure(figsize=(300 * MM, H["errors"] * MM))
    legend_row(
        fig,
        [
            (T.OUT_RELEVANT, "relevant"),
            (T.OUT_WRONG_MANEUVER, "wrong maneuver, right place"),
            (T.OUT_WRONG_PLACE, "wrong intersection"),
        ],
        x0=0.030,
        y=0.995,
        size=LEGEND_PT,
    )

    ax = fig.add_axes((0.30, 0.125, 0.665, 0.800))
    rows, labels = [], []
    for dname, d in (("nuScenes", nf), ("Honda HDD", hf)):
        for m in reversed(order):
            comp = d["ranked_outcome_composition"]["methods"][jkey[m]]["1"]
            rows.append((dname, m, [comp[c]["mean"] for c, _ in cats]))
            labels.append(T.METHOD_LABEL[m])

    bh, gap_in, gap_between = 0.42, 0.19, 0.58
    yc = []
    y = 0.0
    for i, (dname, m, vals) in enumerate(rows):
        if i and rows[i - 1][0] != dname:
            y += gap_between
        total = sum(vals)
        left = 0.0
        seg_gap = 0.004  # surface gap between stacked segments
        for (cat, color), v in zip(cats, vals):
            frac = v / total
            w = max(frac - seg_gap, 0.0)
            if w > 0:
                ax.add_patch(
                    Rectangle((left, y), w, bh, facecolor=color, edgecolor="none", zorder=3)
                )
            # A "98.9%" label at ANNOT_PT is ~22.6 mm wide against a 199.5 mm axis,
            # i.e. ~0.113 of it, so the segment must be at least that wide or the
            # label spills over its own colour. The threshold tracks the type size.
            if frac > 0.12:
                ax.text(
                    left + frac / 2,
                    y + bh / 2,
                    f"{frac * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=ANNOT_PT,
                    fontweight="bold",
                    color="white",
                    zorder=5,
                )
            left += frac
        yc.append(y + bh / 2)
        y += bh + gap_in

    ax.set_yticks(yc)
    ax.set_yticklabels(labels, fontsize=TICK_PT, color=T.INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=TICK_PT)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.18, y - gap_in + 0.18)
    ax.set_xlabel("share of top-1 retrievals", fontsize=LABEL_PT, color=T.INK_2, labelpad=10)
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_color(T.BASELINE)

    # dataset brackets
    for dname, idxs in (("nuScenes", [0, 1, 2]), ("Honda HDD", [3, 4, 5])):
        y0, y1 = yc[idxs[0]] - bh / 2, yc[idxs[-1]] + bh / 2
        ax.plot(
            [-0.335, -0.335],
            [y0, y1],
            color=T.BASELINE,
            lw=2.2,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            -0.355,
            (y0 + y1) / 2,
            dname,
            rotation=90,
            ha="center",
            va="center",
            fontsize=TITLE_PT,
            fontweight="bold",
            color=T.INK,
            clip_on=False,
        )

    fig.text(
        0.030,
        0.012,
        "Right place, wrong maneuver is never above 0.6%.\n"
        "Global errors are overwhelmingly wrong-location.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        linespacing=1.35,
    )

    out = OUT / "p_errors.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 4. cascade sweep
# ----------------------------------------------------------------------
def fig_cascade() -> Path:
    hdd = load("hdd/bof_dtw_directed_rerank_results.json")
    nus = load("nuscenes/fusion_results.json")

    fig = plt.figure(figsize=(300 * MM, H["cascade"] * MM))
    legend_row(
        fig,
        [(T.BOT, "BoT AP@k"), (T.DTW, "BoT→DTW rerank AP@k")],
        x0=0.105,
        y=0.995,
        size=LEGEND_PT,
    )

    for i, (data, title) in enumerate(((hdd, "Honda HDD"), (nus, "nuScenes"))):
        # Bottom fraction buys room for the two-line caption below the x label;
        # top edge (bottom + height) stays at 0.855 so the panels do not move.
        ax = fig.add_axes((0.105 + i * 0.495, 0.200, 0.385, 0.655))
        ks = sorted(int(k) for k in data["k_sweep"])
        bot = [data["k_sweep"][str(k)]["bot"]["ap"]["mean"] for k in ks]
        dtw = [data["k_sweep"][str(k)]["dtw_rerank"]["ap"]["mean"] for k in ks]
        xs = np.arange(len(ks))

        ticks = list(np.arange(0, 0.351, 0.05))
        hgrid(ax, ticks)

        for ys, color in ((bot, T.BOT), (dtw, T.DTW)):
            ax.plot(xs, ys, color=color, lw=2.6, zorder=3, solid_capstyle="round")
            ax.plot(
                xs,
                ys,
                "o",
                color=color,
                markersize=9,
                markeredgecolor=T.SURFACE,
                markeredgewidth=2.0,
                zorder=4,
            )
        # Direct end labels only (never a number on every point). Each label is
        # ringed in its series colour so identity comes from a swatch rather than
        # from proximity to the nearest line; the glyphs stay INK, per legend_row.
        for ys, color, dy in ((bot, T.BOT, 0.030), (dtw, T.DTW, -0.038)):
            ax.text(
                xs[-1],
                ys[-1] + dy,
                f"{ys[-1]:.3f}",
                ha="right",
                va="center",
                fontsize=ANNOT_PT,
                fontweight="bold",
                color=T.INK,
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor=T.SURFACE,
                    edgecolor=color,
                    linewidth=1.8,
                ),
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in ks], fontsize=TICK_PT)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=TICK_PT)
        ax.set_xlim(-0.30, len(ks) - 0.70)
        ax.set_ylim(0, 0.375)
        ax.set_xlabel("k (candidates reranked)", fontsize=LABEL_PT, color=T.INK_2, labelpad=8)
        if i == 0:
            ax.set_ylabel("AP@k", fontsize=LABEL_PT, color=T.INK_2, labelpad=10)
        ax.set_title(
            title, fontsize=TITLE_PT, fontweight="bold", color=T.INK, pad=10, loc="left"
        )
        despine(ax)
        ax.spines["left"].set_color(T.BASELINE)
        ax.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.105,
        0.022,
        # Two lines: at CAPTION_PT this caption is wider than the figure on one.
        "Recall@k is identical (shared candidate set) —\n"
        "reranking only reorders, and it reorders worse.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
        linespacing=1.35,
    )

    out = OUT / "p_cascade.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 5. third-domain transfer check
# ----------------------------------------------------------------------
def fig_soccernet() -> Path:
    """Paired differences on SoccerNet-v2, as a forest plot.

    A forest plot is the right mark here: every quantity is a paired difference
    with a match-clustered CI, and the only question asked of each row is whether
    its interval clears zero. Filled/hollow carries that, so no palette slot is
    spent on it -- the method colours keep their meaning from the other panels.
    """
    d = load("soccernet/replay_results.json")
    paired = d["paired_same_half_rr"]

    rows = [
        ("encoder_seq_dtw_minus_bot", "Enc-seq DTW − BoT"),
        ("temporal_residual_dtw_minus_bot", "Residual DTW − BoT"),
        ("encoder_seq_dtw_minus_encoder_seq_dtw_shuffled", "Intact − shuffled DTW"),
        ("encoder_seq_dtw_minus_encoder_seq_assignment", "Intact − order-free assign."),
    ]
    n_matches = paired[rows[0][0]]["n_matches"]
    n_queries = d["methods"]["bot"]["same_half_primary"]["aggregate"]["n_queries"]

    fig = plt.figure(figsize=(300 * MM, H["soccernet"] * MM))
    ax = fig.add_axes((0.400, 0.330, 0.420, 0.580))

    ys = list(range(len(rows)))[::-1]  # first row reads at the top
    for yy, (key, label) in zip(ys, rows):
        v = paired[key]
        lo, hi = v["ci"]
        diff = v["difference_a_minus_b"]
        detected = lo > 0.0 or hi < 0.0

        ax.plot([lo, hi], [yy, yy], color=T.INK_2, lw=2.2, solid_capstyle="butt", zorder=3)
        for end in (lo, hi):
            ax.plot([end, end], [yy - 0.17, yy + 0.17], color=T.INK_2, lw=2.2, zorder=3)
        ax.plot(
            [diff],
            [yy],
            "o",
            markersize=13,
            color=T.INK if detected else T.SURFACE,
            markeredgecolor=T.INK,
            markeredgewidth=2.2,
            zorder=4,
        )
        ax.text(
            1.045,
            yy,
            f"{diff:+.3f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=ANNOT_PT,
            fontweight="bold",
            color=T.INK if detected else T.INK_2,
            clip_on=False,
            zorder=6,
        )

    ax.axvline(0.0, color=T.ACCENT, lw=2.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=TICK_PT, color=T.INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_xlim(-0.028, 0.034)
    ax.set_xticks([-0.02, 0.0, 0.02])
    ax.set_xticklabels(["−0.02", "0", "+0.02"], fontsize=TICK_PT)
    ax.set_xlabel(
        "difference in match-macro RR", fontsize=LABEL_PT, color=T.INK_2, labelpad=8
    )
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.015,
        0.020,
        f"SoccerNet-v2 within-match replay grounding: {n_queries:,} queries / "
        f"{n_matches} test matches, same half.\n"
        "Filled = 95% CI excludes zero.  Hollow = no detected difference.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
        linespacing=1.35,
    )

    out = OUT / "p_soccernet.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# 6. companion frozen-encoder evaluations
# ----------------------------------------------------------------------
def load_cohorts() -> dict:
    """Load the self-contained snapshot used by the companion figure.

    Five configurations have values in all three frozen-backbone evaluations and two
    additional configurations have synthetic-copy point estimates only.  The source paths,
    selection rule, and cross-protocol qualifications are recorded in the file.
    """
    return json.loads((Path(__file__).resolve().parent / "frozen_matrix.json").read_text())


# Short axis labels. Anything missing falls back to the cohort's own `metric`
# string from frozen_matrix.json, so adding a cohort cannot break the build.
XLAB = {
    "synthetic_copies": "best F1",
    "hdd": "50-cluster macro-mAP",
    "vcdb": "within-topic AP",
    "soccernet": "match-macro MRR",
    "aria": "order sensitivity ratio",
}


def fig_cohorts() -> Path:
    """Five descriptive evaluations with their protocol limits kept visible.

    The first three panels use common numeric bounds, but not common units.  The
    shared bounds expose VCDB's ceiling compression without implying that F1 and
    AP levels can be compared across datasets.
    """
    d = load_cohorts()
    arms = d["arms"]
    coh = {c["key"]: c for c in d["cohorts"]}
    ys = list(range(len(arms)))[::-1]

    hdd_test = coh["hdd"]["reference_test"]
    vcdb_test = coh["vcdb"]["reference_test"]
    soccernet_cov = coh["soccernet"]["coverage"]
    aria_test = coh["aria"]["threshold_test"]
    vcdb_values = [a["vcdb"]["value"] for a in arms if a["vcdb"].get("value") is not None]
    vcdb_span = max(vcdb_values) - min(vcdb_values)

    fig = plt.figure(figsize=(900 * MM, 142 * MM))
    fig.text(0.012, 0.970, "6", fontsize=21, fontweight="bold", color=T.ACCENT, va="top")
    fig.text(
        0.033, 0.970,
        "Frozen encoders leave an open cross-task retrieval gap",
        fontsize=27, fontweight="bold", color=T.INK, va="top",
    )

    AP = (0.40, 1.03)
    APT = [0.4, 0.6, 0.8, 1.0]
    panels = [
        ("synthetic_copies", 0.175, "Internal synthetic copies\npoint estimates", AP, APT),
        (
            "hdd",
            0.337,
            "Honda HDD\n"
            f"V-JEPA 2 > all {hdd_test['displayed_comparisons']} scored peers",
            AP,
            APT,
        ),
        ("vcdb", 0.499, f"VCDB\nshown spread = {vcdb_span:.4f} AP", AP, APT),
        (
            "soccernet",
            0.661,
            "SoccerNet-v2 (valid split)\n"
            f"{soccernet_cov['scored_rows']}/{soccernet_cov['displayed_rows']} shown rows run",
            (0.10, 0.26),
            [0.12, 0.18, 0.24],
        ),
        (
            "aria",
            0.823,
            "Project Aria (12 recordings)\n"
            f"{aria_test['displayed_clear']}/{aria_test['displayed_cells']} shown readouts "
            "clear parity",
            (0.72, 1.22),
            [0.8, 1.0, 1.2],
        ),
    ]

    for i, (key, left, title, xlim, xticks) in enumerate(panels):
        # Leave a full three-line caption lane below the x labels.
        ax = fig.add_axes((left, 0.315, 0.145, 0.410))
        for y, a in zip(ys, arms):
            cell = a[key]
            if cell.get("value") is None:
                continue  # never run; the gap is the information
            ci = cell.get("ci")
            if ci:
                ax.plot(ci, [y, y], color=T.INK_2, lw=2.0, zorder=3, solid_capstyle="butt")
            ax.plot([cell["value"]], [y], "o", markersize=11, zorder=4,
                    color=T.ACCENT if a["key"] == "vjepa2" else T.INK,
                    markeredgecolor=T.SURFACE, markeredgewidth=1.6)

        ref = coh[key].get("reference_line")
        if ref:
            ax.axvline(ref["value"], color=T.BASELINE, lw=1.8, ls=(0, (5, 4)), zorder=1)
            ax.text(ref["value"], -0.55, f"  {ref['label']}", ha="left", va="bottom",
                    fontsize=CAPTION_PT, style="italic", color=T.INK_2, zorder=5)

        ax.set_yticks(ys)
        ax.set_yticklabels([a["label"] for a in arms] if i == 0 else [],
                           fontsize=TICK_PT, color=T.INK)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_ylim(-0.62, len(arms) - 0.38)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=TICK_PT)
        ax.set_xlabel(XLAB.get(key, coh[key]["metric"]), fontsize=CAPTION_PT,
                      color=T.INK_2, labelpad=6)
        ax.set_title(title, fontsize=TICK_PT, fontweight="bold", color=T.INK, pad=8, loc="left")
        despine(ax, keep=("bottom",))
        ax.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.012, 0.015,
        "Separate protocols; shared bounds are not shared units. Cross-model levels are "
        "descriptive: sampling, resolution, tokenization, and Aria readout vary.\n"
        "Internal-copy evaluation: no CIs. HDD: "
        f"{hdd_test['displayed_separated']}/{hdd_test['displayed_comparisons']} shown V-JEPA 2 "
        f"contrasts exclude 0. VCDB: all {vcdb_test['source_comparisons']} registered reference "
        "contrasts include 0.\nSoccerNet-v2 valid: 6,376 same-half queries / 100 matches; "
        f"{soccernet_cov['scored_rows']}/{soccernet_cov['displayed_rows']} shown rows run; "
        f"95% CIs for {soccernet_cov['cells_with_ci']}/{soccernet_cov['scored_rows']}. "
        "Gaps = not run. Snapshot da580f9.",
        fontsize=CAPTION_PT, style="italic", color=T.INK_2, va="bottom",
    )

    out = OUT / "p_cohorts.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# New cross-task poster synthesis
# ----------------------------------------------------------------------
def _snapshot() -> tuple[dict, dict, list[dict]]:
    d = load_cohorts()
    return d, {c["key"]: c for c in d["cohorts"]}, d["arms"]


def _dot_rows(ax, arms, key: str, ys, *, xlim, xticks, values=True) -> None:
    for y, arm in zip(ys, arms):
        cell = arm[key]
        value = cell.get("value")
        if value is None:
            continue
        ci = cell.get("ci")
        if ci:
            ax.plot(ci, [y, y], color=T.INK_2, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.plot(
            [value],
            [y],
            "o",
            markersize=11,
            color=T.ACCENT if arm["key"] == "vjepa2" else T.INK,
            markeredgecolor=T.SURFACE,
            markeredgewidth=1.6,
            zorder=4,
        )
        if values:
            span = xlim[1] - xlim[0]
            anchor = ci[1] if ci else value
            xtext = anchor + 0.018 * span
            ha = "left"
            if xtext > xlim[1] - 0.07 * span:
                anchor = ci[0] if ci else value
                xtext = anchor - 0.018 * span
                ha = "right"
            ax.text(
                xtext,
                y,
                f"{value:.3f}",
                ha=ha,
                va="center",
                fontsize=ANNOT_PT,
                fontweight="bold",
                color=T.INK,
                zorder=5,
            )
    ax.set_ylim(-0.62, len(arms) - 0.38)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=TICK_PT)
    ax.set_yticks(ys)
    ax.set_yticklabels([a["label"] for a in arms], fontsize=TICK_PT, color=T.INK)
    ax.tick_params(axis="y", length=0, pad=8)
    despine(ax, keep=("bottom",))
    ax.spines["bottom"].set_color(T.BASELINE)


def fig_copy_hdd() -> Path:
    """Copy robustness and maneuver retrieval, with gaps left explicit."""
    _, cohorts, arms = _snapshot()
    ys = list(range(len(arms)))[::-1]
    fig = plt.figure(figsize=(300 * MM, H["copy_hdd"] * MM))

    specs = [
        ("synthetic_copies", 0.585, (0.60, 1.02), [0.6, 0.8, 1.0], "best F1"),
        ("hdd", 0.190, (0.40, 1.02), [0.4, 0.6, 0.8, 1.0], "50-cluster macro-mAP"),
    ]
    for key, bottom, xlim, ticks, xlabel in specs:
        ax = fig.add_axes((0.355, bottom, 0.625, 0.305))
        _dot_rows(ax, arms, key, ys, xlim=xlim, xticks=ticks)
        ax.set_xlabel(xlabel, fontsize=LABEL_PT, color=T.INK_2, labelpad=7)
        ax.set_title(cohorts[key]["label"], fontsize=TITLE_PT, fontweight="bold",
                     color=T.INK, pad=8, loc="left")
        if key == "hdd":
            ref = cohorts[key]["reference_line"]
            ax.axvline(ref["value"], color=T.BASELINE, lw=1.8, ls=(0, (5, 4)), zorder=1)
            ax.text(ref["value"], -0.56, "  chance", ha="left", va="bottom",
                    fontsize=CAPTION_PT, style="italic", color=T.INK_2)

    fig.text(
        0.015,
        0.012,
        "Different metrics: compare within-panel ordering, not raw levels. Gaps = not run.\n"
        "Synthetic copies: point estimates only. On HDD, all four shown V-JEPA 2 peer\n"
        "contrasts exclude zero under paired cluster resampling.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
        linespacing=1.35,
    )
    out = OUT / "p_copy_hdd.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def fig_soccernet_evidence() -> Path:
    """Separate event-identity level from the effect of explicit ordering."""
    _, cohorts, all_arms = _snapshot()
    by_key = {a["key"]: a for a in all_arms}
    arms = [by_key[k] for k in ("dinov3", "radio", "vjepa2")]
    ys = list(range(len(arms)))[::-1]
    cohort = cohorts["soccernet"]

    fig = plt.figure(figsize=(300 * MM, H["soccernet_evidence"] * MM))
    ax = fig.add_axes((0.355, 0.530, 0.625, 0.360))
    _dot_rows(ax, arms, "soccernet", ys, xlim=(0.105, 0.265), xticks=[0.12, 0.18, 0.24])
    ref = cohort["reference_line"]
    ax.axvline(ref["value"], color=T.BASELINE, lw=1.8, ls=(0, (5, 4)), zorder=1)
    ax.text(ref["value"], -0.56, "  chance", ha="left", va="bottom",
            fontsize=CAPTION_PT, style="italic", color=T.INK_2)
    ax.set_xlabel("valid match-macro MRR", fontsize=LABEL_PT, color=T.INK_2, labelpad=7)
    ax.set_title("Event identity — unordered matching", fontsize=TITLE_PT,
                 fontweight="bold", color=T.INK, pad=8, loc="left")

    test = cohort["order_test"]
    ax2 = fig.add_axes((0.355, 0.205, 0.625, 0.145))
    lo, hi = test["ci"]
    diff = test["difference"]
    ax2.axvline(0, color=T.ACCENT, lw=1.8, zorder=1)
    ax2.plot([lo, hi], [0, 0], color=T.INK_2, lw=2.2, solid_capstyle="butt", zorder=3)
    for end in (lo, hi):
        ax2.plot([end, end], [-0.12, 0.12], color=T.INK_2, lw=2.2, zorder=3)
    ax2.plot([diff], [0], "o", markersize=12, color=T.SURFACE,
             markeredgecolor=T.INK, markeredgewidth=2.0, zorder=4)
    ax2.text(-0.0072, 0, "V-JEPA 2\nordered − unordered", ha="right", va="center",
             fontsize=TICK_PT, color=T.INK)
    ax2.text(0.0068, 0, f"{diff:+.4f}", ha="right", va="center",
             fontsize=ANNOT_PT, fontweight="bold", color=T.INK_2)
    ax2.set_xlim(-0.007, 0.007)
    ax2.set_ylim(-0.36, 0.36)
    ax2.set_yticks([])
    ax2.set_xticks([-0.005, 0, 0.005])
    ax2.set_xticklabels(["−.005", "0", "+.005"], fontsize=TICK_PT)
    ax2.set_xlabel("ordered-alignment minus unordered-Chamfer MRR",
                   fontsize=CAPTION_PT, color=T.INK_2, labelpad=5)
    despine(ax2, keep=("bottom",))
    ax2.spines["bottom"].set_color(T.BASELINE)

    fig.text(
        0.015,
        0.012,
        "6,376 same-half queries / 100 valid matches; V-JEPA 2 is a point estimate.\n"
        "Top: cross-model levels are descriptive. Bottom: paired over complete matches.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
    )
    out = OUT / "p_soccernet_evidence.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def fig_aria_order() -> Path:
    """Predeclared order-sensitivity readouts on the public displayed subset."""
    _, cohorts, all_arms = _snapshot()
    by_key = {a["key"]: a for a in all_arms}
    arms = [by_key[k] for k in ("levjepa", "dinov3", "radio", "vjepa2", "internvideo_large")]
    ys = list(range(len(arms)))[::-1]

    fig = plt.figure(figsize=(300 * MM, H["aria_order"] * MM))
    ax = fig.add_axes((0.355, 0.255, 0.625, 0.650))
    _dot_rows(ax, arms, "aria", ys, xlim=(0.75, 1.16), xticks=[0.8, 0.9, 1.0, 1.1])
    ax.axvline(1.0, color=T.BASELINE, lw=1.8, ls=(0, (5, 4)), zorder=1)
    ax.text(1.0, -0.56, "  parity", ha="left", va="bottom",
            fontsize=CAPTION_PT, style="italic", color=T.INK_2)
    ax.set_xlabel("order sensitivity / within-record change", fontsize=LABEL_PT,
                  color=T.INK_2, labelpad=7)

    fig.text(
        0.015,
        0.012,
        "768 four-second windows / 12 recordings. Pass requires the entire 95% CI above 1.0.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
    )
    out = OUT / "p_aria_order.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def fig_evidence_map() -> Path:
    """Task-level verdicts that preserve uncertainty instead of forcing ranks."""
    fig = plt.figure(figsize=(900 * MM, H["evidence_map"] * MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.965, "6", fontsize=21, fontweight="bold", color=T.ACCENT, va="top")
    ax.text(0.035, 0.965, "Evidence map: five protocols, five different limits",
            fontsize=27, fontweight="bold", color=T.INK, va="top")

    cards = [
        ("Internal synthetic copies", "5 points", "No paired intervals"),
        ("Honda HDD", "4 / 4", "Shown V-JEPA 2 peer\ncontrasts exclude zero"),
        ("VCDB", "0 / 7", "Registered reference\ncontrasts exclude zero"),
        ("SoccerNet-v2", "−0.0014", "Ordered − unordered;\n95% CI crosses zero"),
        ("Project Aria", "0 / 5", "Shown readouts clear\nparity"),
    ]
    gap = 0.012
    left0 = 0.012
    card_w = (0.976 - 4 * gap) / 5
    for i, (title, value, note) in enumerate(cards):
        x = left0 + i * (card_w + gap)
        box = FancyBboxPatch(
            (x, 0.205), card_w, 0.575,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=T.PLANE,
            edgecolor=T.RULE,
            linewidth=1.3,
        )
        ax.add_patch(box)
        ax.text(x + 0.015, 0.710, title, transform=ax.transAxes,
                fontsize=18, fontweight="bold", color=T.INK, va="top")
        ax.text(x + 0.015, 0.535, value, transform=ax.transAxes,
                fontsize=31, fontweight="bold",
                color=T.ACCENT if i in (1, 3) else T.INK, va="center")
        ax.text(x + 0.015, 0.365, note, transform=ax.transAxes,
                fontsize=16, color=T.INK_2, va="center", linespacing=1.25)

    ax.text(
        0.012,
        0.055,
        "Counts answer different questions and must not be averaged into one leaderboard. "
        "Cross-model levels remain descriptive where sampling, resolution, or readout differ. "
        "Snapshot da580f9.",
        fontsize=CAPTION_PT,
        style="italic",
        color=T.INK_2,
        va="bottom",
    )
    out = OUT / "p_evidence_map.png"
    fig.savefig(out, dpi=DPI, facecolor=T.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    for fn in (
        fig_scorecard,
        fig_diagnostic_pair,
    ):
        p = fn()
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

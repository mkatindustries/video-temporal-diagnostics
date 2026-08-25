"""Render the Video4Real poster charts at print resolution.

Every value is read from the tracked result JSONs under results/ -- nothing is
hardcoded, so a rerun cannot leave the poster silently stale.

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
    fig = plt.figure(figsize=(300 * MM, 138 * MM))

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
    ratio = (box_w * 300.0) / (box_h * 118.0)
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

    fig = plt.figure(figsize=(300 * MM, 400 * MM))
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
                # Three ANNOT_PT labels will not sit side by side over three bars:
                # "0.955" is about 24 mm against a ~22 mm bar pitch, and in the
                # conditional panel all three bars are nearly the same height, so
                # the labels land at the same y and collide. Lifting the middle one
                # leaves the outer two a full two pitches apart, which cannot touch.
                lift = 0.055 if mi == 1 else 0.0
                ax.text(
                    x + bw / 2,
                    max(ci[1], mean) + 0.030 + lift,
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

    fig = plt.figure(figsize=(300 * MM, 360 * MM))
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
        "Right place, wrong maneuver is never above 0.6%.\nThe loss is location, not order.",
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

    fig = plt.figure(figsize=(300 * MM, 266 * MM))
    legend_row(
        fig,
        [(T.BOT, "BoT AP@k"), (T.DTW, "BoT→DTW rerank AP@k")],
        x0=0.105,
        y=0.995,
        size=LEGEND_PT,
    )

    for i, (data, title) in enumerate(((hdd, "Honda HDD"), (nus, "nuScenes"))):
        ax = fig.add_axes((0.105 + i * 0.495, 0.155, 0.385, 0.700))
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    for fn in (fig_schematic, fig_reversal, fig_errors, fig_cascade):
        p = fn()
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

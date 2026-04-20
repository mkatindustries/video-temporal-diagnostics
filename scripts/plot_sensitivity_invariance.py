"""Generate the sensitivity-vs-invariance diagnostic scatter plot.

x-axis: Temporal sensitivity (1 - s_rev)
y-axis: Copy robustness (VCDB AP)
Annotations: HDD maneuver discrimination AP
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Data ──────────────────────────────────────────────────────────────
#   (label, temporal_sensitivity, vcdb_ap, hdd_ap, marker, color)
methods = [
    ("DINOv3 BoF", 0.000, 0.979, 0.530, "o", "#1f77b4"),
    ("DINOv3 Chamfer", 0.0001, 0.989, 0.559, "s", "#1f77b4"),
    ("DINOv3 Temp. Deriv.", 0.500, 0.775, 0.498, "^", "#ff7f0e"),
    ("DINOv3 Attn. Traj.", 0.808, 0.700, 0.507, "D", "#ff7f0e"),
    ("V-JEPA 2 BoT", 0.016, 0.838, 0.825, "o", "#2ca02c"),
    ("V-JEPA 2 Temp. Res.", 0.301, 0.698, 0.956, "^", "#2ca02c"),
    ("ViCLIP", 0.004, 0.907, 0.542, "h", "#d62728"),
    ("X-CLIP", 0.020, 0.960, None, "P", "#9467bd"),
    ("TARA (chiral)", 0.045, None, 0.547, "h", "#e377c2"),
    ("PL-Stitch BoF", 0.000, None, 0.478, "o", "#bcbd22"),
    ("PL-Stitch DTW", 0.994, None, 0.540, "^", "#bcbd22"),
]

fig, ax = plt.subplots(figsize=(9, 6.5))

# ── Plot points ──────────────────────────────────────────────────────
for label, ts, vcdb, hdd, marker, color in methods:
    if vcdb is None:
        continue  # HDD-only methods plotted separately below
    size = 200 if hdd is not None else 140
    ax.scatter(
        ts,
        vcdb,
        s=size,
        marker=marker,
        color=color,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

# ── Labels with manual offsets ───────────────────────────────────────
# (label, ts, vcdb, offset_x, offset_y, ha, va, annotation_extra)
label_specs = [
    ("DINOv3 Chamfer", 0.0001, 0.989, 0.06, 0.005, "left", "center", "HDD 0.559"),
    ("DINOv3 BoF", 0.000, 0.979, 0.06, -0.005, "left", "center", "HDD 0.530"),
    ("X-CLIP", 0.020, 0.960, 0.06, 0.000, "left", "center", None),
    ("ViCLIP", 0.004, 0.907, 0.06, 0.000, "left", "center", "HDD 0.542"),
    ("V-JEPA 2 BoT", 0.016, 0.838, 0.06, 0.000, "left", "center", "HDD 0.825"),
    ("DINOv3 Temp. Deriv.", 0.500, 0.775, 0.03, 0.018, "left", "bottom", "HDD 0.498"),
    ("DINOv3 Attn. Traj.", 0.808, 0.700, -0.03, 0.025, "right", "bottom", "HDD 0.507"),
    ("V-JEPA 2 Temp. Res.", 0.301, 0.698, -0.03, -0.025, "right", "top", "HDD 0.956"),
]

for label, ts, vcdb, ox, oy, ha, va, extra in label_specs:
    txt = label
    if extra:
        txt += f"  ({extra})"
    ax.annotate(
        txt,
        (ts, vcdb),
        xytext=(ts + ox, vcdb + oy),
        fontsize=8.5,
        ha=ha,
        va=va,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )

# ── Region annotations ──────────────────────────────────────────────
ax.text(
    0.55,
    0.975,
    "Copy-robust / order-invariant\n(scene matching)",
    fontsize=9,
    fontstyle="italic",
    color="#4466aa",
    ha="center",
    va="bottom",
    transform=ax.transData,
)
ax.text(
    0.88,
    0.68,
    "Temporally sensitive\n(manipulation detection)",
    fontsize=9,
    fontstyle="italic",
    color="#bb6622",
    ha="right",
    va="bottom",
    transform=ax.transData,
)

# ── VLM bridge annotation ──────────────────────────────────────────
# VLM vision towers are order-invariant when pooled (temporal sens ≈ 0)
# and have no VCDB AP; place text box in the bottom-left quadrant.
ax.annotate(
    "VLM vision towers (no VCDB data)\n"
    "pooled: HDD AP $\\approx$ 0.52  |  seq DTW: HDD AP $\\approx$ 0.55\n"
    "2D per-frame encoders fail on motion regardless of comparator",
    xy=(0.08, 0.635),
    fontsize=7.5,
    ha="left",
    va="top",
    color="#666666",
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.85
    ),
)

# ── Temporal-training baselines (HDD-only, no VCDB) ─────────────────
ax.annotate(
    "Temporal-training baselines (HDD only, no VCDB data)\n"
    "TARA (chiral): HDD 0.547, $s_{\\mathrm{rev}}$ 0.955\n"
    "PL-Stitch BoF: HDD 0.478, $s_{\\mathrm{rev}}$ 1.000\n"
    "PL-Stitch DTW: HDD 0.540, $s_{\\mathrm{rev}}$ 0.006",
    xy=(0.55, 0.635),
    fontsize=7.5,
    ha="left",
    va="top",
    color="#666666",
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="#fdf6ec", edgecolor="#cccccc", alpha=0.85
    ),
)

# ── Axes ─────────────────────────────────────────────────────────────
ax.set_xlabel("Temporal Sensitivity  $(1 - s_{\\mathrm{rev}})$", fontsize=12)
ax.set_ylabel("Copy Robustness  (VCDB AP)", fontsize=12)
ax.set_xlim(-0.05, 1.0)
ax.set_ylim(0.60, 1.05)
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_title(
    "Sensitivity–Invariance Trade-off in Video Retrieval",
    fontsize=13,
    fontweight="bold",
)

# ── Legend ────────────────────────────────────────────────────────────
# Build legend handles from the plotted method table so marker/color pairs
# stay consistent with the scatter points.
legend_elements = [
    Line2D(
        [0],
        [0],
        linestyle="None",
        marker=marker,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9,
        label=label,
    )
    for label, _ts, _vcdb, _hdd, marker, color in methods
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=8,
    ncol=1,
    handletextpad=0.5,
    framealpha=0.9,
)

plt.tight_layout()
plt.savefig("figures/sensitivity_invariance_tradeoff.png", dpi=200, bbox_inches="tight")
print("Saved: figures/sensitivity_invariance_tradeoff.png")

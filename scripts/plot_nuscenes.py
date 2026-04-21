"""Regenerate nuScenes maneuver discrimination bar chart.

Uses the same matplotlib Set1 palette as plot_vcdb_reversal.py so that
shared methods share colors across figures. Rotates x-axis labels so
multi-word names (Bag of Tokens, Encoder-Seq DTW, Temporal Res.) no
longer collide.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
fig_dir = project_root / "figures"

set1 = plt.get_cmap("Set1").colors

# (label, AP, AUC, color)
# Colors mirror plot_vcdb_reversal.py; Encoder-Seq DTW uses set1[7] (pink)
# since it does not appear in VCDB reversal.
methods = [
    ("Bag of Frames",          0.613, 0.589, set1[0]),
    ("Chamfer",                0.612, 0.595, set1[1]),
    ("Temporal Derivative",    0.623, 0.574, set1[2]),
    ("Attention Trajectory",   0.584, 0.568, set1[3]),
    ("V-JEPA 2 Bag of Tokens", 0.791, 0.741, set1[4]),
    ("V-JEPA 2 Enc-Seq DTW",   0.867, 0.840, set1[7]),
    ("V-JEPA 2 Temporal Res.", 0.815, 0.775, set1[6]),
]

labels = [m[0] for m in methods]
aps    = [m[1] for m in methods]
aucs   = [m[2] for m in methods]
colors = [m[3] for m in methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))


def draw(ax, values, ylabel, title):
    bars = ax.bar(range(len(methods)), values, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend(loc="upper left")


draw(ax1, aps,  "Average Precision", "Maneuver Discrimination: AP")
draw(ax2, aucs, "ROC-AUC",           "Maneuver Discrimination: AUC")

fig.suptitle("nuScenes: Left Turn vs Right Turn at Same Intersection",
             fontsize=14, fontweight="bold")
fig.tight_layout()

path = fig_dir / "nuscenes_maneuver_discrimination.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

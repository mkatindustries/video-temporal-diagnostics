"""Regenerate VCDB reversal attack bar chart with unified model-family colors.

Values match the most recent eval_vcdb_reversal.py run. Colors follow the same
scheme used by plot_hdd_all_methods.py and plot_sensitivity_invariance.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

project_root = Path(__file__).parent.parent
fig_dir = project_root / "figures"

# Matplotlib Set1 palette (qualitative, one color per method).
set1 = plt.get_cmap("Set1").colors

# (label, normal_ap, reversed_ap, normal_auc, reversed_auc, color)
methods = [
    ("Bag of Frames",          0.979, 0.979, 0.974, 0.974, set1[0]),
    ("Chamfer",                0.989, 0.989, 0.990, 0.990, set1[1]),
    ("Temporal Derivative",    0.775, 0.685, 0.701, 0.659, set1[2]),
    ("Attention Trajectory",   0.700, 0.542, 0.637, 0.522, set1[3]),
    ("V-JEPA 2 Bag of Tokens", 0.837, 0.832, 0.805, 0.802, set1[4]),
    ("V-JEPA 2 Temporal Res.", 0.698, 0.651, 0.654, 0.412, set1[6]),
]

labels      = [m[0] for m in methods]
normal_aps  = [m[1] for m in methods]
revd_aps    = [m[2] for m in methods]
normal_aucs = [m[3] for m in methods]
revd_aucs   = [m[4] for m in methods]
colors      = [m[5] for m in methods]

x = np.arange(len(methods))
bw = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


def draw(ax, normal, reversed_, ylabel, title):
    bn = ax.bar(x - bw / 2, normal,   bw, color=colors, edgecolor="black", linewidth=0.5)
    br = ax.bar(x + bw / 2, reversed_, bw, color=colors, edgecolor="black", linewidth=0.5,
                alpha=0.5, hatch="//")
    for bar, v in zip(bn, normal):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, v in zip(br, reversed_):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(handles=[
        Patch(facecolor="#cccccc", edgecolor="black", linewidth=0.8, label="Normal"),
        Patch(facecolor="#cccccc", edgecolor="black", linewidth=0.8, alpha=0.5,
              hatch="//", label="Reversed"),
    ], fontsize=10)


draw(ax1, normal_aps,  revd_aps,  "Average Precision", "Average Precision")
draw(ax2, normal_aucs, revd_aucs, "ROC-AUC",           "ROC-AUC")

fig.suptitle("VCDB Reversal Attack: Normal vs Reversed Copy Detection",
             fontsize=14, fontweight="bold")
fig.tight_layout()

path = fig_dir / "vcdb_reversal_attack.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

"""Regenerate HDD maneuver discrimination bar chart with all methods.

Combines DINOv3, V-JEPA 2, ViCLIP, TARA, and PL-Stitch results.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
fig_dir = project_root / "figures"

# ── Colors ───────────────────────────────────────────────────────────
# DINOv3 (4 bars): shades of orange/red via Oranges colormap
# V-JEPA 2 (3 bars): shades of green via Greens colormap
oranges = plt.get_cmap("Oranges")
greens  = plt.get_cmap("Greens")
dino_colors   = [oranges(0.45), oranges(0.60), oranges(0.75), oranges(0.90)]
jepa_colors   = [greens(0.50),  greens(0.70),  greens(0.90)]
viclip_color  = "#377eb8"   # blue  (own)
tara_color    = "#984ea3"   # purple (own)

# ── Collect results ──────────────────────────────────────────────────
# Order matches Table 2 in the paper
methods = [
    ("DINOv3\nTemp. Deriv.", 0.498, 0.526, dino_colors[0]),
    ("DINOv3\nAttn. Traj.",  0.507, 0.541, dino_colors[1]),
    ("DINOv3\nBoF",          0.530, 0.542, dino_colors[2]),
    ("DINOv3\nChamfer",      0.559, 0.577, dino_colors[3]),
    ("ViCLIP",               0.542, 0.549, viclip_color),
    ("TARA\n(chiral)",       0.547, 0.600, tara_color),
]

# Load TARA results (overwrite placeholder if JSON available)
tara_path = project_root / "datasets" / "tara_hdd_results.json"
if tara_path.exists():
    with open(tara_path) as f:
        tara = json.load(f)
    for i, (name, ap, auc, color) in enumerate(methods):
        if "TARA" in name:
            methods[i] = (name, tara["hdd"]["ap"], tara["hdd"]["auc"], color)

# Add V-JEPA 2 methods (always last, best performers)
methods.extend([
    ("V-JEPA 2\nBoT",         0.825, 0.836, jepa_colors[0]),
    ("V-JEPA 2\nEnc-Seq DTW", 0.942, 0.935, jepa_colors[1]),
    ("V-JEPA 2\nTemp. Res.",  0.956, 0.952, jepa_colors[2]),
])

labels = [m[0] for m in methods]
aps = [m[1] for m in methods]
aucs = [m[2] for m in methods]
colors = [m[3] for m in methods]

# ── Plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

# AP
bars = ax1.bar(range(len(methods)), aps, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
for bar, val in zip(bars, aps):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
    )
ax1.set_ylabel("Average Precision", fontsize=12)
ax1.set_title("Maneuver Discrimination: AP", fontsize=13)
ax1.set_ylim(0, min(1.0, max(aps) * 1.15 + 0.05))
ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
ax1.legend()

# AUC
bars = ax2.bar(range(len(methods)), aucs, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
for bar, val in zip(bars, aucs):
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
    )
ax2.set_ylabel("ROC-AUC", fontsize=12)
ax2.set_title("Maneuver Discrimination: AUC", fontsize=13)
ax2.set_ylim(0, min(1.0, max(aucs) * 1.15 + 0.05))
ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
ax2.legend()

fig.suptitle(
    "Honda HDD: Left Turn vs Right Turn at Same Intersection",
    fontsize=14, fontweight="bold",
)
fig.tight_layout()

path = fig_dir / "hdd_maneuver_discrimination.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

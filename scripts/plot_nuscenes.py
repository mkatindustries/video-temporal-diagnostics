"""Regenerate the nuScenes maneuver-discrimination bar chart from results.

Reads the tracked, location-aware result JSON rather than transcribing values.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

project_root = Path(__file__).parent.parent
fig_dir = project_root / "figures"

results = json.loads((project_root / "results/nuscenes/intersection_results.json").read_text())
method_specs = [
    ("bag_of_frames", "Bag of Frames", "#e74c3c"),
    ("chamfer", "Chamfer", "#1abc9c"),
    ("temporal_derivative", "Temporal Derivative", "#2ecc71"),
    ("attention_trajectory", "Attention Trajectory", "#3498db"),
    ("vjepa2_bag_of_tokens", "V-JEPA 2 Bag of Tokens", "#9b59b6"),
    ("vjepa2_encoder_seq_dtw", "V-JEPA 2 Enc-Seq DTW", "#8e44ad"),
    ("vjepa2_encoder_seq_dtw_shuffled", "V-JEPA 2 Shuffled Enc.", "#34495e"),
    ("vjepa2_encoder_seq_assignment", "V-JEPA 2 Enc. Assignment", "#16a085"),
    ("vjepa2_temporal_residual", "V-JEPA 2 Temporal Res.", "#f39c12"),
    ("vjepa2_temporal_residual_shuffled", "V-JEPA 2 Shuffled Res.", "#7f8c8d"),
    ("vjepa2_temporal_residual_assignment", "V-JEPA 2 Res. Assignment", "#c0392b"),
]
methods = [
    (label, results[key]["ap"], results[key]["auc"], color)
    for key, label, color in method_specs
    if key in results
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

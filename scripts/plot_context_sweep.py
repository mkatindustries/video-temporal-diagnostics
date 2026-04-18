#!/usr/bin/env python3
"""Generate context sweep figure for paper: AP vs context_sec for temporal residual vs bag-of-tokens."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("datasets/hdd/context_sec_sweep_results.json") as f:
    data = json.load(f)

residual = data["vjepa2_temporal_residual"]
bot = data["vjepa2_bag_of_tokens"]

ctx = [d["context_sec"] for d in residual]
res_ap = [d["ap"] for d in residual]
res_lo = [d["ci_low"] for d in residual]
res_hi = [d["ci_high"] for d in residual]
bot_ap = [d["ap"] for d in bot]
bot_lo = [d["ci_low"] for d in bot]
bot_hi = [d["ci_high"] for d in bot]
fps_eff = [d["fps_eff_mean"] for d in residual]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(ctx, res_ap, "o-", color="#2196F3", linewidth=2, markersize=7, label="Temporal Residual", zorder=3)
ax.fill_between(ctx, res_lo, res_hi, color="#2196F3", alpha=0.15, zorder=2)

ax.plot(ctx, bot_ap, "s-", color="#FF9800", linewidth=2, markersize=7, label="Bag-of-Tokens", zorder=3)
ax.fill_between(ctx, bot_lo, bot_hi, color="#FF9800", alpha=0.15, zorder=2)

# Annotate fps_eff above each x tick
for i, (x, fps) in enumerate(zip(ctx, fps_eff)):
    ax.annotate(f"{fps:.1f}", (x, max(res_ap[i], bot_ap[i]) + 0.015),
                ha="center", va="bottom", fontsize=7.5, color="#666")

ax.set_xlabel("Context window (seconds)", fontsize=11)
ax.set_ylabel("Maneuver discrimination AP", fontsize=11)
ax.set_title("V-JEPA 2 Context Sweep on Honda HDD\n(64 frames, no padding)", fontsize=12)
ax.set_xticks(ctx)
ax.set_ylim(0.55, 1.0)
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.3)

# Secondary x-axis label for fps_eff
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ctx)
ax2.set_xticklabels([f"{f:.1f}" for f in fps_eff], fontsize=8)
ax2.set_xlabel("Effective FPS (64 frames / duration)", fontsize=9, color="#666")
ax2.tick_params(colors="#666")

plt.tight_layout()
plt.savefig("figures/hdd_context_sweep.png", dpi=200, bbox_inches="tight")
print("Saved figures/hdd_context_sweep.png")

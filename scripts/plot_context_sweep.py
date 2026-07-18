#!/usr/bin/env python3
"""Plot AP against per-side context padding for residual and bag-of-tokens."""

import json

import matplotlib.pyplot as plt

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
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    ctx,
    res_ap,
    "o-",
    color="#2196F3",
    linewidth=2,
    markersize=7,
    label="Temporal Residual",
    zorder=3,
)
ax.fill_between(ctx, res_lo, res_hi, color="#2196F3", alpha=0.15, zorder=2)

ax.plot(
    ctx, bot_ap, "s-", color="#FF9800", linewidth=2, markersize=7, label="Bag-of-Tokens", zorder=3
)
ax.fill_between(ctx, bot_lo, bot_hi, color="#FF9800", alpha=0.15, zorder=2)

ax.set_xlabel("Per-side context padding c (seconds)", fontsize=11)
ax.set_ylabel("Maneuver discrimination AP", fontsize=11)
ax.set_title(
    "V-JEPA 2 Context Sweep on Honda HDD\n(64 frames; duration = maneuver duration + 2c)",
    fontsize=12,
)
ax.set_xticks(ctx)
ax.set_ylim(0.55, 1.0)
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/hdd_context_sweep.png", dpi=200, bbox_inches="tight")
print("Saved figures/hdd_context_sweep.png")

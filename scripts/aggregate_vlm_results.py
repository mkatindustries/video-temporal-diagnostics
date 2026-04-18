#!/usr/bin/env python3
"""Aggregate multi-VLM temporal order results into combined table + figures.

Reads per-VLM result JSONs (temporal_order_results_{family}.json) and produces:
1. Combined CSV table for paper
2. Bar chart: balanced accuracy per VLM x prompt variant
3. s_rev comparison: vision_repr + LLM repr layers across all VLMs + DINOv3/V-JEPA 2

Usage:
    python scripts/aggregate_vlm_results.py \\
        --results-dir ./data/epic_kitchens

    # Custom output directory
    python scripts/aggregate_vlm_results.py \\
        --results-dir ./data/epic_kitchens \\
        --output-dir figures
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VLM_FAMILIES = ["qwen3", "gemma4", "llava-video"]


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all available per-family result JSONs."""
    results = {}
    for family in VLM_FAMILIES:
        path = results_dir / f"temporal_order_results_{family}.json"
        if path.exists():
            with open(path) as f:
                results[family] = json.load(f)
            print(f"  Loaded {family}: {path}")

    # Also load the base results (DINOv3 + V-JEPA 2)
    base_path = results_dir / "temporal_order_results.json"
    if base_path.exists():
        with open(base_path) as f:
            results["_base"] = json.load(f)
        print(f"  Loaded base results: {base_path}")

    return results


def write_combined_csv(results: dict[str, dict], output_path: Path) -> None:
    """Write combined CSV with generative + embedding results across VLMs."""
    rows = []

    # Header
    header = [
        "VLM Family",
        "Model",
        # Generative (mean across prompts)
        "Balanced Acc (mean)",
        "Balanced Acc (std)",
        # Per-prompt
        "Prompt 0 Acc",
        "Prompt 1 Acc",
        "Prompt 2 Acc",
        # Text-only
        "Text-Only Bias (FORWARD)",
        # Integrity
        "Integrity Forward",
        "Integrity Reverse",
        "Integrity Scramble k4",
        "Integrity Scramble k16",
        "Consistency %",
        # Embeddings
        "Vision Repr s_rev",
        "LLM Last s_rev",
        "LLM Mid s_rev",
        "LLM Penultimate s_rev",
    ]

    for family in VLM_FAMILIES:
        if family not in results:
            continue
        data = results[family]
        meta = data.get("metadata", {})
        gen = data.get("vlm_generative", {})
        emb = data.get("vlm_embedding", {})

        row = [family, meta.get("vlm_model", "")]

        # Generative
        row.append(gen.get("mean_balanced_acc", ""))
        row.append(gen.get("std_balanced_acc", ""))

        # Per-prompt
        per_prompt = gen.get("per_prompt", {})
        for i in range(3):
            key = f"prompt_{i}"
            if key in per_prompt:
                row.append(per_prompt[key].get("balanced_acc", ""))
            else:
                row.append("")

        # Text-only
        text_only = gen.get("text_only_baseline", {})
        row.append(text_only.get("mean_bias_rate_forward", ""))

        # Integrity
        integrity = gen.get("integrity", {})
        row.append(integrity.get("forward_acc", ""))
        row.append(integrity.get("reverse_acc", ""))
        row.append(integrity.get("scramble_k4_acc", ""))
        row.append(integrity.get("scramble_k16_acc", ""))
        row.append(integrity.get("consistency_pct", ""))

        # Embeddings
        row.append(emb.get("vision_repr", {}).get("s_rev_mean", ""))
        row.append(emb.get("llm_repr_last", {}).get("s_rev_mean", ""))
        row.append(emb.get("llm_repr_mid", {}).get("s_rev_mean", ""))
        row.append(emb.get("llm_repr_penultimate", {}).get("s_rev_mean", ""))

        rows.append(row)

    # Add DINOv3 / V-JEPA 2 baseline rows from any available result
    any_result = next(iter(results.values()), None)
    if any_result and "order_sensitivity" in any_result:
        order = any_result["order_sensitivity"]
        for method_key, label in [
            ("bag_of_frames", "DINOv3 BoF"),
            ("chamfer", "DINOv3 Chamfer"),
            ("temporal_derivative", "DINOv3 Temp. Deriv."),
            ("attention_trajectory", "DINOv3 Attn. Traj."),
            ("vjepa2_bag_of_tokens", "V-JEPA 2 BoT"),
            ("vjepa2_temporal_residual", "V-JEPA 2 Residual"),
        ]:
            if method_key in order:
                s_rev = order[method_key].get("s_rev_mean", "")
                baseline_row = [label, ""] + [""] * 13 + [s_rev, "", "", ""]
                rows.append(baseline_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  CSV saved to {output_path}")


def plot_balanced_accuracy(results: dict[str, dict], output_path: Path) -> None:
    """Bar chart: balanced accuracy per VLM x prompt variant."""
    families_with_gen = [
        f for f in VLM_FAMILIES if f in results and "vlm_generative" in results[f]
    ]

    if not families_with_gen:
        print("  No generative results found, skipping accuracy plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n_families = len(families_with_gen)
    n_prompts = 3
    width = 0.2
    x = np.arange(n_families)

    for prompt_idx in range(n_prompts):
        accs = []
        for family in families_with_gen:
            gen = results[family]["vlm_generative"]
            per_prompt = gen.get("per_prompt", {})
            key = f"prompt_{prompt_idx}"
            accs.append(per_prompt.get(key, {}).get("balanced_acc", 0))

        offset = (prompt_idx - 1) * width
        ax.bar(
            x + offset,
            accs,
            width,
            label=f"Prompt {prompt_idx}",
            edgecolor="black",
            alpha=0.8,
        )

    # Add mean as marker
    for i, family in enumerate(families_with_gen):
        gen = results[family]["vlm_generative"]
        mean_acc = gen.get("mean_balanced_acc", 0)
        std_acc = gen.get("std_balanced_acc", 0)
        ax.errorbar(
            i, mean_acc, yerr=std_acc, fmt="ko", markersize=8, capsize=4, zorder=5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(families_with_gen, fontsize=11)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("EPIC-Kitchens: VLM Forward/Reverse Accuracy per Prompt Variant")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_s_rev_comparison(results: dict[str, dict], output_path: Path) -> None:
    """s_rev comparison across all methods: DINOv3, V-JEPA 2, and VLM layers."""
    # Collect all (label, s_rev, ci) tuples
    entries = []

    # Baseline methods from any available result
    any_result = next(iter(results.values()), None)
    if any_result and "order_sensitivity" in any_result:
        order = any_result["order_sensitivity"]
        baseline_methods = [
            ("bag_of_frames", "DINOv3\nBoF"),
            ("chamfer", "DINOv3\nChamfer"),
            ("temporal_derivative", "DINOv3\nTemp Deriv"),
            ("attention_trajectory", "DINOv3\nAttn Traj"),
            ("vjepa2_bag_of_tokens", "V-JEPA 2\nBoT"),
            ("vjepa2_temporal_residual", "V-JEPA 2\nResidual"),
        ]
        for key, label in baseline_methods:
            if key in order:
                s_rev = order[key].get("s_rev_mean", None)
                ci = order[key].get("s_rev_ci", [s_rev, s_rev])
                if s_rev is not None:
                    entries.append((label, s_rev, ci, "steelblue"))

    # VLM embedding methods
    vlm_colors = {
        "qwen3": "coral",
        "gemma4": "mediumseagreen",
        "llava-video": "mediumpurple",
    }
    for family in VLM_FAMILIES:
        if family not in results:
            continue
        emb = results[family].get("vlm_embedding", {})
        color = vlm_colors.get(family, "gray")
        for layer_key, layer_label in [
            ("vision_repr", "Vision"),
            ("llm_repr_last", "LLM Last"),
            ("llm_repr_mid", "LLM Mid"),
            ("llm_repr_penultimate", "LLM Penult"),
        ]:
            if layer_key in emb:
                s_rev = emb[layer_key].get("s_rev_mean", None)
                ci = emb[layer_key].get("s_rev_ci", [s_rev, s_rev])
                if s_rev is not None:
                    label = f"{family}\n{layer_label}"
                    entries.append((label, s_rev, ci, color))

    if not entries:
        print("  No s_rev data found, skipping comparison plot")
        return

    labels, s_revs, cis, colors = zip(*entries)
    ci_lows = [s - c[0] for s, c in zip(s_revs, cis)]
    ci_highs = [c[1] - s for s, c in zip(s_revs, cis)]

    fig, ax = plt.subplots(figsize=(max(12.0, len(entries) * 0.8), 5))
    x = np.arange(len(entries))
    ax.bar(
        x,
        s_revs,
        yerr=[ci_lows, ci_highs],
        capsize=3,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("s_rev (cosine sim fwd vs rev)")
    ax.set_title("EPIC-Kitchens: s_rev Across All Methods")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-VLM temporal order results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing temporal_order_results_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for figures and CSV",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("No result files found!")
        return

    print(f"\nFound {len(results)} result files")

    # Combined CSV
    write_combined_csv(results, output_dir / "epic_vlm_combined_results.csv")

    # Balanced accuracy chart
    plot_balanced_accuracy(results, output_dir / "epic_vlm_balanced_accuracy.png")

    # s_rev comparison
    plot_s_rev_comparison(results, output_dir / "epic_vlm_s_rev_comparison.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

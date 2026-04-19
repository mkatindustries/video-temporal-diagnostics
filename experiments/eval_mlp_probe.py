#!/usr/bin/env python3
"""MLP Probe on Cached LLM Hidden States.

Extends the linear probe (eval_epic_linear_probe.py) with a 2-layer MLP
to test whether temporal order is nonlinearly encoded in LLM hidden states.

If the MLP also fails → the "not hidden, gone" claim is airtight.
If the MLP succeeds → temporal signal is nonlinearly encoded but
inaccessible to standard readouts (still interesting, different narrative).

Loads cached hidden states from the linear probe SLURM runs.
No GPU needed (400 samples, CPU is fine).

Usage:
    python experiments/eval_mlp_probe.py
    python experiments/eval_mlp_probe.py --hidden-dim 512 --n-folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Pooling (same as eval_epic_linear_probe.py)
# ---------------------------------------------------------------------------

POOLING_STRATEGIES = [
    "mean",
    "max",
    "first",
    "last",
    "concat_first_last",
    "vision_mean",
    "vision_concat_fl",
]


def apply_pooling(
    hidden: torch.Tensor,
    vision_mask: torch.Tensor | None,
    strategy: str,
) -> np.ndarray | None:
    hidden = hidden.float()

    if strategy == "mean":
        return F.normalize(hidden.mean(dim=0), dim=0).numpy()
    elif strategy == "max":
        return F.normalize(hidden.max(dim=0).values, dim=0).numpy()
    elif strategy == "first":
        return F.normalize(hidden[0], dim=0).numpy()
    elif strategy == "last":
        return F.normalize(hidden[-1], dim=0).numpy()
    elif strategy == "concat_first_last":
        v = torch.cat([hidden[0], hidden[-1]])
        return F.normalize(v, dim=0).numpy()
    elif strategy == "vision_mean":
        if vision_mask is None or vision_mask.sum() == 0:
            return None
        vis_hidden = hidden[vision_mask]
        return F.normalize(vis_hidden.mean(dim=0), dim=0).numpy()
    elif strategy == "vision_concat_fl":
        if vision_mask is None or vision_mask.sum() < 2:
            return None
        vis_hidden = hidden[vision_mask]
        v = torch.cat([vis_hidden[0], vis_hidden[-1]])
        return F.normalize(v, dim=0).numpy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MLP probe on cached LLM hidden states"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="datasets/epic_kitchens/feature_cache",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cache_dir = project_root / args.cache_dir

    families = []
    for f in sorted(cache_dir.glob("linear_probe_*_hidden_states.pt")):
        family = f.stem.replace("linear_probe_", "").replace("_hidden_states", "")
        families.append((family, f))

    if not families:
        print(f"No cached hidden states found in {cache_dir}")
        return

    print("=" * 70)
    print("MLP PROBE ON LLM HIDDEN STATES: FORWARD vs REVERSE")
    print(f"  MLP: 2 layers, {args.hidden_dim} hidden units, ReLU")
    print(f"  CV: {args.n_folds}-fold GroupKFold (fwd/rev pairs kept together)")
    print(f"  Families: {[f for f, _ in families]}")
    print("=" * 70)

    all_results = {}

    for family, cache_path in families:
        print(f"\n{'=' * 50}")
        print(f"  {family}")
        print("=" * 50)

        print(f"  Loading {cache_path.name}...")
        cached = torch.load(cache_path, weights_only=False)
        fwd_states = cached["forward"]
        rev_states = cached["reverse"]
        seq_ids = cached["sequence_ids"]
        print(f"  {len(seq_ids)} sequences loaded")

        # Build feature matrices
        strategy_features: dict[str, list] = {s: [] for s in POOLING_STRATEGIES}
        labels = []

        for seq_id in seq_ids:
            for direction, states in [("fwd", fwd_states), ("rev", rev_states)]:
                state = states[seq_id]
                hidden = state["hidden_last"]
                vision_mask = state.get("vision_mask")

                labels.append(0 if direction == "fwd" else 1)
                for strategy in POOLING_STRATEGIES:
                    vec = apply_pooling(hidden, vision_mask, strategy)
                    strategy_features[strategy].append(vec)

        labels = np.array(labels)
        # Groups for GroupKFold: each sequence contributes 2 samples (fwd+rev)
        # that must stay in the same fold to avoid sequence-identity leakage
        all_groups = np.repeat(np.arange(len(seq_ids)), 2)
        n_samples = len(labels)
        print(f"  {n_samples} samples ({n_samples // 2} fwd + {n_samples // 2} rev)")

        # Evaluate both linear and MLP for comparison
        family_results = {}
        gkf = GroupKFold(n_splits=args.n_folds)

        for strategy in POOLING_STRATEGIES:
            vecs = strategy_features[strategy]
            if any(v is None for v in vecs):
                n_none = sum(1 for v in vecs if v is None)
                if n_none > len(vecs) * 0.5:
                    print(f"  {strategy:>20s}: SKIPPED ({n_none}/{len(vecs)} None)")
                    continue
                valid_idx = [i for i, v in enumerate(vecs) if v is not None]
                X = np.stack([vecs[i] for i in valid_idx])
                y = labels[valid_idx]
                groups = all_groups[valid_idx]
            else:
                X = np.stack(vecs)
                y = labels
                groups = all_groups

            # Linear probe (logistic regression)
            linear_accs = []
            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])
                from sklearn.linear_model import LogisticRegression

                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                clf.fit(X_tr, y[train_idx])
                linear_accs.append(clf.score(X_te, y[test_idx]))

            # MLP probe
            mlp_accs = []
            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])

                mlp = MLPClassifier(
                    hidden_layer_sizes=(args.hidden_dim, args.hidden_dim),
                    activation="relu",
                    max_iter=args.max_iter,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                )
                mlp.fit(X_tr, y[train_idx])
                mlp_accs.append(mlp.score(X_te, y[test_idx]))

            linear_arr = np.array(linear_accs)
            mlp_arr = np.array(mlp_accs)

            family_results[strategy] = {
                "linear_mean": float(np.mean(linear_arr)),
                "linear_std": float(np.std(linear_arr)),
                "linear_folds": [float(a) for a in linear_accs],
                "mlp_mean": float(np.mean(mlp_arr)),
                "mlp_std": float(np.std(mlp_arr)),
                "mlp_folds": [float(a) for a in mlp_accs],
            }

            above = "***" if float(np.mean(mlp_arr)) > 0.55 else ""
            print(
                f"  {strategy:>20s}:  "
                f"linear={np.mean(linear_arr):.3f}±{np.std(linear_arr):.3f}  "
                f"MLP={np.mean(mlp_arr):.3f}±{np.std(mlp_arr):.3f}  "
                f"{above}"
            )

        all_results[family] = family_results

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    for family, results in all_results.items():
        print(f"\n{family}:")
        for strategy, r in results.items():
            mlp_mean: float = r["mlp_mean"]  # pyrefly: ignore
            linear_mean: float = r["linear_mean"]  # pyrefly: ignore
            delta = mlp_mean - linear_mean
            flag = "***" if mlp_mean > 0.55 else ""
            print(
                f"  {strategy:>20s}:  "
                f"linear={r['linear_mean']:.3f}  "
                f"MLP={r['mlp_mean']:.3f}  "
                f"Δ={delta:+.3f}  {flag}"
            )

    # Save
    out_path = project_root / "datasets" / "epic_kitchens" / "mlp_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

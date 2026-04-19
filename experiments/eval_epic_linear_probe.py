#!/usr/bin/env python3
"""Linear Probe on VLM Hidden States for Temporal Order Classification.

Tests whether temporal order information is present in VLM hidden states
but destroyed by mean pooling. For each VLM, extracts per-position hidden
states for forward and reverse EPIC-Kitchens sequences, then trains a
logistic regression classifier (5-fold CV) to predict temporal direction.

Multiple pooling strategies are compared:
  - mean: Mean over all positions
  - max: Max over all positions
  - first: First token only
  - last: Last token only
  - concat_first_last: Concatenation of first and last tokens
  - vision_mean: Mean over vision-token positions only
  - vision_concat_fl: First + last vision-token positions

If accuracy > chance: temporal signal IS present in hidden states, just
not surfaced by mean pooling (readout problem).
If accuracy ≈ chance: the LLM genuinely destroys temporal information.

Usage:
    # Single VLM (SLURM job)
    python experiments/eval_epic_linear_probe.py \\
        --epic-dir datasets/epic_kitchens --vlm-family qwen3

    # Smoke test
    python experiments/eval_epic_linear_probe.py \\
        --epic-dir datasets/epic_kitchens --vlm-family llava-video \\
        --max-sequences 10

    # All 3 VLMs sequentially (not recommended — run as separate jobs)
    python experiments/eval_epic_linear_probe.py \\
        --epic-dir datasets/epic_kitchens --vlm-family qwen3 gemma4 llava-video
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Reuse VLM adapters and frame loading from eval_epic_temporal_order
from eval_epic_temporal_order import (
    DESCRIBE_PROMPT,
    VLM_ADAPTERS,
    VLM_DEFAULT_PATHS,
    VLMAdapter,
    compute_fps_eff,
    load_sequences,
    sample_canonical_frames,
)


# ---------------------------------------------------------------------------
# Hidden state extraction — returns full per-position states, not pooled
# ---------------------------------------------------------------------------


def extract_hidden_states(
    adapter: VLMAdapter,
    model: torch.nn.Module,
    processor: object,
    pil_frames: list[Image.Image],
    device: torch.device,
    fps_eff: float,
) -> dict[str, torch.Tensor | None] | None:
    """Extract full per-position LLM hidden states for a single sequence.

    Returns dict with:
        'hidden_last': (seq_len, D) — last layer hidden states
        'vision_mask': (seq_len,) bool — True at vision-token positions
    or None on failure.
    """
    try:
        inputs = adapter.prepare_inputs(
            processor, pil_frames, DESCRIBE_PROMPT, device, fps=fps_eff
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Last layer hidden states
            hidden_last = outputs.hidden_states[-1][0]  # (seq_len, D)

            # Identify vision token positions
            input_ids = inputs.get("input_ids")
            vision_mask = _find_vision_mask(model, input_ids, hidden_last)

        return {
            "hidden_last": hidden_last.cpu(),
            "vision_mask": vision_mask.cpu() if vision_mask is not None else None,
        }
    except Exception as e:
        warnings.warn(f"extract_hidden_states failed: {e}")
        return None


def _find_vision_mask(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None,
    hidden: torch.Tensor,
) -> torch.Tensor | None:
    """Find vision token positions in hidden states.

    Returns a boolean mask of shape (hidden_seq_len,) indicating vision
    positions, or None if detection fails.
    """
    if input_ids is None:
        return None

    ids = input_ids[0]  # (seq_len_in,)
    seq_len_in = len(ids)
    seq_len_out = hidden.shape[0]

    # Try known vision placeholder token IDs
    candidate_ids = []

    # Model-specific token IDs
    img_tok_id = getattr(model.config, "image_token_index", None)
    if img_tok_id is not None:
        candidate_ids.append(img_tok_id)

    # Qwen: <|video_pad|>=151656, <|image_pad|>=151655
    # LLaVA: image_token_index from config, or 32000/32001
    # Gemma: image_token_index from config, or 258880/262144
    candidate_ids.extend([151656, 151655, 32000, 32001, 258880, 262144, 256000])

    vision_positions_input = None
    n_placeholders = 0
    for cid in candidate_ids:
        mask = ids == cid
        n = int(mask.sum().item())
        if n > 0:
            vision_positions_input = mask
            n_placeholders = n
            break

    if vision_positions_input is None:
        return None

    # Build output-space vision mask
    if seq_len_out > seq_len_in:
        # Vision tokens were expanded during merge
        text_len = seq_len_in - n_placeholders
        n_vision_expanded = seq_len_out - text_len

        positions = torch.where(vision_positions_input)[0]
        first_vis_pos = int(positions[0].item())

        out_mask = torch.zeros(seq_len_out, dtype=torch.bool, device=hidden.device)
        out_mask[first_vis_pos : first_vis_pos + n_vision_expanded] = True
        return out_mask
    else:
        # 1:1 mapping — same length
        return vision_positions_input.to(hidden.device)


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------


def apply_pooling(
    hidden: torch.Tensor,
    vision_mask: torch.Tensor | None,
    strategy: str,
    keep_tensor: bool = False,
) -> np.ndarray | torch.Tensor | None:
    """Apply a pooling strategy to hidden states.

    Args:
        hidden: (seq_len, D)
        vision_mask: (seq_len,) bool or None
        strategy: one of mean, max, first, last, concat_first_last,
                  vision_mean, vision_concat_fl
        keep_tensor: if True, return a torch.Tensor instead of numpy array

    Returns:
        1D numpy array (or torch.Tensor if keep_tensor) or None.
    """
    # Cast to float32 for numpy compatibility (hidden states may be bfloat16)
    hidden = hidden.float()

    if strategy == "mean":
        v = F.normalize(hidden.mean(dim=0), dim=0)
    elif strategy == "max":
        v = F.normalize(hidden.max(dim=0).values, dim=0)
    elif strategy == "first":
        v = F.normalize(hidden[0], dim=0)
    elif strategy == "last":
        v = F.normalize(hidden[-1], dim=0)
    elif strategy == "concat_first_last":
        v = F.normalize(torch.cat([hidden[0], hidden[-1]]), dim=0)
    elif strategy == "vision_mean":
        if vision_mask is None or vision_mask.sum() == 0:
            return None
        vis_hidden = hidden[vision_mask]
        v = F.normalize(vis_hidden.mean(dim=0), dim=0)
    elif strategy == "vision_concat_fl":
        if vision_mask is None or vision_mask.sum() < 2:
            return None
        vis_hidden = hidden[vision_mask]
        v = F.normalize(torch.cat([vis_hidden[0], vis_hidden[-1]]), dim=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return v if keep_tensor else v.numpy()


POOLING_STRATEGIES = [
    "mean",
    "max",
    "first",
    "last",
    "concat_first_last",
    "vision_mean",
    "vision_concat_fl",
]


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------


def evaluate_linear_probe(
    features: dict[str, np.ndarray | torch.Tensor | None],
    labels: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    c_values: list[float] | None = None,
) -> dict[str, dict | None]:
    """Train logistic regression with GroupKFold CV.

    Uses GroupKFold so forward/reverse pairs of the same sequence stay
    in the same fold, preventing sequence-identity leakage.

    Args:
        features: Dict mapping strategy name -> (N, D) feature matrix.
        labels: (N,) binary labels (0=forward, 1=reverse).
        groups: (N,) group IDs (sequence index, so fwd/rev pairs share a group).
        n_folds: Number of CV folds.
        seed: Random seed.
        c_values: Regularization strengths to sweep. Reports best across sweep.

    Returns:
        Dict mapping strategy name -> best result across C values.
    """
    if c_values is None:
        c_values = [1.0]

    results: dict[str, dict | None] = {}
    gkf = GroupKFold(n_splits=n_folds)

    for strategy, X_raw in features.items():
        if X_raw is None:
            results[strategy] = None
            continue

        X: np.ndarray = X_raw.numpy() if isinstance(X_raw, torch.Tensor) else X_raw

        best_result = None
        best_mean_acc = -1.0
        per_c_accs: dict[float, float] = {}

        for c_val in c_values:
            fold_accs = []
            for train_idx, test_idx in gkf.split(X, labels, groups=groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf = LogisticRegression(
                    max_iter=1000,
                    C=c_val,
                    solver="lbfgs",
                    random_state=seed,
                )
                clf.fit(X_train, y_train)
                fold_accs.append(clf.score(X_test, y_test))

            fold_accs_arr = np.array(fold_accs)
            mean_acc = float(np.mean(fold_accs_arr))
            per_c_accs[c_val] = mean_acc

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_result = {
                    "mean_acc": mean_acc,
                    "std_acc": float(np.std(fold_accs_arr)),
                    "per_fold": [float(a) for a in fold_accs],
                    "best_C": c_val,
                    "per_c_accs": {str(k): v for k, v in per_c_accs.items()},
                }

        # Attach final per_c_accs to best result
        if best_result is not None:
            best_result["per_c_accs"] = {str(k): v for k, v in per_c_accs.items()}

        results[strategy] = best_result

    return results


def _gpu_logistic_regression(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    C: float,
    max_iter: int = 1000,
) -> float:
    D = X_train.shape[1]
    mu = X_train.mean(dim=0)
    std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    w = torch.zeros(D, device=X_train.device, dtype=X_train.dtype, requires_grad=True)
    b = torch.zeros(1, device=X_train.device, dtype=X_train.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y_f = y_train.float()

    def closure():
        opt.zero_grad()
        logits = X_train @ w + b
        loss = loss_fn(logits, y_f) + (0.5 / C) * w.dot(w)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        preds = (X_test @ w + b) > 0
        acc = preds.eq(y_test.bool()).float().mean().item()
    return acc


def evaluate_linear_probe_gpu(
    features: dict[str, torch.Tensor | np.ndarray | None],
    labels: torch.Tensor,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    c_values: list[float] | None = None,
) -> dict[str, dict | None]:
    if c_values is None:
        c_values = [1.0]

    results: dict[str, dict | None] = {}
    gkf = GroupKFold(n_splits=n_folds)
    labels_np = labels.cpu().numpy()

    for strategy, X_raw in features.items():
        if X_raw is None:
            results[strategy] = None
            continue

        X = X_raw if isinstance(X_raw, torch.Tensor) else torch.from_numpy(X_raw)
        X_np = X.cpu().numpy()
        best_result = None
        best_mean_acc = -1.0
        per_c_accs: dict[float, float] = {}

        for c_val in c_values:
            fold_accs = []
            for train_idx, test_idx in gkf.split(X_np, labels_np, groups=groups):
                train_idx_t = torch.tensor(train_idx, device=X.device)
                test_idx_t = torch.tensor(test_idx, device=X.device)
                acc = _gpu_logistic_regression(
                    X[train_idx_t], labels[train_idx_t],
                    X[test_idx_t], labels[test_idx_t],
                    C=c_val,
                )
                fold_accs.append(acc)

            fold_accs_arr = np.array(fold_accs)
            mean_acc = float(np.mean(fold_accs_arr))
            per_c_accs[c_val] = mean_acc

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_result = {
                    "mean_acc": mean_acc,
                    "std_acc": float(np.std(fold_accs_arr)),
                    "per_fold": [float(a) for a in fold_accs],
                    "best_C": c_val,
                    "per_c_accs": {str(k): v for k, v in per_c_accs.items()},
                }

        if best_result is not None:
            best_result["per_c_accs"] = {str(k): v for k, v in per_c_accs.items()}

        results[strategy] = best_result

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe on VLM hidden states for temporal order"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default="datasets/epic_kitchens",
        help="Path to EPIC-Kitchens data directory",
    )
    parser.add_argument(
        "--vlm-family",
        type=str,
        nargs="+",
        default=["qwen3"],
        choices=list(VLM_DEFAULT_PATHS.keys()),
        help="VLM family/families to evaluate",
    )
    parser.add_argument(
        "--vlm-path",
        type=str,
        default=None,
        help="Override model path (only for single --vlm-family)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=200,
        help="Maximum number of sequences",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="Canonical frame count for VLM input",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="Regularization strengths C to sweep (default: 0.01 0.1 1 10 100)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Skip model loading; require cached hidden states",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    epic_dir = project_root / args.epic_dir
    device = torch.device(args.device)

    print("=" * 70)
    print("LINEAR PROBE ON VLM HIDDEN STATES: FORWARD vs REVERSE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load sequences (skipped when --probe-only)
    # ------------------------------------------------------------------
    if args.probe_only:
        print("\nStep 1: --probe-only, skipping video loading")
        sequences = []
    else:
        print("\nStep 1: Loading sequences...")
        sequences = load_sequences(epic_dir, max_sequences=args.max_sequences)
        n_seq = len(sequences)
        print(f"  Using {n_seq} sequences")

    # ------------------------------------------------------------------
    # Step 2: For each VLM, extract hidden states and evaluate
    # ------------------------------------------------------------------
    all_results = {}

    for family in args.vlm_family:
        print(f"\n{'=' * 70}")
        print(f"VLM: {family}")
        print("=" * 70)

        model_path = args.vlm_path if args.vlm_path else VLM_DEFAULT_PATHS[family]

        # Check for cached hidden states
        cache_dir = project_root / "datasets" / "epic_kitchens" / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"linear_probe_{family}_hidden_states.pt"

        if cache_path.exists():
            map_loc = "cuda" if args.probe_only and device.type == "cuda" else "cpu"
            print(f"\n  Loading cached hidden states from {cache_path} (map_location={map_loc})")
            cached = torch.load(cache_path, weights_only=False, map_location=map_loc)
            fwd_states = cached["forward"]
            rev_states = cached["reverse"]
            seq_ids = cached["sequence_ids"]
        elif args.probe_only:
            print(f"  ERROR: --probe-only but no cache at {cache_path}")
            continue
        else:
            adapter = VLM_ADAPTERS[family]()

            print(f"\nStep 2a: Loading {family} from {model_path}...")
            t0 = time.time()
            model, processor = adapter.load(model_path, device)
            print(f"  Loaded in {time.time() - t0:.1f}s")

            # Extract hidden states for forward and reverse
            print(f"\nStep 2b: Extracting hidden states ({family})...")
            fwd_states = {}  # seq_id -> hidden state dict
            rev_states = {}
            failed = 0

            for seq in tqdm(sequences, desc=f"{family} hidden states"):
                seq_id = seq["sequence_id"]
                try:
                    pil_frames, timestamps = sample_canonical_frames(
                        seq["video_path"],
                        seq["start_sec"],
                        seq["stop_sec"],
                        n_frames=args.n_frames,
                        max_resolution=518,
                    )
                    if len(pil_frames) < 3:
                        failed += 1
                        continue

                    fps_eff = compute_fps_eff(timestamps)

                    # Forward
                    fwd = extract_hidden_states(
                        adapter, model, processor, pil_frames, device, fps_eff
                    )
                    if fwd is None:
                        failed += 1
                        continue

                    # Reverse
                    rev_frames = pil_frames[::-1]
                    rev = extract_hidden_states(
                        adapter, model, processor, rev_frames, device, fps_eff
                    )
                    if rev is None:
                        failed += 1
                        continue

                    fwd_states[seq_id] = fwd
                    rev_states[seq_id] = rev

                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"  WARNING: {seq_id} failed: {e}")
                    continue

            seq_ids = sorted(fwd_states.keys())
            print(
                f"  Extracted: {len(seq_ids)}/{n_seq} sequences "
                f"({failed} failed)"
            )

            # Cache
            torch.save(
                {
                    "forward": fwd_states,
                    "reverse": rev_states,
                    "sequence_ids": seq_ids,
                },
                cache_path,
            )
            print(f"  Cached to {cache_path}")

            del model, processor
            torch.cuda.empty_cache()

        if len(seq_ids) < 10:
            print(f"  WARNING: Only {len(seq_ids)} sequences — skipping evaluation")
            continue

        # ------------------------------------------------------------------
        # Step 3: Build feature matrices per pooling strategy
        # ------------------------------------------------------------------
        print(f"\nStep 3: Building feature matrices ({family})...")

        use_gpu_probe = args.probe_only and device.type == "cuda"

        # Labels: 0 = forward, 1 = reverse
        # Each sequence contributes one forward and one reverse sample
        strategy_features: dict[str, list] = {
            s: [] for s in POOLING_STRATEGIES
        }
        labels = []

        for seq_id in seq_ids:
            for direction, states in [("fwd", fwd_states), ("rev", rev_states)]:
                state = states[seq_id]
                hidden = state["hidden_last"]
                vision_mask = state.get("vision_mask")

                label = 0 if direction == "fwd" else 1
                labels.append(label)

                for strategy in POOLING_STRATEGIES:
                    # pyrefly: ignore [bad-argument-type]
                    vec = apply_pooling(hidden, vision_mask, strategy,
                                        keep_tensor=use_gpu_probe)
                    strategy_features[strategy].append(vec)

        labels_np = np.array(labels)
        # Groups for GroupKFold: fwd/rev pairs of the same sequence must stay together
        all_groups = np.repeat(np.arange(len(seq_ids)), 2)
        n_samples = len(labels_np)
        print(f"  Total samples: {n_samples} ({n_samples // 2} fwd + {n_samples // 2} rev)")

        # Stack into matrices, dropping strategies where some samples failed
        feature_matrices_gpu: dict[str, torch.Tensor | None] = {}
        feature_matrices: dict[str, np.ndarray | None] = {}
        if use_gpu_probe:
            for strategy in POOLING_STRATEGIES:
                vecs = strategy_features[strategy]
                if any(v is None for v in vecs):
                    n_none = sum(1 for v in vecs if v is None)
                    if n_none > len(vecs) * 0.5:
                        print(f"  WARNING: {strategy} has {n_none}/{len(vecs)} None — skipping")
                        feature_matrices_gpu[strategy] = None
                    else:
                        valid_vecs = [v for v in vecs if v is not None]
                        feature_matrices_gpu[strategy] = torch.stack(valid_vecs)
                else:
                    feature_matrices_gpu[strategy] = torch.stack(vecs)
            labels_t = torch.tensor(labels_np, device=device, dtype=torch.long)
        else:
            for strategy in POOLING_STRATEGIES:
                vecs = strategy_features[strategy]
                if any(v is None for v in vecs):
                    n_none = sum(1 for v in vecs if v is None)
                    if n_none > len(vecs) * 0.5:
                        print(f"  WARNING: {strategy} has {n_none}/{len(vecs)} None — skipping")
                        feature_matrices[strategy] = None
                    else:
                        valid_vecs = [v for v in vecs if v is not None]
                        # pyrefly: ignore [no-matching-overload]
                        feature_matrices[strategy] = np.stack(valid_vecs)
                else:
                    # pyrefly: ignore [no-matching-overload]
                    feature_matrices[strategy] = np.stack(vecs)

        # ------------------------------------------------------------------
        # Step 4: Evaluate linear probes
        # ------------------------------------------------------------------
        print(f"\nStep 4: Evaluating linear probes ({family}, {args.n_folds}-fold CV, C={args.c_values})...")

        # For strategies with filtered samples, we need separate label arrays
        family_results: dict[str, dict | None] = {}
        for strategy in POOLING_STRATEGIES:
            if use_gpu_probe:
                X = feature_matrices_gpu.get(strategy)
            else:
                X = feature_matrices.get(strategy)
            if X is None:
                family_results[strategy] = None
                print(f"  {strategy:>20s}: SKIPPED (insufficient data)")
                continue

            # Get matching labels and groups
            vecs = strategy_features[strategy]
            if any(v is None for v in vecs):
                valid_idx = [i for i, v in enumerate(vecs) if v is not None]
                y_np = labels_np[valid_idx]
                g = all_groups[valid_idx]
            else:
                y_np = labels_np
                g = all_groups

            if use_gpu_probe:
                y_t = torch.tensor(y_np, device=device, dtype=torch.long)
                result = evaluate_linear_probe_gpu(
                    {strategy: X}, y_t, groups=g, n_folds=args.n_folds,
                    c_values=args.c_values,
                )
            else:
                result = evaluate_linear_probe(
                    {strategy: X}, y_np, groups=g, n_folds=args.n_folds,
                    c_values=args.c_values,
                )
            r = result[strategy]
            family_results[strategy] = r

            if r is not None:
                c_str = f"  C={r['best_C']}" if len(args.c_values) > 1 else ""
                print(
                    f"  {strategy:>20s}: acc={r['mean_acc']:.3f} ± {r['std_acc']:.3f}{c_str}  "
                    f"(folds: {', '.join(f'{a:.3f}' for a in r['per_fold'])})"
                )

        all_results[family] = {
            "pooling_strategies": family_results,
            "n_sequences": len(seq_ids),
            "n_samples": n_samples,
            "n_frames": args.n_frames,
            "n_folds": args.n_folds,
            "c_values": args.c_values,
            "model_path": model_path,
        }

    # ------------------------------------------------------------------
    # Step 5: Summary and save
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    for family, res in all_results.items():
        print(f"\n{family} ({res['n_sequences']} sequences, {res['n_samples']} samples):")
        pooling = res["pooling_strategies"]
        for strategy in POOLING_STRATEGIES:
            r = pooling.get(strategy) if isinstance(pooling, dict) else None
            if r is None:
                print(f"  {strategy:>20s}: SKIPPED")
            else:
                above_chance = "***" if r["mean_acc"] > 0.55 else ""
                c_info = f"  (best C={r['best_C']})" if "best_C" in r else ""
                print(
                    f"  {strategy:>20s}: {r['mean_acc']:.3f} ± {r['std_acc']:.3f} "
                    f"{above_chance}{c_info}"
                )

    # Save results
    output_dir = project_root / "datasets" / "epic_kitchens"
    output_dir.mkdir(parents=True, exist_ok=True)
    families_str = "_".join(args.vlm_family)
    output_path = output_dir / f"linear_probe_{families_str}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Interpretation guide
    print("\nInterpretation:")
    print("  acc ≈ 0.50: No temporal signal in hidden states (true invariance)")
    print("  acc > 0.55: Temporal signal present but not surfaced by pooling")
    print("  acc > 0.70: Strong temporal signal — readout problem confirmed")
    print("  *** marks strategies with acc > 0.55")


if __name__ == "__main__":
    main()

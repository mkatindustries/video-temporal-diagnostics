#!/usr/bin/env python3
"""Gemini 3.1 Pro Generative Probe for Temporal Order on EPIC-Kitchens.

Tests whether Gemini 3.1 Pro (frontier proprietary VLM) can distinguish
forward from reverse video playback. Sends 16 canonical frames as base64
images via the OpenAI-compatible API endpoint.

Same protocol as eval_epic_claude_probe.py (Claude 4.6 Opus):
  - Same 3 causality prompt variants
  - Same EPIC-Kitchens sequences (from temporal_order_sequences.json)
  - Same canonical frame sampling (16 frames, uniform temporal spacing)
  - Forward + reverse for each sequence, report balanced accuracy

Usage:
    # Full run
    python experiments/eval_epic_gemini_probe.py \\
        --epic-dir datasets/epic_kitchens

    # Smoke test (5 sequences)
    python experiments/eval_epic_gemini_probe.py \\
        --epic-dir datasets/epic_kitchens --max-sequences 5

    # Custom API key
    python experiments/eval_epic_gemini_probe.py \\
        --epic-dir datasets/epic_kitchens \\
        --api-key "Bearer LLM|..."
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse shared infrastructure
from eval_epic_claude_probe import encode_frame_base64
from eval_epic_temporal_order import (
    DEFAULT_CAUSALITY_PROMPTS,
    load_sequences,
    parse_forward_reverse,
    sample_canonical_frames,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://api.llama.com/experimental/compat/openai/v1"
DEFAULT_MODEL = "gemini-3-1-pro-preview-genai"
# Gemini 3.1 Pro is a reasoning model: thinking tokens consume part of
# max_tokens, so we need a larger budget than Claude's 64.
DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# API client (handles reasoning-model response shape)
# ---------------------------------------------------------------------------


def call_gemini_api(
    frames_b64: list[str],
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """Send frames + prompt to Gemini via OpenAI-compatible API.

    Handles reasoning models where the response may lack a ``message``
    field if all tokens are consumed by thinking.
    """
    import requests

    content: list[dict] = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }

    url = f"{api_base}/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message")
            if message is None or "content" not in message:
                raise ValueError(
                    f"No output text (reasoning model may have exhausted token budget). "
                    f"finish_reason={choice.get('finish_reason')}"
                )
            return message["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [retry] Attempt {attempt + 1} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("All retries exhausted")  # unreachable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Gemini 3.1 Pro generative probe for temporal order"
    )
    parser.add_argument(
        "--epic-dir",
        type=str,
        default="datasets/epic_kitchens",
        help="Path to EPIC-Kitchens data directory",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (with 'Bearer ' prefix). Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model identifier",
    )
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
        help="Canonical frame count",
    )
    parser.add_argument(
        "--frame-max-size",
        type=int,
        default=768,
        help="Max pixel dimension for frame encoding (reduces API payload)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (rate limiting)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    epic_dir = project_root / args.epic_dir

    # Resolve API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set GEMINI_API_KEY env var.")
        return
    if not api_key.startswith("Bearer "):
        api_key = f"Bearer {api_key}"

    print("=" * 70)
    print("GEMINI 3.1 PRO: TEMPORAL ORDER GENERATIVE PROBE")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  API base: {args.api_base}")
    print(f"  Frames per sequence: {args.n_frames}")
    print(f"  Frame max size: {args.frame_max_size}px")

    # ------------------------------------------------------------------
    # Step 1: Load sequences
    # ------------------------------------------------------------------
    print("\nStep 1: Loading sequences...")
    sequences = load_sequences(epic_dir, max_sequences=args.max_sequences)
    n_seq = len(sequences)
    print(f"  Using {n_seq} sequences")

    # ------------------------------------------------------------------
    # Step 2: Verify API connectivity
    # ------------------------------------------------------------------
    print("\nStep 2: Verifying API connectivity...")
    try:
        import requests

        test_payload = {
            "model": args.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": "Say OK."}],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key,
        }
        resp = requests.post(
            f"{args.api_base}/chat/completions",
            json=test_payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            test_response = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(f"  ERROR: Unexpected response structure: {json.dumps(data, indent=2)[:500]}")
            return
        print(f"  API OK. Test response: {test_response!r}")
    except Exception as e:
        print(f"  ERROR: API connectivity failed: {e}")
        return

    # ------------------------------------------------------------------
    # Step 3: Run generative probes
    # ------------------------------------------------------------------
    causality_prompts = DEFAULT_CAUSALITY_PROMPTS
    n_prompts = len(causality_prompts)

    print(f"\nStep 3: Running forward/reverse probes ({n_prompts} prompts × {n_seq} sequences)...")

    results_path = project_root / "datasets" / "epic_kitchens" / "gemini_probe_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = results_path.with_suffix(".partial.json")
    completed: dict[str, dict] = {}
    if partial_path.exists():
        with open(partial_path) as f:
            completed = json.load(f)
        print(f"  Resuming from {len(completed)} completed sequence-prompt pairs")

    per_prompt_results = {}
    all_balanced_accs = []
    total_api_calls = 0
    total_errors = 0

    for prompt_idx, prompt in enumerate(causality_prompts):
        print(f"\n  --- Prompt {prompt_idx} ---")
        print(f"  {prompt[:80]}...")

        fwd_correct = 0
        rev_correct = 0
        fwd_total = 0
        rev_total = 0

        for seq in tqdm(sequences, desc=f"Prompt {prompt_idx}"):
            seq_id = seq["sequence_id"]
            cache_key = f"{seq_id}_p{prompt_idx}"

            # Check if already completed
            if cache_key in completed:
                cached = completed[cache_key]
                pred_fwd = cached.get("pred_fwd")
                pred_rev = cached.get("pred_rev")
                if pred_fwd is not None:
                    fwd_total += 1
                    if pred_fwd == "FORWARD":
                        fwd_correct += 1
                if pred_rev is not None:
                    rev_total += 1
                    if pred_rev == "REVERSE":
                        rev_correct += 1
                continue

            try:
                pil_frames, timestamps = sample_canonical_frames(
                    seq["video_path"],
                    seq["start_sec"],
                    seq["stop_sec"],
                    n_frames=args.n_frames,
                    max_resolution=518,
                )
                if len(pil_frames) < 3:
                    continue

                frames_b64 = [
                    encode_frame_base64(f, max_size=args.frame_max_size)
                    for f in pil_frames
                ]

                # Forward probe
                resp_fwd = call_gemini_api(
                    frames_b64, prompt,
                    args.api_base, api_key, args.model,
                )
                pred_fwd = parse_forward_reverse(resp_fwd)
                total_api_calls += 1

                if args.rate_limit > 0:
                    time.sleep(args.rate_limit)

                if fwd_total + rev_total < 6 and prompt_idx == 0:
                    print(f"    [debug] fwd resp: {resp_fwd[:120]!r} -> {pred_fwd}")

                if pred_fwd is not None:
                    fwd_total += 1
                    if pred_fwd == "FORWARD":
                        fwd_correct += 1

                # Reverse probe
                rev_frames_b64 = list(reversed(frames_b64))
                resp_rev = call_gemini_api(
                    rev_frames_b64, prompt,
                    args.api_base, api_key, args.model,
                )
                pred_rev = parse_forward_reverse(resp_rev)
                total_api_calls += 1

                if args.rate_limit > 0:
                    time.sleep(args.rate_limit)

                if fwd_total + rev_total < 6 and prompt_idx == 0:
                    print(f"    [debug] rev resp: {resp_rev[:120]!r} -> {pred_rev}")

                if pred_rev is not None:
                    rev_total += 1
                    if pred_rev == "REVERSE":
                        rev_correct += 1

                completed[cache_key] = {
                    "resp_fwd": resp_fwd,
                    "resp_rev": resp_rev,
                    "pred_fwd": pred_fwd,
                    "pred_rev": pred_rev,
                }

                if total_api_calls % 20 == 0:
                    with open(partial_path, "w") as f:
                        json.dump(completed, f)

            except Exception as e:
                total_errors += 1
                if total_errors <= 5:
                    print(f"    [error] {seq_id}: {e}")
                continue

        fwd_acc = fwd_correct / max(fwd_total, 1)
        rev_acc = rev_correct / max(rev_total, 1)
        balanced_acc = (fwd_acc + rev_acc) / 2

        per_prompt_results[f"prompt_{prompt_idx}"] = {
            "balanced_acc": round(balanced_acc, 4),
            "fwd_acc": round(fwd_acc, 4),
            "rev_acc": round(rev_acc, 4),
            "fwd_total": fwd_total,
            "rev_total": rev_total,
            "prompt_text": prompt,
        }
        all_balanced_accs.append(balanced_acc)
        print(
            f"  Prompt {prompt_idx}: fwd={fwd_acc:.3f} ({fwd_correct}/{fwd_total}), "
            f"rev={rev_acc:.3f} ({rev_correct}/{rev_total}), "
            f"balanced={balanced_acc:.3f}"
        )

    with open(partial_path, "w") as f:
        json.dump(completed, f)

    # ------------------------------------------------------------------
    # Step 4: Results
    # ------------------------------------------------------------------
    mean_balanced_acc = float(np.mean(all_balanced_accs)) if all_balanced_accs else 0.0
    std_balanced_acc = float(np.std(all_balanced_accs)) if all_balanced_accs else 0.0

    print(f"\n{'=' * 70}")
    print("RESULTS: GEMINI 3.1 PRO TEMPORAL ORDER PROBE")
    print("=" * 70)
    print(f"  Mean balanced acc: {mean_balanced_acc:.3f} ± {std_balanced_acc:.3f}")
    print(f"  API calls: {total_api_calls}")
    print(f"  Errors: {total_errors}")

    for prompt_key, r in per_prompt_results.items():
        print(
            f"  {prompt_key}: balanced={r['balanced_acc']:.3f}  "
            f"fwd={r['fwd_acc']:.3f}  rev={r['rev_acc']:.3f}  "
            f"(n_fwd={r['fwd_total']}, n_rev={r['rev_total']})"
        )

    print(f"\n  Context (other VLMs from paper):")
    print(f"    Qwen3-VL-8B:      ~0.50 balanced acc")
    print(f"    Gemma-4-31B:      ~0.50 balanced acc")
    print(f"    LLaVA-Video-7B:   ~0.54 balanced acc")
    print(f"    Claude 4.6 Opus:  ~0.545 balanced acc")
    print(f"    Gemini 3.1 Pro:    {mean_balanced_acc:.2f} balanced acc  {'<- at chance' if mean_balanced_acc < 0.56 else '<- ABOVE CHANCE'}")

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    output = {
        "model": args.model,
        "api_base": args.api_base,
        "n_sequences": n_seq,
        "n_frames": args.n_frames,
        "mean_balanced_acc": round(mean_balanced_acc, 4),
        "std_balanced_acc": round(std_balanced_acc, 4),
        "per_prompt": per_prompt_results,
        "total_api_calls": total_api_calls,
        "total_errors": total_errors,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if partial_path.exists():
        partial_path.unlink()
        print(f"  Cleaned up {partial_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

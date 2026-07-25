#!/usr/bin/env python3
"""V-JEPA 2 feature extraction for the SoccerNet replay-grounding event-retrieval
protocol, driven by an immutable extraction plan.

For each clip in the plan (replay-query spans + live-event positive windows) this
produces the same three V-JEPA 2 features the driving evaluators use — mean_emb
(BoT), encoder_seq [32,1024], temporal_residual [16,1024] — and writes a cache
keyed by ``clip_id`` and bound to the plan's ``plan_sha256`` for provenance.

Gated: :func:`build_extraction_plan` calls ``require_locked_window_policy``, so
this refuses to run unless the manifest's window policy is locked/approved. This
is the **frozen-feature** arm only; the SONAR2-PE / learned-SDM arms live on the
parallel temporal-SDM line, not here. ``--limit`` caps clips for a GPU canary.

Usage:
    python experiments/extract_soccernet_features.py \
        --soccernet-dir <root> --split test --out-cache <path.pt> [--limit N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from common import (
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
    build_temporal_masks,
    load_clip_vjepa2,
)
from soccernet_plan import build_extraction_plan, write_feature_cache, write_plan
from tqdm import tqdm


def load_vjepa2(device: torch.device):
    """Load V-JEPA 2, honoring VTD_MODEL_DIR for the offline compute-node cache."""
    from transformers import AutoModel, AutoVideoProcessor

    path = VJEPA2_MODEL_NAME
    model_dir = os.environ.get("VTD_MODEL_DIR")
    if model_dir:
        local = Path(model_dir) / VJEPA2_MODEL_NAME.split("/")[-1]
        if local.exists():
            path = str(local)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(path, trust_remote_code=True)
    return model, processor


@torch.no_grad()
def extract_clip(model, processor, frames, device, context_mask, target_mask, n_target) -> dict:
    """One clip's (mean_emb, encoder_seq, temporal_residual) — mirrors
    eval_hdd_encoder_seq.extract_vjepa2_all_features so the features are identical
    to the driving datasets'."""
    inputs = processor(videos=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    enc = model(**inputs, skip_predictor=True)
    tokens = enc.last_hidden_state[0]  # (T*S, D)
    mean_emb = F.normalize(tokens.mean(dim=0), dim=0)
    encoder_seq = tokens.reshape(VJEPA2_T_PATCHES, VJEPA2_SPATIAL, -1).mean(dim=1)  # (32, D)
    pred = model(**inputs, context_mask=[context_mask], target_mask=[target_mask])
    p = pred.predictor_output.last_hidden_state[0].reshape(n_target, VJEPA2_SPATIAL, -1)
    g = pred.predictor_output.target_hidden_state[0].reshape(n_target, VJEPA2_SPATIAL, -1)
    residual = (p - g).mean(dim=1)  # (n_target, D)
    # encoder_seq/residual are TEMPORAL-major (time outer, spatial inner): V-JEPA 2's token
    # order is [T, S], the same layout common.build_temporal_masks relies on. Assert the
    # temporal length so a wrong reshape (e.g. space-major) is caught, not silently averaged.
    assert encoder_seq.shape[0] == VJEPA2_T_PATCHES and residual.shape[0] == n_target, (
        f"unexpected temporal length: encoder_seq={tuple(encoder_seq.shape)} "
        f"residual={tuple(residual.shape)}")
    return {"mean_emb": mean_emb.cpu(), "encoder_seq": encoder_seq.cpu(),
            "temporal_residual": residual.cpu()}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soccernet-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--arm", default="vjepa2_encoder_seq", choices=["vjepa2_encoder_seq"])
    p.add_argument("--split", default="test")
    p.add_argument("--out-cache", type=Path, required=True)
    p.add_argument("--plan-out", type=Path, default=None, help="also write the immutable plan JSON")
    p.add_argument("--limit", type=int, default=0, help="cap #clips for a GPU canary (0 = all)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    manifest_path = args.manifest or (
        args.soccernet_dir / "replay_event_manifest_v1_seed42.json")
    # Gate: raises SystemExit unless the window policy is locked/approved.
    plan = build_extraction_plan(manifest_path, args.arm, args.split)
    if args.plan_out:
        print(f"wrote plan {write_plan(plan, args.plan_out)}")
    clips = plan["clips"][: args.limit] if args.limit else plan["clips"]
    print(f"plan {plan['plan_sha256'][:12]} | arm={args.arm} split={args.split} | "
          f"{len(clips)} clips (window {plan['window_policy']['pre_s']},"
          f"{plan['window_policy']['post_s']}s)")

    device = torch.device(args.device)
    model, processor = load_vjepa2(device)
    n_ctx = VJEPA2_T_PATCHES // 2
    n_tgt = VJEPA2_T_PATCHES - n_ctx
    context_mask, target_mask = build_temporal_masks(n_ctx, device)

    features: dict[str, dict] = {}
    failed = 0
    for c in tqdm(clips, desc="SoccerNet V-JEPA2"):
        video = args.soccernet_dir / c["video"]
        t0, t1 = c["span_ms"][0] / 1000.0, c["span_ms"][1] / 1000.0
        try:
            frames, _ = load_clip_vjepa2(str(video), t0, t1)
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue
            feat = extract_clip(model, processor, frames, device, context_mask, target_mask, n_tgt)
            if not all(bool(torch.isfinite(v).all()) for v in feat.values()):
                failed += 1  # non-finite features would invert into a best rank; drop the clip
                continue
            features[c["clip_id"]] = feat
        except Exception:
            failed += 1
            continue
    print(f"extracted {len(features)}/{len(clips)} ({failed} failed/short)")

    blob = write_feature_cache(args.out_cache, plan, features,
                               canary=bool(args.limit), limit=args.limit)
    print(f"wrote {args.out_cache} (complete={blob['complete']} canary={blob['canary']} "
          f"{blob['n_features']}/{blob['n_expected']} rows)")


if __name__ == "__main__":
    main()

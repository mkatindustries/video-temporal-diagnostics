#!/usr/bin/env python3
"""Something-Something V2 motion-direction evaluation.

Tests whether the feature-vs-comparator decomposition that lands on Honda
HDD (driving) generalizes to a different domain (tabletop manipulation,
crowd-sourced video). For each chiral SSv2 template pair (e.g. "Pushing X
left to right" / "Pushing X right to left"), we form pairs:

    positive : two videos from the SAME template (same direction)
    negative : two videos from the CHIRAL TWIN template (opposite direction)

If BoF / Chamfer collapse to ~chance AP, the motion-direction collapse
generalizes beyond dashcams. If V-JEPA 2 + DTW recovers AP, the
comparator-vs-encoder finding generalizes.

This script supports a "spike" mode: process only N videos, time the
extraction, and project the full ETA. Use it for the 4-hour feasibility
gate before committing to a full run.

Usage:
    # Spike (4-hour budget, ~200 videos):
    python experiments/eval_ssv2_motion_direction.py \\
        --manifest datasets/ssv2/validation_manifest.json \\
        --spike-budget-hours 4 \\
        --max-videos 200

    # Full run (after spike confirms feasibility):
    python experiments/eval_ssv2_motion_direction.py \\
        --manifest datasets/ssv2/validation_manifest.json \\
        --device cuda

Reuses extract_clip_features / extract_vjepa2_features from
eval_hdd_intersections by adapting them to operate on whole videos rather
than sub-clips of a longer recording.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# Reuse existing infrastructure
from eval_hdd_intersections import (
    DINOV3_MODEL_NAME,
    VJEPA2_MODEL_NAME,
    VJEPA2_NUM_FRAMES,
    VJEPA2_SPATIAL,
    VJEPA2_T_PATCHES,
    bootstrap_ap,
    build_temporal_masks,
    load_clip,
    load_clip_vjepa2,
)
from video_retrieval.fingerprints import (
    TemporalDerivativeFingerprint,
    TrajectoryFingerprint,
)
from video_retrieval.fingerprints.dtw import dtw_distance_batch
from video_retrieval.models import DINOv3Encoder


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


@dataclass
class SSv2Clip:
    video_id: str
    template_id: int
    chiral_pair_id: int
    chiral_direction: str
    video_path: str


def load_manifest(path: Path) -> tuple[list[SSv2Clip], list[dict]]:
    """Load a manifest produced by scripts/setup_ssv2.py."""
    with open(path) as f:
        m = json.load(f)
    clips = [
        SSv2Clip(
            video_id=e["video_id"],
            template_id=int(e["template_id"]),
            chiral_pair_id=int(e["chiral_pair_id"]),
            chiral_direction=e["chiral_direction"],
            video_path=e["video_path"],
        )
        for e in m["entries"]
    ]
    return clips, m["chiral_pairs"]


def get_video_duration(video_path: str) -> float:
    """Return duration in seconds. SSv2 webms are usually 2-5s."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    container.close()
    return duration


# ---------------------------------------------------------------------------
# Feature extraction (full-clip variants of the HDD functions)
# ---------------------------------------------------------------------------


def extract_dinov3_full_clip(
    encoder: DINOv3Encoder,
    clips: list[SSv2Clip],
    target_fps: float = 3.0,
    max_resolution: int = 518,
) -> dict[int, dict]:
    """Extract DINOv3 features over the full duration of each SSv2 clip."""
    features: dict[int, dict] = {}
    failed = 0

    for i, clip in enumerate(tqdm(clips, desc="DINOv3")):
        try:
            duration = get_video_duration(clip.video_path)
            if duration < 0.5:
                failed += 1
                continue
            frames = load_clip(
                clip.video_path,
                start_sec=0.0,
                end_sec=duration,
                target_fps=target_fps,
                max_resolution=max_resolution,
            )
            if len(frames) < 3:
                failed += 1
                continue

            emb = encoder.encode_frames(frames)
            centroids = encoder.get_attention_centroids(frames)
            mean_emb = F.normalize(emb.mean(dim=0), dim=0)

            features[i] = {
                "embeddings": emb,
                "centroids": centroids,
                "mean_emb": mean_emb,
            }
        except Exception:
            failed += 1
            continue

    print(f"  DINOv3 extracted: {len(features)}/{len(clips)} ({failed} failed)")
    return features


def extract_vjepa2_full_clip(
    model: torch.nn.Module,
    processor: object,
    clips: list[SSv2Clip],
    device: torch.device,
) -> dict[int, dict]:
    """Extract V-JEPA 2 BoT + temporal residuals over the full clip duration."""
    n_context_steps = VJEPA2_T_PATCHES // 2
    n_target_steps = VJEPA2_T_PATCHES - n_context_steps
    context_mask, target_mask = build_temporal_masks(n_context_steps, device)

    features: dict[int, dict] = {}
    failed = 0

    for i, clip in enumerate(tqdm(clips, desc="V-JEPA 2")):
        try:
            duration = get_video_duration(clip.video_path)
            if duration < 0.5:
                failed += 1
                continue
            frames, _ = load_clip_vjepa2(
                clip.video_path, start_sec=0.0, end_sec=duration
            )
            if len(frames) < VJEPA2_NUM_FRAMES:
                failed += 1
                continue

            # pyrefly: ignore [not-callable]
            inputs = processor(videos=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                enc_out = model(**inputs, skip_predictor=True)
                encoder_tokens = enc_out.last_hidden_state[0]
                mean_emb = F.normalize(encoder_tokens.mean(dim=0), dim=0)

                pred_out = model(
                    **inputs,
                    context_mask=[context_mask],
                    target_mask=[target_mask],
                )
                predicted = pred_out.predictor_output.last_hidden_state[0]
                ground_truth = pred_out.predictor_output.target_hidden_state[0]
                predicted = predicted.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                ground_truth = ground_truth.reshape(n_target_steps, VJEPA2_SPATIAL, -1)
                residual = (predicted - ground_truth).mean(dim=1)

            features[i] = {
                "mean_emb": mean_emb.cpu(),
                "temporal_residual": residual.cpu(),
            }
        except Exception:
            failed += 1
            continue

    print(f"  V-JEPA 2 extracted: {len(features)}/{len(clips)} ({failed} failed)")
    return features


# ---------------------------------------------------------------------------
# Pairwise similarities
# ---------------------------------------------------------------------------


def enumerate_chiral_pairs(
    clips: list[SSv2Clip],
    feature_indices: set[int],
) -> tuple[list[int], list[int], list[int]]:
    """Enumerate (a, b, gt) for all within-chiral-pair video pairs.

    Within one chiral pair (e.g. left_to_right + right_to_left):
        positive: same direction (same template)
        negative: opposite direction (chiral twin template)

    We restrict to within-pair comparisons so visual content is matched
    (both clips show "pushing X" — only direction differs).
    """
    by_pair: dict[int, list[int]] = defaultdict(list)
    for idx, clip in enumerate(clips):
        if idx in feature_indices:
            by_pair[clip.chiral_pair_id].append(idx)

    a_idx, b_idx, gts = [], [], []
    for pair_id, indices in by_pair.items():
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                ia, ib = indices[ai], indices[bi]
                a_idx.append(ia)
                b_idx.append(ib)
                gts.append(int(clips[ia].template_id == clips[ib].template_id))
    return a_idx, b_idx, gts


def compute_similarities(
    clips: list[SSv2Clip],
    dinov3: dict[int, dict],
    vjepa2: dict[int, dict] | None,
    device: torch.device,
) -> dict[str, tuple[list[float], list[int]]]:
    common = set(dinov3.keys())
    if vjepa2 is not None:
        common &= set(vjepa2.keys())
    a_idx, b_idx, gts = enumerate_chiral_pairs(clips, common)

    n_pos = sum(gts)
    n_neg = len(gts) - n_pos
    print(f"  pairs: total={len(gts)} pos={n_pos} neg={n_neg}")
    if not gts:
        return {}

    deriv_fp = TemporalDerivativeFingerprint()
    traj_fp = TrajectoryFingerprint()
    deriv_fps = {i: deriv_fp.compute_fingerprint(dinov3[i]["embeddings"]) for i in dinov3}
    traj_fps = {i: traj_fp.compute_fingerprint(dinov3[i]["centroids"]) for i in dinov3}

    print("  bag-of-frames ...")
    ma = torch.stack([dinov3[i]["mean_emb"] for i in a_idx]).to(device)
    mb = torch.stack([dinov3[i]["mean_emb"] for i in b_idx]).to(device)
    bof = (ma * mb).sum(dim=1).cpu().tolist()

    print("  Chamfer ...")
    chamfer = []
    for ia, ib in zip(a_idx, b_idx):
        ea = dinov3[ia]["embeddings"].to(device)
        eb = dinov3[ib]["embeddings"].to(device)
        sm = torch.mm(ea, eb.t())
        chamfer.append(((sm.max(dim=1).values.mean() + sm.max(dim=0).values.mean()) / 2).item())

    print("  temporal-derivative DTW ...")
    da = [deriv_fps[i].to(device) for i in a_idx]
    db = [deriv_fps[i].to(device) for i in b_idx]
    deriv_sims = torch.exp(-dtw_distance_batch(da, db, normalize=False)).cpu().tolist()

    print("  attention-trajectory DTW ...")
    ta = [traj_fps[i].to(device) for i in a_idx]
    tb = [traj_fps[i].to(device) for i in b_idx]
    traj_sims = torch.exp(-5.0 * dtw_distance_batch(ta, tb, normalize=True)).cpu().tolist()

    out: dict[str, tuple[list[float], list[int]]] = {
        "bag_of_frames": (bof, gts),
        "chamfer": (chamfer, gts),
        "temporal_derivative": (deriv_sims, gts),
        "attention_trajectory": (traj_sims, gts),
    }

    if vjepa2 is not None:
        print("  V-JEPA 2 BoT ...")
        va = torch.stack([vjepa2[i]["mean_emb"] for i in a_idx]).to(device)
        vb = torch.stack([vjepa2[i]["mean_emb"] for i in b_idx]).to(device)
        bot = (va * vb).sum(dim=1).cpu().tolist()

        print("  V-JEPA 2 temporal residual DTW ...")
        ra = [vjepa2[i]["temporal_residual"].to(device) for i in a_idx]
        rb = [vjepa2[i]["temporal_residual"].to(device) for i in b_idx]
        res_sims = torch.exp(-dtw_distance_batch(ra, rb, normalize=True)).cpu().tolist()

        out["vjepa2_bag_of_tokens"] = (bot, gts)
        out["vjepa2_temporal_residual"] = (res_sims, gts)

    return out


# ---------------------------------------------------------------------------
# Spike-mode timing
# ---------------------------------------------------------------------------


def time_extraction_spike(
    clips: list[SSv2Clip],
    n_probe: int,
    encoder: DINOv3Encoder,
    vjepa2_model: torch.nn.Module | None,
    vjepa2_processor: object,
    device: torch.device,
) -> dict:
    """Time the first n_probe extractions and project total ETA.

    Returns timings + projection. Caller decides whether to continue.
    """
    probe = clips[:n_probe]
    t0 = time.time()
    dinov3 = extract_dinov3_full_clip(encoder, probe)
    t_dinov3 = time.time() - t0

    t_vjepa2 = None
    if vjepa2_model is not None:
        t1 = time.time()
        _ = extract_vjepa2_full_clip(vjepa2_model, vjepa2_processor, probe, device)
        t_vjepa2 = time.time() - t1

    n_done = max(1, len(dinov3))
    per_clip_dinov3 = t_dinov3 / n_done
    per_clip_vjepa2 = (t_vjepa2 / n_done) if t_vjepa2 is not None else None

    n_total = len(clips)
    eta_dinov3 = per_clip_dinov3 * n_total
    eta_vjepa2 = (per_clip_vjepa2 * n_total) if per_clip_vjepa2 is not None else None
    eta_total = eta_dinov3 + (eta_vjepa2 or 0.0)

    return {
        "n_probed": n_probe,
        "n_dinov3_ok": n_done,
        "t_dinov3_sec": t_dinov3,
        "t_vjepa2_sec": t_vjepa2,
        "per_clip_dinov3_sec": per_clip_dinov3,
        "per_clip_vjepa2_sec": per_clip_vjepa2,
        "n_total_clips": n_total,
        "eta_dinov3_hours": eta_dinov3 / 3600.0,
        "eta_vjepa2_hours": (eta_vjepa2 / 3600.0) if eta_vjepa2 is not None else None,
        "eta_total_hours": eta_total / 3600.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_model_path(model_name: str) -> str:
    import os
    from pathlib import Path as _P
    md = os.environ.get("VTD_MODEL_DIR")
    if md:
        local = _P(md) / model_name.split("/")[-1]
        if local.exists():
            return str(local)
    return model_name


def main():
    parser = argparse.ArgumentParser(description="SSv2 motion-direction eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap the number of videos processed (useful for the spike).",
    )
    parser.add_argument(
        "--spike-budget-hours",
        type=float,
        default=None,
        help="If set, only run the timing probe (no full extraction). "
             "Reports projected ETA and aborts if it exceeds the budget.",
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=20,
        help="Number of clips to time when projecting ETA (default: 20).",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-vjepa2", action="store_true",
                        help="DINOv3 only (faster spike).")
    args = parser.parse_args()

    print(f"Loading manifest from {args.manifest} ...")
    clips, chiral_pairs = load_manifest(args.manifest)
    if args.max_videos:
        clips = clips[: args.max_videos]
    print(f"  {len(clips)} clips across {len(chiral_pairs)} chiral pairs")

    device = torch.device(args.device)
    print(f"Loading DINOv3 ({DINOV3_MODEL_NAME}) ...")
    encoder = DINOv3Encoder(device=args.device, model_name=DINOV3_MODEL_NAME)

    vjepa2_model: torch.nn.Module | None = None
    vjepa2_processor = None
    if not args.skip_vjepa2:
        print(f"Loading V-JEPA 2 ({VJEPA2_MODEL_NAME}) ...")
        from transformers import AutoModel, AutoVideoProcessor
        vjepa2_path = _resolve_model_path(VJEPA2_MODEL_NAME)
        vjepa2_model = AutoModel.from_pretrained(
            vjepa2_path, trust_remote_code=True
        ).to(device).eval()
        vjepa2_processor = AutoVideoProcessor.from_pretrained(
            vjepa2_path, trust_remote_code=True
        )

    # ---- Spike mode: time the probe and stop ----
    if args.spike_budget_hours is not None:
        print(f"\n=== SPIKE MODE: probing {args.probe_size} clips, "
              f"budget {args.spike_budget_hours}h ===")
        timings = time_extraction_spike(
            clips, args.probe_size, encoder, vjepa2_model, vjepa2_processor, device
        )
        print(json.dumps(timings, indent=2))
        if timings["eta_total_hours"] > args.spike_budget_hours:
            print(
                f"\nABORT: projected ETA {timings['eta_total_hours']:.2f}h "
                f"exceeds budget {args.spike_budget_hours}h."
            )
            print("Options: reduce --videos-per-template in setup_ssv2.py, "
                  "or run --skip-vjepa2 first.")
            return 1
        print(
            f"\nPROCEED: projected ETA "
            f"{timings['eta_total_hours']:.2f}h < {args.spike_budget_hours}h."
        )
        return 0

    # ---- Full run: extract, compare, evaluate ----
    cache_dir = args.cache_dir or args.manifest.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dinov3_cache = cache_dir / "dinov3.pt"
    vjepa2_cache = cache_dir / "vjepa2.pt"

    if dinov3_cache.exists():
        print(f"Loading cached DINOv3 features from {dinov3_cache}")
        dinov3 = torch.load(dinov3_cache, weights_only=False)
    else:
        dinov3 = extract_dinov3_full_clip(encoder, clips)
        torch.save(
            {k: {fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv
                 for fk, fv in v.items()} for k, v in dinov3.items()},
            dinov3_cache,
        )
    del encoder
    torch.cuda.empty_cache()

    vjepa2 = None
    if vjepa2_model is not None:
        if vjepa2_cache.exists():
            print(f"Loading cached V-JEPA 2 features from {vjepa2_cache}")
            vjepa2 = torch.load(vjepa2_cache, weights_only=False)
        else:
            vjepa2 = extract_vjepa2_full_clip(
                vjepa2_model, vjepa2_processor, clips, device
            )
            torch.save(
                {k: {fk: fv.cpu() if isinstance(fv, torch.Tensor) else fv
                     for fk, fv in v.items()} for k, v in vjepa2.items()},
                vjepa2_cache,
            )
        del vjepa2_model
        torch.cuda.empty_cache()

    print("\n--- pairwise similarities ---")
    scores = compute_similarities(clips, dinov3, vjepa2, device)

    print("\n=== RESULTS: SSv2 motion-direction (within-chiral-pair) ===")
    results: dict[str, dict] = {}
    for method, (sc, gt) in scores.items():
        s = np.array(sc)
        y = np.array(gt)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ap, lo, hi = bootstrap_ap(s, y)
        auc = roc_auc_score(y, s)
        results[method] = {
            "ap": float(ap), "ap_ci_low": float(lo), "ap_ci_high": float(hi),
            "auc": float(auc),
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        }
        print(f"  {method:<28}  AP={ap:.4f} [{lo:.4f}, {hi:.4f}]  AUC={auc:.4f}")

    out_path = args.out or args.manifest.parent / "ssv2_motion_direction_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_clips": len(clips),
                "n_chiral_pairs": len(chiral_pairs),
                "methods": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

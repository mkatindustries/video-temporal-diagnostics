#!/usr/bin/env bash
#SBATCH --job-name=soccernet-replay
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# SoccerNet replay-grounding frozen-feature event retrieval.
#   1. Extract V-JEPA 2 features per the immutable plan (GATED: refuses unless the
#      manifest's window policy is locked/approved AND its canary evidence re-grounds).
#   2. Evaluate BoT / encoder-seq DTW / temporal-residual DTW / shuffled-DTW (ordering
#      control) / assignment over the within-match gallery (fail-closed on canary/incomplete).
#
# Submit (account/qos at submit time, per slurm_jobs/README.md):
#   SOCCERNET_DIR=<root> sbatch slurm_jobs/rerun_soccernet.sbatch
# GPU canary before the full run (extraction only; cache is canary-tagged, unpublishable):
#   SOCCERNET_DIR=<root> LIMIT=24 sbatch slurm_jobs/rerun_soccernet.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${SOCCERNET_DIR:?Set SOCCERNET_DIR to the SoccerNet replay-grounding dataset root}"
SPLIT="${SPLIT:-test}"
ARM="${ARM:-vjepa2_encoder_seq}"
LIMIT="${LIMIT:-0}"   # >0 = GPU canary: extraction capped, cache tagged canary-only (not publishable)
RESULTS_OUTPUT="${RESULTS_OUTPUT:-$REPO_ROOT/results/soccernet/replay_results.json}"

FEATURE_CACHE="${FEATURE_CACHE:-$REPO_ROOT/datasets/soccernet/${ARM}_${SPLIT}.pt}"
if [[ "$LIMIT" -gt 0 ]]; then
    FEATURE_CACHE="${FEATURE_CACHE%.pt}.canary.pt"  # canary cache can never shadow the full one
fi

# This frozen-feature evaluation uses a dedicated manifest so another experiment cannot
# overwrite its locked window policy or invalidate the audit artifact.
MANIFEST="${MANIFEST:-$SOCCERNET_DIR/replay_event_manifest_v1_seed42_frozenfeat.json}"

# 1. Extraction — gated on the locked+re-grounded window policy via build_extraction_plan.
"$PYTHON" experiments/extract_soccernet_features.py \
    --soccernet-dir "$SOCCERNET_DIR" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" --arm "$ARM" \
    --out-cache "$FEATURE_CACHE" \
    --plan-out "$REPO_ROOT/results/soccernet/plans" \
    --limit "$LIMIT" \
    --device cuda

# 2. Eval — skipped for a canary (its cache is fail-closed against publication).
if [[ "$LIMIT" -gt 0 ]]; then
    echo "canary run (LIMIT=$LIMIT): extraction validated; eval skipped (cache is canary-tagged)."
else
    "$PYTHON" experiments/eval_soccernet_replay.py \
        --soccernet-dir "$SOCCERNET_DIR" \
        --manifest "$MANIFEST" \
        --split "$SPLIT" --arm "$ARM" \
        --feature-cache "$FEATURE_CACHE" \
        --output "$RESULTS_OUTPUT"
fi

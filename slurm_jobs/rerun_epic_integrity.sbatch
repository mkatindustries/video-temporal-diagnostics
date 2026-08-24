#!/usr/bin/env bash
#SBATCH --job-name=epic-integrity
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#
# EPIC-Kitchens INTACT/TAMPERED integrity probe -- re-run for the WITHDRAWN claim.
#
# Why this exists: the transcribed integrity aggregate in
# results/epic/vlm_prompt_results.json applied `fraction_tampered = 1 - acc` to all four
# conditions, but eval_epic_temporal_order.py records `forward_acc` as the fraction judged
# INTACT while `reverse_acc` / `scramble_kN_acc` are already the fraction judged TAMPERED.
# The reversed and scramble columns are therefore believed inverted, the per-clip responses
# were not retained, and the claim was withdrawn from both manuscripts. This job regenerates
# the probe from the same manifest so the numbers can be recomputed rather than guessed.
# Read the recorded `*_acc` values from the output JSON directly; do NOT re-apply a blanket
# `1 - acc` when transcribing them.
#
# Submit (account/partition/qos at submit time, per slurm_jobs/README.md):
#   EPIC_DIR=<root> sbatch slurm_jobs/rerun_epic_integrity.sbatch
# One family only:
#   EPIC_DIR=<root> sbatch --array=0 slurm_jobs/rerun_epic_integrity.sbatch
#
# Note: Gemma-4-31B needs an 80GB-class GPU; request one at submit time if the default
# --gres=gpu:1 does not resolve to one on this cluster.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${EPIC_DIR:?Set EPIC_DIR to the EPIC-Kitchens data root}"

# --vlm-family choices come from VLM_ADAPTERS in experiments/eval_epic_temporal_order.py.
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) VLM_FAMILY="qwen3" ;;
    1) VLM_FAMILY="gemma4" ;;
    2) VLM_FAMILY="llava-video" ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

# The eval script hardcodes <EPIC_DIR>/annotations/temporal_order_sequences.json, which
# download_epic_kitchens.py --setup writes as a symlink to the versioned build. Pin the
# original run's manifest by checking the symlink target instead of passing a flag.
EXPECTED_MANIFEST="${EXPECTED_MANIFEST:-temporal_order_sequences_v1_len6-15_narr2-3_seed42.json}"
MANIFEST_LINK="$EPIC_DIR/annotations/temporal_order_sequences.json"
if [[ ! -e "$MANIFEST_LINK" ]]; then
    echo "Manifest not found: $MANIFEST_LINK" >&2
    echo "Run: python scripts/download_epic_kitchens.py --epic-dir $EPIC_DIR --setup" >&2
    exit 3
fi
RESOLVED_MANIFEST="$(basename "$(readlink -f "$MANIFEST_LINK")")"
if [[ "$RESOLVED_MANIFEST" != "$EXPECTED_MANIFEST" ]]; then
    echo "Manifest mismatch: $MANIFEST_LINK -> $RESOLVED_MANIFEST" >&2
    echo "Expected $EXPECTED_MANIFEST (the manifest of the withdrawn run)." >&2
    echo "Override with EXPECTED_MANIFEST=... only if the change is intended." >&2
    exit 3
fi

# eval_epic_temporal_order.py overwrites several git-tracked figures unconditionally, including
# Figure 4 of the paper (epic_temporal_order_sensitivity.png) and the per-family integrity plots
# that are the surviving evidence the published table was inverted. Archive every one of them
# before the script runs; cp -n so a re-submission never overwrites the archived original.
ARCHIVE_DIR="$REPO_ROOT/results/epic/prewithdrawal_figures"
mkdir -p "$ARCHIVE_DIR"
for FIG in \
    "epic_${VLM_FAMILY}_temporal_integrity.png" \
    "epic_${VLM_FAMILY}_prompt_variance.png" \
    "epic_temporal_order_sensitivity.png" \
    "epic_temporal_order_retrieval.png" \
    "epic_temporal_order_s_rev_distributions.png"; do
    [[ -f "$REPO_ROOT/figures/$FIG" ]] && cp -n "$REPO_ROOT/figures/$FIG" "$ARCHIVE_DIR/$FIG"
done
echo "Archived pre-rerun figures to $ARCHIVE_DIR"
echo "NOTE: this job rewrites tracked files under figures/. Review 'git diff --stat figures/'"
echo "      afterwards and keep only the plots you intend to update."

# --vlm-integrity-probe only takes effect inside the generative step, so --vlm-generative is
# required as well; without it the probe silently no-ops. --skip-vjepa2 drops the V-JEPA 2
# passes, which this probe does not use (DINOv3 extraction has no skip flag and still runs).
"$PYTHON" experiments/eval_epic_temporal_order.py \
    --epic-dir "$EPIC_DIR" \
    --device cuda \
    --skip-vjepa2 \
    --max-sequences "${MAX_SEQUENCES:-500}" \
    --vlm-family "$VLM_FAMILY" \
    --vlm-generative \
    --vlm-integrity-probe \
    --vlm-scramble-levels ${VLM_SCRAMBLE_LEVELS:-4 16} \
    --vlm-num-frames "${VLM_NUM_FRAMES:-16}" \
    --vlm-text-only-baseline

# eval_epic_temporal_order.py has no output-path flag: it always writes
# <EPIC_DIR>/temporal_order_results_<family>.json, which lives in the gitignored dataset
# tree. Copy the compact result into the tracked results/ path under a distinct name so it
# cannot shadow results/epic/temporal_order_results.json.
SRC_RESULTS="$EPIC_DIR/temporal_order_results_${VLM_FAMILY}.json"
DST_RESULTS="${RESULTS_OUTPUT:-$REPO_ROOT/results/epic/integrity_rerun_${VLM_FAMILY}.json}"
if [[ -f "$SRC_RESULTS" ]]; then
    mkdir -p "$(dirname "$DST_RESULTS")"
    cp "$SRC_RESULTS" "$DST_RESULTS"
    echo "Copied $SRC_RESULTS -> $DST_RESULTS"
else
    echo "Expected results file not written: $SRC_RESULTS" >&2
    exit 4
fi

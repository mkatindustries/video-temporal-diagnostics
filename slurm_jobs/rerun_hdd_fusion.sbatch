#!/usr/bin/env bash
#SBATCH --job-name=hdd-fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${HDD_DIR:?Set HDD_DIR to the Honda HDD dataset root}"
FEATURE_CACHE="${FEATURE_CACHE:-$REPO_ROOT/datasets/hdd/vjepa2_encoder_features.pt}"
DIST_CACHE="${DIST_CACHE:-$REPO_ROOT/datasets/hdd/fusion_score_cache.pt}"
RESULTS_OUTPUT="${RESULTS_OUTPUT:-$REPO_ROOT/results/hdd/fusion_results.json}"

"$PYTHON" experiments/eval_hdd_fusion.py \
    --hdd-dir "$HDD_DIR" \
    --device cuda \
    --max-clusters "${MAX_CLUSTERS:-50}" \
    --feature-cache "$FEATURE_CACHE" \
    --dist-cache "$DIST_CACHE" \
    --output "$RESULTS_OUTPUT" \
    --composition-k 1 10 \
    --n-bootstrap "${N_BOOTSTRAP:-2000}" \
    --dtw-batch-size "${DTW_BATCH_SIZE:-256}"

#!/usr/bin/env bash
#SBATCH --job-name=epic-residual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${EPIC_DIR:?Set EPIC_DIR to the EPIC-Kitchens data root}"

"$PYTHON" experiments/eval_epic_temporal_order.py \
    --epic-dir "$EPIC_DIR" \
    --device cuda \
    --max-sequences "${MAX_SEQUENCES:-500}"

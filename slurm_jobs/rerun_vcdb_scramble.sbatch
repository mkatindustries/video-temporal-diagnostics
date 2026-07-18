#!/usr/bin/env bash
#SBATCH --job-name=vcdb-scramble
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${VCDB_DIR:?Set VCDB_DIR to the root containing annotation/ and core_dataset/}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        "$PYTHON" experiments/eval_vcdb_scramble_multiseed.py \
            --vcdb-dir "$VCDB_DIR" \
            --device cuda \
            --n-seeds "${N_SEEDS:-10}"
        ;;
    1)
        "$PYTHON" experiments/eval_vcdb_scramble_raw.py \
            --vcdb-dir "$VCDB_DIR" \
            --device cuda
        ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

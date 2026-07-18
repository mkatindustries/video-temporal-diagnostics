#!/usr/bin/env bash
#SBATCH --job-name=nuscenes-temporal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${NUSCENES_DIR:?Set NUSCENES_DIR to the nuScenes dataset root}"

"$PYTHON" experiments/eval_nuscenes_intersections.py \
    --nuscenes-dir "$NUSCENES_DIR" \
    --version "${NUSCENES_VERSION:-v1.0-trainval}" \
    --device cuda \
    --max-clusters "${MAX_CLUSTERS:-50}"

"$PYTHON" experiments/eval_grouped_pair_bootstrap.py \
    --pairs-json "$NUSCENES_DIR/pair_scores.json" \
    --output "$NUSCENES_DIR/cluster_bootstrap_results.json" \
    --n-resamples "${N_BOOTSTRAP:-2000}" \
    --paired-methods vjepa2_encoder_seq_dtw vjepa2_bag_of_tokens \
    --paired-methods vjepa2_encoder_seq_dtw vjepa2_temporal_residual \
    --paired-methods vjepa2_temporal_residual vjepa2_bag_of_tokens

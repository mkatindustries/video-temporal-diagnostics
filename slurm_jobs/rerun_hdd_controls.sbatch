#!/usr/bin/env bash
#SBATCH --job-name=hdd-controls
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

"$PYTHON" -m pytest tests/test_dtw.py -k 'Shuffled or Assignment' -v

"$PYTHON" experiments/eval_hdd_encoder_seq.py \
    --hdd-dir "$HDD_DIR" \
    --device cuda \
    --max-clusters "${MAX_CLUSTERS:-50}" \
    --feature-cache "$FEATURE_CACHE"

"$PYTHON" experiments/eval_cluster_bootstrap.py \
    --hdd-dir "$HDD_DIR" \
    --pairs-json "$HDD_DIR/encoder_seq_pair_scores.json" \
    --max-clusters "${MAX_CLUSTERS:-50}" \
    --n-resamples "${N_BOOTSTRAP:-2000}" \
    --paired-methods encoder_seq_dtw bot_cosine \
    --paired-methods encoder_seq_dtw encoder_seq_dtw_shuffled \
    --paired-methods encoder_seq_dtw encoder_seq_assignment \
    --paired-methods temporal_residual_dtw encoder_seq_dtw \
    --paired-methods temporal_residual_dtw bot_cosine \
    --paired-methods temporal_residual_dtw temporal_residual_dtw_shuffled \
    --paired-methods temporal_residual_dtw temporal_residual_assignment

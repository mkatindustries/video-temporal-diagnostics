#!/usr/bin/env bash
#SBATCH --job-name=nuscenes-fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm_jobs/common.sh"

: "${NUSCENES_DIR:?Set NUSCENES_DIR to the nuScenes dataset root}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"

# Reuse the V-JEPA 2 feature cache built by eval_nuscenes_intersections; only the
# full-gallery DTW matrix is (re)computed. DIST_CACHE lets smoke and full runs use
# separate score caches.
DEFAULT_DIST="$NUSCENES_DIR/feature_cache/nuscenes_fusion_score_cache_${NUSCENES_VERSION}.pt"

"$PYTHON" experiments/eval_nuscenes_fusion.py \
    --nuscenes-dir "$NUSCENES_DIR" \
    --version "$NUSCENES_VERSION" \
    --device cuda \
    --max-clusters "${MAX_CLUSTERS:-50}" \
    --dist-cache "${DIST_CACHE:-$DEFAULT_DIST}" \
    --n-bootstrap "${N_BOOTSTRAP:-2000}" \
    --dtw-batch-size "${DTW_BATCH_SIZE:-256}"

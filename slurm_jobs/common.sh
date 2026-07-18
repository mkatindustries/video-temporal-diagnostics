#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
else
    CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
    if [[ -f "$CONDA_SH" ]]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "${CONDA_ENV:-video_retrieval}"
    fi
    PYTHON="${PYTHON:-python}"
fi

export PYTHONPATH="$REPO_ROOT/experiments:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -c "import torch; print(f'python ready; torch={torch.__version__}; cuda={torch.cuda.is_available()}')"

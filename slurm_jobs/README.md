# Corrected Cluster Runs

Submit these jobs from the repository root after creating the log directory:

```bash
mkdir -p slurm_logs

VCDB_DIR=/datasets/VCDB sbatch slurm_jobs/rerun_vcdb_scramble.sbatch
HDD_DIR=/datasets/HDD sbatch slurm_jobs/rerun_hdd_retrieval.sbatch
NUSCENES_DIR=/datasets/nuScenes sbatch slurm_jobs/rerun_nuscenes.sbatch
EPIC_DIR=/datasets/EPIC_KITCHENS sbatch slurm_jobs/rerun_epic_residual.sbatch
```

The scripts intentionally omit site-specific account, partition, and QoS values. Add them
with `sbatch --account=... --partition=... --qos=...` as required. Set `PYTHON_BIN` to an
environment-specific Python executable, or set `CONDA_SH` and `CONDA_ENV` (default:
`video_retrieval`). Model downloads can be redirected with the experiment-specific
environment variables documented in `REPRODUCIBILITY.md`.

## Outputs To Integrate

1. VCDB extracted-embedding multi-seed gradient and raw-frame gradient: replace the pending
   scramble table, figure, and text. Confirm each partition has chunk sizes differing by at
   most one.
2. HDD directed retrieval: integrate macro AP@k, recall@k, MRR, and cluster intervals from
   `bof_dtw_directed_rerank_results.json`, including the full-gallery encoder-sequence-DTW
   baseline. Do not restore the withdrawn unordered-pair table.
3. HDD paired contrasts: integrate AP-difference intervals from
   `cluster_bootstrap_results.json`, especially encoder-sequence DTW minus BoT.
4. EPIC residual: replace the pending V-JEPA 2 residual `s_rev` value. Do not add a balanced
   accuracy column for embedding similarities.
5. nuScenes: integrate the paired cluster-bootstrap contrasts from
   `cluster_bootstrap_results.json`; do not infer a paired difference from marginal intervals.

Before updating the paper, archive the command, git commit, SLURM log, output JSON, model
checkpoint revision, and dataset version for every run.

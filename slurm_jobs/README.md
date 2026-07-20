# Corrected Cluster Runs

These are the exact jobs used for the validated 2026-07-18 rerun. Compact outputs and
run provenance are tracked under `results/`; dataset-local caches and per-pair files are
intentionally excluded.

Submit these jobs from the repository root after creating the log directory:

```bash
mkdir -p slurm_logs

VCDB_DIR=/datasets/VCDB sbatch slurm_jobs/rerun_vcdb_scramble.sbatch
HDD_DIR=/datasets/HDD sbatch slurm_jobs/rerun_hdd_retrieval.sbatch
HDD_DIR=/datasets/HDD sbatch slurm_jobs/rerun_hdd_fusion.sbatch
NUSCENES_DIR=/datasets/nuScenes sbatch slurm_jobs/rerun_nuscenes.sbatch
EPIC_DIR=/datasets/EPIC_KITCHENS sbatch slurm_jobs/rerun_epic_residual.sbatch
```

For the Video4Real error-composition result, rerun the two fusion jobs after the
feature and distance caches exist. They write compact summaries directly to the
tracked `results/` paths:

```bash
HDD_DIR=/datasets/HDD sbatch slurm_jobs/rerun_hdd_fusion.sbatch
NUSCENES_DIR=/datasets/nuScenes sbatch slurm_jobs/rerun_nuscenes_fusion.sbatch
```

The fusion jobs also evaluate full-gallery temporal-residual DTW. A validated legacy
BoT/encoder-DTW score cache is augmented in place, so only the missing residual matrix is
computed. Do not set `--rebuild-dist-cache` unless the existing encoder-DTW matrix is stale.

After both jobs finish, generate the figure on a CPU node and rebuild the paper:

```bash
python scripts/plot_video4real_figures.py error-composition
make video4real
```

The scripts intentionally omit site-specific account, partition, and QoS values. Add them
with `sbatch --account=... --partition=... --qos=...` as required. Set `PYTHON_BIN` to an
environment-specific Python executable, or set `CONDA_SH` and `CONDA_ENV` (default:
`video_retrieval`). Model downloads can be redirected with the experiment-specific
environment variables documented in `REPRODUCIBILITY.md`.

## Validated Outputs

1. VCDB uses near-equal scramble chunks and 10 seeds; chunk sizes differ by at most one.
2. HDD reports directed query-wise AP@k, recall@k, MRR, full-gallery baselines, and paired
   intersection-cluster contrasts. The withdrawn unordered-pair table must not be restored.
3. HDD held-out score fusion selects `alpha=0.95` in all 50 folds and provides no detected
   mAP improvement over BoT (+0.0010 [-0.0031, 0.0036]).
4. EPIC reports V-JEPA 2 temporal-residual `s_rev` under sequence DTW, without treating it
   as an embedding classification accuracy.
5. nuScenes reports paired intersection-cluster contrasts rather than differences inferred
   from marginal intervals.
6. Full-gallery temporal-residual DTW reaches mAP 0.164 on HDD and 0.122 on nuScenes, below
   BoT by -0.091 [-0.116, -0.077] and -0.193 [-0.270, -0.135], respectively.

See `results/PROVENANCE.md` for generating commits, SLURM job IDs, dataset versions, model
revisions, and the recorded software environment.

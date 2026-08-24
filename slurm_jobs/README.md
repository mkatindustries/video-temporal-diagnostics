# Corrected Cluster Runs

These are the submission scripts for the corrected cluster reruns. Compact outputs and run
provenance are tracked under `results/`; dataset-local caches and per-pair files are
intentionally excluded.

`<YOUR_DATA_ROOT>` below is a placeholder: these jobs carry no default dataset locations, so
every `*_DIR` variable must point at the operator's own copy of the dataset, obtained under
its own license. Substitute your paths before submitting.

Submit these jobs from the repository root after creating the log directory:

```bash
mkdir -p slurm_logs

VCDB_DIR=<YOUR_DATA_ROOT>/vcdb sbatch slurm_jobs/rerun_vcdb_scramble.sbatch
HDD_DIR=<YOUR_DATA_ROOT>/hdd sbatch slurm_jobs/rerun_hdd_retrieval.sbatch
HDD_DIR=<YOUR_DATA_ROOT>/hdd sbatch slurm_jobs/rerun_hdd_controls.sbatch
HDD_DIR=<YOUR_DATA_ROOT>/hdd sbatch slurm_jobs/rerun_hdd_fusion.sbatch
EPIC_DIR=<YOUR_DATA_ROOT>/epic_kitchens sbatch slurm_jobs/rerun_epic_residual.sbatch
EPIC_DIR=<YOUR_DATA_ROOT>/epic_kitchens sbatch slurm_jobs/rerun_epic_integrity.sbatch   # re-run of the withdrawn integrity probe
SOCCERNET_DIR=<YOUR_DATA_ROOT>/soccernet sbatch slurm_jobs/rerun_soccernet.sbatch
```

`VCDB_DIR` must be the directory that directly contains VCDB's `annotation/` and
`core_dataset/` subdirectories, and `HDD_DIR` must be the HDD release root that contains
`release_2019_07_08/`, `features/`, and `labels/` — in both cases the level that holds those
entries, not a wrapper directory above it.

The nuScenes fusion run consumes the feature cache produced by the temporal run, so submit
it as an `afterok` dependency rather than starting both jobs concurrently:

```bash
nuscenes_job=$(NUSCENES_DIR=<YOUR_DATA_ROOT>/nuscenes \
  sbatch --parsable slurm_jobs/rerun_nuscenes.sbatch)
NUSCENES_DIR=<YOUR_DATA_ROOT>/nuscenes \
  sbatch --dependency="afterok:${nuscenes_job}" slurm_jobs/rerun_nuscenes_fusion.sbatch
```

`NUSCENES_DIR` must be the extracted nuScenes root that actually contains `v1.0-trainval/`,
`samples/`, `sweeps/`, and `maps/`. A sibling path whose name differs only by a `_data` suffix
(or the reverse) is a common trap on shared filesystems: one is the real extraction and the
other is an empty stub or a README-only placeholder left over from the download step. Confirm
`ls "$NUSCENES_DIR"/v1.0-trainval` succeeds before submitting, or the temporal job fails after
queueing. The same check applies to `HDD_DIR`.

For the HDD Video4Real error-composition result, rerun its fusion job after the feature and
distance caches exist. The nuScenes fusion job is already included in the dependency chain
above. Both write compact summaries directly to the tracked `results/` paths:

```bash
HDD_DIR=<YOUR_DATA_ROOT>/hdd sbatch slurm_jobs/rerun_hdd_fusion.sbatch
```

The fusion jobs also evaluate full-gallery temporal-residual DTW. A compatible
BoT/encoder-DTW score cache is augmented in place, so only the missing residual matrix is
computed. Incompatible feature or score caches are rejected by their metadata checks.

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
6. Full-gallery temporal-residual DTW reaches mAP 0.164 on HDD and 0.136 on nuScenes, below
   BoT by -0.091 [-0.116, -0.077] and -0.196 [-0.271, -0.136], respectively.

See `results/PROVENANCE.md` for generating commits, SLURM job IDs, dataset versions, model
revisions, and the recorded software environment.

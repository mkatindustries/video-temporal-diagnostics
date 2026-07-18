# Results Provenance

Compact result summaries for the corrected temporal-diagnostics rerun (2026-07-18).
Caches (`*.pt`), large per-pair score files (`pair_scores.json`,
`encoder_seq_pair_scores.json`), and raw SLURM logs are intentionally **not** tracked.

Shared configuration for every artifact below:

- **Model revisions:** DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` snapshot
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`; V-JEPA 2 `facebook/vjepa2-vitl-fpc64-256`
  snapshot `b3c1679b7c34d3255ef3547f27c7b226aefab26f`.
- **Environment:** conda env `video_retrieval` (Python 3.11.15, torch 2.10.0+cu128,
  transformers 5.6.0.dev0); SLURM account `dream`, qos `h200_comm_shared`, partition `h200`.

| Artifact | Generating commit | SLURM job | Dataset version |
|----------|-------------------|-----------|-----------------|
| `vcdb/vcdb_scramble_multiseed.json` | `c2daec7` | `9634576_0` | VCDB core_dataset |
| `vcdb/raw_frame_scramble_results.json` | `c2daec7` | `9634576_1` | VCDB core_dataset |
| `nuscenes/intersection_results.json` | `290619c` | `9636031` | nuScenes v1.0-trainval |
| `nuscenes/cluster_bootstrap_results.json` | `290619c` | `9636031` | nuScenes v1.0-trainval |
| `hdd/bof_dtw_directed_rerank_results.json` | `290619c` | `9636095` | HDD release_2019_07_08 |
| `hdd/cluster_bootstrap_results.json` | `290619c` | `9636095` | HDD release_2019_07_08 |
| `epic/temporal_order_results.json` | `c2daec7` | `9634579` | EPIC temporal_order_sequences_v1_len6-15_narr2-3_seed42 |

Notes:
- nuScenes and HDD were **reruns** at `290619c` after the int64 JSON-serialization fix
  (`7e67fe7`) and the HDD feature-cache-reuse fix (`290619c`). Their original runs
  (`9634578`, `9634577`) failed and are superseded.
- VCDB (multiseed + raw) and EPIC ran cleanly at the original submission commit `c2daec7`.
- Exact evaluation commands and requested GPU, CPU, memory, and time resources are preserved in `slurm_jobs/`.

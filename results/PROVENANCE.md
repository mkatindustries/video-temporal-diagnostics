# Results Provenance

Compact result summaries for the corrected temporal-diagnostics reruns (2026-07-18 through
2026-07-20).
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
| `hdd/fusion_results.json` | `13250dd` | `9674478` | HDD release_2019_07_08 |
| `nuscenes/fusion_results.json` | `13250dd` | `9674479` | nuScenes v1.0-trainval |

Notes:
- nuScenes and HDD were **reruns** at `290619c` after the int64 JSON-serialization fix
  (`7e67fe7`) and the HDD feature-cache-reuse fix (`290619c`). Their original runs
  (`9634578`, `9634577`) failed and are superseded.
- VCDB (multiseed + raw) and EPIC ran cleanly at the original submission commit `c2daec7`.
- `hdd/fusion_results.json` is the held-out leave-one-cluster-out score fusion (BoT × encoder-seq
  DTW) at commit `b72592e`; its `bot_full_gallery`/`encoder_seq_dtw_full_gallery` baselines match
  `hdd/bof_dtw_directed_rerank_results.json` at reported precision (0.2556 / 0.1765 mAP). Honest null:
  fused mAP 0.2566, fused−BoT +0.0010 (95% CI [−0.0031, 0.0036]); α*=0.95 in all 50 folds. Regenerated
  at commit `7539555` (job 9654434) to add the global paired contrast encoder-seq DTW − BoT
  = −0.0790 (95% CI [−0.1059, −0.0617]); all other values reproduced identically.
- `nuscenes/fusion_results.json` applies the same directed-retrieval + held-out fusion protocol
  to nuScenes (commit `894edc3`, job `9645008`): 264 segments from 50 clusters; 222 eligible
  queries from 40 clusters. It replicates HDD's conditional-vs-global reversal — full-gallery
  BoT mAP 0.3150 [0.254, 0.395] vs encoder-seq DTW 0.1406 [0.106, 0.183] (paired diff
  −0.1745 [−0.245, −0.117]); the BoT→DTW cascade lowers AP at every k. Leave-one-cluster-out
  fusion selected α*=1.0 in all 40 folds, so the fused ranking is identical to BoT (fused−BoT
  difference exactly 0). This is the first nuScenes directed full-gallery evaluation, so the
  BoT/DTW mAPs here have no earlier counterpart to cross-check against.
- Jobs 9671544 (HDD) and 9671547 (nuScenes), committed as `18eee91`, reproduced the fusion
  metrics and added the Video4Real ranked-outcome decomposition. At top 1,
  encoder-sequence DTW minus BoT increases
  wrong-intersection retrieval by +0.1417 [0.1081, 0.1869] on HDD and +0.3063 [0.2290, 0.3874]
  on nuScenes. Same-intersection/wrong-maneuver outcomes are at most 0.45% for all top-1 rows
  (and at most 1.04% through top 10), localizing nearly all observed errors to location selection.
  HDD recomputed the full evaluation-gallery DTW matrix because its score cache was absent;
  nuScenes reused its cache.
- Jobs 9674478 (HDD) and 9674479 (nuScenes), generated from `13250dd`, reused the validated
  BoT/encoder-DTW caches and added full-gallery temporal-residual DTW. Previous metrics reproduced
  unchanged. Residual DTW reaches mAP 0.1644 on HDD and 0.1217 on nuScenes, below BoT by
  -0.0912 [-0.1162, -0.0767] and -0.1933 [-0.2695, -0.1346], respectively. Its top-1
  wrong-intersection fraction is 0.4632 on HDD and 0.8649 on nuScenes; same-intersection,
  wrong-maneuver errors remain at or below 0.06%. The updated result JSONs and regenerated
  `figures/v4r_error_composition.png` were committed in `11a10d0`.
- Exact evaluation commands and requested GPU, CPU, memory, and time resources are preserved in `slurm_jobs/`.

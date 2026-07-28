# Results Provenance

Compact result summaries for the corrected temporal-diagnostics reruns (2026-07-18 through
2026-07-28).
Caches (`*.pt`), large per-pair score files (`pair_scores.json`,
`encoder_seq_pair_scores.json`), and raw SLURM logs are intentionally **not** tracked.

Shared configuration for every artifact below:

- **Model revisions:** DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` snapshot
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`; V-JEPA 2 `facebook/vjepa2-vitl-fpc64-256`
  snapshot `b3c1679b7c34d3255ef3547f27c7b226aefab26f`.
- **Environment:** conda env `video_retrieval` (Python 3.11.15, torch 2.10.0+cu128,
  transformers 5.6.0.dev0); GPU jobs used one H200-class accelerator unless noted otherwise.

| Artifact | Generating commit | SLURM job | Dataset version |
|----------|-------------------|-----------|-----------------|
| `vcdb/vcdb_scramble_multiseed.json` | `c2daec7` | `9634576_0` | VCDB core_dataset |
| `vcdb/raw_frame_scramble_results.json` | `c2daec7` | `9634576_1` | VCDB core_dataset |
| `nuscenes/intersection_results.json` | `597ace9` | `9937532` | nuScenes v1.0-trainval |
| `nuscenes/cluster_bootstrap_results.json` | `597ace9` | `9937532` | nuScenes v1.0-trainval |
| `hdd/bof_dtw_directed_rerank_results.json` | `290619c` | `9636095` | HDD release_2019_07_08 |
| `hdd/encoder_seq_results.json` | `f1f4a7c` | `9910735` | HDD release_2019_07_08 |
| `hdd/cluster_bootstrap_results.json` | `f1f4a7c` | `9910735` | HDD release_2019_07_08 |
| `epic/temporal_order_results.json` | `c2daec7` | `9634579` | EPIC temporal_order_sequences_v1_len6-15_narr2-3_seed42 |
| `epic/vlm_prompt_results.json` | this commit (artifact consolidation) | historical cluster aggregates; raw job IDs unavailable | EPIC temporal_order_sequences_v1_len6-15_narr2-3_seed42 |
| `hdd/fusion_results.json` | `13250dd` | `9674478` | HDD release_2019_07_08 |
| `nuscenes/fusion_results.json` | `597ace9` | `9937533` | nuScenes v1.0-trainval |
| `hdd/conditional_querywise_results.json` | `9236c63` | local CPU, 2026-07-27 | HDD release_2019_07_08 |
| `nuscenes/conditional_querywise_results.json` | `597ace9` | `9937533` | nuScenes v1.0-trainval |

Notes:
- nuScenes and HDD were **reruns** at `290619c` after the int64 JSON-serialization fix
  (`7e67fe7`) and the HDD feature-cache-reuse fix (`290619c`). Their original runs
  (`9634578`, `9634577`) failed and are superseded.
- Job 9910735 (HDD), generated from `f1f4a7c`, added shuffled-DTW and order-free assignment
  controls. Encoder-sequence DTW minus shuffled DTW is +0.0360 [−0.0006, 0.0539], while
  temporal-residual DTW minus shuffled DTW is +0.0288 [−0.0075, 0.0426].
- Jobs 9937532 and 9937533, generated from `597ace9`, supersede all earlier nuScenes results.
  DBSCAN now runs independently within each nuScenes map's local coordinate frame, yielding
  244 evaluation segments in 50 mixed clusters (824 pooled pairs) and 197 eligible directed
  queries from 37 clusters. Encoder-sequence DTW minus shuffled DTW is +0.0627
  [0.0225, 0.1017]; temporal-residual DTW minus shuffled DTW is +0.0299
  [−0.0157, 0.0680]. DTW does not significantly outperform assignment; residual assignment
  exceeds residual DTW by 0.0474 [0.0171, 0.0796]. Both jobs exited successfully.
- VCDB (multiseed + raw) and EPIC ran cleanly at the original submission commit `c2daec7`.
- `epic/vlm_prompt_results.json` is a compact aggregate-only record: it contains no per-clip
  responses. The rounded prompt metrics and integrity condition rates were transcribed from two
  contemporaneous paper artifacts because the original generative aggregate files are absent from
  this checkout. Their SHA-256 values are embedded in the JSON. Three separately hashed
  dataset-local result files authenticate the 500-sequence protocol, prompts, and model identities;
  they contain embedding results and are explicitly not represented as metric sources.
- `hdd/fusion_results.json` is the held-out leave-one-cluster-out score fusion (BoT × encoder-seq
  DTW) at commit `b72592e`; its `bot_full_gallery`/`encoder_seq_dtw_full_gallery` baselines match
  `hdd/bof_dtw_directed_rerank_results.json` at reported precision (0.2556 / 0.1765 mAP). Honest null:
  fused mAP 0.2566, fused−BoT +0.0010 (95% CI [−0.0031, 0.0036]); α*=0.95 in all 50 folds. Regenerated
  at commit `7539555` (job 9654434) to add the global paired contrast encoder-seq DTW − BoT
  = −0.0790 (95% CI [−0.1059, −0.0617]); all other values reproduced identically.
- `nuscenes/fusion_results.json` applies the same directed-retrieval + held-out fusion protocol
  to nuScenes. The location-aware rerun (job 9937533) gives full-gallery BoT mAP 0.3326
  [0.2713, 0.4127] vs encoder-sequence DTW 0.1595 [0.1222, 0.2043] (paired difference
  −0.1732 [−0.2404, −0.1176]); the BoT→DTW cascade lowers AP at every k. Leave-one-cluster-out
  fusion selected α*=1.0 in all 37 folds, so the fused ranking is identical to BoT.
- The Video4Real ranked-outcome decomposition localizes nearly all top-1 errors to location
  selection. Encoder-sequence DTW minus BoT increases wrong-intersection retrieval by +0.1417
  [0.1081, 0.1869] on HDD and +0.2944 [0.2171, 0.3693] on corrected nuScenes. The corresponding
  residual-DTW increases are +0.172 [0.137, 0.214] and +0.320 [0.241, 0.396].
- Full-gallery temporal-residual DTW reaches mAP 0.1644 on HDD and 0.1364 on corrected
  nuScenes, below BoT by -0.0912 [-0.1162, -0.0767] and -0.1962
  [-0.2710, -0.1357], respectively. Its top-1 wrong-intersection fraction is 0.4632 on HDD
  and 0.8325 on nuScenes; same-intersection/wrong-maneuver errors remain at or below 0.06%.
- The conditional query-wise artifacts use the same directed AP definition and eligible query
  sets as the full-gallery fusion runs, but restrict each gallery to the query's intersection
  cluster. Their source score-cache SHA-256, cache version, and feature-cache identity metadata
  are embedded in each JSON. The corrected nuScenes artifact was generated by job 9937533.
- Exact evaluation commands and requested GPU, CPU, memory, and time resources are preserved in `slurm_jobs/`.

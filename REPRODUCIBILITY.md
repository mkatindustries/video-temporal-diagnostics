# Reproducibility Guide: Diagnosing Temporal Sensitivity in Video Retrieval Pipelines

Commands, output artifacts, and environment details for *Diagnosing Temporal Sensitivity in Video Retrieval Pipelines*. Feature caches and datasets are not committed, so this is not a one-command exact reproduction. Corrected compact summaries and provenance are tracked under `results/`; the exact cluster jobs that generated them are under `slurm_jobs/`.

## Environment

```bash
conda activate video_retrieval
pip install -e .              # Core package
pip install -e '.[vlm]'      # VLM experiment support (transformers, accelerate, etc.)
```

**Key dependencies** (from `pyproject.toml`):
- Python >= 3.10
- torch >= 2.0.0, torchvision >= 0.15.0, transformers >= 4.56.0
- numpy >= 1.24.0, scipy >= 1.11.0, scikit-learn >= 1.3.0
- av >= 10.0.0, opencv-python >= 4.8.0

**Hardware:** NVIDIA GPU (A100/H100/H200 recommended). Most experiments require 1 GPU; VLM experiments require 1-2 GPUs depending on model size. Linear and MLP probes run on CPU; bootstrap CIs run on CPU but are GPU-accelerated when CUDA is available.

**Model weights:** HuggingFace IDs: `facebook/dinov3-vitl16-pretrain-lvd1689m`, `facebook/vjepa2-vitl-fpc64-256`, `Qwen/Qwen3-VL-8B-Instruct`, `google/gemma-4-31B-it`, `llava-hf/LLaVA-Video-7B-Qwen2-hf`. Pass `--model-path` to experiment scripts if weights are pre-downloaded to a local path. Claude Opus 4.6 and Gemini 3.1 Pro are API-only (see experiments 32--33).

## Table and Figure Reference Map

Numbering follows the compiled paper (`paper.pdf`).

Main body tables: Table 1 (VCDB Reversal Attack), Table 2 (HDD Maneuver), Table 3 (BoT→DTW Rerank Sweep on HDD), Table 4 (VLM Generative), Table 5 (VLM Embedding s_rev).

Main body figures: Figure 1 (corrected balanced Scramble Gradient), Figure 2 (HDD Maneuver Discrimination).

Appendix tables include the FPS cap, context sweep, layer ablation, cross-method summaries, scene matching, integrity probe, negative-sampling sensitivity, descriptive pair-bootstrap intervals, paired cluster-bootstrap contrasts, corrected directed retrieval, vision-token probes, corrected multi-seed scramble, exploratory linear/MLP probes, computational costs, failure-mode taxonomy, and licenses.

Appendix figures: Figure 3 (Reversal Attack bar chart), Figure 4 (EPIC Sensitivity), Figure 5 (HDD Qualitative), Figure 6 (nuScenes Maneuver). The corrected multi-seed scramble plot is Figure 1 in the main body.

## Headline Results

### 1. Copy Detection (VCDB) — Table 9 (VCDB column)

```bash
python experiments/eval_vcdb.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `datasets/vcdb/eval_results.json`, `figures/vcdb_benchmark.png`

### 2. Reversal Attack (VCDB) — Table 1, Figure 3 (Appendix B)

```bash
python experiments/eval_vcdb_reversal.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `datasets/vcdb/reversal_attack_results.json`, `figures/vcdb_reversal_attack.png`

### 3. Temporal Scramble Gradient (VCDB) — Table 19 (Appendix R), Figure 1

The implementation uses near-equal chunk partitions (`numpy.array_split`). Results produced before this correction are invalid when the frame count is not divisible by `K`. The corrected evaluation uses 10 permutation seeds for the extracted-embedding variant and also re-extracts features after raw-frame scrambling.

```bash
python experiments/eval_vcdb_scramble_multiseed.py \
    --vcdb-dir /path/to/vcdb --device cuda --n-seeds 10

python experiments/eval_vcdb_scramble_raw.py \
    --vcdb-dir /path/to/vcdb --device cuda
```

**Output:** `results/scramble_multiseed/vcdb_scramble_multiseed.json`, `<vcdb-dir>/raw_frame_scramble_results.json`, and `figures/vcdb_scramble_gradient_errorbars.png`. Final compact copies are tracked under `results/vcdb/`.

### 4. Maneuver Discrimination (Honda HDD) — Table 2, Figure 2 (main body)

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd
```

**Output:** `figures/hdd_maneuver_discrimination.png`, `figures/hdd_similarity_distributions.png`

### 5. Context Window Sweep (Honda HDD) — Table 7 (Appendix C)

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd \
    --context-sec-sweep 1.0 2.0 3.0 4.0 6.5 8.0
```

**Output:** `datasets/hdd/context_sec_sweep_results.json`

### 6. V-JEPA 2 Encoder-Sequence DTW Baseline (HDD) — Table 2 (Encoder-Seq row), §3.2

Disentangles feature vs. comparator contribution. Encoder patches spatially averaged per timestep, compared via DTW.

```bash
python experiments/eval_hdd_encoder_seq.py \
    --hdd-dir /path/to/hdd
```

**Output:** `datasets/hdd/encoder_seq_results.json`, `datasets/hdd/encoder_seq_pair_scores.json`, and `datasets/hdd/vjepa2_encoder_features.pt` (paths follow `--hdd-dir`/`--feature-cache`).

### 6b. BoT→DTW Two-Stage Retriever (HDD) — corrected directed protocol

Tests whether the pair-diagnostic gain composes into a query-wise retriever. For each query, the gallery contains every other cached segment. Relevance requires the same intersection cluster and maneuver label; outside-cluster candidates remain negatives. BoT selects directional top-k candidates and encoder-sequence DTW reranks them. The script reports macro truncated AP@k (normalized by all relevant gallery items), recall@k, and MRR, plus cluster-bootstrap intervals.

```bash
python experiments/eval_hdd_bof_dtw_rerank.py \
    --hdd-dir /path/to/hdd
```

Requires the feature cache produced by experiment 6 (default `datasets/hdd/vjepa2_encoder_features.pt`). Pass the same path to `--feature-cache` if it is stored elsewhere.

**Output:** `datasets/hdd/bof_dtw_directed_rerank_results.json`

The former unordered-pair survivor-AP/RRF table is withdrawn. It used an either-endpoint survival rule, excluded out-of-cluster negatives, and could report AP above recall. In the corrected run, full-gallery BoT mAP is 0.256 [0.218, 0.288], while full-gallery encoder-sequence DTW mAP is 0.177 [0.135, 0.212]. DTW reranking lowers AP and MRR at every tested `k`; see `results/hdd/bof_dtw_directed_rerank_results.json`.

### 7. FPS Downsample Sweep (Honda HDD) — Appendix A

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd \
    --fps-downsample 2 5 10 15 30
```

**Output:** `datasets/hdd/fps_downsample_results.json`

### 8. FPS Cap Invariance Verification — Table 6 (Appendix A)

```bash
python scripts/verify_fps_cap_invariance.py \
    --hdd-dir /path/to/hdd
```

**Output:** Console (no GPU required)

### 9. Cross-Dataset Validation (nuScenes) — §3.2.2, Figure 6 (Appendix P)

```bash
python experiments/eval_nuscenes_intersections.py \
    --nuscenes-dir /path/to/nuscenes
```

**Output:** `<nuscenes-dir>/intersection_results.json`, `figures/nuscenes_maneuver_discrimination.png`

### 10. VLM Vision Tower Bridge (Honda HDD) — Table 10 (VLM summary, HDD column)

Evaluates VLM vision tower embeddings (SigLIP, CLIP) and LLM hidden states on HDD maneuver discrimination.

```bash
# Gemma-4 SigLIP + LLM hidden state
python experiments/eval_hdd_vlm_bridge.py \
    --hdd-dir /path/to/hdd --vlm-family gemma4 --include-baselines --extract-llm

# LLaVA CLIP + LLM hidden state
python experiments/eval_hdd_vlm_bridge.py \
    --hdd-dir /path/to/hdd --vlm-family llava-video --include-baselines --extract-llm

# Qwen3 SigLIP + LLM hidden state
python experiments/eval_hdd_vlm_bridge.py \
    --hdd-dir /path/to/hdd --vlm-family qwen3 --include-baselines --extract-llm
```

**Output:** `datasets/hdd/vlm_bridge_{gemma4,llava-video,qwen3}_results.json`

### 11. Multi-VLM Temporal Order Probes (EPIC-Kitchens) — Tables 4-5, Figure 4 (Appendix E)

**Generative probes (all 3 VLMs):**
```bash
for family in qwen gemma llava; do
    python experiments/eval_epic_temporal_order.py \
        --vlm-family $family --vlm-generative --vlm-integrity-probe
done
```

**Embedding probes (Gemma + LLaVA):**
```bash
for family in gemma llava; do
    python experiments/eval_epic_temporal_order.py \
        --vlm-family $family --vlm-embeddings
done
```

**Layer ablation (Table 8, Appendix D):**
```bash
for family in qwen gemma llava; do
    python experiments/eval_epic_temporal_order.py \
        --vlm-family $family --vlm-embeddings --vlm-layer-ablation
done
```

**Output:** `datasets/epic_kitchens/temporal_order_results*.json`, `figures/epic_temporal_order_sensitivity.png` (Figure 4, Appendix E)

### 12. Scene Retrieval (Nymeria) — Table 11 (Appendix F)

```bash
python experiments/eval_nymeria_activities.py \
    --nymeria-dir /path/to/nymeria
```

**Output:** `figures/nymeria_*.png`

### 13. Multi-Domain Retrieval (MUVR) — Table 11 (Appendix F)

```bash
python experiments/eval_muvr.py \
    --partition news \
    --muvr-dir /path/to/muvr
```

**Output:** `figures/muvr_news_*.png`

### 14. nuScenes VLM Vision Tower Bridge — Table 10 (nuScenes column)

```bash
# Gemma-4 SigLIP + LLM hidden state
python experiments/eval_nuscenes_vlm_bridge.py \
    --nuscenes-dir /path/to/nuscenes --vlm-family gemma4 --extract-llm

# LLaVA CLIP + LLM hidden state
python experiments/eval_nuscenes_vlm_bridge.py \
    --nuscenes-dir /path/to/nuscenes --vlm-family llava-video --extract-llm

# Qwen3-VL SigLIP + LLM hidden state
python experiments/eval_nuscenes_vlm_bridge.py \
    --nuscenes-dir /path/to/nuscenes --vlm-family qwen3 --extract-llm
```

**Output:** `datasets/nuscenes/vlm_bridge_{gemma4,llava-video,qwen3}_results.json`

### 15. VCDB VLM Generative Probes — supplementary (not in main-body tables)

```bash
python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family gemma4 --max-clips 500

python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family llava-video --max-clips 500

python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family qwen3 --max-clips 500
```

**Output:** `datasets/vcdb/vlm_probes_{gemma4,llava-video,qwen3}_results.json`

### 16. VCDB VLM Vision Tower Bridge — Table 10 (VCDB column)

Evaluates VLM vision tower embeddings and LLM hidden states on VCDB copy detection.

```bash
for family in gemma4 llava-video qwen3; do
    python experiments/eval_vcdb_vlm_bridge.py \
        --vcdb-dir /path/to/vcdb/core_dataset --vlm-family $family
done
```

**Output:** `datasets/vcdb/vlm_bridge_{gemma4,llava-video,qwen3}_results.json`, `*_pair_scores.json`

### 17. HDD VLM Generative Probes — supplementary (not in main-body tables)

Forward/reverse direction classification on HDD maneuver segments.

```bash
for family in gemma4 llava-video qwen3; do
    python experiments/eval_hdd_vlm_generative.py \
        --hdd-dir /path/to/hdd --vlm-family $family
done
```

**Output:** `datasets/hdd/vlm_generative_{gemma4,llava-video,qwen3}_results.json`

### 18. nuScenes VLM Generative Probes — supplementary (not in main-body tables)

Forward/reverse direction classification on nuScenes maneuver segments.

```bash
for family in gemma4 llava-video qwen3; do
    python experiments/eval_nuscenes_vlm_generative.py \
        --nuscenes-dir /path/to/nuscenes --vlm-family $family --version v1.0-trainval
done
```

**Output:** `datasets/nuscenes/vlm_generative_{gemma4,llava-video,qwen3}_results.json`

### 19. V-JEPA 2 Comparator Ablation (Honda HDD) — Appendix

```bash
python experiments/eval_hdd_residual_ablation.py \
    --hdd-dir /path/to/hdd --device cuda
```

**Output:** `datasets/hdd/residual_ablation_results.json`

### 20. HDD Cross-Session Validation — Appendix H

```bash
python experiments/eval_hdd_cross_session.py \
    --hdd-dir /path/to/hdd --device cuda
```

**Output:** `datasets/hdd/cross_session_results.json`

### 21. HDD Standard Retrieval Protocol — Appendix

```bash
python experiments/eval_hdd_retrieval_protocol.py \
    --hdd-dir /path/to/hdd --device cuda
```

**Output:** `datasets/hdd/retrieval_protocol_results.json`

### 22. VCDB Extended Attack Suite — Appendix

```bash
python experiments/eval_vcdb_attack_suite.py \
    --vcdb-dir /path/to/vcdb/core_dataset --attacks all
```

**Output:** `datasets/vcdb/attack_suite_results.json`

### 23. Bootstrap Confidence Intervals — Table 15 (Appendix L)

Requires `pair_scores.json` files generated by the eval scripts above.

```bash
python experiments/bootstrap_cis.py \
    --benchmark vcdb --pairs-json datasets/vcdb/pair_scores.json \
    --output-json datasets/vcdb/bootstrap_cis.json

python experiments/bootstrap_cis.py \
    --benchmark hdd --pairs-json /path/to/hdd/pair_scores.json \
    --output-json /path/to/hdd/bootstrap_cis.json

python experiments/bootstrap_cis.py \
    --benchmark nuscenes --pairs-json /path/to/nuscenes/pair_scores.json \
    --output-json /path/to/nuscenes/bootstrap_cis.json

python experiments/bootstrap_cis.py \
    --benchmark nymeria --pairs-json /path/to/nymeria/pair_scores.json \
    --output-json /path/to/nymeria/bootstrap_cis.json

python experiments/bootstrap_cis.py \
    --benchmark muvr --pairs-json /path/to/muvr/pair_scores_news.json \
    --output-json /path/to/muvr/bootstrap_cis_news.json
```

**Output:** Bootstrap CI JSON files; GPU-accelerated when CUDA is available.

### 24. Cluster-Level Block Bootstrap — uncertainty appendix

Resamples GPS clusters (not pairs) to account for within-cluster dependence. The corrected script also reports paired AP differences between methods on the same resampled clusters; marginal intervals alone do not test method differences.

```bash
python experiments/eval_cluster_bootstrap.py \
    --hdd-dir /path/to/hdd --nuscenes-dir /path/to/nuscenes
```

**Output:** `datasets/hdd/cluster_bootstrap_results.json`. The nuScenes job writes aligned
cluster IDs and runs `eval_grouped_pair_bootstrap.py`, producing
`datasets/nuscenes/cluster_bootstrap_results.json` (or the corresponding dataset-root path).

### 25. V-JEPA 2 VCDB Bootstrap CIs — Table 15 (Appendix L)

Extracts V-JEPA 2 features for VCDB and computes BoT + temporal residual pair scores with bootstrap CIs.

```bash
python experiments/eval_vjepa2_vcdb_bootstrap.py \
    --vcdb-dir /path/to/vcdb/core_dataset
```

**Output:** `datasets/vcdb/pair_scores_vjepa2.json`, `datasets/vcdb/bootstrap_cis_vjepa2.json`

### 26. Scramble Multi-Seed — Figure 1 and Table 19 (Appendix R)

10 independent shuffle seeds per K level to quantify variance.

```bash
python experiments/eval_vcdb_scramble_multiseed.py \
    --vcdb-dir /path/to/vcdb --n-seeds 10
```

**Output:** `results/scramble_multiseed/vcdb_scramble_multiseed.json`,
`figures/vcdb_scramble_gradient_errorbars.png`. The finalized compact summary is
tracked at `results/vcdb/vcdb_scramble_multiseed.json`.

### 27. Raw-Frame Scramble (V-JEPA 2, VCDB) — Appendix R

Re-runs V-JEPA 2 encoder+predictor on chunk-shuffled raw video frames (not extracted embeddings). Tests whether the encoder produces different residuals from temporally disrupted input.

```bash
python experiments/eval_vcdb_scramble_raw.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `/path/to/vcdb/raw_frame_scramble_results.json`

### 28. Left-vs-Right Only AP (HDD) — §3.2

Filters HDD evaluation to left/right turns only (excludes intersection-passing).

```bash
python experiments/eval_hdd_left_vs_right.py \
    --hdd-dir /path/to/hdd
```

**Output:** `/path/to/hdd/left_right_results.json`

### 29. Optical Flow Baseline (HDD) — §3.2

RAFT optical flow baseline for maneuver discrimination.

```bash
python experiments/eval_hdd_optical_flow.py \
    --hdd-dir /path/to/hdd
```

**Output:** `/path/to/hdd/optical_flow_results.json`

### 30. Linear Probe on LLM Hidden States — Table 20 (Appendix S)

Tests whether temporal signal is present in per-position LLM hidden states. Uses 5-fold GroupKFold CV (fwd/rev pairs kept together) with logistic regression.

```bash
# One job per VLM (can run in parallel)
python experiments/eval_epic_linear_probe.py \
    --epic-dir /path/to/epic_kitchens --vlm-family qwen3

python experiments/eval_epic_linear_probe.py \
    --epic-dir /path/to/epic_kitchens --vlm-family gemma4

python experiments/eval_epic_linear_probe.py \
    --epic-dir /path/to/epic_kitchens --vlm-family llava-video
```

**Output:** `datasets/epic_kitchens/linear_probe_{qwen3,gemma4,llava-video}.json`, `datasets/epic_kitchens/feature_cache/linear_probe_*_hidden_states.pt`

### 31. MLP Probe on LLM Hidden States — Table 20 (Appendix S)

2-layer MLP (256 hidden units, ReLU) on cached hidden states from the linear probe runs. Uses 5-fold GroupKFold CV. CPU only (~2 minutes total).

```bash
python experiments/eval_mlp_probe.py
```

**Output:** `datasets/epic_kitchens/mlp_probe_results.json`

### 32. Claude Opus 4.6 Generative Probe — §3.3, Table 4

Proprietary VLM temporal order probe via OpenAI-compatible API. Requires API access and `CLAUDE_API_KEY` environment variable.

```bash
export CLAUDE_API_KEY="Bearer YOUR_API_KEY"
python experiments/eval_epic_claude_probe.py \
    --epic-dir /path/to/epic_kitchens
```

**Output:** `datasets/epic_kitchens/claude_probe_results.json`

### 33. Gemini 3.1 Pro Generative Probe — §3.3, Table 4

Reasoning-model temporal order probe via OpenAI-compatible API. Same protocol as Claude. Requires `GEMINI_API_KEY` environment variable.

```bash
export GEMINI_API_KEY="Bearer YOUR_API_KEY"
python experiments/eval_epic_gemini_probe.py \
    --epic-dir /path/to/epic_kitchens
```

**Output:** `datasets/epic_kitchens/gemini_probe_results.json`

### 34. ViCLIP Evaluation — Table 9, §5

ViCLIP (ViT-L, InternVid-10M) as a video-native single-vector baseline across VCDB, HDD, and EPIC-Kitchens. Tests whether contrastive video-text pretraining changes the temporal blindness story.

```bash
python experiments/eval_viclip.py \
    --vcdb-dir /path/to/vcdb/core_dataset \
    --hdd-dir /path/to/hdd \
    --epic-dir /path/to/epic_kitchens \
    --benchmarks all
```

**Output:** `datasets/viclip_results.json`

### 35. TARA Evaluation (Honda HDD) — Table 2, Table 9

TARA (Tarsier-7B MLLM trained with chiral negatives) as a single-vector encoder on HDD. Tests whether temporal training helps under cosine similarity.

```bash
python experiments/eval_hdd_tara.py \
    --hdd-dir /path/to/hdd \
    --model-path ~/src/TARA
```

**Output:** `datasets/tara_hdd_results.json`

### 36. PL-Stitch Evaluation (Honda HDD) — Table 2, Table 9

PL-Stitch (ViT-Base with Plackett-Luce temporal ranking) as a per-frame encoder on HDD. Enables feature-vs-comparator decomposition with 4 similarity methods (BoF, Chamfer, temporal derivative DTW, raw DTW).

```bash
python experiments/eval_hdd_pl_stitch.py \
    --hdd-dir /path/to/hdd \
    --weights ~/src/PL-Stitch/pl_lemon.pth
```

**Output:** `datasets/pl_stitch_hdd_results.json`

### 37. OrderedMaxSim Comparator Ablation — DINOv3 on HDD (Supplementary)

Tests whether a monotonicity-penalised late-interaction comparator (OrderedMaxSim) on frozen DINOv3 per-frame CLS tokens can close the comparator gap on HDD.

```bash
python experiments/eval_hdd_ordered_maxsim.py \
    --hdd-dir /path/to/hdd
```

**Output:** `/path/to/hdd/ordered_maxsim_ablation_results.json`, `datasets/dinov3_hdd_frame_features.pt` (cached features)

### 38. OrderedMaxSim Comparator Ablation — V-JEPA 2 on HDD (Supplementary)

Same comparator suite on V-JEPA 2 encoder-sequence tokens (32×1024 per segment). Includes DTW as reference upper bound.

```bash
python experiments/eval_hdd_ordered_maxsim_vjepa2.py \
    --hdd-dir /path/to/hdd
```

**Output:** `/path/to/hdd/ordered_maxsim_vjepa2_ablation_results.json`, `datasets/vjepa2_hdd_encoder_features.pt` (cached features)

### 39. OrderedMaxSim on VCDB — Both Backbones (Supplementary)

Runs the full comparator suite on VCDB copy detection using pre-cached DINOv3 and V-JEPA 2 features. No feature extraction needed.

```bash
python experiments/eval_vcdb_ordered_maxsim.py \
    --vcdb-dir /path/to/vcdb/core_dataset
```

**Output:** `/path/to/vcdb/core_dataset/ordered_maxsim_ablation_results.json`

### 40. Monotonicity Violation Diagnostic (Supplementary)

Counts MaxSim argmax monotonicity violations per VCDB pair to explain why OrderedMaxSim = Chamfer. Two variants: positive pairs only, and positive vs. negative pairs.

```bash
# Positive pairs only
python experiments/vcdb_violation_diagnostic.py \
    --vcdb-dir /path/to/vcdb/core_dataset

# Positive vs. negative pairs
python experiments/vcdb_violation_pos_neg.py \
    --vcdb-dir /path/to/vcdb/core_dataset
```

**Output:** Console (no saved artifact)

### 41. DTW similarity transform

For fixed positive alpha, `exp(-alpha * distance)` is strictly monotone and therefore cannot change AP except through numerical ties. The former empirical sweep mixed preprocessing across result tables and is not part of the paper's evidence.

## Output Artifact Index

| Artifact | Paper Reference |
|----------|-----------------|
| `datasets/vcdb/eval_results.json` | Table 9 (VCDB column) |
| `datasets/vcdb/reversal_attack_results.json` | Table 1 |
| `results/vcdb/vcdb_scramble_multiseed.json` | Corrected multi-seed Table 19 (Appendix R) |
| `results/vcdb/raw_frame_scramble_results.json` | Appendix R (raw-frame scramble) |
| `datasets/hdd/context_sec_sweep_results.json` | Table 7 (Appendix C) |
| `datasets/hdd/fps_downsample_results.json` | Table 6 (Appendix A) |
| `datasets/hdd/encoder_seq_results.json` | Table 2 (Encoder-Seq row) |
| `results/hdd/bof_dtw_directed_rerank_results.json` | Corrected directed retrieval tables |
| `datasets/hdd/vlm_bridge_*_results.json` | Table 10 (HDD column) |
| `results/hdd/cluster_bootstrap_results.json` | Grouped marginal and paired AP intervals |
| `results/epic/temporal_order_results.json` | Corrected EPIC residual result; Tables 4-5 |
| `datasets/epic_kitchens/linear_probe_*.json` | Table 20 (Appendix S) |
| `datasets/epic_kitchens/mlp_probe_results.json` | Table 20 (Appendix S, MLP columns) |
| `figures/vcdb_reversal_attack.png` | Figure 3 (Appendix B) |
| `figures/epic_temporal_order_sensitivity.png` | Figure 4 (Appendix E) |
| `figures/hdd_maneuver_discrimination.png` | Figure 2 (main body) |
| `figures/nuscenes_maneuver_discrimination.png` | Figure 6 (Appendix P) |
| `datasets/nuscenes/vlm_bridge_*_results.json` | Table 10 (nuScenes column) |
| `results/nuscenes/intersection_results.json` | Corrected nuScenes pair diagnostic |
| `results/nuscenes/cluster_bootstrap_results.json` | Grouped marginal and paired AP intervals |
| `datasets/vcdb/vlm_bridge_*_results.json` | Table 10 (VCDB column) |
| `datasets/vcdb/vlm_probes_*_results.json` | Supplementary (VCDB VLM generative) |
| `datasets/hdd/vlm_generative_*_results.json` | Supplementary (HDD VLM generative) |
| `datasets/nuscenes/vlm_generative_*_results.json` | Supplementary (nuScenes VLM generative) |
| `datasets/hdd/residual_ablation_results.json` | Appendix (comparator ablation) |
| `datasets/hdd/cross_session_results.json` | Appendix H (cross-session) |
| `datasets/hdd/retrieval_protocol_results.json` | Appendix (retrieval metrics) |
| `datasets/vcdb/attack_suite_results.json` | Appendix (attack suite) |
| `datasets/vcdb/pair_scores_vjepa2.json` | Table 15 (Appendix L) |
| `results/PROVENANCE.md` | Commit, job, dataset, environment, and model revision map |
| `figures/vcdb_scramble_gradient_errorbars.png` | Figure 1; source values in Table 19 (Appendix R) |
| `datasets/hdd/left_right_results.json` | §3.2 (left-vs-right AP) |
| `datasets/hdd/optical_flow_results.json` | §3.2 (optical flow baseline) |
| `datasets/epic_kitchens/claude_probe_results.json` | §3.3, Table 4 (Claude probe) |
| `datasets/epic_kitchens/gemini_probe_results.json` | §3.3, Table 4 (Gemini probe) |
| `datasets/viclip_results.json` | Table 9, §5 (ViCLIP) |
| `datasets/tara_hdd_results.json` | Table 2, Table 9 (TARA) |
| `datasets/pl_stitch_hdd_results.json` | Table 2, Table 9 (PL-Stitch) |
| `datasets/hdd/ordered_maxsim_ablation_results.json` | Supplementary (DINOv3 OrderedMaxSim ablation) |
| `datasets/hdd/ordered_maxsim_vjepa2_ablation_results.json` | Supplementary (V-JEPA 2 OrderedMaxSim ablation) |
| `datasets/vcdb/ordered_maxsim_ablation_results.json` | Supplementary (VCDB OrderedMaxSim ablation) |

## Paper Compilation

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

# Reproducibility Guide: The Temporal Blind Spot in Video Retrieval

Exact commands, output artifacts, and environment details for reproducing all results in *The Temporal Blind Spot in Video Retrieval: Diagnosing Temporal Sensitivity*.

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

**Hardware:** NVIDIA GPU (A100/H100/H200 recommended). Most experiments require 1 GPU; VLM experiments require 1-2 GPUs depending on model size. Probes (linear + MLP) and bootstrap CIs are CPU-only.

**Model weights:** HuggingFace IDs: `facebook/dinov3-vitl16-pretrain-lvd1689m`, `facebook/vjepa2-vitl-fpc64-256`, `Qwen/Qwen3-VL-8B-Instruct`, `google/gemma-4-31B-it`, `llava-hf/LLaVA-Video-7B-Qwen2-hf`. Pass `--model-path` to experiment scripts if weights are pre-downloaded to a local path.

## Table and Figure Reference Map

Main body tables: Table 1 (Reversal Attack), Table 2 (HDD Maneuver), Table 3 (Context Sweep), Table 4 (VLM Generative), Table 5 (VLM Embedding s_rev), Table 6 (Cross-Method Summary).

Main body figures: Figure 1 (Reversal Attack), Figure 2 (Scramble Gradient), Figure 3 (HDD Maneuver), Figure 4 (Context Sweep), Figure 5 (Sensitivity-Invariance Trade-off).

Appendix tables: Table 7 (FPS Cap), Table 8 (Layer Ablation), Table 9 (Scene Matching), Table 10 (Qwen VCDB), Table 11 (Integrity Probe), Table 12 (Neg-Sampling Sensitivity), Table 13 (Bootstrap CIs), Table 14 (Vision-Token Probes), Table 15 (Scramble Data), Table 16 (Linear+MLP Probes), Table 17 (Computational Costs).

Appendix figures: Figure 6 (EPIC Sensitivity), Figure 7 (HDD Qualitative), Figure 8 (nuScenes Maneuver), Figure 9 (Scramble Multi-Seed).

## Headline Results

### 1. Copy Detection (VCDB) — Table 6 (VCDB column)

```bash
python experiments/eval_vcdb.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `datasets/vcdb/eval_results.json`, `figures/vcdb_benchmark.png`

### 2. Reversal Attack (VCDB) — Table 1, Figure 1

```bash
python experiments/eval_vcdb_reversal.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `datasets/vcdb/reversal_attack_results.json`, `figures/vcdb_reversal_attack.png`

### 3. Temporal Scramble Gradient (VCDB) — Table 15 (Appendix M), Figure 2

```bash
python experiments/eval_vcdb_scramble.py \
    --vcdb-dir /path/to/vcdb
```

**Output:** `datasets/vcdb/scramble_gradient_results.json`, `figures/vcdb_scramble_gradient.png`

### 4. Maneuver Discrimination (Honda HDD) — Table 2, Figure 3

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd
```

**Output:** `figures/hdd_maneuver_discrimination.png`, `figures/hdd_similarity_distributions.png`

### 5. Context Window Sweep (Honda HDD) — Table 3, Figure 4

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd \
    --context-sec-sweep 1.0 2.0 3.0 4.0 6.5 8.0
```

**Output:** `datasets/hdd/context_sec_sweep_results.json`

**Figure generation:**
```bash
python scripts/plot_context_sweep.py
```

**Output:** `figures/hdd_context_sweep.png`

### 6. V-JEPA 2 Encoder-Sequence DTW Baseline (HDD) — Table 2 (Encoder-Seq row), §3.2

Disentangles feature vs. comparator contribution. Encoder patches spatially averaged per timestep, compared via DTW.

```bash
python experiments/eval_hdd_encoder_seq.py \
    --hdd-dir /path/to/hdd
```

**Output:** `datasets/hdd/encoder_seq_results.json`

### 7. FPS Downsample Sweep (Honda HDD) — Appendix A

```bash
python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd \
    --fps-downsample 2 5 10 15 30
```

**Output:** `datasets/hdd/fps_downsample_results.json`

### 8. FPS Cap Invariance Verification — Table 7 (Appendix A)

```bash
python scripts/verify_fps_cap_invariance.py \
    --hdd-dir /path/to/hdd
```

**Output:** Console (no GPU required)

### 9. Cross-Dataset Validation (nuScenes) — §3.2.2, Figure 8 (Appendix K)

```bash
python experiments/eval_nuscenes_intersections.py \
    --nuscenes-dir /path/to/nuscenes
```

**Output:** `<nuscenes-dir>/intersection_results.json`, `figures/nuscenes_maneuver_discrimination.png`

### 10. VLM Vision Tower Bridge (Honda HDD) — Table 6 (VLM rows)

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

### 11. Multi-VLM Temporal Order Probes (EPIC-Kitchens) — Tables 4-5, Figure 6 (Appendix B)

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

**Layer ablation (Table 8, Appendix B):**
```bash
for family in qwen gemma llava; do
    python experiments/eval_epic_temporal_order.py \
        --vlm-family $family --vlm-embeddings --vlm-layer-ablation
done
```

**Output:** `datasets/epic_kitchens/temporal_order_results*.json`, `figures/epic_temporal_order_sensitivity.png`

### 12. Scene Retrieval (Nymeria) — Table 9 (Appendix C)

```bash
python experiments/eval_nymeria_activities.py \
    --nymeria-dir /path/to/nymeria
```

**Output:** `figures/nymeria_*.png`

### 13. Multi-Domain Retrieval (MUVR) — Table 9 (Appendix C)

```bash
python experiments/eval_muvr.py \
    --partition news \
    --muvr-dir /path/to/muvr
```

**Output:** `figures/muvr_news_*.png`

### 14. nuScenes VLM Vision Tower Bridge — Table 6 (VLM rows)

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

### 15. VCDB VLM Generative Probes — Table 6 (VLM gen rows)

```bash
python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family gemma4 --max-clips 500

python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family llava-video --max-clips 500

python experiments/eval_vcdb_vlm_probes.py \
    --vcdb-dir /path/to/vcdb/core_dataset --vlm-family qwen3 --max-clips 500
```

**Output:** `datasets/vcdb/vlm_probes_{gemma4,llava-video,qwen3}_results.json`

### 16. VCDB VLM Vision Tower Bridge — Table 6 (VLM vision rows)

Evaluates VLM vision tower embeddings and LLM hidden states on VCDB copy detection.

```bash
for family in gemma4 llava-video qwen3; do
    python experiments/eval_vcdb_vlm_bridge.py \
        --vcdb-dir /path/to/vcdb/core_dataset --vlm-family $family
done
```

**Output:** `datasets/vcdb/vlm_bridge_{gemma4,llava-video,qwen3}_results.json`, `*_pair_scores.json`

### 17. HDD VLM Generative Probes — Table 6 (VLM gen rows)

Forward/reverse direction classification on HDD maneuver segments.

```bash
for family in gemma4 llava-video qwen3; do
    python experiments/eval_hdd_vlm_generative.py \
        --hdd-dir /path/to/hdd --vlm-family $family
done
```

**Output:** `datasets/hdd/vlm_generative_{gemma4,llava-video,qwen3}_results.json`

### 18. nuScenes VLM Generative Probes — Table 6 (VLM gen rows)

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

### 20. HDD Cross-Session Validation — Appendix G

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

### 23. Bootstrap Confidence Intervals — Table 13 (Appendix I)

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

### 24. Cluster-Level Block Bootstrap — Table 13 footnote (Appendix I)

Resamples GPS clusters (not pairs) to produce CIs that account for within-cluster dependence. CIs are 7-32× wider than pair-level.

```bash
python experiments/eval_cluster_bootstrap.py \
    --hdd-dir /path/to/hdd --nuscenes-dir /path/to/nuscenes
```

**Output:** `datasets/hdd/cluster_bootstrap_cis.json`, `datasets/nuscenes/cluster_bootstrap_cis.json`

### 25. V-JEPA 2 VCDB Bootstrap CIs — Table 13 (Appendix I)

Extracts V-JEPA 2 features for VCDB and computes BoT + temporal residual pair scores with bootstrap CIs.

```bash
python experiments/eval_vjepa2_vcdb_bootstrap.py \
    --vcdb-dir /path/to/vcdb/core_dataset
```

**Output:** `datasets/vcdb/pair_scores_vjepa2.json`, `datasets/vcdb/bootstrap_cis_vjepa2.json`

### 26. Scramble Multi-Seed — Figure 9 (Appendix M)

10 independent shuffle seeds per K level to quantify variance.

```bash
python experiments/eval_vcdb_scramble_multiseed.py \
    --vcdb-dir /path/to/vcdb/core_dataset --n-seeds 10
```

**Output:** `datasets/vcdb/scramble_multiseed_results.json`, `figures/vcdb_scramble_multiseed.png`

### 27. Raw-Frame Scramble (V-JEPA 2, VCDB) — Appendix M

Re-runs V-JEPA 2 encoder+predictor on chunk-shuffled raw video frames (not extracted embeddings). Tests whether the encoder produces different residuals from temporally disrupted input.

```bash
python experiments/eval_vcdb_scramble_raw.py \
    --vcdb-dir /path/to/vcdb/core_dataset
```

**Output:** `datasets/vcdb/scramble_raw_results.json`

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

### 30. Linear Probe on LLM Hidden States — Table 16 (Appendix N)

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

### 31. MLP Probe on LLM Hidden States — Table 16 (Appendix N)

2-layer MLP (256 hidden units, ReLU) on cached hidden states from the linear probe runs. Uses 5-fold GroupKFold CV. CPU only (~2 minutes total).

```bash
python experiments/eval_mlp_probe.py
```

**Output:** `datasets/epic_kitchens/mlp_probe_results.json`

### 32. Claude 4.6 Opus Generative Probe — §3.3, Table 4

Proprietary VLM temporal order probe via OpenAI-compatible API. Requires API access and `CLAUDE_API_KEY` environment variable.

```bash
export CLAUDE_API_KEY="Bearer YOUR_API_KEY"
python experiments/eval_epic_claude_probe.py \
    --epic-dir /path/to/epic_kitchens
```

**Output:** `datasets/epic_kitchens/claude_probe_results.json`

### 33. ViCLIP Evaluation — Table 6, §5

ViCLIP (ViT-L, InternVid-10M) as a video-native single-vector baseline across VCDB, HDD, and EPIC-Kitchens. Tests whether contrastive video-text pretraining changes the temporal blindness story.

```bash
python experiments/eval_viclip.py \
    --vcdb-dir /path/to/vcdb/core_dataset \
    --hdd-dir /path/to/hdd \
    --epic-dir /path/to/epic_kitchens \
    --benchmarks all
```

**Output:** `datasets/viclip_results.json`

## Output Artifact Index

| Artifact | Paper Reference |
|----------|-----------------|
| `datasets/vcdb/eval_results.json` | Table 6 (VCDB column) |
| `datasets/vcdb/reversal_attack_results.json` | Table 1 |
| `datasets/vcdb/scramble_gradient_results.json` | Table 15 (Appendix M) |
| `datasets/vcdb/scramble_raw_results.json` | Appendix M (raw-frame scramble) |
| `datasets/hdd/context_sec_sweep_results.json` | Table 3 |
| `datasets/hdd/fps_downsample_results.json` | Table 7 (Appendix A) |
| `datasets/hdd/encoder_seq_results.json` | Table 2 (Encoder-Seq row) |
| `datasets/hdd/vlm_bridge_*_results.json` | Table 6 (VLM bridge rows) |
| `datasets/hdd/cluster_bootstrap_cis.json` | Table 13 footnote (Appendix I) |
| `datasets/epic_kitchens/temporal_order_results*.json` | Tables 4-5, Table 8 (Appendix B) |
| `datasets/epic_kitchens/linear_probe_*.json` | Table 16 (Appendix N) |
| `datasets/epic_kitchens/mlp_probe_results.json` | Table 16 (Appendix N, MLP columns) |
| `figures/vcdb_reversal_attack.png` | Figure 1 |
| `figures/vcdb_scramble_gradient.png` | Figure 2 |
| `figures/epic_temporal_order_sensitivity.png` | Figure 6 (Appendix B) |
| `figures/hdd_maneuver_discrimination.png` | Figure 3 |
| `figures/hdd_context_sweep.png` | Figure 4 |
| `figures/nuscenes_maneuver_discrimination.png` | Figure 8 (Appendix K) |
| `figures/sensitivity_invariance_tradeoff.png` | Figure 5 |
| `datasets/nuscenes/vlm_bridge_*_results.json` | Table 6 (nuScenes VLM rows) |
| `datasets/nuscenes/cluster_bootstrap_cis.json` | Table 13 footnote (Appendix I) |
| `datasets/vcdb/vlm_bridge_*_results.json` | Table 6 (VCDB VLM rows) |
| `datasets/vcdb/vlm_probes_*_results.json` | Table 6 (VLM gen rows) |
| `datasets/hdd/vlm_generative_*_results.json` | Table 6 (HDD VLM gen rows) |
| `datasets/nuscenes/vlm_generative_*_results.json` | Table 6 (nuScenes VLM gen rows) |
| `datasets/hdd/residual_ablation_results.json` | Appendix (comparator ablation) |
| `datasets/hdd/cross_session_results.json` | Appendix G (cross-session) |
| `datasets/hdd/retrieval_protocol_results.json` | Appendix (retrieval metrics) |
| `datasets/vcdb/attack_suite_results.json` | Appendix (attack suite) |
| `datasets/vcdb/pair_scores_vjepa2.json` | Table 13 (Appendix I) |
| `datasets/vcdb/scramble_multiseed_results.json` | Appendix M (multi-seed) |
| `figures/vcdb_scramble_multiseed.png` | Figure 9 (Appendix M) |
| `datasets/hdd/left_right_results.json` | §3.2 (left-vs-right AP) |
| `datasets/hdd/optical_flow_results.json` | §3.2 (optical flow baseline) |
| `datasets/epic_kitchens/claude_probe_results.json` | §3.3, Table 4 (Claude probe) |
| `datasets/viclip_results.json` | Table 6, §5 (ViCLIP) |

## Paper Compilation

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

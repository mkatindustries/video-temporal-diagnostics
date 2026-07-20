# Diagnosing Temporal Sensitivity in Video Retrieval Pipelines

Diagnostic code for *Diagnosing Temporal Sensitivity in Video Retrieval Pipelines*. The project evaluates temporal signals for video deduplication and retrieval, including cases where semantic descriptors assign similar scores to recordings with different motion or ordering.

## Abstract

Scalable video retrieval often uses global descriptors that are insensitive to motion direction. This repository implements three diagnostics for locating that behavior: a temporal scramble gradient, a forward/reverse score under a declared comparator, and a controlled feature-by-comparator factorial evaluated on one shared pair set. Exact permutation invariance applies to symmetric comparisons of fixed independently encoded elements. Contextual video tokens and VLM outputs instead require empirical tests. Scores from cosine and DTW are comparator-specific and must not be put on one numerical scale.

The paper reports pooled pair-classification diagnostics across seven benchmarks. These are not standard query-wise retrieval metrics. The old HDD reranking results and unbalanced-chunk scramble results were withdrawn and replaced by corrected runs. Compact summaries and provenance are tracked under `results/`; the exact rerun jobs are under `slurm_jobs/`.

## The Problem

Two cyclists record themselves biking through New York City. Both pass through Central Park, producing similar frame-level embeddings. A naive semantic deduplication system flags them as duplicates, but they are entirely different videos.

```
Cyclist A: Harlem → Central Park → Financial District
Cyclist B: Chelsea → Central Park → Queens
```

We need signals that capture **direction of travel**, **temporal sequence**, and **motion patterns** to tell them apart.

## Current Evidence

Selected point estimates from the valid pair diagnostics:

| Task | Winner | Score |
|------|--------|-------|
| Copy detection (VCDB) | Chamfer/BoF | AP 0.989 |
| Reversal diagnostic (EPIC) | DINOv3 attention trajectory | DTW-derived s_rev 0.192 |
| Maneuver discrimination (HDD) | V-JEPA 2 temporal residual | AP 0.956 |
| Cross-dataset maneuver (nuScenes) | V-JEPA 2 encoder-sequence DTW | AP 0.867 |
| Scene retrieval (Nymeria) | BoF | AP 0.485 |
| Multi-domain retrieval (MUVR News) | Chamfer | AP 0.746 |
| VLM direct direction prompts (open models) | Prompt-dependent, near chance | 0.50--0.54 balanced accuracy |
| VLM integrity prompt (Qwen/Gemma) | Detects forward vs reverse | 0.879 / 0.845 balanced accuracy |
| LLM fixed-vector probes | Exploratory, no reliable evidence | best observed 0.560 across many configs |
| V-JEPA 2 encoder-sequence DTW (HDD) | Controlled comparator contrast | AP 0.942 |
| Directed BoT-to-DTW retrieval (HDD) | BoT beats encoder-sequence DTW globally | full-gallery mAP 0.256 vs. 0.177 |
| Top-1 retrieval error composition | Wrong-intersection errors dominate | at least 98.9% of errors across HDD/nuScenes methods |
| Balanced-chunk scramble (VCDB) | BoF / Chamfer / BoT remain flat | max std over 10 seeds 0.0101 |
| V-JEPA 2 reversal (EPIC) | Temporal residual under DTW | s_rev 0.0033 [0.0031, 0.0034] |

On the conditional HDD pair task, replacing pooled cosine with encoder-sequence DTW closes 89% of the observed BoT-to-residual AP gap as a descriptive point estimate; a paired intersection-cluster bootstrap estimates encoder-sequence DTW minus BoT at +0.117 [0.044, 0.127]. The result does not compose into global retrieval: encoder-sequence DTW has lower full-gallery mAP than BoT (0.177 vs. 0.256) and lowers AP and MRR throughout the rerank sweep. VLM findings are readout- and prompt-dependent: mean-pooled cosine changes little, sequence DTW detects changes, direct direction prompts are weak, and integrity prompts are much stronger.

## Methods

| Method | Signal | Diagnostic property |
|--------|--------|---------------------|
| **Attention Trajectories** | Spatial center-of-mass of DINOv3 attention maps via DTW | Sequence comparator |
| **Temporal Derivatives** | d(embedding)/d(frame) via DTW | Sequence comparator |
| **V-JEPA 2 Temporal Residual** | Prediction-error sequences via DTW | Sequence comparator |
| Bag-of-Frames | Mean CLS embedding cosine similarity | Exactly invariant for independent frame embeddings |
| Chamfer Similarity | Per-frame best-match average | Exactly invariant for a fixed frame set |
| V-JEPA 2 Bag-of-Tokens | Mean-pooled contextual encoder tokens | Empirically tested; not invariant by theorem |
| VLM Vision Pooled | Mean-pooled vision tower embeddings | Empirically tested |
| VLM Vision Seq DTW | Per-frame vision tower embeddings via DTW | Sequence comparator |
| VLM LLM Hidden State | Mean-pooled LLM hidden states | Empirically tested |

## Installation

```bash
pip install -e .                    # Core package
pip install -e '.[vlm]'            # + VLM experiment support
```

Requires Python 3.10+, CUDA GPU recommended.

## Diagnostic Toolkit

The scramble gradient, reversal test, and feature-by-comparator factorial are packaged as reusable evaluation components.

**Python API:**

```python
from video_retrieval.diagnostics import temporal_report

report = temporal_report(emb_a, emb_b, pairs, similarity_fn)
print(report["scramble_gradient"]["verdict"])  # "order-sensitive" or "no-detected-sensitivity"
```

**CLI:**

```bash
temporal-diag scramble-gradient \
    --embeddings-a features_a.pt --embeddings-b features_b.pt \
    --pairs pairs.csv --similarity cosine --k-values 1 4 16

temporal-diag s-rev --embeddings features.pt --similarity dtw

temporal-diag decompose \
    --baseline-embeddings baseline.pt --alternative-embeddings alternative.pt \
    --pairs pairs.csv --baseline-comparator cosine --alternative-comparator dtw

temporal-diag report \
    --embeddings-a features_a.pt --embeddings-b features_b.pt \
    --pairs pairs.csv --similarity cosine --output report.json
```

Embeddings are `{video_id: (T, D)}` dicts saved as `.pt` files. Pairs are CSVs with columns `id_a, id_b, label`.

## Quick Start

```bash
python experiments/eval_vcdb.py                  # VCDB copy detection benchmark
python experiments/eval_vcdb_scramble.py         # Temporal scramble sensitivity
python experiments/eval_vcdb_vlm_bridge.py       # VLM vision tower + LLM on VCDB
python experiments/eval_vcdb_vlm_probes.py       # VLM generative probes on VCDB
python experiments/eval_hdd_intersections.py     # HDD maneuver discrimination
python experiments/eval_hdd_vlm_bridge.py        # VLM vision tower + LLM on HDD
python experiments/eval_hdd_vlm_generative.py    # VLM generative probes on HDD
python experiments/eval_nuscenes_intersections.py # nuScenes cross-dataset validation
python experiments/eval_nuscenes_vlm_bridge.py   # VLM vision tower + LLM on nuScenes
python experiments/eval_nuscenes_vlm_generative.py # VLM generative probes on nuScenes
python experiments/eval_epic_temporal_order.py   # EPIC-Kitchens multi-VLM probes
python experiments/eval_epic_linear_probe.py     # Linear probe on LLM hidden states
python experiments/eval_mlp_probe.py              # MLP probe on LLM hidden states (GroupKFold)
python experiments/eval_epic_claude_probe.py     # Claude Opus 4.6 API probe
python experiments/eval_epic_gemini_probe.py     # Gemini 3.1 Pro API probe
python experiments/eval_hdd_left_vs_right.py     # Left-vs-right only HDD evaluation
python experiments/eval_hdd_encoder_seq.py       # V-JEPA 2 encoder-seq DTW ablation on HDD
python experiments/eval_hdd_fusion.py            # Held-out BoT/DTW global-retrieval fusion
python experiments/eval_hdd_optical_flow.py      # Optical flow (RAFT) baseline on HDD
python experiments/eval_vcdb_scramble_multiseed.py # Multi-seed scramble gradient
python experiments/eval_vcdb_scramble_raw.py     # Raw-frame scramble (V-JEPA 2 re-extraction)
python experiments/eval_vjepa2_vcdb_bootstrap.py # V-JEPA 2 VCDB bootstrap CIs
python experiments/eval_cluster_bootstrap.py     # Cluster-level block bootstrap CIs
python experiments/eval_viclip.py                # ViCLIP video-native baseline (all benchmarks)
python experiments/eval_hdd_tara.py              # TARA chiral-trained MLLM on HDD
python experiments/eval_hdd_pl_stitch.py         # PL-Stitch temporal ranking on HDD
python experiments/eval_hdd_ordered_maxsim.py    # OrderedMaxSim comparator ablation (DINOv3, HDD)
python experiments/eval_hdd_ordered_maxsim_vjepa2.py # OrderedMaxSim comparator ablation (V-JEPA 2, HDD)
python experiments/eval_vcdb_ordered_maxsim.py   # OrderedMaxSim on VCDB (both backbones, cached features)
python experiments/vcdb_violation_diagnostic.py  # Monotonicity violation count (positive pairs)
python experiments/vcdb_violation_pos_neg.py     # Violation diagnostic: positive vs negative pairs
```

## Benchmarks

| Dataset | Size | Task |
|---------|------|------|
| **VCDB** | 528 videos, 28 categories | Copy detection, reversal attack, scramble gradient |
| **Honda HDD** | 128 sessions, dashcam | Maneuver discrimination (left/right turn at same intersection) |
| **nuScenes** | 850 scenes, multi-sensor | Cross-dataset maneuver discrimination |
| **SSv2** | 400 clips, chiral template pairs | Cross-domain motion-direction retrieval |
| **EPIC-Kitchens-100** | 500 cooking sequences | Multi-VLM temporal order probes (generative + embedding) |
| **Nymeria** | Aria egocentric recordings | Activity scene retrieval |
| **MUVR** | Music/dance/news | Multi-domain video retrieval |

## Models

| Model | HuggingFace ID | Type |
|-------|----------------|------|
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Per-frame self-supervised (300M params, 1024-dim) |
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` | Video masked prediction (64 frames, 1024-dim) |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | VLM (Qwen2 backbone + native video) |
| Gemma 4 31B | `google/gemma-4-31B-it` | VLM (SigLIP vision + Gemma LLM) |
| LLaVA-Video 7B | `llava-hf/LLaVA-Video-7B-Qwen2-hf` | VLM (CLIP vision + Qwen2 LLM) |
| Claude Opus 4.6 | API-only (`claude-4-6-opus-genai`) | Proprietary VLM (generative probe only) |
| Gemini 3.1 Pro | API-only (`gemini-3-1-pro-preview-genai`) | Proprietary reasoning VLM (generative probe only) |
| ViCLIP ViT-L | `OpenGVLab/ViCLIP` (local weights) | Video-native contrastive (InternVid-10M, 768-dim) |
| TARA (Tarsier-7B) | `bpiyush/TARA` (local weights) | Chiral-trained MLLM (16 frames, 4096-dim) |
| PL-Stitch ViT-B | `visurg/PL-Stitch` (`pl_lemon.pth`) | Temporal ranking pretrained (per-frame, 768-dim) |

## Paper

LaTeX draft in `paper/paper.tex`. Title: *"Diagnosing Temporal Sensitivity in Video Retrieval Pipelines"*

The Video4Real extended abstract is `paper/video4real.tex`. Build it from the repository root:

```bash
make video4real
```

For a clean rebuild, run `make clean-video4real video4real`.

## License

This code is released under the [MIT License](LICENSE). Note that the datasets and model weights used in experiments carry their own licenses:

| Asset | License |
|-------|---------|
| nuScenes | CC-BY-NC-SA 4.0 |
| EPIC-Kitchens-100 | Non-commercial academic |
| VCDB, Honda HDD | Research-only |
| Nymeria, MUVR | Research-only |
| V-JEPA 2 (Meta) | CC-BY-NC 4.0 |
| DINOv3, Gemma 4, Qwen3-VL, LLaVA-Video | See respective model cards |

Users must independently obtain datasets and model weights under their respective licenses.

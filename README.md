# The Temporal Blind Spot in Video Retrieval

Diagnostic code for *The Temporal Blind Spot in Video Retrieval: Diagnosing Temporal Sensitivity*. Video deduplication and retrieval using **non-semantic temporal signals**. Semantic embeddings alone can't distinguish videos that share content but represent different recordings — this project explores signals that capture *how* a video moves through content space, not just *what* it contains.

## The Problem

Two cyclists record themselves biking through New York City. Both pass through Central Park, producing similar frame-level embeddings. A naive semantic deduplication system flags them as duplicates — but they're entirely different videos.

```
Cyclist A: Harlem → Central Park → Financial District
Cyclist B: Chelsea → Central Park → Queens
```

We need signals that capture **direction of travel**, **temporal sequence**, and **motion patterns** to tell them apart.

## Core Finding: The Temporal Blind Spot

No single method works across all retrieval paradigms — this is the "Triad" thesis:

| Task | Winner | Score |
|------|--------|-------|
| Copy detection (VCDB) | Chamfer/BoF | AP 0.989 |
| Reversal detection (VCDB) | DINOv3 attention trajectory | s_rev 0.192 |
| Maneuver discrimination (HDD) | V-JEPA 2 temporal residual | AP 0.956 |
| Cross-dataset maneuver (nuScenes) | V-JEPA 2 temporal residual | AP 0.815 |
| Scene retrieval (Nymeria) | BoF | AP 0.485 |
| Multi-domain retrieval (MUVR News) | Chamfer | AP 0.746 |
| VLM reversal (3 VLMs, EPIC-Kitchens) | All at chance | 0.50--0.54 bal. acc. |
| VLM reversal (Claude 4.6 Opus, EPIC) | At chance | ~0.52 bal. acc. |
| VLM reversal (Gemini 3.1 Pro, EPIC) | Marginal | ~0.60 bal. acc. |
| VLM vision tower s_rev | Order-invariant | ~1.0 |
| VLM vision sequence DTW s_rev | Order-sensitive | ~0.49 |
| VLM LLM hidden state (VCDB) | Strong copy detector | AP 0.84--0.92 |
| VLM LLM hidden state (HDD) | At chance | AP ~0.51 |
| LLM hidden state probes (linear + MLP) | All within chance | best 0.560 (126 configs) |
| Optical flow RAFT (HDD) | Below chance | AP 0.478 |
| HDD left-vs-right only | V-JEPA 2 temporal residual | AP 0.960 |
| V-JEPA 2 Encoder-Seq DTW (HDD) | Disentangles feature vs comparator | AP 0.942 |
| Raw-frame scramble (V-JEPA 2, VCDB) | Monotonic degradation | AP 0.705→0.652 |
| ViCLIP (InternVid-10M, all benchmarks) | Order-invariant | VCDB 0.907, HDD 0.542, s_rev 0.996 |

Order-invariant methods (bag-of-frames, Chamfer) excel at copy detection but are completely blind to temporal manipulation (reversal, scrambling). Order-aware methods (attention trajectory DTW, temporal derivatives) detect manipulation but sacrifice copy detection accuracy. VLM vision towers contain per-frame temporal signal but destroy it through pooling — no standard readout from the LLM backbone recovers it (126 linear + MLP probe configurations at chance via GroupKFold CV). Yet per-position hidden states *do* retain order under full-sequence DTW — symmetric aggregation, not the LLM backbone itself, is the bottleneck. A frontier proprietary model (Claude 4.6 Opus) fares no better than 7B open-weight models; a reasoning model (Gemini 3.1 Pro) shows marginal improvement (~0.60 balanced accuracy) but remains far from reliable. On HDD, an encoder-sequence DTW baseline (AP=0.942) shows that most of the bag-of-tokens→residual gap comes from the comparator (cosine→DTW), though the residual adds a further 1.4 points. No existing method spans both the copy detection and motion retrieval regimes.

## Methods

| Method | Signal | Order-Aware? |
|--------|--------|:------------:|
| **Attention Trajectories** | Spatial center-of-mass of DINOv3 attention maps via DTW | Yes |
| **Temporal Derivatives** | d(embedding)/d(frame) via DTW | Yes |
| **V-JEPA 2 Temporal Residual** | Prediction error sequences via DTW | Yes |
| Bag-of-Frames | Mean CLS embedding cosine similarity | No |
| Chamfer Similarity | Per-frame best-match average | No |
| V-JEPA 2 Bag-of-Tokens | Mean-pooled encoder tokens | No |
| VLM Vision Pooled | Mean-pooled vision tower embeddings (SigLIP/CLIP) | No |
| VLM Vision Seq DTW | Per-frame vision tower embeddings via DTW | Yes |
| VLM LLM Hidden State | Mean-pooled LLM hidden states | No |

## Installation

```bash
pip install -e .                    # Core package
pip install -e '.[vlm]'            # + VLM experiment support
```

Requires Python 3.10+, CUDA GPU recommended.

## Diagnostic Toolkit

The `temporal-diag` CLI packages the scramble gradient and reversal test as reusable evaluation components (see paper Section 5).

**Python API:**

```python
from video_retrieval.diagnostics import temporal_report

report = temporal_report(emb_a, emb_b, pairs, similarity_fn)
print(report["scramble_gradient"]["verdict"])  # "order-sensitive" or "order-invariant"
```

**CLI:**

```bash
temporal-diag scramble-gradient \
    --embeddings-a features_a.pt --embeddings-b features_b.pt \
    --pairs pairs.csv --similarity cosine --k-values 1 4 16

temporal-diag s-rev --embeddings features.pt --similarity dtw

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
python experiments/eval_epic_claude_probe.py     # Claude 4.6 Opus API probe
python experiments/eval_epic_gemini_probe.py     # Gemini 3.1 Pro API probe
python experiments/eval_hdd_left_vs_right.py     # Left-vs-right only HDD evaluation
python experiments/eval_hdd_encoder_seq.py       # V-JEPA 2 encoder-seq DTW ablation on HDD
python experiments/eval_hdd_optical_flow.py      # Optical flow (RAFT) baseline on HDD
python experiments/eval_vcdb_scramble_multiseed.py # Multi-seed scramble gradient
python experiments/eval_vcdb_scramble_raw.py     # Raw-frame scramble (V-JEPA 2 re-extraction)
python experiments/eval_vjepa2_vcdb_bootstrap.py # V-JEPA 2 VCDB bootstrap CIs
python experiments/eval_cluster_bootstrap.py     # Cluster-level block bootstrap CIs
python experiments/eval_viclip.py                # ViCLIP video-native baseline (all benchmarks)
```

## Benchmarks

| Dataset | Size | Task |
|---------|------|------|
| **VCDB** | 528 videos, 28 categories | Copy detection, reversal attack, scramble gradient |
| **Honda HDD** | 128 sessions, dashcam | Maneuver discrimination (left/right turn at same intersection) |
| **nuScenes** | 850 scenes, multi-sensor | Cross-dataset maneuver discrimination |
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
| Claude 4.6 Opus | API-only (`claude-4-6-opus-genai`) | Proprietary VLM (generative probe only) |
| Gemini 3.1 Pro | API-only (`gemini-3-1-pro-preview-genai`) | Proprietary reasoning VLM (generative probe only) |
| ViCLIP ViT-L | `OpenGVLab/ViCLIP` (local weights) | Video-native contrastive (InternVid-10M, 768-dim) |

## Paper

LaTeX draft in `paper/paper.tex`. Title: *"The Temporal Blind Spot in Video Retrieval: Diagnosing Temporal Sensitivity"*

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

# When Conditional Sequence Matching Does Not Transfer to Global Video Retrieval

Evaluation code and diagnostics for *When Conditional Sequence Matching Does Not Transfer to
Global Video Retrieval*, Arjang Talattof — accepted for presentation at the
[Video4Real workshop](https://sites.google.com/utwente.nl/video4real/home) at ECCV 2026
(Malmö, 9 September 2026). The extended abstract is `paper/video4real.tex`; its checked-in TeX
still uses the anonymous review style. The poster is under `poster/`.

The current poster has three deliberately separate evidence blocks: a six-recipe public-facing
slice of a complete 13-recipe × 5-cohort DRT production scorecard, two source-backed pooled
vision-tower baselines, and the matched conditional/global retrieval diagnostic from the extended
abstract. It builds from self-contained tracked snapshots plus the corrected result JSONs under
`results/`; the upstream raw DRT artifact is not included in this repository.

The project evaluates temporal signals for video deduplication and retrieval, including cases
where semantic descriptors assign similar scores to recordings with different motion or ordering.

## Abstract

Scalable video retrieval favors pooled descriptors, but pooling can blur motion direction.
Sequence comparators such as dynamic time warping (DTW) over per-frame features can distinguish
maneuvers *within* a known location. We ask whether this within-location advantage survives when
the search spans many locations. Under a matched query-wise protocol, V-JEPA 2 encoder-sequence
DTW beats a bag-of-tokens (BoT) cosine baseline within intersections (Honda HDD 0.955 vs. 0.923
mAP; nuScenes 0.922 vs. 0.855), yet loses over the full evaluation gallery (HDD 0.177 vs. 0.256;
nuScenes 0.159 vs. 0.333). Separate order-ablation controls do not support a general
order-specific explanation: intact DTW detectably beats shuffled DTW only for nuScenes encoder
features; order-free assignment differs detectably only once, improving nuScenes
temporal-residual AP. Across all six top-1 rankings, at least 98.9% of errors come from the wrong
intersection. A BoT→DTW cascade lowers AP at every evaluated k; leakage-safe fusion shows no
detected gain on HDD (+0.001, 95% CI [−0.003, 0.004]) and equals BoT on nuScenes. For these
datasets, scores, and a linear fusion rule, fine-grained sequence matching is useful conditionally
but insufficient for global retrieval. We release the diagnostics and evaluation protocol.

### Protocols are not interchangeable

`paper/` also carries a longer companion manuscript covering the same code base across seven
benchmarks. It is not under review. It reports pooled pair-classification diagnostics, which are
not standard query-wise retrieval metrics, so its numbers and the extended abstract's are not
comparable. The extended abstract instead headlines a matched query-wise protocol on HDD/nuScenes
(query-macro mAP over the same eligible-query set for both conditional and global retrieval),
reserving the pooled-pair protocol for its shuffled-DTW/assignment order controls. A separate
SoccerNet-v2 within-match transfer check uses its own match-macro MRR protocol. None of these
protocols share a numerical scale and should not be compared directly.

The old HDD reranking results and unbalanced-chunk scramble results were withdrawn and replaced
by corrected runs. Compact summaries and provenance are tracked under `results/`; the exact
rerun jobs are under `slurm_jobs/`.

The poster's DRT scorecard is a separate production evaluation, not an extension of the paper's
matched retrieval experiment. Its five columns use different tasks and metrics and must not be
compared numerically across columns. In particular, its SoccerNet-v2 lane uses one normalized clip
vector with cosine similarity (6,376 same-half headline queries plus 28 cross-half queries), while
the paper's SoccerNet-v2 transfer check uses sequence DTW and BoT on 6,277 same-half queries. The
two SoccerNet-v2 result sets are both retained, but their values and protocol qualifications are
not interchangeable.

## Current Poster

The complete DRT source scorecard contains 65/65 measured cells (13 fixed recipes on five cohorts).
The poster shows six public-facing recipes—InternVideo-Next L, SAM3 Perception Encoder, Internal
copy detector, DINOv3, V-JEPA 2, and LeVJEPA—and recomputes ranks only among those six. The cohorts
are Internal synthetic copies (best F1), VCDB (copy AP), SoccerNet-v2 (match-macro MRR), Honda HDD
(global-pair AP), and Project Aria (false-merge rate; lower is better).

`poster/production_scorecard.json` is the self-contained scorecard snapshot, including source
hashes, selection rules, and protocol boundaries. Its point estimates do not have paired
confidence intervals. `poster/large_model_baselines.json` supplies a separate inset for pooled
Gemma 4 and LLaVA-Video vision-tower embeddings; those are not end-to-end language-model results
and do not participate in the scorecard ranks. The conditional/global diagnostic is independently
loaded from `results/hdd/` and `results/nuscenes/`.

## The Problem

Two cyclists record themselves biking through New York City. Both pass through Central Park, producing similar frame-level embeddings. A naive semantic deduplication system flags them as duplicates, but they are entirely different videos.

```
Cyclist A: Harlem → Central Park → Financial District
Cyclist B: Chelsea → Central Park → Queens
```

We need signals that capture **direction of travel**, **temporal sequence**, and **motion patterns** to tell them apart.

## Current Evidence

Selected findings from the project's distinct evaluation protocols are listed below. Values from
different rows are not necessarily on a common scale.

| Evaluation | Selected method or finding | Value and status |
|------------|----------------------------|------------------|
| Copy detection (VCDB) | DINOv3 Chamfer; BoF is 0.979 | AP 0.989 |
| Reversal diagnostic (EPIC) | DINOv3 attention trajectory | DTW-derived s_rev 0.192 |
| Maneuver discrimination (HDD) | V-JEPA 2 temporal residual | AP 0.956 |
| Cross-dataset maneuver (nuScenes) | V-JEPA 2 encoder-sequence DTW, the primary-method leader | AP 0.863; order-free assignment control 0.875 |
| Scene retrieval (Nymeria) | BoF | AP 0.485; manuscript/figure only, with no tracked numeric result JSON |
| Multi-domain retrieval (MUVR News) | Chamfer | AP 0.746; manuscript/figure only, with no tracked numeric result JSON |
| VLM direct direction prompts (open models) | Prompt-dependent, near chance | 0.503--0.538 balanced accuracy; aggregate transcription without raw responses |
| VLM integrity prompt (Qwen/Gemma) | **Withdrawn** — transcription inverted three of four conditions | see the companion manuscript's "VLM Temporal Integrity Probe (Withdrawn)" appendix |
| LLM fixed-vector probes | Exploratory, no reliable evidence | best observed 0.560 across many configs; manuscript-reported artifact is absent from this checkout |
| V-JEPA 2 encoder-sequence DTW (HDD) | Controlled comparator contrast | AP 0.942 |
| Directed BoT-to-DTW retrieval (HDD) | BoT beats encoder-sequence DTW globally | full-gallery mAP 0.256 vs. 0.177 |
| Top-1 retrieval error composition | Wrong-intersection errors dominate | at least 98.9% of errors across HDD/nuScenes methods |
| Balanced-chunk scramble (VCDB) | BoF / Chamfer / BoT remain flat | max std over 10 seeds 0.0101 |
| V-JEPA 2 reversal (EPIC) | Temporal residual under DTW | s_rev 0.0033 [0.0031, 0.0034] |
| Within-match transfer check (SoccerNet-v2, Video4Real) | No detected gain over BoT | match-macro MRR 0.155 vs. 0.153 |

On the pooled HDD pair diagnostic, replacing pooled cosine with encoder-sequence DTW closes 89% of the observed BoT-to-residual AP gap as a descriptive point estimate; a paired intersection-cluster bootstrap estimates encoder-sequence DTW minus BoT at +0.117 [0.044, 0.127]. The result does not compose into global retrieval: encoder-sequence DTW has lower full-gallery mAP than BoT (0.177 vs. 0.256) and lowers AP and MRR throughout the rerank sweep. VLM findings are readout- and prompt-dependent: mean-pooled cosine changes little, sequence DTW detects changes, and direct direction prompts are weak with answer priors that shift strongly across phrasings. The INTACT/TAMPERED integrity-prompt result previously reported here is **withdrawn**: its transcribed aggregate applied a uniform `1 - accuracy` conversion to all four conditions, which is correct only for the forward condition, and the raw per-clip responses were not retained. Re-run with `slurm_jobs/rerun_epic_integrity.sbatch` to restore it.

The corrected HDD, nuScenes, VCDB scramble, EPIC reversal, and SoccerNet-v2 compact results have
tracked machine-readable artifacts and provenance under `results/`. The qualifications in the
table above are intentional for results that currently survive only in manuscript text, figures,
or aggregate transcriptions.

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
pip install -e '.[poster]'         # + poster rendering
pip install -e '.[dev]'            # + tests and development tools
```

Requires Python 3.10+. The diagnostic toolkit and poster build can run on CPU; most experiment
entry points default to CUDA, and the VLM experiments generally need one or two GPUs. Datasets,
feature caches, and model weights are not included. VLM adapters load weights from the local cache,
and exact reruns should use the model revisions and environment recorded in
`results/PROVENANCE.md` rather than relying only on the package's minimum dependency bounds.

## Diagnostic Toolkit

The scramble gradient, reversal test, and feature-by-comparator factorial are packaged as reusable evaluation components.

**Python API:**

```python
from video_retrieval.diagnostics import temporal_report

report = temporal_report(emb_a, emb_b, pairs, similarity_fn)
print(report["scramble_gradient"]["verdict"])
# "order-sensitive", "no-detected-sensitivity", or "inconclusive"
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

## Experiment Workflows

Experiment scripts are not zero-configuration commands: they require locally obtained datasets,
and many require feature caches, model weights, or explicit model-family flags. The commands below
show the minimum dataset arguments for representative evaluations. See `REPRODUCIBILITY.md` for
the authoritative command sequence, expected artifacts, resource requirements, and superseded-run
warnings.

For VCDB, `/path/to/vcdb` below means the directory containing both `annotation/` and
`core_dataset/`. The EPIC command shown runs the encoder temporal-order evaluation; add an explicit
`--vlm-family` and `--vlm-embeddings` and/or `--vlm-generative` for a VLM run.

```bash
python experiments/eval_vcdb.py \
    --vcdb-dir /path/to/vcdb

python experiments/eval_vcdb_reversal.py \
    --vcdb-dir /path/to/vcdb

python experiments/eval_hdd_intersections.py \
    --hdd-dir /path/to/hdd

python experiments/eval_nuscenes_intersections.py \
    --nuscenes-dir /path/to/nuscenes --version v1.0-trainval

python experiments/eval_epic_temporal_order.py \
    --epic-dir /path/to/epic_kitchens --max-sequences 500

python experiments/eval_ssv2_motion_direction.py \
    --manifest /path/to/ssv2/validation_manifest.json

python experiments/eval_nymeria_activities.py \
    --nymeria-dir /path/to/nymeria

python experiments/eval_muvr.py \
    --partition news --muvr-dir /path/to/muvr
```

The matched conditional/global headline is reproduced by the two fusion jobs; each also invokes
`eval_conditional_querywise.py` on the matching score cache:

```bash
HDD_DIR=/path/to/hdd sbatch slurm_jobs/rerun_hdd_fusion.sbatch
NUSCENES_DIR=/path/to/nuscenes sbatch slurm_jobs/rerun_nuscenes_fusion.sbatch
```

SoccerNet-v2 is a gated, multi-step workflow with a manually reviewed train-only canary; follow
`REPRODUCIBILITY.md` rather than invoking its extraction and evaluation scripts without arguments.
The old `eval_vcdb_scramble.py` is retained only as a historical single-permutation implementation;
the valid result uses `eval_vcdb_scramble_multiseed.py` plus `eval_vcdb_scramble_raw.py`.

## Benchmarks

| Dataset | Evaluation scope | Task |
|---------|------------------|------|
| **VCDB** | 528 core videos; the poster production snapshot has 527 successfully encoded videos | Copy detection, reversal attack, scramble gradient |
| **Honda HDD** | 128 sessions; 1,687 retrieval segments in the corrected run | Maneuver discrimination and global retrieval |
| **nuScenes** | v1.0-trainval; 244 segments and 197 eligible directed queries in the corrected run | Cross-dataset maneuver discrimination and global retrieval |
| **SSv2** | Selected split of 400 clips in chiral template pairs | Cross-domain motion-direction retrieval |
| **EPIC-Kitchens-100** | 500 sequences from 19 source videos | Multi-VLM temporal-order probes |
| **Nymeria** | 100 sessions, 8,780 segments | Activity scene retrieval |
| **MUVR News** | 9,958 videos, 74 topics | Multi-domain video retrieval |
| **SoccerNet-v2** | 100 test matches; protocol-specific query counts are stated above | Within-match replay-grounding event retrieval |
| **Project Aria** | 78,324 confirmed-negative pairs in the poster snapshot | False-merge evaluation at transferred thresholds |
| **Internal synthetic copies** | Controlled synthetic-copy evaluation; row count withheld | Copy-detection best F1 in the poster scorecard |

## Models

| Model | Identifier or source | Type |
|-------|----------------------|------|
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Per-frame self-supervised (300M params, 1024-dim) |
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` | Video masked prediction (64 frames, 1024-dim) |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | Qwen3-VL native-video model |
| Gemma 4 31B | `google/gemma-4-31B-it` | VLM (SigLIP vision + Gemma LLM) |
| LLaVA-Video 7B | `llava-hf/LLaVA-Video-7B-Qwen2-hf` | VLM (CLIP vision + Qwen2 LLM) |
| Claude Opus 4.6 | API alias used by the experiment script (`claude-4-6-opus-genai`) | Proprietary VLM (generative probe only) |
| Gemini 3.1 Pro | API alias used by the experiment script (`gemini-3-1-pro-preview-genai`) | Proprietary reasoning VLM (generative probe only) |
| ViCLIP ViT-L | local OpenGVLab/ViCLIP checkout and weights | Video-native contrastive (InternVid-10M, 768-dim) |
| TARA (Tarsier-7B) | local TARA checkout and weights | Chiral-trained MLLM (16 frames, 4096-dim) |
| PL-Stitch ViT-B | local PL-Stitch checkout and `pl_lemon.pth` | Temporal ranking pretrained (per-frame, 768-dim) |

The poster scorecard additionally uses InternVideo-Next L, SAM3 Perception Encoder, Internal copy
detector, and LeVJEPA under their public-facing names. See `poster/production_scorecard.json` for
the exact selected recipes and per-cohort protocol qualifications.

## Paper and poster

- `paper/video4real.tex` — *"When Conditional Sequence Matching Does Not Transfer to Global Video
  Retrieval,"* Arjang Talattof. Extended abstract, accepted for presentation at
  [Video4Real](https://sites.google.com/utwente.nl/video4real/home) at ECCV 2026 (Malmö,
  9 September 2026). Per the workshop call, accepted abstracts are excluded from the ECCV
  proceedings. The checked-in TeX remains in anonymous review format.
- `poster/` — the 36 × 24 inch workshop poster described above. See `poster/README.md` for
  provenance, print specifications, and preflight checks.
- `paper/neurips.tex` — *"Diagnosing Temporal Sensitivity in Video Retrieval Pipelines."* Longer
  companion manuscript over seven benchmarks. It is not under review and currently uses anonymous
  draft formatting. Some appendix artifacts are not reproducible from this checkout; see the
  caveats in `REPRODUCIBILITY.md`.

Build both manuscripts from the repository root:

```bash
make papers
```

Individual targets are `make video4real` and `make neurips`. For a clean rebuild,
run `make clean-papers papers`. These targets require a working LaTeX installation.

Build the poster from the repository root (no GPU, datasets, or LaTeX required):

```bash
python poster/charts.py
python poster/build_poster.py
```

The final print file is `poster/build/video4real_poster_36x24in.pdf`; generated files under
`poster/build/` are intentionally ignored by Git.

## License

This code is released under the [MIT License](LICENSE). Note that the datasets and model weights used in experiments carry their own licenses:

| Asset | License |
|-------|---------|
| nuScenes | CC-BY-NC-SA 4.0 |
| EPIC-Kitchens-100 | CC BY-NC 4.0 |
| VCDB | Research-only; no warranty (Fudan University) |
| Honda HDD | HRI-USA Data Use Agreement (non-commercial) |
| Something-Something V2 | See the official dataset terms |
| Nymeria | CC BY-NC 4.0 |
| MUVR | Research-only (authors' terms) |
| SoccerNet-v2 | Non-commercial academic (registration/EULA required) |
| Project Aria | See the official dataset terms |
| V-JEPA 2 (Meta) | CC-BY-NC 4.0 |
| DINOv3 | Meta FAIR non-commercial license |
| Qwen3-VL-8B | Apache 2.0 |
| Gemma 4 | Gemma Terms of Use |
| LLaVA-Video | Apache 2.0 code; Llama 2 terms for weights |
| ViCLIP / InternVid | Apache 2.0 code; CC BY-NC 4.0 weights |
| Other model weights and hosted APIs | See their respective distribution terms and model cards |

Users must independently obtain datasets and model weights under their respective licenses.

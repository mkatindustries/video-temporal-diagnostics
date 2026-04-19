# TemporalDiag

Diagnostic toolkit for temporal sensitivity in video retrieval embeddings.

Two complementary probes, usable from Python or the `temporal-diag` CLI:

- **Scramble gradient** — chunk-shuffle query-side embeddings at increasing `K`
  and re-score. Flat AP across `K` ⇒ order-invariant. Monotonic drop ⇒
  sequence-aware.
- **Reversal sensitivity (`s_rev`)** — similarity between a clip and its
  time-reversed copy. `s_rev ≈ 1.0` ⇒ temporally blind.

Both probes operate on pre-extracted embeddings, so the toolkit is
encoder-agnostic: DINOv3, V-JEPA 2, ViCLIP, or any VLM vision tower output will
work as long as you can produce a `(T, D)` tensor per video.

## Install

Editable install from the repo root:

```bash
pip install -e .
```

This registers the `temporal-diag` CLI (see `pyproject.toml`).

## Python API

```python
from video_retrieval.diagnostics import (
    compute_s_rev,
    scramble_gradient,
    temporal_report,
)

# embeddings_a, embeddings_b: {video_id: Tensor of shape (T, D)}
# pairs: [(id_a, id_b, label), ...] where label ∈ {0, 1}
# similarity_fn: (Tensor, Tensor) -> float

report = temporal_report(
    embeddings_a, embeddings_b, pairs,
    similarity_fn=my_cosine_fn,
    k_values=[1, 4, 16],
)
# report["scramble_gradient"]["ap_scores"]     -> [AP@1, AP@4, AP@16]
# report["scramble_gradient"]["verdict"]       -> "order-invariant" | "order-sensitive"
# report["reversal_sensitivity"]["mean"]       -> mean s_rev
```

To run either probe in isolation:

```python
sr = compute_s_rev(embeddings, similarity_fn)
sg = scramble_gradient(embeddings_a, embeddings_b, pairs, similarity_fn)
```

## CLI

Embeddings are expected as a `.pt` file containing a `{video_id: (T, D) Tensor}`
dict (load via `torch.load`, `weights_only=True`). Pairs are a CSV with columns
`id_a,id_b,label`.

```bash
# Full report (scramble gradient + s_rev)
temporal-diag report \
    --embeddings-a features_a.pt \
    --embeddings-b features_b.pt \
    --pairs pairs.csv \
    --similarity cosine \
    --k-values 1 4 16 \
    --output report.json

# Just the scramble gradient
temporal-diag scramble-gradient \
    --embeddings-a features_a.pt --embeddings-b features_b.pt \
    --pairs pairs.csv --similarity dtw

# Just s_rev
temporal-diag s-rev --embeddings features.pt --similarity cosine
```

Built-in similarity functions: `cosine` (mean-pooled cosine), `dtw`
(`exp(-dtw_distance)` on the `(T, D)` sequence). For a custom comparator, use
the Python API and pass your own callable.

## Interpreting the results

| Signal | Order-invariant method | Order-sensitive method |
|---|---|---|
| `ap_scores` across K | flat (≈ constant) | monotonic drop |
| `verdict` | `"order-invariant"` | `"order-sensitive"` (AP drop > 0.05) |
| `s_rev` (pooled / cosine) | ≈ 1.0 | < 1.0 |
| `s_rev` (sequence / DTW) | ≈ 1.0 | well below 1.0 |

The scramble and reversal probes use different similarity functions
(cosine vs. DTW), so `s_rev` values are not directly comparable across probe
families — see Eq. (1) in the paper.

## Reproducing paper numbers

The VCDB scramble gradient figure and the `s_rev` rows of the cross-method
summary table were produced by these same functions, called from scripts under
`experiments/`. See `REPRODUCIBILITY.md` at the repo root for the end-to-end
pipeline (data download, embedding extraction, evaluation).

## Citation

If you use this toolkit, please cite:

```
@article{talattof2026temporal,
  title  = {The Temporal Blind Spot in Video Retrieval: Diagnosing Temporal Sensitivity},
  author = {Talattof, Arjang},
  year   = {2026}
}
```

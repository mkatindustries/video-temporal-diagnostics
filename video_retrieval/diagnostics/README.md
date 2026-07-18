# TemporalDiag

Diagnostic toolkit for temporal sensitivity in video retrieval embeddings.

Three complementary diagnostics, usable from Python or the `temporal-diag` CLI:

- **Scramble gradient** — chunk-shuffle query-side embeddings into near-equal
  chunks at increasing `K` and re-score. A flat curve means this intervention
  detected no sensitivity; it does not prove invariance.
- **Reversal sensitivity (`s_rev`)** — similarity between a clip and its
  time-reversed copy under a declared comparator. Values are descriptive on
  that comparator's scale.
- **Feature-by-comparator factorial** — evaluate multiple feature sets and
  comparators on one shared pair set. This controls pair composition but does
  not by itself establish causal attribution.

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
    feature_comparator_decomposition,
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
# report["scramble_gradient"]["verdict"]       -> "no-detected-sensitivity" | "order-sensitive"
# report["reversal_sensitivity"]["mean"]       -> mean s_rev
```

To run either probe in isolation:

```python
sr = compute_s_rev(embeddings, similarity_fn)
sg = scramble_gradient(embeddings_a, embeddings_b, pairs, similarity_fn)
factorial = feature_comparator_decomposition(
    {"baseline": embeddings_a, "alternative": embeddings_b},
    {"cosine": cosine_fn, "dtw": dtw_fn},
    pairs,
)
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

# Feature-by-comparator factorial
temporal-diag decompose \
    --baseline-embeddings baseline.pt \
    --alternative-embeddings alternative.pt \
    --pairs pairs.csv \
    --baseline-comparator cosine \
    --alternative-comparator dtw
```

Built-in similarity functions are `cosine` (mean-pooled cosine), `chamfer`
(symmetric maximum cosine), and `dtw` (`exp(-dtw_distance)` on the `(T, D)`
sequence). For a custom comparator, use the Python API and pass a callable.

## Interpreting the results

| Signal | No detected sensitivity | Detected sensitivity |
|---|---|---|
| `ap_scores` across K | flat (≈ constant) | monotonic drop |
| `verdict` | `"no-detected-sensitivity"` | `"order-sensitive"` (AP drop > 0.05) |
| `s_rev` (pooled / cosine) | ≈ 1.0 | < 1.0 |
| `s_rev` (sequence / DTW) | ≈ 1.0 | well below 1.0 |

Cosine and DTW `s_rev` values are not directly comparable. DTW preprocessing
and the exponential scale parameter also affect the number, so rank only
results produced by the same probe definition.

## Reproducing paper numbers

Corrected VCDB scramble, EPIC residual, HDD retrieval, and paired-bootstrap
summaries are tracked under `results/`, with provenance in
`results/PROVENANCE.md`. See `REPRODUCIBILITY.md` for data, cache, and output
paths and `slurm_jobs/` for the exact rerun jobs.

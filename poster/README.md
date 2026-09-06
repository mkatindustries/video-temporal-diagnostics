# Video4Real @ ECCV 2026 poster

Poster for *When Conditional Sequence Matching Does Not Transfer to Global Video
Retrieval*. Its main result
is a six-recipe public-safe slice of the complete 13-by-5 DRT production-recipe
scorecard; section 2 is a separate matched conditional/global retrieval diagnostic from
`paper/video4real.tex`. A separate top inset reports the two source-backed
large-model vision-tower baselines. Workshop: Wednesday 9 September 2026, PM.

## Build

The poster needs `reportlab`, `qrcode`, and Pillow on top of the base install. They are
declared as the `poster` extra, so from an activated `video_retrieval` env:

```bash
pip install -e ".[poster]"
```

`pip install -e .` and `pip install -e ".[vlm]"` do **not** cover them — the
poster extra is separate.

If you would rather install through conda:

```bash
conda install -n video_retrieval --freeze-installed -c conda-forge reportlab qrcode pillow
```

Keep `--freeze-installed` to avoid opportunistic updates to packages already in
the environment. Poster generation consumes frozen JSON summaries; activating
this environment does not imply that every upstream artifact was produced with
its current package state. `results/PROVENANCE.md` records the environments for
the tracked diagnostic runs.

Then, from the repository root:

```bash
conda activate video_retrieval
python poster/charts.py        # -> p_scorecard.png + p_diagnostic_pair.png at 300 dpi
python poster/build_poster.py  # -> poster/build/video4real_poster_36x24in.pdf
```

Both scripts resolve data and output paths from `__file__`; the commands above
still use relative script paths and therefore assume the repository root. From
another directory, invoke each script by its absolute path. No GPU, dataset, or
LaTeX build is needed. The scorecard and large-model inset read render-ready
snapshots in `poster/`; the separate conditional/global figure reads result JSONs
under `results/`.

The page is drawn directly with ReportLab, which gives exact millimetre control
over trim, bleed, and crop marks; the poster build does not depend on the paper's
LaTeX toolchain.

`production_scorecard.json` records the immutable source hash, source identity,
completion-bundle hash, public display labels, display selection, rank rules, and
protocol qualifications. Its upstream raw artifact is not part of this
repository. The scorecard and matched diagnostic are intentionally not
numerically combined. The snapshots are self-contained for rendering, but not
for re-deriving every value: some upstream scorecard, vision-tower, narrative,
and controlled-follow-up evidence is represented only by source identifiers and
hashes rather than tracked raw artifacts.

## Print specification

The current build targets a standard North American 36 × 24 inch landscape
poster. It is reflowed natively at 3:2; it is not a scaled version of the earlier
1400 × 1000 mm artwork.

| Requirement | Status |
|---|---|
| 1:1 scale, 36 × 24 inch landscape | explicit 914.4 × 609.6 mm CropBox and TrimBox |
| 5 mm bleed | explicit 924.4 × 619.6 mm BleedBox (5 mm each edge) |
| Crop marks | 12 mm marks, offset 3 mm beyond the bleed, in a 956.4 × 651.6 mm MediaBox |
| Fonts embedded or outlined | 3 subset-embedded TrueType faces, no base-14 refs |
| Images ≥ 100 DPI at 1:1 | charts at 300 DPI; logo and QR exceed 300 effective DPI |
| CMYK (Fogra 39) | **not applied — see below** |

**Colour.** The PDF is RGB. A faithful Fogra 39 conversion needs that ICC
profile, which is not available in this environment; converting without it would
shift every hue. Give the print shop the RGB PDF and let them convert with the
correct profile — that is the normal workflow and yields a better result than a
blind conversion.

**Print handling.** The CropBox and TrimBox are the exact 36 × 24 inch finished
size. The MediaBox is 37.65 × 25.65 inches because it retains bleed, crop marks,
and their surrounding slug. A shop using the full MediaBox therefore needs
larger stock or roll paper and should trim to the TrimBox at 100%. For direct
output on exact 36 × 24 inch stock, ask for a trim-only export rather than
scaling the MediaBox down. Do not use “fit to page,” crop to fill, or non-uniform
scaling.

Note that the ECCV on-site printing deadline was 21 August 2026 and has passed,
so this most likely goes to a local printer instead.

## Before printing — check

`build_poster.py` top matter:

- `AUTHORS` — `Arjang Talattof`, sole author. No placeholders remain in the PDF.
- `AFFIL` — empty, so no affiliation line is printed. Set the string to add one;
  it renders after the name.
- `CODE_URL` — the public research landing page encoded by the QR. Confirm that
  it is live before printing.

## Design

Palette and mark specifications are centralized in `tokens.py`. The recorded
separation checks against a white print surface are:

```
methods   #2a78d6,#eb6834,#1baf7a   all-pairs  CVD ΔE 9.2   normal-vision ΔE 24.0
outcomes  #008300,#eda100,#e34948   adjacent   CVD ΔE 15.3  normal-vision ΔE 20.8
```

Both sets include one color below 3:1 against white. The current poster does not
use the method-color set; it is retained for auxiliary figures. In the rendered
outcome chart, the wide green and red segments carry values, while the very thin
amber same-place/wrong-maneuver segments are not directly labelled; their meaning
and upper bound are stated in the figure caption. The palette checks therefore
describe color separation, not universal direct labelling or text contrast.

## Files

| File | Role |
|---|---|
| `tokens.py` | palette + ink tokens shared by both scripts |
| `charts.py` | renders the production scorecard and compact matched diagnostic |
| `production_scorecard.json` | public-safe scorecard snapshot with source and protocol provenance |
| `large_model_baselines.json` | source-backed Gemma 4 and LLaVA-Video vision-tower baselines |
| `frozen_matrix.json` | provenance snapshot behind the controlled-follow-up summaries manually mirrored in Section 3; not read at build time |
| `ECCV_Color Logo_2026.png` | tracked workshop logo placed in the poster header |
| `build_poster.py` | page geometry, typography, layout, crop marks |
| `build/` | generated output (gitignored; the printable PDF is not included by a Git push) |

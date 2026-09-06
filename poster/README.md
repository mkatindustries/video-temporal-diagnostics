# Video4Real @ ECCV 2026 poster

Poster for *When Conditional Sequence Matching Does Not Transfer to Global Video
Retrieval*. Its main result
is a public-safe view of the complete DRT production-recipe scorecard; section 2
is a separate matched conditional/global retrieval diagnostic from
`paper/video4real.tex`. A separate top inset reports the two source-backed
large-model vision-tower baselines. Workshop: Wednesday 9 September 2026, PM.

## Build

The poster needs `reportlab` and `qrcode` on top of the base install. They are
declared as the `poster` extra, so from an activated `video_retrieval` env:

```bash
pip install -e ".[poster]"
```

`pip install -e .` and `pip install -e ".[vlm]"` do **not** cover them — the
poster extra is separate.

If you would rather install through conda:

```bash
conda install -n video_retrieval --freeze-installed -c conda-forge reportlab qrcode
```

Keep `--freeze-installed`. Without it the solver also bumps `openssl` and
`ca-certificates` in the env that produced every tracked result; with it, nothing
already installed is touched, so `torch` and `transformers` stay exactly as
`results/PROVENANCE.md` records them.

Then, from anywhere:

```bash
conda activate video_retrieval
python poster/charts.py        # -> poster/build/p_*.png at 300 dpi
python poster/build_poster.py  # -> poster/build/video4real_poster_36x24in.pdf
```

Both scripts resolve their paths from `__file__`, so the working directory does
not matter. No GPU, dataset, or LaTeX install is needed. The scorecard and
large-model inset read the self-contained snapshots in `poster/`; the separate
conditional/global figure reads result JSONs under `results/`.

A beamerposter/tikzposter route is not available here: both conda TeX Live
installs are binaries-only, with no `.sty`, no `.cls` and no format sources, so
no LaTeX document can be compiled at all. Drawing the page directly with
reportlab also gives exact millimetre control over trim, bleed, and crop marks.

`production_scorecard.json` records the immutable source hash, source identity,
completion-bundle hash, public name mapping, display selection, rank rules, and
protocol qualifications. Its upstream raw artifact is not part of this
repository. The scorecard and matched diagnostic are intentionally not
numerically combined.

## Print specification

The current build targets a standard North American 36 × 24 inch landscape
poster. It is reflowed natively at 3:2; it is not a scaled version of the earlier
1400 × 1000 mm artwork.

| Requirement | Status |
|---|---|
| 1:1 scale, 36 × 24 inch landscape | explicit 914.4 × 609.6 mm CropBox and TrimBox |
| 5 mm bleed | explicit 924.4 × 619.6 mm BleedBox (5 mm each edge) |
| Crop marks | complete 12 mm marks, offset 3 mm outside trim, in a 946.4 × 641.6 mm MediaBox |
| Fonts embedded or outlined | 3 subset-embedded TrueType faces, no base-14 refs |
| Images ≥ 100 DPI at 1:1 | charts at 300 DPI; logo and QR exceed 300 effective DPI |
| CMYK (Fogra 39) | **not applied — see below** |

**Colour.** The PDF is RGB. A faithful Fogra 39 conversion needs that ICC
profile, which is not available in this environment; converting without it would
shift every hue. Give the print shop the RGB PDF and let them convert with the
correct profile — that is the normal workflow and yields a better result than a
blind conversion.

**Print handling.** Send the 36 × 24 inch PDF at 100%. Its default CropBox is
the exact finished size; the MediaBox retains bleed and crop marks for the shop.
Do not use “fit to page,” crop to fill, or non-uniform scaling.

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

Palette and mark specs follow the `dataviz` skill's reference palette (light
mode). Both categorical sets were checked with its validator against a white
print surface and pass every hard gate:

```
methods   #2a78d6,#eb6834,#1baf7a   all-pairs  CVD ΔE 9.2   normal-vision ΔE 24.0
outcomes  #008300,#eda100,#e34948   adjacent   CVD ΔE 15.3  normal-vision ΔE 20.8
```

Both raise the validator's contrast WARN (one slot below 3:1 on white), which
obligates visible direct labels — every mark on the poster carries one.

## Files

| File | Role |
|---|---|
| `tokens.py` | palette + ink tokens shared by both scripts |
| `charts.py` | renders the production scorecard and compact matched diagnostic |
| `production_scorecard.json` | public-safe scorecard snapshot with source and protocol provenance |
| `large_model_baselines.json` | source-backed Gemma 4 and LLaVA-Video vision-tower baselines |
| `frozen_matrix.json` | provenance snapshot for separate controlled follow-ups; not rendered here |
| `build_poster.py` | page geometry, typography, layout, crop marks |
| `build/` | generated output (gitignored) |

# Video4Real @ ECCV 2026 poster

Poster for *When Conditional Sequence Matching Does Not Transfer to Global Video
Retrieval* (`paper/video4real.tex`). Workshop: Wednesday 9 September 2026, PM.

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
python poster/build_poster.py  # -> poster/build/video4real_poster_1400x1000mm.pdf
```

Both scripts resolve their paths from `__file__`, so the working directory does
not matter. No GPU, no dataset, and no LaTeX is needed — only the result JSONs
tracked under `results/`, so this builds from a bare clone.

A beamerposter/tikzposter route is not available here: both conda TeX Live
installs are binaries-only, with no `.sty`, no `.cls` and no format sources, so
no LaTeX document can be compiled at all. Drawing the page directly with
reportlab also gives exact millimetre control over trim, bleed, and crop marks.

`charts.py` reads every number from the tracked JSONs under `results/` — nothing
is hardcoded, so regenerating a result and re-running the two scripts cannot
leave the poster silently stale.

## Print specification

Checked against the ECCV / Nordic Expo Service brief:

| Requirement | Status |
|---|---|
| 1:1 scale, 1400 × 1000 mm landscape | page is 1410 × 1010 mm = trim + 5 mm bleed |
| 5–10 mm bleed | 5 mm on every edge |
| Crop marks | 12 mm marks, offset 3 mm outside the trim |
| Fonts embedded or outlined | 3 subset-embedded TrueType faces, no base-14 refs |
| Images ≥ 100 DPI at 1:1 | charts at 300 DPI, QR at 287 DPI |
| CMYK (Fogra 39) | **not applied — see below** |

**Colour.** The PDF is RGB. A faithful Fogra 39 conversion needs that ICC
profile, which is not available in this environment; converting without it would
shift every hue. Give the print shop the RGB PDF and let them convert with the
correct profile — that is the normal workflow and yields a better result than a
blind conversion.

**Filename.** The brief wants `{PAPER_ID}Lastname{WIDTH}x{HEIGHT}mm.pdf`. Rename
on submission, e.g. `12_Talattof_1400x1000mm.pdf` (submission ID 12).

Note that the ECCV on-site printing deadline was 21 August 2026 and has passed,
so this most likely goes to a local printer instead.

## Before printing — check

`build_poster.py` top matter:

- `AUTHORS` — `Arjang Talattof`, sole author. No placeholders remain in the PDF.
- `AFFIL` — empty, so no affiliation line is printed. Set the string to add one;
  it renders after the name.
- `CODE_URL` — the GitHub URL, which the QR code encodes. The paper itself points
  at an anonymised `anonymous.4open.science` mirror, so confirm this repository is
  public and live before printing the QR.

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
| `charts.py` | renders the four chart PNGs from `results/` |
| `build_poster.py` | page geometry, typography, layout, crop marks |
| `build/` | generated output (gitignored) |

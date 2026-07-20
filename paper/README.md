# Manuscripts

This directory contains two active papers:

- `neurips.tex`: NeurIPS 2026 Evaluations & Datasets submission.
- `video4real.tex`: Video4Real at ECCV 2026 extended abstract.

Build both from the repository root:

```bash
make papers
```

Build or clean one manuscript with `make neurips`, `make video4real`,
`make clean-neurips`, or `make clean-video4real`.

Supporting files are intentionally kept beside their manuscript:

- `checklist.tex` and `neurips_2026.sty` support `neurips.tex`.
- `video4real.bib` and the pinned `eccv2026/` template support
  `video4real.tex`.

PDFs and LaTeX build artifacts are generated locally and ignored by Git.

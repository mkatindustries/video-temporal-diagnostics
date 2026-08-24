"""Shared design tokens for the Video4Real poster.

Palette values come from the dataviz reference palette (light mode) and were
checked with its validator against a white print surface:

  methods  #2a78d6,#eb6834,#1baf7a   all-pairs: CVD dE 9.2, normal-vision dE 24.0
  outcomes #008300,#eda100,#e34948   adjacent:  CVD dE 15.3, normal-vision dE 20.8

Both pass every hard gate. Both raise the contrast WARN (one slot below 3:1 on
white), which obligates visible direct labels -- every mark on this poster
carries one, so the relief rule is satisfied.
"""

from __future__ import annotations

# --- ink ---------------------------------------------------------------
SURFACE = "#ffffff"  # paper
PLANE = "#f7f7f5"  # panel wash, one step off paper
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RULE = "#d8d7d0"

# --- categorical: retrieval methods ------------------------------------
BOT = "#2a78d6"
DTW = "#eb6834"
RESID = "#1baf7a"

METHOD_COLOR = {
    "bot_cosine": BOT,
    "encoder_seq_dtw": DTW,
    "temporal_residual_dtw": RESID,
}
METHOD_LABEL = {
    "bot_cosine": "BoT (cosine)",
    "encoder_seq_dtw": "Encoder-seq DTW",
    "temporal_residual_dtw": "Temporal-residual DTW",
}

# --- categorical: top-1 outcome composition ----------------------------
OUT_RELEVANT = "#008300"
OUT_WRONG_MANEUVER = "#eda100"
OUT_WRONG_PLACE = "#e34948"

# --- accent used for the "this is the point" callouts ------------------
ACCENT = "#4a3aa7"

FONT = "DejaVu Sans"

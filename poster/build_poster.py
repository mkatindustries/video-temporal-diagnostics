"""Build the Video4Real @ ECCV 2026 poster as a print-ready PDF.

    conda activate video_retrieval
    python poster/charts.py && python poster/build_poster.py

Needs reportlab and qrcode on top of the project env:
    conda install -n video_retrieval --freeze-installed -c conda-forge reportlab qrcode

Output: poster/build/video4real_poster_1400x1000mm.pdf

Geometry follows the ECCV / Nordic Expo Service brief: 1400 x 1000 mm landscape
trim, 1:1 scale, 5 mm bleed on every edge, crop marks outside the trim. Fonts are
embedded TrueType (DejaVu Sans), not base-14 references.

COLOUR: this writes RGB. The printer brief asks for CMYK (Fogra 39); a faithful
conversion needs that ICC profile, which is not available here, and a naive
conversion would shift every hue. Hand the RGB PDF to the print shop and let them
convert with the correct profile -- that is the normal workflow and gives a better
result than converting blind.
"""

from __future__ import annotations

from pathlib import Path

import qrcode
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import tokens as T

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# ---------------------------------------------------------------- geometry
TRIM_W, TRIM_H = 1400.0, 1000.0
BLEED = 5.0
MARK_LEN, MARK_OFF = 12.0, 3.0

MARGIN = 40.0
COL_W = 300.0
GUTTER = 40.0
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(4)]

HEADER_TOP = TRIM_H
HEADER_H = 124.0
RULE_Y = TRIM_H - HEADER_H
STATS_TOP = RULE_Y - 20.0
STATS_H = 100.0
BODY_TOP = STATS_TOP - STATS_H - 24.0
FOOTER_Y = 34.0

# ---------------------------------------------------------------- content
AUTHORS = "Arjang Talattof"
# No affiliation was supplied, so none is printed. To add one, put the string here
# and it will render after the name.
AFFIL = ""
CODE_URL = "https://github.com/mkatindustries/video-temporal-diagnostics"

TITLE = "When Conditional Sequence Matching Does Not\nTransfer to Global Video Retrieval"
VENUE = "Video4Real @ ECCV 2026"

TAKEAWAY_BODY = (
    "A sequence score that wins inside a place can lose across places if it is not itself "
    "place-discriminative. Conditional benchmarks do not predict global retrieval — report "
    "both galleries."
)
# SoccerNet used to live here as a parenthetical; it now has its own panel, so this
# keeps only the limitation that panel cannot speak to.
SCOPE = (
    "Scope: two driving datasets, their 50 largest mixed-direction clusters, one backbone, the "
    "tested DTW variants, one linear fusion family. The SoccerNet-v2 check is within-match only "
    "and does not test cross-match search."
)

# Vertical budget for the full-width closing band: it hangs BAND_GAP below the
# shortest column and may not intrude on FOOTER_CEIL (the top of the QR block).
BAND_GAP = 16.0
FOOTER_CEIL = FOOTER_Y + 54.0


def font(name: str) -> str:
    return name


def register_fonts() -> None:
    mpl_ttf = (
        Path(__import__("matplotlib").__file__).parent / "mpl-data" / "fonts" / "ttf"
    )
    for face, fn in (
        ("DejaVu", "DejaVuSans.ttf"),
        ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
        ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(face, str(mpl_ttf / fn)))


# ---------------------------------------------------------------- helpers
def X(x: float) -> float:
    return (x + BLEED) * mm


def Y(y: float) -> float:
    return (y + BLEED) * mm


def lh(size_pt: float, mult: float = 1.34) -> float:
    """Line height in MILLIMETRES for a point size.

    Every vertical advance on this canvas is in mm while font sizes are in pt;
    mixing the two silently blows the layout apart, so all leading goes through here.
    """
    return size_pt * mult / 72.0 * 25.4


def wrap(text: str, face: str, size: float, width_mm: float) -> list[str]:
    """Greedy word wrap; honours explicit newlines."""
    out: list[str] = []
    limit = width_mm * mm
    for para in text.split("\n"):
        words, line = para.split(), ""
        for w in words:
            trial = f"{line} {w}".strip()
            if pdfmetrics.stringWidth(trial, face, size) <= limit or not line:
                line = trial
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def para(c, text, x, y, w, size=25, mult=1.34, face="DejaVu", color=T.INK_2) -> float:
    """Draw a wrapped paragraph from a top-left anchor; return the new y (mm)."""
    c.setFont(face, size)
    c.setFillColor(HexColor(color))
    step = lh(size, mult)
    for line in wrap(text, face, size, w):
        y -= step
        c.drawString(X(x), Y(y), line)
    return y


def bullets(c, items, x, y, w, size=24, mult=1.30, gap=4.5) -> float:
    ind = size * 0.88 / 72.0 * 25.4
    for item in items:
        c.setFont("DejaVu-Bold", size)
        c.setFillColor(HexColor(T.ACCENT))
        c.drawString(X(x), Y(y - lh(size, mult)), "\u2022")
        y = para(c, item, x + ind, y, w - ind, size=size, mult=mult)
        y -= gap
    return y


def section_head(c, n: int, text: str, x: float, y: float, w: float) -> float:
    ind = 11.0
    c.setFont("DejaVu-Bold", 26)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawString(X(x), Y(y - lh(38)), f"{n}")
    c.setFont("DejaVu-Bold", 38)
    c.setFillColor(HexColor(T.INK))
    yy = y
    for line in wrap(text, "DejaVu-Bold", 38, w - ind):
        yy -= lh(38)
        c.drawString(X(x + ind), Y(yy), line)
    yy -= 7
    c.setStrokeColor(HexColor(T.ACCENT))
    c.setLineWidth(2.4)
    c.line(X(x), Y(yy), X(x + w), Y(yy))
    return yy - 13


def h_para(text, w, size=25, mult=1.34, face="DejaVu") -> float:
    return len(wrap(text, face, size, w)) * lh(size, mult)


def h_bullets(items, w, size=24, mult=1.30, gap=4.5) -> float:
    ind = size * 0.88 / 72.0 * 25.4
    return sum(h_para(i, w - ind, size, mult) for i in items) + gap * len(items)


def boxed(c, x, y_top, w, title, body, *, bullet=True, size=21, pad=15.0,
          fill=T.PLANE, stroke=None, title_size=24, title_color=None,
          body_color=T.INK_2) -> float:
    """Panel sized to its own content, so nothing can overflow the box."""
    inner = w - 2 * pad
    h = pad + lh(title_size) + 5.0
    h += h_bullets(body, inner, size=size) if bullet else h_para(body, inner, size=size)
    h += pad
    panel(c, x, y_top - h, w, h, fill=fill, stroke=stroke)

    iy = y_top - pad
    c.setFont("DejaVu-Bold", title_size)
    c.setFillColor(HexColor(title_color or T.INK))
    iy -= lh(title_size)
    c.drawString(X(x + pad), Y(iy), title)
    iy -= 5.0
    if bullet:
        bullets(c, body, x + pad, iy, inner, size=size)
    else:
        para(c, body, x + pad, iy, inner, size=size, color=body_color)
    return y_top - h


def panel(c, x, y, w, h, fill=T.PLANE, stroke=None) -> None:
    c.setFillColor(HexColor(fill))
    if stroke:
        c.setStrokeColor(HexColor(stroke))
        c.setLineWidth(1.6)
    c.roundRect(X(x), Y(y), w * mm, h * mm, 5 * mm, stroke=1 if stroke else 0, fill=1)


CHARTS = (
    "p_schematic.png",
    "p_reversal.png",
    "p_errors.png",
    "p_cascade.png",
    "p_soccernet.png",
)


def require_charts() -> None:
    missing = [c for c in CHARTS if not (BUILD / c).is_file()]
    if missing:
        raise SystemExit(
            "Missing chart images: " + ", ".join(missing) + "\n"
            "Render them first:  python poster/charts.py"
        )


def image(c, name: str, x: float, y_top: float, w: float) -> float:
    """Place a chart PNG at 1:1 in the column; return the y below it."""
    from PIL import Image

    p = BUILD / name
    iw, ih = Image.open(p).size
    h = w * ih / iw
    c.drawImage(str(p), X(x), Y(y_top - h), w * mm, h * mm, mask=None)
    return y_top - h


def stat_tile(c, x, y, w, h, label, value, note, hero=False) -> None:
    panel(c, x, y, w, h, fill="#ffffff", stroke=T.RULE)
    pad = 14.0
    top = y + h - pad

    c.setFont("DejaVu", 21)
    c.setFillColor(HexColor(T.INK_2))
    top -= lh(21)
    c.drawString(X(x + pad), Y(top), label)

    vsize = 78 if hero else 54
    c.setFont("DejaVu-Bold", vsize)
    c.setFillColor(HexColor(T.ACCENT if hero else T.INK))
    top -= lh(vsize, 1.12)
    c.drawString(X(x + pad), Y(top), value)

    para(c, note, x + pad, top - 5, w - 2 * pad, size=18, color=T.INK_2)


# ---------------------------------------------------------------- marks
def crop_marks(c) -> None:
    c.setStrokeColor(Color(0, 0, 0))
    c.setLineWidth(0.5)
    for cx, cy, dx, dy in (
        (0, 0, -1, 0),
        (0, 0, 0, -1),
        (TRIM_W, 0, 1, 0),
        (TRIM_W, 0, 0, -1),
        (0, TRIM_H, -1, 0),
        (0, TRIM_H, 0, 1),
        (TRIM_W, TRIM_H, 1, 0),
        (TRIM_W, TRIM_H, 0, 1),
    ):
        x0 = cx + dx * MARK_OFF
        y0 = cy + dy * MARK_OFF
        c.line(X(x0), Y(y0), X(x0 + dx * MARK_LEN), Y(y0 + dy * MARK_LEN))


def qr_png(url: str, path: Path) -> Path:
    q = qrcode.QRCode(box_size=20, border=1, error_correction=qrcode.ERROR_CORRECT_M)
    q.add_data(url)
    q.make(fit=True)
    q.make_image(fill_color="black", back_color="white").save(path)
    return path


# ---------------------------------------------------------------- sections
def draw_header(c) -> None:
    y = HEADER_TOP - 20
    c.setFont("DejaVu-Bold", 26)
    c.setFillColor(HexColor(T.ACCENT))
    y -= lh(26)
    c.drawString(X(MARGIN), Y(y), VENUE.upper())

    c.setFont("DejaVu-Bold", 76)
    c.setFillColor(HexColor(T.INK))
    y -= 6
    for line in TITLE.split("\n"):
        y -= lh(76, 1.16)
        c.drawString(X(MARGIN), Y(y), line)

    c.setFont("DejaVu", 29)
    c.setFillColor(HexColor(T.INK_2))
    y -= lh(29, 1.55)
    byline = f"{AUTHORS}      {AFFIL}".rstrip()
    c.drawString(X(MARGIN), Y(y), byline)

    c.setStrokeColor(HexColor(T.RULE))
    c.setLineWidth(2.5)
    c.line(X(MARGIN), Y(RULE_Y), X(TRIM_W - MARGIN), Y(RULE_Y))


def draw_stats(c) -> None:
    y = STATS_TOP - STATS_H
    tiles = [
        (
            "Top-1 errors that are the wrong intersection",
            "≥ 98.9%",
            "Across all six top-1 rankings. Right place / wrong maneuver never exceeds 0.6%.",
            True,
        ),
        (
            "Conditional gain, DTW − BoT",
            "+0.032 / +0.067",
            "HDD / nuScenes mAP. Both paired CIs exclude zero — sequence matching wins here.",
            False,
        ),
        (
            "Global gain, DTW − BoT",
            "−0.079 / −0.173",
            "Same queries, same comparators, bigger gallery. The sign flips.",
            False,
        ),
        (
            "Held-out fusion weight α*",
            "0.95 / 1.00",
            "Leave-one-cluster-out fusion puts effectively all weight on appearance.",
            False,
        ),
    ]
    for i, (label, value, note, hero) in enumerate(tiles):
        stat_tile(c, COL_X[i], y, COL_W, STATS_H, label, value, note, hero=hero)


def draw_col1(c) -> float:
    x, w = COL_X[0], COL_W
    y = section_head(c, 1, "The question", x, BODY_TOP, w)
    y = para(
        c,
        "Scalable video retrieval pools a clip into one descriptor and ranks by cosine, "
        "which can blur motion direction. Comparing per-frame feature sequences with DTW "
        "fixes that — within a known location.",
        x, y, w,
    )
    y -= 9
    y = para(
        c,
        "We ask whether that advantage survives when the same query must be found in a "
        "gallery spanning many locations.",
        x, y, w, face="DejaVu-Oblique", color=T.INK,
    )
    y -= 16
    y = image(c, "p_schematic.png", x, y, w)
    y -= 18
    y = boxed(
        c, x, y, w, "Protocol",
        [
            "Maneuver segments clustered into intersections by DBSCAN on GPS "
            "(ε ≈ 30 m); a cluster is kept only if it holds both left and right turns.",
            "Relevant = same intersection AND same maneuver. Intersection identity is "
            "never given to either scorer.",
            "Honda HDD: 1,687 segments, 1,673 eligible queries, 50 clusters. "
            "nuScenes: 244 segments, 197 queries, 37 clusters.",
            "All 95% intervals from 2,000 intersection-cluster bootstrap resamples.",
        ],
    )
    y -= 18
    return boxed(
        c, x, y, w, "Three comparators, one backbone",
        [
            "BoT — V-JEPA 2 mean-pools every patch token to one vector; cosine "
            "similarity. Indexable.",
            "Encoder-seq DTW — patches averaged per temporal position to a (32, 1024) "
            "trajectory, compared by DTW.",
            "Temporal-residual DTW — predictor−target differences at 16 target "
            "positions, compared by DTW.",
            "DTW cost is normalised by T₁+T₂ and mapped through exp(−d), so scores from "
            "different comparators are not on one scale.",
        ],
    )


def draw_col2(c) -> float:
    x, w = COL_X[1], COL_W
    y = section_head(c, 2, "Conditional gains reverse globally", x, BODY_TOP, w)
    y = para(
        c,
        "Same queries, same relevance rule, same comparators — only the gallery changes.",
        x, y, w, face="DejaVu-Oblique", color=T.INK,
    )
    y -= 12
    y = image(c, "p_reversal.png", x, y, w)
    y -= 16

    rows = [
        ("Enc-seq DTW − BoT, conditional", "+0.032 [0.016, 0.042]", "+0.067 [0.036, 0.101]"),
        ("Enc-seq DTW − BoT, global", "−0.079 [−0.106, −0.062]", "−0.173 [−0.240, −0.118]"),
        ("Residual DTW − BoT, global", "−0.091 [−0.116, −0.077]", "−0.196 [−0.271, −0.136]"),
    ]
    pad, rsize = 15.0, 17.0
    row_h = lh(rsize, 1.85)
    h = pad + lh(24) + 6 + lh(rsize, 1.5) + len(rows) * row_h + pad
    panel(c, x, y - h, w, h, fill=T.PLANE)

    iy = y - pad
    c.setFont("DejaVu-Bold", 24)
    c.setFillColor(HexColor(T.INK))
    iy -= lh(24)
    c.drawString(X(x + pad), Y(iy), "Paired difference, 95% cluster CI")
    iy -= 6

    widest = max(
        pdfmetrics.stringWidth(v, "DejaVu-Bold", rsize) for _, a, b in rows for v in (a, b)
    ) / mm
    col_b = x + w - pad
    col_a = col_b - (widest + 9.0)
    c.setFont("DejaVu", 18)
    c.setFillColor(HexColor(T.INK_2))
    iy -= lh(rsize, 1.5)
    c.drawRightString(X(col_a), Y(iy), "Honda HDD")
    c.drawRightString(X(col_b), Y(iy), "nuScenes")

    for i, (label, a, b) in enumerate(rows):
        iy -= row_h
        if i:
            c.setStrokeColor(HexColor(T.RULE))
            c.setLineWidth(1.0)
            c.line(X(x + pad), Y(iy + row_h * 0.62), X(x + w - pad), Y(iy + row_h * 0.62))
        c.setFont("DejaVu", rsize)
        c.setFillColor(HexColor(T.INK_2))
        c.drawString(X(x + pad), Y(iy), label)
        c.setFont("DejaVu-Bold", rsize)
        c.setFillColor(HexColor(T.INK))
        c.drawRightString(X(col_a), Y(iy), a)
        c.drawRightString(X(col_b), Y(iy), b)

    return y - h


def draw_col3(c) -> float:
    x, w = COL_X[2], COL_W
    y = section_head(c, 3, "The loss is location, not order", x, BODY_TOP, w)
    y = para(
        c,
        "Decomposing every top-1 retrieval separates two different failures: the wrong "
        "maneuver at the right intersection, and the wrong intersection altogether.",
        x, y, w,
    )
    y -= 14
    y = image(c, "p_errors.png", x, y, w)
    y -= 18
    return boxed(
        c, x, y, w, "Order controls do not explain it",
        [
            "Intact DTW beats its shuffled control detectably on nuScenes only "
            "(+0.063 [0.023, 0.102]); the HDD interval crosses zero.",
            "Order-free assignment differs detectably just once — and it makes "
            "nuScenes residual features better, not worse.",
            "So the conditional gain is finer per-frame matching more than intact "
            "order, and neither buys location discrimination.",
        ],
    )


def draw_col4(c) -> float:
    x, w = COL_X[3], COL_W
    y = section_head(c, 4, "Neither remedy recovers the gap", x, BODY_TOP, w)
    y = image(c, "p_cascade.png", x, y, w)
    y -= 16

    y = boxed(
        c, x, y, w, "Leakage-safe fusion",
        "Leave-one-cluster-out selects α* = 0.95 in all 50 HDD folds and 1.00 in all 37 "
        "nuScenes folds. Fused mAP shows no detected gain on HDD (+0.001 [−0.003, 0.004]) "
        "and collapses exactly onto BoT on nuScenes.",
        bullet=False,
    )
    y -= 14

    # The takeaway used to sit here. It is the poster's conclusion, so it now runs
    # full width across the foot of the page and this column carries the
    # third-domain check instead.
    y = section_head(c, 5, "Does it hold outside driving?", x, y, w)
    return image(c, "p_soccernet.png", x, y, w)


def draw_takeaway(c, y_top: float) -> float:
    """Full-width closing band. Returns its bottom edge in mm."""
    bottom = boxed(
        c, MARGIN, y_top, TRIM_W - 2 * MARGIN, "Takeaway", TAKEAWAY_BODY,
        bullet=False, size=24, title_size=28, pad=12.0,
        fill="#f1efff", stroke=T.ACCENT, title_color=T.ACCENT, body_color=T.INK,
    )
    if bottom < FOOTER_CEIL:
        raise SystemExit(
            f"Takeaway band runs to {bottom:.1f} mm, under the {FOOTER_CEIL:.1f} mm footer "
            "ceiling -- a column grew and the band no longer clears the QR block. "
            "Shorten a column, or reduce BAND_GAP."
        )
    return bottom


def draw_footer(c) -> None:
    qr = qr_png(CODE_URL, BUILD / "_qr.png")
    s = 62.0
    qx = TRIM_W - MARGIN - s
    c.drawImage(str(qr), X(qx), Y(FOOTER_Y - 14), s * mm, s * mm)

    c.setFont("DejaVu", 21)
    c.setFillColor(HexColor(T.INK_2))
    c.drawRightString(
        X(qx - 16),
        Y(FOOTER_Y + 26),
        "Code, evaluation protocol, and every result JSON behind these numbers:",
    )
    c.setFont("DejaVu-Bold", 21)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawRightString(X(qx - 16), Y(FOOTER_Y - 2), CODE_URL)

    # Scope fills the footer's empty left half. The takeaway band's stroked border
    # now separates the footer, so the old hairline rule here would just be noise.
    para(c, SCOPE, MARGIN, FOOTER_Y + 46, 940.0, size=18, color=T.INK_2)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    require_charts()
    register_fonts()
    out = BUILD / "video4real_poster_1400x1000mm.pdf"
    c = canvas.Canvas(
        str(out),
        pagesize=((TRIM_W + 2 * BLEED) * mm, (TRIM_H + 2 * BLEED) * mm),
        initialFontName="DejaVu",
        initialFontSize=12,
    )
    c.setTitle("When Conditional Sequence Matching Does Not Transfer to Global Video Retrieval")
    c.setAuthor(AUTHORS)

    # bleed fill, then trim-area paper
    c.setFillColor(HexColor(T.SURFACE))
    c.rect(0, 0, (TRIM_W + 2 * BLEED) * mm, (TRIM_H + 2 * BLEED) * mm, stroke=0, fill=1)

    draw_header(c)
    draw_stats(c)
    # The band hangs off the shortest column rather than a fixed y, so editing any
    # column's copy cannot silently drive it into the footer -- draw_takeaway raises
    # instead.
    bottoms = [draw_col1(c), draw_col2(c), draw_col3(c), draw_col4(c)]
    draw_takeaway(c, min(bottoms) - BAND_GAP)
    draw_footer(c)
    crop_marks(c)

    c.showPage()
    c.save()
    print(f"wrote {out}")
    print(f"  page  {(TRIM_W + 2 * BLEED):.0f} x {(TRIM_H + 2 * BLEED):.0f} mm "
          f"(trim {TRIM_W:.0f} x {TRIM_H:.0f} + {BLEED:.0f} mm bleed)")


if __name__ == "__main__":
    main()

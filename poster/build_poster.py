"""Build the Video4Real @ ECCV 2026 poster as a print-ready PDF.

    conda activate video_retrieval
    python poster/charts.py && python poster/build_poster.py

Needs reportlab and qrcode on top of the project env:
    conda install -n video_retrieval --freeze-installed -c conda-forge reportlab qrcode

Output: poster/build/video4real_poster_1400x1000mm.pdf

Geometry follows the ECCV / Nordic Expo Service brief: 1400 x 1000 mm landscape
trim, 1:1 scale, 5 mm bleed on every edge, and a separate slug for crop marks.
The PDF encodes distinct MediaBox, BleedBox, and TrimBox values. Fonts are embedded
TrueType (DejaVu Sans), not base-14 references.

COLOUR: this writes RGB. The printer brief asks for CMYK (Fogra 39); a faithful
conversion needs that ICC profile, which is not available here, and a naive
conversion would shift every hue. Hand the RGB PDF to the print shop and let them
convert with the correct profile -- that is the normal workflow and gives a better
result than converting blind.
"""

from __future__ import annotations

import json
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
# The marks extend MARK_OFF + MARK_LEN outside trim.  One additional millimetre
# keeps their outer endpoints off the MediaBox boundary.  This is a slug, not
# extra bleed: TrimBox and BleedBox below retain the requested physical sizes.
MEDIA_MARGIN = MARK_OFF + MARK_LEN + 1.0
PAGE_W = TRIM_W + 2 * MEDIA_MARGIN
PAGE_H = TRIM_H + 2 * MEDIA_MARGIN

MARGIN = 40.0
COL_W = 300.0
GUTTER = 40.0
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(4)]

HEADER_TOP = TRIM_H
# Tightened from 124 / 20 / 100 / 24. Trimming the top matter buys height in
# every column at once, which is the only lever that does -- shrinking charts
# hits column 1 almost immediately, since its only figure is the schematic.
HEADER_H = 114.0
RULE_Y = TRIM_H - HEADER_H
STATS_TOP = RULE_Y - 14.0
STATS_H = 100.0
BODY_TOP = STATS_TOP - STATS_H - 18.0
FOOTER_Y = 34.0

# ---------------------------------------------------------------- content
AUTHORS = "Arjang Talattof"
# No affiliation was supplied, so none is printed. To add one, put the string here
# and it will render after the name.
AFFIL = ""
# The QR and the printed link both resolve here. This is the research landing
# page, not the code repository -- that is one click away from it, and a shorter
# payload makes a coarser QR that scans from further back.
CODE_URL = "https://mkat.fyi/research/"

TITLE = "When Conditional Sequence Matching Does Not\nTransfer to Global Video Retrieval"
VENUE = "Video4Real @ ECCV 2026"

TAKEAWAY_BODY = (
    "No frozen recipe is best across all five tasks. Strong copy or event-identity point "
    "estimates do not guarantee maneuver retrieval or safe deletion. The next step is a "
    "target-trained safety head or a fusion of appearance and temporal features, evaluated "
    "under protocol-matched comparisons."
)
SCOPE = (
    "Scorecard snapshot 91eaebaf; six approved recipes shown from a complete 65/65 matrix. "
    "Values and ranks are descriptive point estimates within the six shown; metrics have no "
    "shared scale. The vision-tower inset and Section 2 are separate diagnostics; neither "
    "reranks it."
)

QR_SIZE = 52.0
QR_Y = FOOTER_Y - 14.0
FOOTER_CEIL = QR_Y + QR_SIZE + 6.0
MIN_FOOTER_GAP = 8.0


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
    return (x + MEDIA_MARGIN) * mm


def Y(y: float) -> float:
    return (y + MEDIA_MARGIN) * mm


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
    "p_scorecard.png",
    "p_diagnostic_pair.png",
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


LOGO = HERE / "ECCV_Color Logo_2026.png"
# 84 mm is the practical ceiling: the logo hangs 16 mm below HEADER_TOP and must
# stay clear of the rule at RULE_Y, which leaves 14 mm of air at this height.
LOGO_H = 84.0


def logo_trimmed():
    """Trim the ECCV logo's padding; return (path, aspect) or None if absent.

    The supplied PNG is mostly margin -- the artwork covers roughly the middle
    60% -- so placing it as-is would waste header width and sit visibly off. The
    build degrades to no logo rather than failing, so a clone without the asset
    still produces a poster.
    """
    if not LOGO.is_file():
        return None
    from PIL import Image, ImageChops

    im = Image.open(LOGO).convert("RGBA")
    lo, _ = im.getchannel("A").getextrema()
    if lo < 255:
        box = im.getchannel("A").getbbox()  # transparent padding
    else:
        rgb = im.convert("RGB")
        box = ImageChops.difference(rgb, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box is None:
        return None
    im = im.crop(box)
    out = BUILD / "_eccv_logo_trimmed.png"
    im.save(out)
    return out, im.width / im.height


def qr_png(url: str, path: Path) -> Path:
    q = qrcode.QRCode(box_size=20, border=1, error_correction=qrcode.ERROR_CORRECT_M)
    q.add_data(url)
    q.make(fit=True)
    q.make_image(fill_color="black", back_color="white").save(path)
    return path


# ---------------------------------------------------------------- sections
def draw_header(c) -> None:
    # Top offset and byline leading are tightened to keep the byline clear of the
    # rule at RULE_Y. HEADER_H came down to 114 to free column height, which moved
    # the rule up into the name; this buys the clearance back inside the header
    # instead of taking it from the columns.
    y = HEADER_TOP - 12
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
    y -= lh(29, 1.35)
    byline = f"{AUTHORS}      {AFFIL}".rstrip()
    c.drawString(X(MARGIN), Y(y), byline)

    # Logo goes top-right: the ECCV template puts it top-left, but this title is
    # left-aligned rather than centred, so left would collide with it.
    lg = logo_trimmed()
    if lg is not None:
        path, aspect = lg
        w = LOGO_H * aspect
        c.drawImage(str(path), X(TRIM_W - MARGIN - w), Y(HEADER_TOP - 16 - LOGO_H),
                    w * mm, LOGO_H * mm, mask="auto")

    c.setStrokeColor(HexColor(T.RULE))
    c.setLineWidth(2.5)
    c.line(X(MARGIN), Y(RULE_Y), X(TRIM_W - MARGIN), Y(RULE_Y))


def draw_context_strip(c) -> None:
    """Replace disconnected headline numbers with task and evidence context."""
    y = STATS_TOP - STATS_H
    gap = 20.0
    left_w = 790.0
    right_x = MARGIN + left_w + gap
    right_w = TRIM_W - MARGIN - right_x

    panel(c, MARGIN, y, left_w, STATS_H, fill="#ffffff", stroke=T.RULE)
    c.setFont("DejaVu-Bold", 24)
    c.setFillColor(HexColor(T.INK))
    c.drawString(X(MARGIN + 14), Y(STATS_TOP - 18), "Five cohorts, four deployment decisions")

    task_cards = [
        ("COPY IDENTITY", "Controlled + real copies", "Internal synthetic F1  ·  VCDB AP"),
        ("EVENT IDENTITY", "Exact replay", "SoccerNet-v2 match-macro MRR"),
        ("MANEUVER RETRIEVAL", "Global driving pairs", "Honda HDD global AP"),
        ("DELETION SAFETY", "Confirmed negatives", "Project Aria false-merge rate ↓"),
    ]
    inner_x = MARGIN + 14
    inner_w = left_w - 28
    cell_w = inner_w / len(task_cards)
    for i, (eyebrow, title, detail) in enumerate(task_cards):
        cx = inner_x + i * cell_w
        if i:
            c.setStrokeColor(HexColor(T.RULE))
            c.setLineWidth(1.1)
            c.line(X(cx - 9), Y(y + 14), X(cx - 9), Y(STATS_TOP - 31))
        c.setFont("DejaVu-Bold", 16)
        c.setFillColor(HexColor(T.ACCENT))
        c.drawString(X(cx), Y(STATS_TOP - 39), eyebrow)
        c.setFont("DejaVu-Bold", 22)
        c.setFillColor(HexColor(T.INK))
        c.drawString(X(cx), Y(STATS_TOP - 55), title)
        para(c, detail, cx, STATS_TOP - 59, cell_w - 16, size=18, mult=1.20)

    baselines = json.loads((HERE / "large_model_baselines.json").read_text())
    panel(c, right_x, y, right_w, STATS_H, fill="#f1efff", stroke=T.ACCENT)
    c.setFont("DejaVu-Bold", 24)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawString(X(right_x + 14), Y(STATS_TOP - 18), "Large-model vision baselines")
    c.setFont("DejaVu", 17)
    c.setFillColor(HexColor(T.INK_2))
    c.drawRightString(X(right_x + right_w - 14), Y(STATS_TOP - 18), "VCDB AP     HDD AP")

    yy = STATS_TOP - 45
    for model in baselines["models"]:
        c.setFont("DejaVu-Bold", 21)
        c.setFillColor(HexColor(T.INK))
        c.drawString(X(right_x + 14), Y(yy), model["label"])
        c.drawRightString(
            X(right_x + right_w - 14),
            Y(yy),
            f"{model['vcdb_ap']:.4f}       {model['hdd_ap']:.4f}",
        )
        yy -= 18

    para(
        c,
        "Pooled vision towers, not language-model outputs; same scorecard populations and "
        "metrics, separate source, excluded from v8 ranks.",
        right_x + 14,
        yy + 3,
        right_w - 28,
        size=17,
        mult=1.18,
        color=T.INK_2,
    )


def draw_scorecard(c) -> float:
    x = MARGIN
    w = TRIM_W - 2 * MARGIN
    y = section_head(c, 1, "DRT production-recipe scorecard", x, BODY_TOP, w)
    y = para(
        c,
        "The complete scorecard evaluates 13 fixed recipes on five cohorts (65/65 cells). "
        "We show six approved recipes from one immutable snapshot; every rank is explicitly "
        "limited to the six shown.",
        x,
        y,
        w,
        size=24,
    )
    y -= 8
    return image(c, "p_scorecard.png", x, y, w)


def draw_reversal_panel(c, y_top: float) -> float:
    x, w = MARGIN, 900.0
    y = section_head(c, 2, "Conditional gains reverse globally", x, y_top, w)
    y = para(
        c,
        "A separate matched V-JEPA 2 diagnostic holds features, queries, relevance, and scorer "
        "pair fixed; only the gallery expands. Left: the sign flips. Right: the global misses "
        "shift overwhelmingly to other locations.",
        x,
        y,
        w,
        size=23,
        face="DejaVu-Oblique",
        color=T.INK,
    )
    y -= 8
    y = image(c, "p_diagnostic_pair.png", x, y, w)
    y -= 8
    return boxed(
        c,
        x,
        y,
        w,
        "What the decomposition establishes",
        "Across all six tracked method×dataset rows, at least 98.9% of top-1 errors are "
        "wrong-location. This localizes the observed failure to place discrimination; it does "
        "not establish a specific DTW scoring mechanism.",
        bullet=False,
        size=19,
        title_size=23,
        pad=13,
    )


def draw_synthesis(c, y_top: float) -> float:
    x, w = 980.0, 380.0
    y = section_head(c, 3, "Open gap: identity, motion, safety", x, y_top, w)
    y = para(
        c,
        "The highest shown point estimate changes by task: InternVideo-Next L on both copy "
        "cohorts, SAM3 Perception Encoder on SoccerNet-v2, V-JEPA 2 on HDD, and LeVJEPA on "
        "Aria.",
        x,
        y,
        w,
        size=23,
    )
    y -= 8
    y = boxed(
        c,
        x,
        y,
        w,
        "Separate evidence — do not mix",
        [
            "The scorecard is descriptive and recipe-level, not a controlled backbone "
            "ablation; it has no common paired intervals.",
            "Gemma 4 and LLaVA-Video above are pooled vision towers, not language-model "
            "outputs, and are excluded from scorecard ranks.",
            "Section 2 uses query-macro mAP with cluster resampling; its values and intervals "
            "do not annotate the scorecard's global AP lane.",
        ],
        size=18,
        title_size=24,
        pad=13,
    )
    y -= 8
    y = boxed(
        c,
        x,
        y,
        w,
        "Controlled follow-ups",
        [
            "VCDB within-topic: 0/7 registered reference contrasts separate.",
            "SoccerNet-v2 window Chamfer: ordered − unordered is −0.0014; its 95% CI "
            "crosses zero.",
            "Project Aria order sensitivity: 0/5 shown intervals clear parity.",
        ],
        size=18,
        title_size=24,
        pad=13,
        fill="#ffffff",
        stroke=T.RULE,
    )
    y -= 8
    return boxed(
        c,
        x,
        y,
        w,
        "Takeaway",
        TAKEAWAY_BODY,
        bullet=False,
        size=21,
        title_size=27,
        pad=14,
        fill="#f1efff",
        stroke=T.ACCENT,
        title_color=T.ACCENT,
        body_color=T.INK,
    )


def draw_footer(c) -> None:
    qr = qr_png(CODE_URL, BUILD / "_qr.png")
    qx = TRIM_W - MARGIN - QR_SIZE
    c.drawImage(str(qr), X(qx), Y(QR_Y), QR_SIZE * mm, QR_SIZE * mm)

    c.setFont("DejaVu", 25)
    c.setFillColor(HexColor(T.INK_2))
    c.drawRightString(
        X(qx - 16),
        Y(FOOTER_Y + 26),
        "MKAT INDUSTRIES LLC",
    )
    c.setFont("DejaVu-Bold", 21)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawRightString(X(qx - 16), Y(FOOTER_Y - 2), CODE_URL)

    # Scope fills the footer's empty left half and wraps to a readable measure.
    para(c, SCOPE, MARGIN, FOOTER_Y + 34, 820.0, size=18, color=T.INK_2)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    require_charts()
    register_fonts()
    out = BUILD / "video4real_poster_1400x1000mm.pdf"
    c = canvas.Canvas(
        str(out),
        pagesize=(PAGE_W * mm, PAGE_H * mm),
        trimBox=(
            MEDIA_MARGIN * mm,
            MEDIA_MARGIN * mm,
            (MEDIA_MARGIN + TRIM_W) * mm,
            (MEDIA_MARGIN + TRIM_H) * mm,
        ),
        bleedBox=(
            (MEDIA_MARGIN - BLEED) * mm,
            (MEDIA_MARGIN - BLEED) * mm,
            (MEDIA_MARGIN + TRIM_W + BLEED) * mm,
            (MEDIA_MARGIN + TRIM_H + BLEED) * mm,
        ),
        initialFontName="DejaVu",
        initialFontSize=12,
    )
    c.setTitle("When Conditional Sequence Matching Does Not Transfer to Global Video Retrieval")
    c.setAuthor(AUTHORS)

    # bleed fill, then trim-area paper
    c.setFillColor(HexColor(T.SURFACE))
    c.rect(0, 0, PAGE_W * mm, PAGE_H * mm, stroke=0, fill=1)

    draw_header(c)
    draw_context_strip(c)
    scorecard_bottom = draw_scorecard(c)
    lower_top = scorecard_bottom - 16.0
    bottoms = [draw_reversal_panel(c, lower_top), draw_synthesis(c, lower_top)]
    gap = min(bottoms) - FOOTER_CEIL
    if gap < MIN_FOOTER_GAP:
        raise SystemExit(
            f"Only {gap:.1f} mm of air above the footer, under the "
            f"{MIN_FOOTER_GAP:.1f} mm minimum. Lower-panel bottoms: "
            + ", ".join(f"{i + 1}:{b:.1f}" for i, b in enumerate(bottoms))
            + f" mm; footer ceiling {FOOTER_CEIL:.1f} mm. Shorten the lowest panel."
        )
    draw_footer(c)
    crop_marks(c)

    c.showPage()
    c.save()
    print(f"wrote {out}")
    print(f"  media {PAGE_W:.0f} x {PAGE_H:.0f} mm; trim {TRIM_W:.0f} x {TRIM_H:.0f} mm; "
          f"bleed {BLEED:.0f} mm each edge; crop-mark slug {MEDIA_MARGIN - BLEED:.0f} mm")
    print(f"  scorecard bottom {scorecard_bottom:.1f} mm; lower panels "
          + ", ".join(f"{i + 1}:{b:.1f}" for i, b in enumerate(bottoms))
          + f" mm; footer air {gap:.1f} mm")


if __name__ == "__main__":
    main()

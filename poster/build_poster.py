"""Build the Video4Real @ ECCV 2026 poster as a print-ready PDF.

    conda activate video_retrieval
    python poster/charts.py && python poster/build_poster.py

Needs reportlab and qrcode on top of the project env:
    conda install -n video_retrieval --freeze-installed -c conda-forge reportlab qrcode

Output: poster/build/video4real_poster_36x24in.pdf

Geometry targets a 36 x 24 inch landscape print: 914.4 x 609.6 mm trim at
1:1 scale, 5 mm bleed on every edge, and a separate slug for crop marks.
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
TRIM_W, TRIM_H = 36 * 25.4, 24 * 25.4
BLEED = 5.0
MARK_LEN, MARK_OFF = 12.0, BLEED + 3.0
# The marks extend MARK_OFF + MARK_LEN outside trim.  One additional millimetre
# keeps their outer endpoints off the MediaBox boundary.  This is a slug, not
# extra bleed: TrimBox and BleedBox below retain the requested physical sizes.
MEDIA_MARGIN = MARK_OFF + MARK_LEN + 1.0
PAGE_W = TRIM_W + 2 * MEDIA_MARGIN
PAGE_H = TRIM_H + 2 * MEDIA_MARGIN

MARGIN = 18.0
LOWER_LEFT_W = 570.0
LOWER_GUTTER = 18.0
LOWER_RIGHT_X = MARGIN + LOWER_LEFT_W + LOWER_GUTTER
LOWER_RIGHT_W = TRIM_W - MARGIN - LOWER_RIGHT_X

HEADER_TOP = TRIM_H
HEADER_H = 86.0
RULE_Y = TRIM_H - HEADER_H
STATS_TOP = RULE_Y - 8.0
STATS_H = 62.0
BODY_TOP = STATS_TOP - STATS_H - 10.0
FOOTER_Y = 14.0

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
    "Leaders change by task. Bridge the gap with task-trained safety heads or "
    "appearance-temporal fusion under matched protocols."
)
SCOPE = (
    "Six recipes from a complete 65/65 matrix · Point estimates; "
    "ranks only among six shown · Metrics are task-specific · The vision-tower inset and "
    "Section 2 are separate diagnostics."
)

QR_SIZE = 40.0
QR_Y = 14.0
FOOTER_CEIL = QR_Y + QR_SIZE + 5.0
MIN_FOOTER_GAP = 7.0


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
    ind = 8.0
    c.setFont("DejaVu-Bold", 22)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawString(X(x), Y(y - lh(32)), f"{n}")
    c.setFont("DejaVu-Bold", 32)
    c.setFillColor(HexColor(T.INK))
    yy = y
    for line in wrap(text, "DejaVu-Bold", 32, w - ind):
        yy -= lh(32)
        c.drawString(X(x + ind), Y(yy), line)
    yy -= 4
    c.setStrokeColor(HexColor(T.ACCENT))
    c.setLineWidth(1.8)
    c.line(X(x), Y(yy), X(x + w), Y(yy))
    return yy - 8


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
LOGO_H = 50.0


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
    y = HEADER_TOP - 10
    c.setFont("DejaVu-Bold", 20)
    c.setFillColor(HexColor(T.ACCENT))
    y -= lh(20)
    c.drawString(X(MARGIN), Y(y), VENUE.upper())

    c.setFont("DejaVu-Bold", 60)
    c.setFillColor(HexColor(T.INK))
    y -= 3
    for line in TITLE.split("\n"):
        y -= lh(60, 1.07)
        c.drawString(X(MARGIN), Y(y), line)

    c.setFont("DejaVu", 18)
    c.setFillColor(HexColor(T.INK_2))
    y -= lh(18, 1.35)
    byline = f"{AUTHORS}      {AFFIL}".rstrip()
    c.drawString(X(MARGIN), Y(y), byline)

    # Logo goes top-right: the ECCV template puts it top-left, but this title is
    # left-aligned rather than centred, so left would collide with it.
    lg = logo_trimmed()
    if lg is not None:
        path, aspect = lg
        w = LOGO_H * aspect
        c.drawImage(str(path), X(TRIM_W - MARGIN - w), Y(HEADER_TOP - 13 - LOGO_H),
                    w * mm, LOGO_H * mm, mask="auto")

    c.setStrokeColor(HexColor(T.RULE))
    c.setLineWidth(2.5)
    c.line(X(MARGIN), Y(RULE_Y), X(TRIM_W - MARGIN), Y(RULE_Y))


def draw_context_strip(c) -> None:
    """Replace disconnected headline numbers with task and evidence context."""
    y = STATS_TOP - STATS_H
    gap = 18.0
    left_w = 552.0
    right_x = MARGIN + left_w + gap
    right_w = TRIM_W - MARGIN - right_x

    panel(c, MARGIN, y, left_w, STATS_H, fill="#ffffff", stroke=T.RULE)
    c.setFont("DejaVu-Bold", 20)
    c.setFillColor(HexColor(T.INK))
    c.drawString(X(MARGIN + 10), Y(STATS_TOP - 14), "Five cohorts, four deployment decisions")

    task_cards = [
        ("Copy identity", "Internal + VCDB"),
        ("Event identity", "SoccerNet-v2"),
        ("Maneuver retrieval", "Honda HDD"),
        ("Deletion safety", "Project Aria  ↓"),
    ]
    inner_x = MARGIN + 10
    inner_w = left_w - 20
    cell_w = inner_w / len(task_cards)
    for i, (title, detail) in enumerate(task_cards):
        cx = inner_x + i * cell_w
        if i:
            c.setStrokeColor(HexColor(T.RULE))
            c.setLineWidth(1.1)
            c.line(X(cx - 6), Y(y + 9), X(cx - 6), Y(STATS_TOP - 25))
        c.setFont("DejaVu-Bold", 18)
        c.setFillColor(HexColor(T.INK))
        c.drawString(X(cx), Y(STATS_TOP - 35), title)
        para(c, detail, cx, STATS_TOP - 38, cell_w - 10, size=16, mult=1.18)

    baselines = json.loads((HERE / "large_model_baselines.json").read_text())
    panel(c, right_x, y, right_w, STATS_H, fill="#f1efff", stroke=T.ACCENT)
    c.setFont("DejaVu-Bold", 20)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawString(X(right_x + 10), Y(STATS_TOP - 14), "Vision-tower baselines")
    c.setFont("DejaVu", 16)
    c.setFillColor(HexColor(T.INK_2))
    c.drawRightString(X(right_x + right_w - 10), Y(STATS_TOP - 14), "VCDB AP    HDD AP")

    yy = STATS_TOP - 30
    for model in baselines["models"]:
        c.setFont("DejaVu-Bold", 18)
        c.setFillColor(HexColor(T.INK))
        c.drawString(X(right_x + 10), Y(yy), model["label"])
        c.drawRightString(
            X(right_x + right_w - 10),
            Y(yy),
            f"{model['vcdb_ap']:.4f}      {model['hdd_ap']:.4f}",
        )
        yy -= 13

    para(
        c,
        "Pooled vision towers · separate source · excluded from scorecard ranks.",
        right_x + 10,
        yy + 4,
        right_w - 20,
        size=16,
        mult=1.15,
        color=T.INK_2,
    )


def draw_scorecard(c) -> float:
    x = MARGIN
    w = TRIM_W - 2 * MARGIN
    y = section_head(c, 1, "DRT production-recipe scorecard", x, BODY_TOP, w)
    y = para(
        c,
        "The complete scorecard evaluates 13 fixed recipes on five cohorts (65/65 cells). "
        "Six approved recipes are shown; every rank is limited to those six.",
        x,
        y,
        w,
        size=20,
    )
    y -= 5
    return image(c, "p_scorecard.png", x, y, w)


def draw_reversal_panel(c, y_top: float) -> float:
    x, w = MARGIN, LOWER_LEFT_W
    y = section_head(c, 2, "Conditional gains reverse globally", x, y_top, w)
    y = para(
        c,
        "A matched V-JEPA 2 diagnostic holds features, queries, relevance, and scorer fixed; "
        "only the gallery expands. Left: the sign flips. Right: global misses land elsewhere.",
        x,
        y,
        w,
        size=20,
        face="DejaVu-Oblique",
        color=T.INK,
    )
    y -= 5
    return image(c, "p_diagnostic_pair.png", x, y, w)


def draw_synthesis(c, y_top: float) -> float:
    x, w = LOWER_RIGHT_X, LOWER_RIGHT_W
    y = section_head(c, 3, "Open gap: identity, motion, safety", x, y_top, w)
    y = para(
        c,
        "Leaders vary by task: InternVideo-Next L on copies, SAM3 Perception Encoder on "
        "SoccerNet-v2, V-JEPA 2 on HDD, and LeVJEPA on Aria.",
        x,
        y,
        w,
        size=20,
    )
    y -= 5
    y = boxed(
        c,
        x,
        y,
        w,
        "Evidence boundaries",
        "Scorecard: recipe-level point estimates; no paired intervals.\n"
        "Vision towers and Section 2 use separate sources and protocols.\n"
        "Controlled checks: VCDB shown spread 0.0054 AP; SoccerNet CI crosses zero; Aria 0/5.",
        bullet=False,
        size=17,
        title_size=20,
        pad=9,
        fill="#ffffff",
        stroke=T.RULE,
    )
    y -= 4
    return boxed(
        c,
        x,
        y,
        w,
        "Takeaway",
        TAKEAWAY_BODY,
        bullet=False,
        size=20,
        title_size=22,
        pad=9,
        fill="#f1efff",
        stroke=T.ACCENT,
        title_color=T.ACCENT,
        body_color=T.INK,
    )


def draw_footer(c) -> None:
    qr = qr_png(CODE_URL, BUILD / "_qr.png")
    qx = TRIM_W - MARGIN - QR_SIZE
    c.drawImage(str(qr), X(qx), Y(QR_Y), QR_SIZE * mm, QR_SIZE * mm)

    c.setFont("DejaVu", 16)
    c.setFillColor(HexColor(T.INK_2))
    c.drawRightString(
        X(qx - 16),
        Y(FOOTER_Y + 22),
        "MKAT INDUSTRIES LLC",
    )
    c.setFont("DejaVu-Bold", 16)
    c.setFillColor(HexColor(T.ACCENT))
    c.drawRightString(X(qx - 10), Y(FOOTER_Y + 3), CODE_URL)

    # Scope fills the footer's empty left half and wraps to a readable measure.
    para(c, SCOPE, MARGIN, FOOTER_Y + 28, 640.0, size=16, color=T.INK_2)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    require_charts()
    register_fonts()
    out = BUILD / "video4real_poster_36x24in.pdf"
    c = canvas.Canvas(
        str(out),
        pagesize=(PAGE_W * mm, PAGE_H * mm),
        cropBox=(
            MEDIA_MARGIN * mm,
            MEDIA_MARGIN * mm,
            (MEDIA_MARGIN + TRIM_W) * mm,
            (MEDIA_MARGIN + TRIM_H) * mm,
        ),
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
    lower_top = scorecard_bottom - 10.0
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
    print(f"  media {PAGE_W:.1f} x {PAGE_H:.1f} mm; trim {TRIM_W:.1f} x {TRIM_H:.1f} mm; "
          f"bleed {BLEED:.0f} mm each edge; crop-mark slug {MEDIA_MARGIN - BLEED:.0f} mm")
    print(f"  scorecard bottom {scorecard_bottom:.1f} mm; lower panels "
          + ", ".join(f"{i + 1}:{b:.1f}" for i, b in enumerate(bottoms))
          + f" mm; footer air {gap:.1f} mm")


if __name__ == "__main__":
    main()

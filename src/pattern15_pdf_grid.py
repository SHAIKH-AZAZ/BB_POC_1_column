"""
pattern15_pdf_grid.py — deterministic grid builder for Pattern 15 (shear-wall schedule).

These CAD PDFs carry the column IDs (SW1..SWn) and the floor-level labels as real
vector TEXT, so we read them with pdfplumber instead of the vision model — this
removes label hallucination entirely (the model never names the columns).

The size / reinforcement / stirrup values are drawn as graphics (not text), so for
those we crop each column cell from the high-res render using the pdfplumber
coordinates, upscale, and let the vision model read only that tight region.

Public entry point: build_pattern15_cells(pdf_path, render_path) -> (list[CropJob], (RW, RH))
where each CropJob is {"label": "SW1", "levels": [..top-to-bottom..], "box": (x0,y0,x1,y1)}.
"""

import os
import re

import pdfplumber
from PIL import Image

# A column mark is a SHORT alphabetic prefix + digits (optional trailing letter):
# SW1, C34, LW1, TA-C1, PC1 ... . This is label-AGNOSTIC — we never assume "SW".
# The real marks are then selected structurally (see _find_mark_words) so title
# noise that happens to match this shape (e.g. "D1"/"D5" from a "TYPE-D1 TO D5"
# title) is rejected.
_MARK_RE = re.compile(r"^[A-Za-z]{1,4}-?\d+[A-Za-z]?$")

# How close (PDF points) a mark must sit to a "COLUMN MARKED" oval to count as a
# real, anchored mark. Comfortably above the genuine mark→oval spread (~40-56)
# and below the distance to unrelated title text (~120+).
_ANCHOR_DIST = 90
# How close COLUMN and MARKED words must be to form one oval anchor.
_OVAL_DIST = 30

# Tokens that make up the (180°-rotated / mirrored) floor-level labels, plus the
# upright spellings as a fallback for non-mirrored sheets.
_LEVEL_TOKENS = {
    "OT", "ROOLF", "DRIHT", "TSRIF", "NOITADNUOF", ".LEVEL", "ECARRET",
    "TO", "FLOOR", "THIRD", "FIRST", "FOUNDATION", "LEVEL", "TERRACE",
}

# Canonical level names for this pattern (3 fixed floor ranges).
_LVL_TERRACE = "THIRD FLOOR TO TERRACE LEVEL"
_LVL_MIDDLE = "FIRST FLOOR TO THIRD FLOOR LEVEL"
_LVL_FOUNDATION = "FOUNDATION TO FIRST FLOOR LEVEL"

# Tuning
_COL_MIN_W = 420       # min column crop width (px)
_COL_MAX_W = 1150      # max column crop width (px)
_CLUSTER_GAP = 400     # vertical gap (px) that marks a cell-block boundary
_LABEL_GAP = 60        # ignore level tokens nearer than this to the SW label


def _prefix(text):
    m = re.match(r"^[A-Za-z]+", text)
    return m.group().upper() if m else ""


def _find_mark_words(words):
    """Select the real column marks STRUCTURALLY (label/value-agnostic).

    A column schedule writes each mark inside a "COLUMN MARKED" oval. So:
      1. candidates = tokens shaped like a mark (short letters + digits),
      2. ovals      = places where a COLUMN word and a MARKED word are co-located,
      3. a candidate is "anchored" if it sits within _ANCHOR_DIST of an oval,
      4. the anchored candidates' PREFIXES define the live mark families,
      5. keep every candidate whose prefix is one of those families.

    Step 5 means a mark whose own oval text is broken/missing in the text layer
    (e.g. one whose "COLUMN MARKED" got split into single glyphs) is still kept
    via its family — while title noise like "D1"/"D5" (no oval, prefix never
    anchored) is dropped. Works for any prefix (SW, C, LW, PC, ...) and for
    sheets that mix several families.
    """
    def c(w):
        return (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2

    cand = [w for w in words if _MARK_RE.match(w["text"])]
    if not cand:
        return []

    cols = [c(w) for w in words if w["text"].upper() == "COLUMN"]
    mkd = [c(w) for w in words
           if w["text"].upper() == "MARKED" or w["text"].upper().startswith("MKD")]
    ovals = [
        (cx, cy) for (cx, cy) in cols
        if any((cx - mx) ** 2 + (cy - my) ** 2 < _OVAL_DIST ** 2 for mx, my in mkd)
    ]

    families = set()
    if ovals:
        for w in cand:
            cx, cy = c(w)
            if any((cx - ox) ** 2 + (cy - oy) ** 2 < _ANCHOR_DIST ** 2
                   for ox, oy in ovals):
                families.add(_prefix(w["text"]))

    if not families:
        # No usable oval anchors (oval text fully broken / non-text PDF): fall
        # back to the dominant prefix family among the candidates.
        from collections import Counter
        counts = Counter(_prefix(w["text"]) for w in cand)
        if counts:
            families = {counts.most_common(1)[0][0]}

    return [w for w in cand if _prefix(w["text"]) in families]


def _demirror(token):
    return token[::-1]


def _letters(token):
    return re.sub(r"[^A-Z]", "", token.upper())


def _column_width(token, all_tokens):
    """Width of a column = distance to the nearest SW label in the same row band."""
    gaps = [
        abs(o["cx"] - token["cx"])
        for o in all_tokens
        if o is not token
        and abs(o["top"] - token["top"]) < 300
        and abs(o["cx"] - token["cx"]) > 50
    ]
    width = min(gaps) if gaps else 900
    return min(max(width, _COL_MIN_W), _COL_MAX_W)


def _single_cell_level(token, width, level_tokens):
    """Identify the ONE level of a repeated single cell from its level keywords.

    The level label is rotated text on the cell's LEFT edge, while the SW mark is
    centred, so the cell extends well to the LEFT of the mark and only slightly to
    its right. Search that asymmetric window (and just above the mark) so we read
    the cell's OWN keywords and not the neighbour's.
    """
    x_lo = token["cx"] - width
    x_hi = token["cx"] + 0.35 * width
    y_lo = token["top"] - 900

    words = set()
    for tok in level_tokens:
        if x_lo < tok["cx"] < x_hi and y_lo < tok["top"] < token["top"]:
            words.add(_letters(_demirror(tok["text"])))
            words.add(_letters(tok["text"]))

    if "FOUNDATION" in words:
        return _LVL_FOUNDATION
    if "TERRACE" in words:
        return _LVL_TERRACE
    return _LVL_MIDDLE


def _crop_box(token, sw_tokens, level_tokens, single_cell):
    """Deterministic crop box for one SW label.

    A label that appears ONCE on the sheet sits below a full vertical STACK of
    level cells (its whole column), so we crop up to the topmost level token in
    its column. A label that REPEATS (single_cell=True) marks one horizontally
    placed cell, so we crop only the nearest cell-cluster above the label
    (bounded by the gap to the block above, which overlaps in x)."""
    width = _column_width(token, sw_tokens)
    xl, xr = token["cx"] - width / 2, token["cx"] + width / 2

    inbox = [
        lt for lt in level_tokens
        if xl < lt["cx"] < xr and lt["top"] < token["top"] - _LABEL_GAP
    ]
    if not inbox:
        return None, []

    tops = sorted((lt["top"] for lt in inbox), reverse=True)
    if single_cell:
        y_top = tops[0]
        for upper, lower in zip(tops, tops[1:]):
            if upper - lower > _CLUSTER_GAP:
                break
            y_top = lower
    else:
        y_top = tops[-1]  # full column -> topmost level token

    cell_tokens = [lt for lt in inbox if lt["top"] >= y_top - 1]
    box = (int(xl), int(max(0, y_top - 40)), int(xr), int(token["top"] - 15))
    return box, cell_tokens


def build_pattern15_cells(pdf_path, render_path):
    """Read SW labels + level labels from the PDF text and compute a crop box per
    column. Returns (jobs, (render_w, render_h))."""
    with Image.open(render_path) as render:
        rw, rh = render.size

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        pw, ph = page.width, page.height
        words = page.extract_words(x_tolerance=2, y_tolerance=2)

    sx, sy = rw / pw, rh / ph

    def to_px(word):
        return {
            "text": word["text"],
            "cx": (word["x0"] + word["x1"]) / 2 * sx,
            "top": word["top"] * sy,
        }

    sw_tokens = [to_px(w) for w in _find_mark_words(words)]
    level_tokens = [to_px(w) for w in words if w["text"] in _LEVEL_TOKENS]

    # A label that repeats marks single horizontally-placed cells; a label that
    # appears once sits below its full vertical stack of cells.
    counts = {}
    for t in sw_tokens:
        counts[t["text"]] = counts.get(t["text"], 0) + 1

    jobs = []
    for token in sorted(sw_tokens, key=lambda t: (t["top"], t["cx"])):
        single_cell = counts[token["text"]] > 1
        box, _ = _crop_box(token, sw_tokens, level_tokens, single_cell)
        if box is None:
            continue
        if single_cell:
            # one repeated cell -> identify its single level from its own keywords
            width = _column_width(token, sw_tokens)
            levels = [_single_cell_level(token, width, level_tokens)]
        else:
            # a label that appears once spans the full 3-level stack
            levels = [_LVL_TERRACE, _LVL_MIDDLE, _LVL_FOUNDATION]
        jobs.append({"label": token["text"], "levels": levels, "box": box})

    return jobs, (rw, rh)


def crop_cell_image(render_path, box, out_dir, label, min_longest=1100):
    """Crop the box from the render, upscale to >= min_longest, save, return path."""
    os.makedirs(out_dir, exist_ok=True)
    with Image.open(render_path).convert("RGB") as render:
        crop = render.crop(box)
        longest = max(crop.size)
        if longest and longest < min_longest:
            scale = min_longest / longest
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.LANCZOS,
            )
        safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "cell"
        path = os.path.join(out_dir, f"{safe}_{box[0]}_{box[1]}.png")
        crop.save(path, "PNG")
    return path

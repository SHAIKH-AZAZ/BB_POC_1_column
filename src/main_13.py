"""
Generalized Column Schedule PDF Extraction Pipeline — PATTERN 13  (v2 — FIXED)
===============================================================================

Fixes vs v1
-----------
1. PIL.Image.MAX_IMAGE_PIXELS = None
       Prevents the DecompressionBombWarning on large-format (A2) PDFs.

2. Default DPI changed to 300
       9925×14034 px at 600 DPI → manageable ~4962×7017 px at 300 DPI.
       Pass --dpi 600 if you need higher resolution.

3. Adaptive coarse min_dist for horizontal-line detection
       min_dist = max(H // 50, 60)
       At 300 DPI that is ≈140 px; at 600 DPI ≈280 px.
       Pattern-13 has TWO visual sub-rows per floor (structural drawing on
       top + text-data on bottom). The fine divider between them is ~100 px
       tall at 300 DPI — well below the adaptive threshold, so it is
       automatically skipped. Only the FLOOR SEPARATOR lines (700+ px apart)
       are kept, giving the correct ~17-row main grid.

4. find_best_grid tolerance raised to 0.60
       The TERRACE and BASEMENT rows are typically taller than the middle
       floors; a 35% tolerance was too tight and failed to accept the full
       run. 60% keeps them all in one grid.

5. Grid selection: prefer LARGEST COVERAGE, require minimum avg_h
       The loop now tracks best_coverage separately so it never regresses to
       a smaller grid found at an earlier threshold, and skips any candidate
       whose average row height is less than H/200 (clearly junk).

6. Text-strip isolation per floor row
       _find_text_strip_y() scans each floor row for the internal divider
       between the structural drawing (upper, cyan) and the text data (lower,
       black text). Only the text portion is sent to GPT-4o. Falls back to
       the bottom 38% of the row if no divider is found.

7. Updated GPT-4o prompts
       Explicitly tell the model that a structural cross-section drawing may
       occupy the upper portion of the image — it should focus only on the
       tabular text in the lower area.

JSON output schema — IDENTICAL to main_11.py
---------------------------------------------
{
  "document": "...",
  "title": "Column Schedule",
  "columns": [
    {
      "column_no":     "SW7",
      "column_name":   "FOURTEENTH_TO_TERRACE",
      "size":          {"width": ..., "depth": ..., "length": ...},
      "reinforcement": [...],
      "stirrups":      {"dia": [...], "spacing": [...]},
      "mix":           "M30" | null,
      "steel_grade":   null
    },
    ...
  ]
}

USAGE
-----
    python src/main_13.py
    python src/main_13.py --pdf pattern-13.pdf
    python src/main_13.py --pdf pattern-13.pdf --dpi 300 --debug

DEPENDENCIES
------------
    pip install pdf2image Pillow scipy numpy openai python-dotenv
    sudo apt-get install poppler-utils
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from scipy.signal import find_peaks

# ── Suppress PIL decompression-bomb error on large-format PDFs ────────────────
Image.MAX_IMAGE_PIXELS = None

# ── Project config ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import INPUT_DIR, OUTPUT_DIR, OPENAI_API_KEY


# ══════════════════════════════════════════════════════════════════════════════
# §1  PDF → Image
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_image(pdf_path: str, dpi: int = 300) -> Image.Image:
    """Render the first page of a PDF at the given DPI."""
    return convert_from_path(pdf_path, dpi=dpi)[0]


# ══════════════════════════════════════════════════════════════════════════════
# §2  Grid-Line Detection
# ══════════════════════════════════════════════════════════════════════════════

def _darkness_peaks(arr2d: np.ndarray, threshold: float, min_distance: int) -> list:
    darkness = np.mean(arr2d < 100, axis=1)
    peaks, _ = find_peaks(darkness, height=threshold, distance=min_distance)
    return peaks.tolist()


def _gradient_peaks(arr2d: np.ndarray, threshold: float, min_distance: int) -> list:
    grad = np.mean(np.abs(np.diff(arr2d.astype(float), axis=0)), axis=1)
    if grad.max() < 1e-6:
        return []
    norm_grad = grad / (grad.max() + 1e-9)
    abs_thr   = max(threshold, norm_grad.mean() + norm_grad.std())
    peaks, _  = find_peaks(norm_grad, height=abs_thr, distance=min_distance)
    return peaks.tolist()


def detect_lines(img: Image.Image,
                 y0: int = 0, y1: int = -1,
                 x0: int = 0, x1: int = -1,
                 h_thr: float = 0.15,
                 v_thr: float = 0.15,
                 min_dist: int = 30) -> tuple:
    arr = np.array(img, dtype=float)
    if y1 < 0: y1 = arr.shape[0]
    if x1 < 0: x1 = arr.shape[1]
    region = arr[y0:y1, x0:x1, :3]
    gray   = np.mean(region, axis=2)
    dark_h     = _darkness_peaks(gray,   h_thr, min_dist)
    grad_h     = _gradient_peaks(gray,   h_thr, min_dist)
    combined_h = cluster_lines(sorted(set(dark_h + grad_h)))
    v_raw = _darkness_peaks(gray.T, v_thr, min_dist)
    return [p + y0 for p in combined_h], [p + x0 for p in v_raw]


def cluster_lines(lines: list, gap: int = 12) -> list:
    if not lines:
        return []
    lines = sorted(lines)
    clusters, group = [], [lines[0]]
    for x in lines[1:]:
        if x - group[-1] <= gap:
            group.append(x)
        else:
            clusters.append(int(round(np.mean(group))))
            group = [x]
    clusters.append(int(round(np.mean(group))))
    return clusters


def find_regular_grid(lines: list, tolerance: float = 0.35) -> list:
    if len(lines) < 3:
        return lines
    lines = sorted(lines)
    gaps  = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
    med   = float(np.median(gaps))
    best_s, best_n = 0, 0
    cur_s,  cur_n  = 0, 1
    for i, g in enumerate(gaps):
        if abs(g - med) / (med + 1e-9) <= tolerance:
            cur_n += 1
        else:
            if cur_n > best_n:
                best_s, best_n = cur_s, cur_n
            cur_s, cur_n = i+1, 1
    if cur_n > best_n:
        best_s, best_n = cur_s, cur_n
    return lines[best_s: best_s + best_n + 1]


def find_best_grid(lines: list, min_rows: int = 3, tolerance: float = 0.60) -> list:
    """
    Find the grid scale that covers the MAXIMUM vertical span.
    tolerance raised to 0.60 for pattern-13 (non-uniform floor heights).
    """
    if len(lines) < min_rows + 1:
        return lines
    lines = sorted(lines)
    gaps  = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
    best_lines, best_coverage = [], 0

    def _evaluate(s: int, n: int):
        nonlocal best_lines, best_coverage
        if n < min_rows: return
        run      = lines[s: s+n]
        coverage = run[-1] - run[0]
        if coverage > best_coverage:
            best_coverage = coverage
            best_lines    = run

    for target in set(gaps):
        cur_s, cur_n = 0, 1
        for i, g in enumerate(gaps):
            if abs(g - target) / (target + 1e-9) <= tolerance:
                cur_n += 1
            else:
                _evaluate(cur_s, cur_n)
                cur_s, cur_n = i+1, 1
        _evaluate(cur_s, cur_n)

    return best_lines if best_lines else lines


def extend_grid_with_trailing_lines(grid: list, all_lines: list,
                                    max_extra_multiplier: float = 2.0) -> list:
    if not grid or len(grid) < 2:
        return grid
    med_gap = np.median([grid[i+1] - grid[i] for i in range(len(grid)-1)])
    last    = grid[-1]
    extras  = sorted(x for x in all_lines
                     if last < x <= last + max_extra_multiplier * med_gap)
    return grid + extras


# ══════════════════════════════════════════════════════════════════════════════
# §3  Text-Strip Isolation  (NEW — pattern-13 specific)
# ══════════════════════════════════════════════════════════════════════════════

def _find_text_strip_y(img_arr: np.ndarray,
                        y0: int, y1: int,
                        x0: int, x1: int) -> int:
    """
    Each pattern-13 floor row has TWO visual sub-rows:
      • UPPER  (~60-70 % of height): structural cross-section drawing in cyan.
      • LOWER  (~30-40 % of height): tabular text data (SIZE, CONC MIX, etc.).

    This function returns the Y coordinate of the DIVIDER between the two,
    so callers can crop only the text portion for GPT-4o extraction.

    Strategy:
      1. Search for a horizontal line (darkness OR gradient peak) inside the
         middle band of the row (40 %–88 % of height) — that is the divider.
      2. If no clear line is found, fall back to 62 % of height from the top
         (i.e. text starts at y0 + 0.62*(y1-y0)).

    Returns an ABSOLUTE y coordinate (text_y0); text ends at y1.
    """
    row_h = y1 - y0
    if row_h < 40:
        return y0  # too small to split

    region = img_arr[y0:y1, x0:x1, :3]
    gray   = np.mean(region, axis=2)

    # Search band: 40 %–88 % of row height
    s_start = int(row_h * 0.40)
    s_end   = int(row_h * 0.88)
    if s_end - s_start < 5:
        return y0 + int(row_h * 0.62)

    band = gray[s_start:s_end]

    # Darkness signal (dark = < 100 / 255)
    darkness = np.mean(band < 100, axis=1)

    # Gradient signal (detects even coloured lines like cyan)
    grad = np.mean(np.abs(np.diff(band.astype(float), axis=0)), axis=1)
    if grad.max() > 1e-6:
        grad_n = grad / grad.max()
    else:
        grad_n = np.zeros(len(grad))

    # Pad gradient to same length as darkness
    if len(grad_n) < len(darkness):
        grad_n = np.append(grad_n, 0.0)

    combined = darkness + 0.5 * grad_n[:len(darkness)]

    if combined.max() < 0.04:
        return y0 + int(row_h * 0.62)  # fallback

    local_y  = int(np.argmax(combined))
    divider  = y0 + s_start + local_y
    return divider


# ══════════════════════════════════════════════════════════════════════════════
# §4  Vision API (OpenAI / GPT-4o)
# ══════════════════════════════════════════════════════════════════════════════

def _to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        raise RuntimeError("Run: pip install openai")


def call_vision(client, img: Image.Image, prompt: str, max_tokens: int = 500) -> str:
    b64  = _to_b64(img)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return resp.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# §5  Label Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _crop_upscale(img: Image.Image, x0: int, y0: int, x1: int, y1: int,
                  upscale: int = 2) -> Image.Image:
    cell = img.crop((x0, y0, x1, y1))
    if upscale > 1:
        cell = cell.resize(
            (cell.width * upscale, cell.height * upscale), Image.LANCZOS)
    return cell


# ─── Column-label helpers ─────────────────────────────────────────────────────

_COL_LABEL_RE = re.compile(r'^[A-Z]{1,3}\d{1,3}[A-Z]?$')

_SKIP_LABEL_WORDS = frozenset({
    "COL", "COLMARK", "COLMARKED", "COLNO", "COLUMN", "COLUMNS",
    "MARKED", "MARK", "SIZE", "CONC", "MIX", "VERT", "REINF", "RING",
    "NO", "NOTE", "NOTES", "SKIP", "NA", "NIA", "NIL",
    "FLOOR", "TO", "TERRACE", "LMR", "GROUND", "BASEMENT",
})


def _normalize_col_label(raw: str) -> str | None:
    if not raw:
        return None
    s = re.sub(r'[\s\u00a0]+', '', raw.strip()).upper().strip('.,;:-/\\()[]{}"\' ')
    if not s:
        return None
    letter_part = re.sub(r'\d', '', s)
    if letter_part in _SKIP_LABEL_WORDS:
        return None
    return s if _COL_LABEL_RE.match(s) else None


def _split_label_group(text: str) -> list:
    """'SW7, SW10, SW45, SW46' → ['SW7', 'SW10', 'SW45', 'SW46']"""
    if not text:
        return []
    raw_tokens = re.split(r'[\s,;/|]+', text.strip())
    out, seen = [], set()
    for tok in raw_tokens:
        norm = _normalize_col_label(tok)
        if norm and norm not in seen:
            seen.add(norm); out.append(norm)
    return out


def extract_col_label_groups_strip(client, img, header_y0, header_y1,
                                    data_col_x, upscale=2) -> list:
    """Read ALL column-label groups from the header strip in one API call."""
    n     = len(data_col_x) - 1
    strip = _crop_upscale(img, data_col_x[0], header_y0,
                          data_col_x[-1], header_y1, upscale)
    prompt = (
        f"This is the column-header row of a structural engineering "
        f"shear-wall / column schedule. It has exactly {n} cells left to right.\n\n"
        "Each cell lists ONE OR MORE wall/column labels, e.g.:\n"
        "  SW1, SW2, SW3, SW4\n"
        "  SW5, SW6, SW7, SW8\n"
        "  SW17, SW18\n\n"
        "Label format = 1-3 capital letters + 1-3 digits (optional suffix letter).\n\n"
        f"Return EXACTLY {n} lines, one per cell, left to right. "
        "Each line lists that cell's labels separated by commas. No other text.\n\n"
        "Example output (3 cells):\n"
        "SW1, SW2, SW3, SW4\n"
        "SW5, SW6, SW7, SW8\n"
        "SW9, SW10"
    )
    resp   = call_vision(client, strip, prompt, max_tokens=400).strip()
    lines  = [ln.strip() for ln in resp.split("\n") if ln.strip()]
    result = [_split_label_group(ln) for ln in lines[:n]]
    while len(result) < n:
        result.append([])
    return result[:n]


def extract_col_label_groups(client, img, header_y0, header_y1,
                              data_col_x, upscale=2) -> list:
    """Per-cell fallback."""
    n      = len(data_col_x) - 1
    result = []
    for i in range(n):
        cell   = _crop_upscale(img, data_col_x[i], header_y0,
                                data_col_x[i+1], header_y1, upscale)
        prompt = (
            "This is a single header cell from a structural-engineering column schedule.\n"
            "It may contain ONE OR MORE labels like 'SW1, SW2, SW3, SW4' or 'SW17, SW18'.\n"
            f"Cell {i+1} of {n}. Format: 1-3 capital letters + 1-3 digits.\n"
            "Return ONLY the labels, comma-separated. If none legible → 'SKIP'."
        )
        raw = call_vision(client, cell, prompt, max_tokens=60).strip().strip('"\'')
        result.append([] if raw.upper() == "SKIP" else _split_label_group(raw))
    return result


# ─── Floor-label helpers ──────────────────────────────────────────────────────

def extract_floor_labels(client, img, label_x0, label_x1,
                          row_y, upscale=2) -> list:
    """
    Read the floor-range label for each row in the label column.
    Returns N strings: label text | 'SKIP' | 'Floor_N' fallback.
    """
    prompt = (
        "This is a single row-header cell from a structural engineering column schedule. "
        "Text is usually printed VERTICALLY (rotated 90°) and describes a FLOOR RANGE.\n\n"
        "Valid floor-range labels (any of these or similar):\n"
        "  'TERRACE TO LMR', 'FOURTEENTH TO TERRACE', 'THIRTEENTH TO FOURTEENTH',\n"
        "  'TWELVETH TO THIRTEENTH', 'ELEVENTH TO TWELVETH', 'TENTH TO ELEVENTH',\n"
        "  'NINTH TO TENTH', 'EIGHTH TO NINTH', 'SEVENTH TO EIGHTH',\n"
        "  'SIXTH TO SEVENTH', 'FIFTH TO SIXTH', 'FOURTH TO FIFTH',\n"
        "  'THIRD TO FOURTH', 'SECOND TO THIRD', 'FIRST TO SECOND',\n"
        "  'GROUND TO FIRST', 'BASEMENT TO GROUND'\n\n"
        "NON-floor cells: 'FOOTING', 'COL. MARKED', 'SIZE', 'CONC. MIX', "
        "'VERT. REINF.', 'RING', repeated header text, or blank.\n\n"
        "If this IS a floor label → return the text EXACTLY as written.\n"
        "If this is NOT → return exactly: SKIP\n"
        "Return nothing else."
    )
    labels = []
    for i in range(len(row_y) - 1):
        cell = _crop_upscale(img, label_x0, row_y[i], label_x1, row_y[i+1], upscale)
        raw  = call_vision(client, cell, prompt, max_tokens=60).strip().strip('"\'')
        if raw.upper() == "SKIP":
            labels.append("SKIP")
        elif raw:
            labels.append(raw)
        else:
            labels.append(f"Floor_{i+1}")
    return labels


def filter_and_recheck_floor_rows(client, img, label_x0, label_x1,
                                   h_grid, raw_labels, upscale=2) -> tuple:
    """Drop SKIPs, re-query Floor_N fallbacks. Returns (row_bounds, labels)."""
    row_bounds, floor_labels = [], []
    for i, label in enumerate(raw_labels):
        y0, y1 = h_grid[i], h_grid[i+1]
        if label == "SKIP":
            print(f"       Row {i+1} ({y0}–{y1} px): excluded (non-floor)")
            continue
        if not label.startswith("Floor_"):
            row_bounds.append((y0, y1)); floor_labels.append(label); continue

        cell  = _crop_upscale(img, label_x0, y0, label_x1, y1, upscale)
        check = call_vision(client, cell,
            "Does this cell contain a floor/level name or range "
            "(e.g. 'GROUND TO FIRST', 'TERRACE TO LMR', '7TH FLOOR')? YES or NO.",
            max_tokens=5).strip().upper()

        if "YES" not in check:
            print(f"       Row {i+1} ({y0}–{y1} px): excluded (confirmed non-floor)")
            continue

        retry = call_vision(client, cell,
            "Read the floor-range label in this cell (may be vertical). "
            "E.g. 'TERRACE TO LMR', 'FOURTEENTH TO TERRACE', 'BASEMENT TO GROUND'. "
            "Return ONLY the label text.",
            max_tokens=60).strip().strip('"\'')
        final = retry if retry and len(retry) < 80 else label
        print(f"       Row {i+1} ({y0}–{y1} px): re-read as '{final}'")
        row_bounds.append((y0, y1)); floor_labels.append(final)

    return row_bounds, floor_labels


# ══════════════════════════════════════════════════════════════════════════════
# §6  Data Cell Extraction
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_CELL = {"SIZE": "---", "CONC_MIX": "---", "VERT_REINF": "---", "RING": "---"}

# Updated prompts: explicitly tell GPT-4o to ignore structural drawings
_DATA_PROMPT_TEMPLATE = (
    "This image shows a cell from a structural-engineering shear-wall / column schedule.\n"
    "Floor range: {floor}.  Wall/Column group: {column}.\n\n"
    "IMPORTANT: The upper portion of the image may contain a structural "
    "cross-section drawing (shapes, dots, reinforcement layout diagrams). "
    "IGNORE that drawing completely.\n\n"
    "Focus ONLY on the TABULAR TEXT in the lower portion of the image, "
    "which contains these four fields:\n"
    '  SIZE         (e.g. "230 X 1500", "230 X 1250", "300 X 900", "AS PER PLAN")\n'
    '  CONC. MIX    (e.g. "M25", "M30", "M35")\n'
    '  VERT. REINF. (e.g. "14-20 TOR + 6-16 TOR", "49-25 TOR", "8-16 TOR")\n'
    '  RING         (e.g. "8 TOR 15 @ 75 + @ 150 + 15 @ 75 C/C, 4 SETS + 1 LINK")\n\n'
    'Return ONLY a JSON object:\n'
    '{{"SIZE":"...","CONC_MIX":"...","VERT_REINF":"...","RING":"..."}}\n'
    'Use "---" for any field that is not visible or unreadable.'
)

_COMBINED_PROMPT_TEMPLATE = (
    "This is a FULL-WIDTH row from a structural-engineering column schedule.\n"
    "Floor range: {floor}.\n\n"
    "IMPORTANT: Ignore any structural cross-section drawings in the upper portion. "
    "Focus ONLY on the tabular text (SIZE, CONC. MIX, VERT. REINF., RING) "
    "in the lower text band.\n\n"
    "This row shows ONE shared specification for ALL columns.\n"
    'Return ONLY a JSON object:\n'
    '{{"SIZE":"...","CONC_MIX":"...","VERT_REINF":"...","RING":"..."}}\n'
    'Use "---" for any field not visible or unreadable.'
)


def _parse_json_response(text: str) -> dict:
    try:
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return dict(_EMPTY_CELL)


def extract_data_cell(client, img: Image.Image, floor: str, column: str) -> dict:
    return _parse_json_response(
        call_vision(client, img,
                    _DATA_PROMPT_TEMPLATE.format(floor=floor, column=column),
                    max_tokens=500))


def extract_combined_row(client, img: Image.Image, floor: str) -> dict:
    return _parse_json_response(
        call_vision(client, img,
                    _COMBINED_PROMPT_TEMPLATE.format(floor=floor),
                    max_tokens=500))


def _is_empty(cell_data: dict) -> bool:
    return all(cell_data.get(k, "---") == "---" for k in _EMPTY_CELL)


# ══════════════════════════════════════════════════════════════════════════════
# §7  Output JSON Parsers  (identical schema to main_11)
# ══════════════════════════════════════════════════════════════════════════════

def parse_size(raw: str) -> dict:
    empty = {"width": None, "depth": None, "length": None}
    if not raw or raw.strip() in ("---", "", "N/A", "n/a"):
        return empty
    parts = re.split(r'\s*[xX×]\s*', raw.strip())
    nums  = []
    for p in parts:
        try:    nums.append(int(float(p.strip())))
        except: pass
    if len(nums) == 2:  return {"width": nums[0], "depth": None,    "length": nums[1]}
    if len(nums) >= 3:  return {"width": nums[0], "depth": nums[1], "length": nums[2]}
    return empty


def parse_reinforcement(raw: str) -> list:
    if not raw or raw.strip() in ("---", ""):
        return []
    result = []
    for part in re.split(r'\s*\+\s*', raw.strip()):
        part = part.strip()
        if not part: continue
        m = re.match(r'(\d+)\s*[-–]\s*(\d+)\s*(?:TOR|T)\b', part, re.IGNORECASE)
        if m:
            result.append(f"{m.group(1)}-{m.group(2)}T")
        else:
            nums = re.findall(r'\d+', part)
            if len(nums) >= 2:
                result.append(f"{nums[0]}-{nums[1]}T")
    return result


def parse_stirrups(raw: str) -> dict:
    empty = {"dia": [], "spacing": []}
    if not raw or raw.strip() in ("---", ""):
        return empty
    primary = re.findall(r'(\d+)\s*(?:TOR|T)\s*(\d+)', raw, re.IGNORECASE)
    dias    = [f"{m[0]}-T{m[1]}" for m in primary]
    if not dias:
        fallback = re.findall(r'(\d+)\s*TOR\b(?!\s*\d)', raw, re.IGNORECASE)
        dias = [f"T{d}" for d in fallback]
    seen, spacings = set(), []
    for n in re.findall(r'(\d+)\s*C/C', raw, re.IGNORECASE):
        if n not in seen: spacings.append(f"{n} C/C"); seen.add(n)
    for n in re.findall(r'@\s*(\d+)', raw):
        if n not in seen: spacings.append(f"{n} C/C"); seen.add(n)
    return {"dia": dias, "spacing": spacings}


def _build_entry(clabel: str, flabel: str, raw: dict, _to_key) -> dict:
    mix_raw = raw.get("CONC_MIX", "---")
    return {
        "column_no":     clabel,
        "column_name":   _to_key(flabel),
        "size":          parse_size(raw.get("SIZE", "---")),
        "reinforcement": parse_reinforcement(raw.get("VERT_REINF", "---")),
        "stirrups":      parse_stirrups(raw.get("RING", "---")),
        "mix":           mix_raw if mix_raw and mix_raw != "---" else None,
        "steel_grade":   None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §8  Debug Overlay
# ══════════════════════════════════════════════════════════════════════════════

def save_debug_overlay(img, h_grid, v_grid, header_band, label_band, out_path):
    W, H    = img.size
    rgba    = img.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr      = ImageDraw.Draw(overlay)
    for y in h_grid:
        dr.line([(0, y), (W, y)], fill=(0, 220, 0, 200), width=3)
    for x in v_grid:
        dr.line([(x, 0), (x, H)], fill=(30, 120, 255, 200), width=3)
    hy0, hy1 = header_band
    dr.rectangle([0, hy0, W, hy1], fill=(255, 220, 0, 60))
    lx0, lx1 = label_band
    dr.rectangle([lx0, 0, lx1, H], fill=(255, 110, 0, 60))
    Image.alpha_composite(rgba, overlay).convert("RGB").save(out_path)
    print(f"       Debug overlay → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# §9  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def extract_column_schedule(pdf_path:    str,
                             output_path: str  = "output.json",
                             dpi:         int  = 300,
                             upscale:     int  = 2,
                             debug:       bool = False) -> dict:
    client = get_client()

    # ── Step 1: Render ────────────────────────────────────────────────────────
    print(f"\n[1/6] Rendering '{Path(pdf_path).name}' at {dpi} DPI …")
    img     = pdf_to_image(pdf_path, dpi=dpi)
    W, H    = img.size
    img_arr = np.array(img, dtype=float)   # kept for sub-row detection
    print(f"       Image: {W} × {H} px")

    # ── Step 2: Adaptive COARSE horizontal-line detection ─────────────────────
    #
    # KEY FIX: Pattern-13 has two visual sub-rows per floor:
    #   • Structural drawing row  (upper, ~65 % of floor height)
    #   • Text-data row           (lower, ~35 % of floor height)
    # The fine divider between them is ~100 px at 300 DPI.
    # The floor separator lines are ~400 px apart at 300 DPI.
    # By setting min_dist = max(H//50, 60) ≈ 140 px at 300 DPI we skip the
    # fine dividers and detect only the main floor separator lines.
    #
    print("[2/6] Detecting horizontal grid lines (coarse, adaptive min_dist) …")

    # FIX: sub-row dividers are ~H/46 px tall (151 px at 300 DPI).
    # Using H//40 (~175 px) ensures they are filtered; floor separators
    # (~H/22 = 318 px) are still easily detected.
    MIN_DIST_H  = max(H // 40, 80)     # ← coarse; skips drawing/text sub-row dividers
    H_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    min_floor_h  = H / 50              # avg floor row ≥ H/50 px

    h_all         = []
    h_grid        = []
    chosen_thr    = H_THRESHOLDS[0]
    best_coverage = 0                  # ← track to avoid regression

    for h_thr in H_THRESHOLDS:
        h_raw, _ = detect_lines(img, h_thr=h_thr, v_thr=0.0, min_dist=MIN_DIST_H)
        cand_all  = cluster_lines(sorted(h_raw))

        if len(cand_all) < 4:
            continue

        # Raised tolerance = 0.60 handles non-uniform row heights (TERRACE,
        # BASEMENT rows are taller / shorter than the typical middle floor).
        cand_grid = find_best_grid(cand_all, min_rows=3, tolerance=0.60)
        n_cand    = len(cand_grid) - 1
        if n_cand < 3:
            continue

        avg_h    = (cand_grid[-1] - cand_grid[0]) / n_cand
        coverage = cand_grid[-1] - cand_grid[0]

        # Only update if this grid is LARGER (more coverage) and has a
        # plausible row height (not junk sub-lines that slipped through).
        if coverage > best_coverage and avg_h >= H / 200:
            best_coverage = coverage
            h_all         = cand_all
            h_grid        = cand_grid
            chosen_thr    = h_thr

        if avg_h >= min_floor_h:
            break                      # good enough — stop trying

    print(f"       {len(h_all)} H-lines  (h_thr={chosen_thr}, min_dist={MIN_DIST_H})")

    # Append any extra row-boundary lines just below the detected grid
    if len(h_grid) >= 2:
        med_row_h = float(np.median(
            [h_grid[i+1] - h_grid[i] for i in range(len(h_grid)-1)]))
        extra = sorted(y for y in h_all
                       if h_grid[-1] < y <= h_grid[-1] + med_row_h * 1.5
                       and y not in h_grid)
        if extra:
            print(f"       Appending {len(extra)} extra row-boundary line(s): y={extra}")
            h_grid += extra

    # ── Step 3: Validate row count ────────────────────────────────────────────
    print("[3/6] Identifying data rows …")
    if len(h_grid) < 3:
        raise ValueError(
            f"Only {len(h_grid)} consistent horizontal grid lines found "
            f"(need ≥ 3). Try --dpi 300 or --debug to inspect.\n"
            f"  H={H}, MIN_DIST_H={MIN_DIST_H}, h_thr={chosen_thr}"
        )

    n_rows = len(h_grid) - 1
    print(f"       {n_rows} data rows  |  Y: {h_grid[0]} – {h_grid[-1]} px  "
          f"|  avg row height: {(h_grid[-1]-h_grid[0])//n_rows} px")

    # ── Step 4: Vertical lines scoped to the table's Y range ─────────────────
    print("[4/6] Detecting vertical grid lines …")
    # FIX: use darkness + gradient for vertical lines (detects coloured borders),
    # and swap find_regular_grid → find_best_grid so we maximise horizontal SPAN
    # (= the actual column boundaries) rather than the most uniform spacing
    # (which often picks noisy internal drawing lines in a narrow range).
    _h_lines_v, v_dark = detect_lines(
        img, y0=h_grid[0], y1=h_grid[-1],
        h_thr=0.0, v_thr=0.15, min_dist=30,
    )
    # Transpose region and run gradient peaks to also catch coloured/cyan borders
    _arr_v = np.array(img, dtype=float)[h_grid[0]:h_grid[-1], :, :3]
    _gray_v = np.mean(_arr_v, axis=2)
    v_grad_raw = _gradient_peaks(_gray_v.T, threshold=0.15, min_distance=30)
    v_raw_all  = cluster_lines(sorted(set(v_dark + v_grad_raw)))

    v_all  = v_raw_all
    # find_best_grid: picks the run with MAX X-span → the true column boundaries.
    # Structural-drawing noise lines only span one cell width (~400 px);
    # real column separators span the full table width (3 000–4 500 px).
    v_core = find_best_grid(v_all, min_rows=3, tolerance=0.60)
    v_grid = extend_grid_with_trailing_lines(v_core, v_all, max_extra_multiplier=3.0)
    print(f"       {len(v_raw_all)} vertical lines after clustering  "
          f"(darkness+gradient)  →  {len(v_core)} in regular grid  "
          f"→  {len(v_grid)} after trailing extension")

    if len(v_grid) < 3:
        raise ValueError(
            f"Only {len(v_grid)} vertical grid lines found (need ≥ 3). "
            "Use --debug or try a different DPI."
        )

    # ── Step 5a: Identify label column + header band ──────────────────────────
    typical_col_w = float(np.median(
        [v_grid[i+1] - v_grid[i] for i in range(len(v_grid)-1)]))

    if v_grid[0] <= int(typical_col_w * 0.5):
        label_x0, label_x1, data_col_x = v_grid[0], v_grid[1], v_grid[1:]
    else:
        label_x1   = v_grid[0]
        label_x0   = max(0, label_x1 - int(typical_col_w * 2.0))
        data_col_x = v_grid

    n_cols = len(data_col_x) - 1
    if n_cols < 1:
        raise ValueError("No data columns detected. Use --debug to inspect.")

    print(f"       {n_cols} data column cells  |  X: {data_col_x[0]} – {data_col_x[-1]} px")

    above_h = [y for y in h_all if y < h_grid[0]]
    below_h = [y for y in h_all if y > h_grid[-1]]

    header_y0 = above_h[-1] if above_h else max(0, h_grid[0]-(h_grid[1]-h_grid[0])//2)
    header_y1 = h_grid[0]
    footer_y0 = h_grid[-1] if below_h else None
    footer_y1 = below_h[0]  if below_h else None

    print(f"       Header Y: {header_y0}–{header_y1}  |  Label X: {label_x0}–{label_x1}")
    if footer_y0:
        print(f"       Footer Y: {footer_y0}–{footer_y1}")

    if debug:
        save_debug_overlay(img, h_grid, v_grid,
                           (header_y0, header_y1), (label_x0, label_x1),
                           str(Path(output_path).with_suffix(".debug.png")))

    # ── Step 5b: Column-label groups ──────────────────────────────────────────
    print("[5/6] Extracting column-label groups and floor labels …")

    def _attempt_col_groups(y0, y1, name):
        print(f"       [{name}] Y {y0}–{y1} …")
        groups = extract_col_label_groups_strip(client, img, y0, y1, data_col_x, upscale)
        while len(groups) < n_cols: groups.append([])
        groups = groups[:n_cols]
        empty_idxs = [i for i, g in enumerate(groups) if not g]
        if empty_idxs:
            print(f"         {len(empty_idxs)} empty → per-cell retry")
            per_cell = extract_col_label_groups(client, img, y0, y1, data_col_x, upscale)
            for idx in empty_idxs:
                if idx < len(per_cell) and per_cell[idx]:
                    groups[idx] = per_cell[idx]
        fb = sum(1 for g in groups if not g)
        for idx in range(n_cols):
            if not groups[idx]:
                groups[idx] = [f"COL_{idx+1}"]; fb += 1
        total = sum(len(g) for g in groups)
        print(f"         Result ({fb} fallback(s), {total} labels): {groups}")
        return groups, fb

    col_groups, fb = _attempt_col_groups(header_y0, header_y1, "header")

    if footer_y0 is not None:
        f_groups, f_fb = _attempt_col_groups(footer_y0, footer_y1, "footer")
        h_total = sum(len(g) for g in col_groups)
        f_total = sum(len(g) for g in f_groups)
        if f_fb < fb or (f_fb == fb and f_total > h_total):
            print("       Footer wins — using footer labels.")
            col_groups, fb = f_groups, f_fb
        else:
            print("       Header kept.")

    # ── Step 5c: Floor labels ─────────────────────────────────────────────────
    raw_floor = extract_floor_labels(
        client, img, label_x0, label_x1, h_grid, upscale)
    print("       Validating floor rows …")
    row_bounds, floor_labels = filter_and_recheck_floor_rows(
        client, img, label_x0, label_x1, h_grid, raw_floor, upscale)

    if not row_bounds:
        raise ValueError("No valid floor rows found. Use --debug to inspect the grid.")

    # Secondary label fallback
    if fb > n_cols // 2 and row_bounds:
        last_y1 = row_bounds[-1][1]
        for i, lbl in enumerate(raw_floor):
            if lbl == "SKIP" and h_grid[i] >= last_y1 - 50:
                sg, sfb = _attempt_col_groups(h_grid[i], h_grid[i+1],
                                              f"skip@{h_grid[i]}")
                if sfb < fb:
                    print(f"       SKIP row better ({sfb}) — using it.")
                    col_groups, fb = sg, sfb
                    break

    def _to_key(s): return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    total_labels = sum(len(g) for g in col_groups)
    n_rows       = len(row_bounds)
    print(f"       ✓ Cells({n_cols})  Labels({total_labels})  Floors({n_rows})")

    # ── Step 6: Extract data cells ────────────────────────────────────────────
    #
    # PATTERN-13 CELL STRUCTURE:
    #   Each floor row = drawing (top) + text data (bottom).
    #   _find_text_strip_y() finds the divider so we only send GPT-4o
    #   the text portion — faster, more accurate, fewer hallucinations.
    #
    total = n_rows * n_cols
    print(f"[6/6] Extracting {total} data cells via Vision API …")

    result = {"document": Path(pdf_path).name, "title": "Column Schedule", "columns": []}
    seen: set = set()
    done = 0

    for r in range(n_rows):
        y0, y1 = row_bounds[r]
        flabel  = floor_labels[r]
        row_raw = []

        # Find text-strip Y for this floor row (shared across all columns)
        text_y0 = _find_text_strip_y(img_arr, y0, y1, data_col_x[0], data_col_x[-1])
        text_y1 = y1
        strip_pct = int(100 * (text_y1 - text_y0) / max(y1 - y0, 1))
        print(f"       Floor '{flabel}': text strip = rows {text_y0}–{text_y1} "
              f"({strip_pct}% of row height)")

        for c in range(n_cols):
            x0, x1  = data_col_x[c], data_col_x[c+1]
            group    = col_groups[c]
            dlabel   = group[0] if group else f"COL_{c+1}"

            # Crop ONLY the text-data portion (ignore the drawing above)
            cell  = _crop_upscale(img, x0, text_y0, x1, text_y1, upscale)
            done += 1
            print(f"       [{done:>3}/{total}]  {flabel} / {dlabel} …",
                  end=" ", flush=True)
            raw = extract_data_cell(client, cell, flabel, dlabel)
            print("✓")
            row_raw.append(raw)

        # Combined-row fallback: > 50 % empty → re-read full text strip
        empty_count = sum(1 for d in row_raw if _is_empty(d))
        if empty_count > n_cols // 2:
            print(f"       ↳ {empty_count}/{n_cols} empty — re-reading full width …",
                  end=" ", flush=True)
            full_row = _crop_upscale(img, data_col_x[0], text_y0,
                                     data_col_x[-1], text_y1, upscale)
            combined = extract_combined_row(client, full_row, flabel)
            print("✓")
            if not _is_empty(combined):
                print(f"       ↳ Broadcasting to all {n_cols} cells.")
                row_raw = [combined] * n_cols

        # BROADCAST: one JSON entry per label in each cell's group
        floor_key = _to_key(flabel)
        for c, raw in enumerate(row_raw):
            group = col_groups[c] if col_groups[c] else [f"COL_{c+1}"]
            for clabel in group:
                key = (floor_key, clabel)
                if key in seen: continue
                seen.add(key)
                result["columns"].append(_build_entry(clabel, flabel, raw, _to_key))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Done!  {len(result['columns'])} entries → {output_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §10  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_output_path(pdf_path: str, output_dir: str) -> str:
    stem = Path(pdf_path).stem
    return os.path.join(output_dir, stem, f"{stem}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Column Schedule Extractor — Pattern 13 (fixed for A2 drawing-+text cells)")
    ap.add_argument("--pdf",        default=None,
                    help="Single PDF filename or full path.")
    ap.add_argument("--input-dir",  default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--dpi",        type=int, default=300,
                    help="Render DPI (default 300; use 150 for faster test runs)")
    ap.add_argument("--upscale",    type=int, default=2,
                    help="Per-cell upscale factor for OCR clarity")
    ap.add_argument("--debug",      action="store_true",
                    help="Save colour-coded grid-overlay PNG")
    args = ap.parse_args()

    if args.pdf:
        candidate = (args.pdf if os.path.isabs(args.pdf)
                     else os.path.join(args.input_dir, args.pdf))
        if not os.path.isfile(candidate):
            print(f"ERROR: PDF not found → {candidate}", file=sys.stderr)
            sys.exit(1)
        pdf_files = [candidate]
    else:
        pdf_files = [str(p) for p in sorted(Path(args.input_dir).glob("*.pdf"))]
        if not pdf_files:
            print(f"No PDFs in {args.input_dir}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    success, failed = [], []

    print(f"\n{'='*60}")
    print(f"  Column Schedule Extractor  (Pattern 13 — v2)")
    print(f"  DPI: {args.dpi}  |  Upscale: {args.upscale}x")
    print(f"{'='*60}")

    for pdf_path in pdf_files:
        out_path = _build_output_path(pdf_path, args.output_dir)
        print(f"\n▶  {Path(pdf_path).name}  →  {Path(out_path).name}")
        try:
            extract_column_schedule(
                pdf_path, out_path, args.dpi, args.upscale, args.debug)
            success.append(Path(pdf_path).name)
        except Exception as exc:
            print(f"\n❌  {Path(pdf_path).name} — {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
            failed.append(Path(pdf_path).name)

    print(f"\n{'='*60}")
    print(f"  {len(success)} succeeded  |  {len(failed)} failed")
    if success: print(f"  ✅  {', '.join(success)}")
    if failed:  print(f"  ❌  {', '.join(failed)}")
    print(f"{'='*60}\n")
    sys.exit(0 if not failed else 1)
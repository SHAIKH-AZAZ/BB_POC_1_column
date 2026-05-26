"""
Generalized Column Schedule PDF Extraction Pipeline
====================================================

Fully dynamic extraction — ZERO hard-coded pixel coordinates.

Works with any structural-engineering column schedule PDF that follows
the same visual pattern, regardless of:
  • Number of floors  (rows in the data grid)
  • Number of columns (C1, C2, C2A … any count, any width)
  • Page size / DPI
  • Combined / merged rows where one spec covers all columns

APPROACH
--------
1. Render PDF page to a high-resolution image (600 DPI by default).
2. Detect horizontal grid lines on the full image → adaptive threshold
   loop (darkness + gradient) to find the longest run of evenly-spaced
   lines (= data rows).
3. Re-detect vertical lines scoped to the table's Y range →
   cluster them (= data column boundaries).
4. Locate the column-header row (above or below the data grid)
   and the floor-label column (left of the first data col).
5. Extract column labels via a single "full-strip" API call first;
   fall back to per-cell with normalization; repair gaps by sequence.
6. For each data row:
   a. Extract individual column cells via GPT-4o.
   b. If >50 % of cells return empty data, the row is a COMBINED row
      (one shared spec for all columns) — re-read the full row width as
      a single image and broadcast the result to every column.
7. Assemble everything into the flat JSON schema.

USAGE
-----
    # Process every PDF inside the configured input folder:
    python src/main_11.py

    # Process a single PDF (path relative to the input folder OR absolute):
    python src/main_11.py --pdf 1.pdf

    # Extra options:
    python src/main_11.py [--dpi 600] [--upscale 2] [--debug]

DEPENDENCIES
------------
    pip install pdf2image Pillow scipy numpy openai python-dotenv
    sudo apt-get install poppler-utils   # or brew install poppler
"""

import argparse
import base64
import json
from extraction_guard import reshape_columns_to_levels
from pattern_batching import atomic_write_json, safe_filename
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from scipy.signal import find_peaks

# ── Project config ────────────────────────────────────────────────────────────
# Allows running from the project root as:  python src/main_11.py
sys.path.insert(0, str(Path(__file__).parent))
from config import INPUT_DIR, OUTPUT_DIR, OPENAI_API_KEY


# ══════════════════════════════════════════════════════════════════════════════
# §1  PDF → Image
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_image(pdf_path: str, dpi: int = 600) -> Image.Image:
    """Render the first page of a PDF at the given DPI."""
    return convert_from_path(pdf_path, dpi=dpi)[0]


# ══════════════════════════════════════════════════════════════════════════════
# §2  Grid-Line Detection
# ══════════════════════════════════════════════════════════════════════════════

def _darkness_peaks(arr2d: np.ndarray,
                    threshold: float,
                    min_distance: int) -> list:
    """
    Return row indices where the fraction of dark pixels (value < 100)
    is a local maximum above `threshold`.
    """
    darkness = np.mean(arr2d < 100, axis=1)
    peaks, _ = find_peaks(darkness, height=threshold, distance=min_distance)
    return peaks.tolist()


def _gradient_peaks(arr2d: np.ndarray,
                    threshold: float,
                    min_distance: int) -> list:
    """
    Return row indices with strong vertical gradient (edge-based detection).

    Useful for coloured grid-lines that are not captured by darkness alone:
    a horizontal line creates a sharp brightness transition even if its
    absolute grayscale value is not very low.

    `threshold` is treated as a fraction of the signal's mean + 1 std-dev
    so it stays scale-independent across different PDF styles.
    """
    grad = np.mean(np.abs(np.diff(arr2d.astype(float), axis=0)), axis=1)
    if grad.max() < 1e-6:
        return []
    norm_grad = grad / (grad.max() + 1e-9)
    abs_thr   = max(threshold, norm_grad.mean() + norm_grad.std())
    peaks, _  = find_peaks(norm_grad, height=abs_thr, distance=min_distance)
    return peaks.tolist()


def detect_lines(img: Image.Image,
                 y0: int = 0,   y1: int = -1,
                 x0: int = 0,   x1: int = -1,
                 h_thr: float = 0.15,
                 v_thr: float = 0.15,
                 min_dist: int = 30) -> tuple:
    """
    Detect horizontal and vertical dark lines within a sub-region of `img`.

    For horizontal lines both darkness-peak and gradient-peak methods are
    used and the results are merged; this makes detection robust for PDFs
    where grid lines are drawn in colour (not pure black).

    Returns (h_lines, v_lines) as absolute pixel positions.
    """
    arr = np.array(img, dtype=float)
    if y1 < 0:
        y1 = arr.shape[0]
    if x1 < 0:
        x1 = arr.shape[1]

    region = arr[y0:y1, x0:x1, :3]
    gray   = np.mean(region, axis=2)          # (H, W) — avoids 3-D scipy bug

    # Horizontal lines — combine darkness + gradient signals, then cluster
    dark_h     = _darkness_peaks(gray,   h_thr, min_dist)
    grad_h     = _gradient_peaks(gray,   h_thr, min_dist)
    combined_h = cluster_lines(sorted(set(dark_h + grad_h)))   # merge near-dups

    # Vertical lines — darkness only (transpose trick)
    v_raw = _darkness_peaks(gray.T, v_thr, min_dist)

    return [p + y0 for p in combined_h], [p + x0 for p in v_raw]


def cluster_lines(lines: list, gap: int = 12) -> list:
    """Merge groups of nearby lines (within `gap` px) into one representative."""
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
    """
    Return the longest contiguous run of lines whose inter-line gaps are
    all within `tolerance` (±35 %) of the median gap.
    """
    if len(lines) < 3:
        return lines

    lines = sorted(lines)
    gaps  = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    med   = float(np.median(gaps))

    best_s, best_n = 0, 0
    cur_s,  cur_n  = 0, 1

    for i, g in enumerate(gaps):
        if abs(g - med) / (med + 1e-9) <= tolerance:
            cur_n += 1
        else:
            if cur_n > best_n:
                best_s, best_n = cur_s, cur_n
            cur_s, cur_n = i + 1, 1

    if cur_n > best_n:
        best_s, best_n = cur_s, cur_n

    return lines[best_s: best_s + best_n + 1]


def find_best_grid(lines: list,
                   min_rows: int = 3,
                   tolerance: float = 0.35) -> list:
    """
    Find the grid scale that covers the MAXIMUM vertical span.

    Tries every unique gap value as a candidate row height and returns
    the run with the greatest total coverage (last_line − first_line).
    This prevents being fooled by the many short sub-row dividers that
    appear inside each floor cell on wide PDFs.

    BUG FIX: `run = lines[s : s+n]`  (not `s+n+1`).
    cur_n counts committed LINES (starts at 1 for the seed line), so
    the correct end index is s+n — not s+n+1 which would include one
    extra line beyond the matched run.
    """
    if len(lines) < min_rows + 1:
        return lines

    lines = sorted(lines)
    gaps  = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]

    best_lines    = []
    best_coverage = 0

    def _evaluate(s: int, n: int) -> None:
        nonlocal best_lines, best_coverage
        if n < min_rows:
            return
        run      = lines[s: s + n]         # FIXED: was lines[s: s+n+1]
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
                cur_s, cur_n = i + 1, 1
        _evaluate(cur_s, cur_n)

    return best_lines if best_lines else lines


def extend_grid_with_trailing_lines(grid: list,
                                    all_lines: list,
                                    max_extra_multiplier: float = 2.0) -> list:
    """
    After finding the regular vertical grid (uniform columns), append any
    extra lines to the right that form a wider final column.

    max_extra_multiplier=2.0 means the last extra column can be at most
    2× the median column width — prevents border/legend lines from being
    mistaken for data columns.
    """
    if not grid or len(grid) < 2:
        return grid

    med_gap = np.median([grid[i + 1] - grid[i] for i in range(len(grid) - 1)])
    last    = grid[-1]

    extras = sorted(
        x for x in all_lines
        if last < x <= last + max_extra_multiplier * med_gap
    )
    return grid + extras


# ══════════════════════════════════════════════════════════════════════════════
# §3  Vision API (OpenAI / GPT-4o)
# ══════════════════════════════════════════════════════════════════════════════

def _to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base-64 PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_client():
    """Instantiate the OpenAI client using the API key from config / .env."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        raise RuntimeError("Run: pip install openai")


def call_vision(client,
                img:        Image.Image,
                prompt:     str,
                max_tokens: int = 500) -> str:
    """Send `img` + `prompt` to GPT-4o and return the text response."""
    b64  = _to_b64(img)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":    f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# §4  Label Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _crop_upscale(img: Image.Image,
                  x0: int, y0: int, x1: int, y1: int,
                  upscale: int = 2) -> Image.Image:
    """Crop a region and upscale it for clearer text recognition."""
    cell = img.crop((x0, y0, x1, y1))
    if upscale > 1:
        cell = cell.resize(
            (cell.width * upscale, cell.height * upscale),
            Image.LANCZOS,
        )
    return cell


# ── Column-label helpers ──────────────────────────────────────────────────────

# Pattern: 1-2 uppercase letters + 1-3 digits + optional 1 uppercase letter
# Examples: C1, C2, C2A, C10, C11, C12, C13, C14, D3, PC1
_COL_LABEL_RE = re.compile(r'^[A-Z]{1,2}\d{1,3}[A-Z]?$')

# Words that are clearly NOT structural-column labels even if they slip through
_SKIP_LABEL_WORDS = frozenset({
    "COL", "COLMARK", "COLNO", "FLOOR", "MARK", "SIZE", "CONC",
    "MIX", "VERT", "REINF", "RING", "NO", "NOTE", "NOTES",
    "SKIP", "NA", "NIA", "NIL",
})


def _normalize_col_label(raw: str) -> str | None:
    """
    Clean and validate a raw column-label string.

    Handles common OCR / model noise:
      'C 1'   → 'C1'
      'c2a'   → 'C2A'
      'C1.'   → 'C1'
      'C10,'  → 'C10'
      'COL.1' → None  (rejected — non-standard)

    Returns None if the string cannot be turned into a valid column label.
    """
    if not raw:
        return None
    s = re.sub(r'[\s\u00a0]+', '', raw.strip())
    s = s.upper()
    s = s.strip('.,;:-/\\()[]{}"\' ')
    if not s:
        return None
    letter_part = re.sub(r'\d', '', s)
    if letter_part in _SKIP_LABEL_WORDS:
        return None
    if _COL_LABEL_RE.match(s):
        return s
    return None


def extract_col_labels_strip(client,
                              img: Image.Image,
                              header_y0: int, header_y1: int,
                              data_col_x: list,
                              upscale: int = 2) -> list:
    """
    Read ALL column labels from the full header strip in a single API call.

    Sending the whole row at once gives the model global context — it can
    use the numeric sequence (C1, C2, C2A, C3, …) to recover labels that
    are partially obscured by column cross-section drawings in individual
    cells.

    Returns a list of N items; each is a normalised label string or None.
    """
    n     = len(data_col_x) - 1
    strip = _crop_upscale(
        img, data_col_x[0], header_y0, data_col_x[-1], header_y1, upscale
    )
    prompt = (
        f"This is the column-header row of a structural engineering column "
        f"schedule. It contains exactly {n} columns arranged left to right.\n\n"
        "Each column is identified by a short code such as:\n"
        "  C1, C2, C2A, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14\n"
        "  (format: capital letter + 1-3 digits + optional suffix letter)\n\n"
        "The label may be printed small, sometimes inside or above a "
        "structural column cross-section drawing within the same cell.\n\n"
        f"List ALL {n} column labels from LEFT to RIGHT, separated by commas. "
        "Use your best guess based on visible text and the expected numeric "
        "sequence. Output ONLY the comma-separated labels — no other text.\n\n"
        "Example for 11 columns:  C1, C2, C2A, C3, C4, C5, C6, C7, C8, C9, C10, C11"
    )
    resp  = call_vision(client, strip, prompt, max_tokens=200).strip()
    parts = re.split(r'[,\n]+', resp)
    result = []
    for p in parts:
        p    = p.strip().strip('"').strip("'").strip()
        norm = _normalize_col_label(p)
        result.append(norm)
    return result[:n]


def extract_col_labels(client,
                       img: Image.Image,
                       header_y0: int, header_y1: int,
                       data_col_x: list,
                       upscale: int = 2) -> list:
    """
    Per-cell fallback: read each column-header cell individually.

    Used to patch any None slots left behind by the strip call.
    Returns a list of N items (normalised str or None).
    """
    n      = len(data_col_x) - 1
    result = []
    for i in range(n):
        cell = _crop_upscale(
            img, data_col_x[i], header_y0, data_col_x[i + 1], header_y1, upscale
        )
        prompt = (
            "This is a single header cell from a structural engineering column "
            "schedule. The cell may contain a small structural column drawing "
            "AND a column label code.\n\n"
            f"This is cell {i + 1} of {n} in the header row.\n\n"
            "The column label is a short code: a capital letter + 1-3 digits + "
            "optional suffix letter  (e.g. C1, C2, C2A, C3, C10, C12, C14).\n\n"
            "It is often printed small near the top, bottom, or centre of the "
            "cell — look carefully even if a drawing is present.\n\n"
            "Return ONLY the label code (e.g. 'C1'). "
            "If you genuinely cannot read it, return 'SKIP'."
        )
        raw  = call_vision(client, cell, prompt, max_tokens=20).strip().strip('"').strip("'")
        norm = _normalize_col_label(raw)
        result.append(norm)
    return result


def _repair_col_sequence(labels: list) -> list:
    """
    Fill in None gaps using neighbour-based sequence logic.

    Supports column numbering like C1, C2, C2A, C3, C4, …
    Only fills single-slot gaps where the inference is unambiguous.

    Examples:
        ['C1', None, 'C2A', 'C3', None, 'C5']
        →  ['C1', 'C2', 'C2A', 'C3', 'C4', 'C5']
    """

    def _parse(lbl: str):
        if not lbl:
            return None
        m = re.match(r'^([A-Z]{1,2})(\d{1,3})([A-Z]?)$', lbl)
        return (m.group(1), int(m.group(2)), m.group(3)) if m else None

    result = list(labels)
    n      = len(result)

    for i in range(n):
        if result[i] is not None:
            continue

        left_lbl  = next((result[j] for j in range(i - 1, -1, -1) if result[j]), None)
        right_lbl = next((result[j] for j in range(i + 1, n)      if result[j]), None)

        left_idx  = next((j for j in range(i - 1, -1, -1) if result[j] is not None), -1)
        right_idx = next((j for j in range(i + 1, n)       if result[j] is not None), -1)
        gap_size  = (right_idx - left_idx - 1) if (left_idx >= 0 and right_idx >= 0) else 0

        if gap_size != 1:
            continue

        lp = _parse(left_lbl)
        rp = _parse(right_lbl)
        if lp is None or rp is None:
            continue

        prefix_l, num_l, suf_l = lp
        prefix_r, num_r, suf_r = rp

        if prefix_l != prefix_r:
            continue

        if suf_l == '' and suf_r == '' and num_r - num_l == 2:
            result[i] = f"{prefix_l}{num_l + 1}"
        elif suf_l == '' and suf_r == '' and num_r - num_l == 1:
            result[i] = f"{prefix_l}{num_l}A"
        elif suf_l != '' and suf_r == '' and num_r - num_l == 2:
            result[i] = f"{prefix_l}{num_l + 1}"

    return result


def _col_sequence_score(labels: list) -> int:
    """
    Count how many adjacent label pairs follow valid C-series ordering.

    Valid consecutive steps:
      C1 → C2        (next integer, no suffix)
      C2 → C2A       (same integer, suffix added)
      C2A → C3       (next integer, previous had suffix)
      C10 → C11      (double-digit, same logic)

    Used to compare header vs footer label sources and pick the one
    that is more internally consistent with the expected label sequence.
    A higher score means the labels form a cleaner sequential run.
    """
    score = 0
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        if not a or not b:
            continue
        ma = re.match(r'^([A-Z]{1,2})(\d{1,3})([A-Z]?)$', a)
        mb = re.match(r'^([A-Z]{1,2})(\d{1,3})([A-Z]?)$', b)
        if not ma or not mb:
            continue
        if ma.group(1) != mb.group(1):
            continue
        na, sa = int(ma.group(2)), ma.group(3)
        nb, sb = int(mb.group(2)), mb.group(3)
        if na == nb and sa == '' and sb != '':          # C2 → C2A
            score += 1
        elif nb == na + 1 and sa == '' and sb == '':    # C2 → C3
            score += 1
        elif nb == na + 1 and sa != '' and sb == '':    # C2A → C3
            score += 1
    return score


# ── Floor-label helpers ───────────────────────────────────────────────────────

def extract_floor_labels(client,
                          img: Image.Image,
                          label_x0: int, label_x1: int,
                          row_y: list,
                          upscale: int = 2) -> list:
    """
    Read the floor/level name from each cell in the label column.

    Returns a list of N strings:
      • Floor label as written  (e.g. "6th FLOOR COLUMN")
      • "SKIP"   — not a floor data row (FOOTING, legend, blank, etc.)
      • "Floor_N" — looks like a floor row but text was unreadable
    """
    prompt = (
        "This is a single row-header cell from a structural engineering "
        "column schedule. The text may be printed vertically (rotated 90°).\n\n"
        "TASK: Decide whether this cell contains a floor or level label.\n\n"
        "Floor labels look like (ANY floor number is valid — not limited to 6):\n"
        "  '13th FLOOR COLUMN', '12th FLOOR COLUMN', '11th FLOOR COLUMN',\n"
        "  '10th FLOOR COLUMN', '9th FLOOR COLUMN',  '8th FLOOR COLUMN',\n"
        "  '7th FLOOR COLUMN',  '6th FLOOR COLUMN',  '5th FLOOR COLUMN',\n"
        "  '4th FLOOR COLUMN',  '3rd FLOOR COLUMN',  '2nd FLOOR COLUMN',\n"
        "  '1st FLOOR COLUMN',  'GROUND FLOOR COLUMN', 'BASE FLOOR COLUMN',\n"
        "  'BASEMENT FLOOR COLUMN', 'PODIUM COLUMN', 'Base Floor Column'.\n"
        "  NOTE: 'BASE' / 'BASE.' can mean BASEMENT. Any ordinal is valid.\n\n"
        "NON-floor cells look like:\n"
        "  'FOOTING', 'PEDESTAL', 'COL. MARK', 'SIZE', 'CONC. MIX',\n"
        "  'VERT. REINF.', 'RING', blank, or a repeat of the column-header.\n\n"
        "If the cell IS a floor label → return the label text EXACTLY as written.\n"
        "If the cell is NOT a floor label → return exactly the word SKIP.\n"
        "Return nothing else."
    )
    labels = []
    for i in range(len(row_y) - 1):
        cell = _crop_upscale(img, label_x0, row_y[i], label_x1, row_y[i + 1], upscale)
        raw  = call_vision(client, cell, prompt, max_tokens=60).strip().strip('"').strip("'")
        if raw.upper() == "SKIP":
            labels.append("SKIP")
        elif raw:
            labels.append(raw)
        else:
            labels.append(f"Floor_{i + 1}")
    return labels


def filter_and_recheck_floor_rows(client,
                                   img: Image.Image,
                                   label_x0: int, label_x1: int,
                                   h_grid: list,
                                   raw_labels: list,
                                   upscale: int = 2) -> tuple:
    """
    Post-process raw floor labels:
      • "SKIP"    → drop the row
      • "Floor_N" → second vision call to confirm and re-read
      • Anything else → keep as-is

    Returns (row_bounds, floor_labels) for valid rows only.
    """
    row_bounds   = []
    floor_labels = []

    for i, label in enumerate(raw_labels):
        y0, y1 = h_grid[i], h_grid[i + 1]

        if label == "SKIP":
            print(f"       Row {i + 1} ({y0}–{y1} px): excluded (non-floor section)")
            continue

        if not label.startswith("Floor_"):
            row_bounds.append((y0, y1))
            floor_labels.append(label)
            continue

        # Fallback label "Floor_N" — verify and re-read
        cell = _crop_upscale(img, label_x0, y0, label_x1, y1, upscale)

        check = call_vision(
            client, cell,
            "Does this cell contain a floor or storey name (e.g. Ground Floor, "
            "1st Floor, Basement, 7th Floor, 13th Floor, or any other level)? "
            "Answer YES or NO only.",
            max_tokens=5,
        ).strip().upper()

        if "YES" not in check:
            print(f"       Row {i + 1} ({y0}–{y1} px): excluded (confirmed non-floor)")
            continue

        retry = call_vision(
            client, cell,
            "Read the floor/level label in this cell (text may be vertical). "
            "Examples: 'BASE FLOOR COLUMN', 'BASEMENT FLOOR COLUMN', "
            "'Base Floor Column', '7th FLOOR COLUMN', '13th FLOOR COLUMN'. "
            "Return ONLY the label text.",
            max_tokens=60,
        ).strip().strip('"').strip("'")

        final_label = retry if retry and len(retry) < 50 else label
        print(f"       Row {i + 1} ({y0}–{y1} px): re-read as '{final_label}'")
        row_bounds.append((y0, y1))
        floor_labels.append(final_label)

    return row_bounds, floor_labels


# ══════════════════════════════════════════════════════════════════════════════
# §5  Data Cell Extraction
# ══════════════════════════════════════════════════════════════════════════════

_EMPTY_CELL = {
    "SIZE":       "---",
    "CONC_MIX":   "---",
    "VERT_REINF": "---",
    "RING":       "---",
}

_DATA_PROMPT_TEMPLATE = (
    "Extract ALL visible text from this structural engineering column "
    "schedule cell.\n"
    "Floor: {floor}.  Column: {column}.\n\n"
    "Fields to extract (copy exactly as written):\n"
    '  SIZE         (e.g. "330 X 1100", "L SHAPE", "AS PER PLAN", "230 X 600")\n'
    '  CONC. MIX    (e.g. "M25", "M30")\n'
    '  VERT. REINF. (e.g. "4-20 TOR + 12-16 TOR", "49-25 TOR", "8-16 TOR")\n'
    '  RING         (e.g. "8 TOR 15 @ 75 + @ 150 + 15 @ 75 C/C, 4 SETS + 1 LINK")\n\n'
    'Return ONLY a JSON object with these exact keys:\n'
    '{{"SIZE":"...","CONC_MIX":"...","VERT_REINF":"...","RING":"..."}}\n'
    'Use "---" for any field that is genuinely not visible or unreadable.'
)

_COMBINED_PROMPT_TEMPLATE = (
    "This is a FULL-WIDTH row from a structural engineering column schedule.\n"
    "Floor: {floor}.\n\n"
    "This row contains ONE shared specification that applies to ALL columns.\n"
    "Extract the following fields (copy exactly as written):\n"
    '  SIZE         (e.g. "AS PER PLAN", "330 X 1100")\n'
    '  CONC. MIX    (e.g. "M25", "M30")\n'
    '  VERT. REINF. (e.g. "49-25 TOR", "14-20 TOR + 6-16 TOR")\n'
    '  RING         (e.g. "10 TOR 15 @ 75 + @ 115 + 15 @ 75 C/C, 9 SETS + 9 LINKS")\n\n'
    'Return ONLY a JSON object:\n'
    '{{"SIZE":"...","CONC_MIX":"...","VERT_REINF":"...","RING":"..."}}\n'
    'Use "---" for any field that is not visible or clearly unreadable.'
)


def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from a vision-model response."""
    try:
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return dict(_EMPTY_CELL)


def extract_data_cell(client, img: Image.Image, floor: str, column: str) -> dict:
    """Send a single data cell image to GPT-4o and return structured data."""
    prompt = _DATA_PROMPT_TEMPLATE.format(floor=floor, column=column)
    return _parse_json_response(call_vision(client, img, prompt, max_tokens=500))


def extract_combined_row(client, img: Image.Image, floor: str) -> dict:
    """
    Read the FULL WIDTH of a data row as one combined cell.

    Called when individual column cells all return empty data, indicating
    the row contains a single shared specification for every column.
    """
    prompt = _COMBINED_PROMPT_TEMPLATE.format(floor=floor)
    return _parse_json_response(call_vision(client, img, prompt, max_tokens=500))


def _is_empty(cell_data: dict) -> bool:
    """Return True if every field in a cell dict is '---' or missing."""
    return all(cell_data.get(k, "---") == "---" for k in _EMPTY_CELL)


# ══════════════════════════════════════════════════════════════════════════════
# §6  Output JSON Parsers
# ══════════════════════════════════════════════════════════════════════════════

def parse_size(raw: str) -> dict:
    empty = {"width": None, "depth": None, "length": None}
    if not raw or raw.strip() in ("---", "", "N/A", "n/a"):
        return empty
    parts = re.split(r'\s*[xX×]\s*', raw.strip())
    nums  = []
    for p in parts:
        try:
            nums.append(int(float(p.strip())))
        except ValueError:
            pass
    if len(nums) == 2:
        return {"width": nums[0], "depth": None, "length": nums[1]}
    if len(nums) >= 3:
        return {"width": nums[0], "depth": nums[1], "length": nums[2]}
    return empty


def parse_reinforcement(raw: str) -> list:
    if not raw or raw.strip() in ("---", ""):
        return []
    result = []
    for part in re.split(r'\s*\+\s*', raw.strip()):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'(\d+)\s*[-–]\s*(\d+)\s*(?:TOR|T)\b', part, re.IGNORECASE)
        if m:
            result.append(f"{m.group(1)}-{m.group(2)}T")
        else:
            nums = re.findall(r'\d+', part)
            if len(nums) >= 2:
                result.append(f"{nums[0]}-{nums[1]}T")
    return result


def parse_stirrups(raw: str) -> dict:
    empty: dict = {"dia": [], "spacing": []}
    if not raw or raw.strip() in ("---", ""):
        return empty
    # Primary pattern: "8 TOR 15 @ 75 C/C" — both bar-size AND count present
    primary = re.findall(r'(\d+)\s*(?:TOR|T)\s*(\d+)', raw, re.IGNORECASE)
    dias = [f"{m[0]}-T{m[1]}" for m in primary]

    # Fallback pattern: "8 TOR @ 125 C/C" — bar-size only, no count after TOR
    # Only fires when the primary pattern found nothing, so there is no double-counting.
    if not dias:
        fallback = re.findall(r'(\d+)\s*TOR\b(?!\s*\d)', raw, re.IGNORECASE)
        dias = [f"T{d}" for d in fallback]
    seen: set      = set()
    spacings: list = []
    for n in re.findall(r'(\d+)\s*C/C', raw, re.IGNORECASE):
        if n not in seen:
            spacings.append(f"{n} C/C")
            seen.add(n)
    for n in re.findall(r'@\s*(\d+)', raw):
        if n not in seen:
            spacings.append(f"{n} C/C")
            seen.add(n)
    return {"dia": dias, "spacing": spacings}


def _build_entry(clabel: str, flabel: str, raw: dict, _to_key) -> dict:
    """Convert raw vision-API dict into the final flat JSON entry."""
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
# §7  Debug Overlay  (optional)
# ══════════════════════════════════════════════════════════════════════════════

def save_debug_overlay(img:         Image.Image,
                        h_grid:      list,
                        v_grid:      list,
                        header_band: tuple,
                        label_band:  tuple,
                        out_path:    str) -> None:
    """Save a colour-coded grid-overlay PNG for inspection."""
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
# §8  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def extract_column_schedule(pdf_path:    str,
                             output_path: str  = "output.json",
                             dpi:         int  = 600,
                             upscale:     int  = 2,
                             debug:       bool = False) -> dict:
    """
    Fully generalized extraction pipeline for structural-engineering
    column schedule PDFs.

    No hard-coded coordinates — all grid boundaries are discovered
    automatically from the image.
    """
    client = get_client()

    # ── Step 1 : Render ───────────────────────────────────────────────────────
    print(f"\n[1/6] Rendering '{Path(pdf_path).name}' at {dpi} DPI …")
    img  = pdf_to_image(pdf_path, dpi=dpi)
    W, H = img.size
    print(f"       Image: {W} × {H} px")

    # ── Step 2 : Adaptive horizontal line detection ───────────────────────────
    print("[2/6] Detecting horizontal grid lines (adaptive threshold) …")

    H_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    min_floor_h  = H / 25
    max_floor_h  = H / 2

    h_all      = []
    h_grid     = []
    chosen_thr = H_THRESHOLDS[0]

    for h_thr in H_THRESHOLDS:
        h_raw, _ = detect_lines(img, h_thr=h_thr, v_thr=0.0, min_dist=30)
        cand_all  = cluster_lines(sorted(h_raw))

        if len(cand_all) < 4:
            h_all = cand_all
            continue

        cand_grid = find_best_grid(cand_all, min_rows=3, tolerance=0.35)
        n_cand    = len(cand_grid) - 1

        if n_cand < 3:
            h_all  = cand_all
            h_grid = cand_grid
            continue

        avg_h      = (cand_grid[-1] - cand_grid[0]) / n_cand
        h_all      = cand_all
        h_grid     = cand_grid
        chosen_thr = h_thr

        if min_floor_h <= avg_h <= max_floor_h:
            break

    print(f"       {len(h_all)} horizontal lines  (h_thr={chosen_thr})")

    # ── Extra row extension ───────────────────────────────────────────────────
    # Catch floor rows that fall just outside the regular grid boundary
    # (e.g. BASE FLOOR row at a slightly different pitch from the main grid).
    # Any line in h_all that sits within 1.5× the median row height below
    # h_grid[-1] is appended — fully generalised, no names hard-coded.
    if len(h_grid) >= 2:
        med_row_h = float(np.median(
            [h_grid[i + 1] - h_grid[i] for i in range(len(h_grid) - 1)]
        ))
        extra_below = sorted(
            y for y in h_all
            if h_grid[-1] < y <= h_grid[-1] + med_row_h * 1.5
            and y not in h_grid
        )
        if extra_below:
            print(
                f"       Found {len(extra_below)} extra row-boundary line(s) just "
                f"below regular grid (y={extra_below}) — appending."
            )
            h_grid = h_grid + extra_below

    # ── Step 3 : Validate row count ───────────────────────────────────────────
    print("[3/6] Identifying data rows …")

    if len(h_grid) < 3:
        raise ValueError(
            f"Only {len(h_grid)} consistent horizontal grid lines found "
            f"(need ≥ 3 for at least 2 data rows). "
            "Use --debug to inspect the detected lines."
        )

    n_rows = len(h_grid) - 1
    print(f"       {n_rows} data rows  |  Y: {h_grid[0]} – {h_grid[-1]} px")

    # ── Step 4 : Detect vertical lines scoped to the table's Y range ──────────
    print("[4/6] Detecting vertical grid lines within table …")
    _, v_raw = detect_lines(
        img,
        y0=h_grid[0], y1=h_grid[-1],
        h_thr=0.0, v_thr=0.20, min_dist=30,
    )
    v_all  = cluster_lines(sorted(v_raw))
    v_core = find_regular_grid(v_all, tolerance=0.40)
    # max_extra_multiplier=2.0 prevents distant border/legend lines from being
    # counted as extra data columns (was 4.0 which over-detected on wide PDFs).
    v_grid = extend_grid_with_trailing_lines(v_core, v_all, max_extra_multiplier=2.0)
    print(f"       {len(v_all)} vertical lines after clustering")

    if len(v_grid) < 3:
        raise ValueError(
            f"Only {len(v_grid)} consistent vertical grid lines found "
            f"(need ≥ 3 for label col + ≥ 1 data col). "
            "Try --debug or adjust thresholds."
        )

    # ── Step 5a : Identify label column and column-header row ─────────────────
    typical_col_width   = float(np.median(
        [v_grid[i + 1] - v_grid[i] for i in range(len(v_grid) - 1)]
    ))
    label_col_threshold = int(typical_col_width * 0.5)

    if v_grid[0] <= label_col_threshold:
        label_x0   = v_grid[0]
        label_x1   = v_grid[1]
        data_col_x = v_grid[1:]
    else:
        label_x1   = v_grid[0]
        label_x0   = max(0, label_x1 - int(typical_col_width * 2.0))
        data_col_x = v_grid

    n_cols = len(data_col_x) - 1
    if n_cols < 1:
        raise ValueError("No data columns detected. Use --debug to inspect.")

    print(f"       {n_cols} data columns  |  X: {data_col_x[0]} – {data_col_x[-1]} px")

    above_h = [y for y in h_all if y < h_grid[0]]
    below_h = [y for y in h_all if y > h_grid[-1]]

    header_y0 = above_h[-1] if above_h else max(0, h_grid[0] - (h_grid[1] - h_grid[0]) // 2)
    header_y1 = h_grid[0]

    footer_y0 = h_grid[-1] if below_h else None
    footer_y1 = below_h[0] if below_h else None

    print(f"       Column-header row  Y: {header_y0} – {header_y1} px")
    if footer_y0 is not None:
        print(f"       Column-footer row  Y: {footer_y0} – {footer_y1} px")
    print(f"       Floor-label column X: {label_x0} – {label_x1} px")

    if debug:
        dbg_path = str(Path(output_path).with_suffix(".debug.png"))
        save_debug_overlay(
            img, h_grid, v_grid,
            (header_y0, header_y1),
            (label_x0,  label_x1),
            dbg_path,
        )

    # ── Step 5b : Extract column labels ──────────────────────────────────────
    print("[5/6] Extracting column and floor labels via Vision API …")

    def _attempt_col_labels(y0: int, y1: int, source_name: str):
        """
        Full extraction pipeline for one header/footer band.
        1. Full-strip call (global context).
        2. Per-cell retry for remaining None slots.
        3. Sequence repair for unambiguous gaps.
        4. Positional fallback COL_{i+1} for anything still None.
        Returns (labels_list, fallback_count).
        """
        print(f"       [{source_name}] Extracting column labels from Y {y0}–{y1} …")

        labels = extract_col_labels_strip(client, img, y0, y1, data_col_x, upscale)
        while len(labels) < n_cols:
            labels.append(None)
        labels = labels[:n_cols]

        none_count = sum(1 for lb in labels if lb is None)
        if none_count:
            print(f"         Strip call: {none_count} slot(s) unresolved → per-cell retry")
            per_cell = extract_col_labels(client, img, y0, y1, data_col_x, upscale)
            for idx in range(n_cols):
                if labels[idx] is None and idx < len(per_cell) and per_cell[idx] is not None:
                    labels[idx] = per_cell[idx]

        labels = _repair_col_sequence(labels)

        for idx in range(n_cols):
            if labels[idx] is None:
                labels[idx] = f"COL_{idx + 1}"

        fb = sum(1 for lb in labels if lb.startswith("COL_"))
        print(f"         Result: {labels}  ({fb} fallback(s))")
        return labels, fb

    # Try header row first
    col_labels, fallback_count = _attempt_col_labels(header_y0, header_y1, "header")

    # ALWAYS try footer when it exists and compare using sequence score.
    # This fixes cases where the header band is tall and contains misleading
    # title-block content that gives valid-looking but wrong labels
    # (e.g. C8/C9/C10 in the header when the real labels are C12/C13/C14).
    if footer_y0 is not None:
        footer_labels, footer_fb = _attempt_col_labels(footer_y0, footer_y1, "footer")
        score_h = _col_sequence_score(col_labels)
        score_f = _col_sequence_score(footer_labels)
        print(
            f"       Header: score={score_h} fallbacks={fallback_count} | "
            f"Footer: score={score_f} fallbacks={footer_fb}"
        )
        # Decision rules (priority order):
        #  1. Fewer fallbacks wins outright.
        #  2. Equal fallbacks → ≥ sequence score wins → prefer footer on tie.
        #  3. Footer sequence score clearly higher (+2) → override header.
        if footer_fb < fallback_count:
            print("       Footer wins: fewer fallbacks — using footer labels.")
            col_labels, fallback_count = footer_labels, footer_fb
        elif footer_fb == fallback_count and score_f >= score_h:
            print("       Footer wins: equal fallbacks, ≥ sequence score — using footer labels.")
            col_labels, fallback_count = footer_labels, footer_fb
        elif score_f > score_h + 1:
            print("       Footer wins: clearly better sequence score — using footer labels.")
            col_labels, fallback_count = footer_labels, footer_fb
        else:
            print(f"       Header kept (score={score_h} ≥ footer={score_f}, "
                  f"fb={fallback_count} ≤ {footer_fb}).")

    # ── Step 5c : Extract and validate floor labels ───────────────────────────
    raw_floor_labels = extract_floor_labels(
        client, img, label_x0, label_x1, h_grid, upscale
    )

    print("       Validating floor rows …")
    row_bounds, floor_labels = filter_and_recheck_floor_rows(
        client, img, label_x0, label_x1, h_grid, raw_floor_labels, upscale
    )

    n_rows = len(row_bounds)
    if n_rows == 0:
        raise ValueError(
            "No valid floor rows found after filtering. Use --debug to inspect."
        )

    # Second-pass column-label fallback: try SKIP rows near the bottom of the
    # data grid in case the label row was labelled SKIP but contains real labels.
    fallback_count = sum(1 for lb in col_labels if lb.startswith("COL_"))
    if fallback_count > n_cols // 2 and row_bounds:
        last_valid_y1  = row_bounds[-1][1]
        skipped_bottom = [
            (h_grid[i], h_grid[i + 1])
            for i, lbl in enumerate(raw_floor_labels)
            if lbl == "SKIP" and h_grid[i] >= last_valid_y1 - 50
        ]
        for sy0, sy1 in skipped_bottom:
            skip_labels, skip_fb = _attempt_col_labels(sy0, sy1, f"skip-row@{sy0}")
            if skip_fb < fallback_count:
                print(f"       SKIP row better ({skip_fb}) — using it as column source.")
                col_labels     = skip_labels
                fallback_count = skip_fb
                break

    print(f"       ✓ Columns ({n_cols}): {col_labels}")
    print(f"       ✓ Floors  ({n_rows}): {floor_labels}")

    def _to_key(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    # ── Step 6 : Extract every data cell ─────────────────────────────────────
    total = n_rows * n_cols
    print(f"[6/6] Extracting {total} data cells via Vision API …")

    result = {
        "document": Path(pdf_path).name,
        "title":    "Column Schedule",
        "columns":  [],
    }

    seen_entries: set = set()
    batch_folder_name = "level_batches"
    batch_folder = os.path.join(os.path.dirname(output_path), batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    manifest = {
        "pattern": 11,
        "mode": "level_batches",
        "levels": floor_labels,
        "batches": [
            {
                "id": f"l{r + 1:03d}",
                "level": floor_labels[r],
                "status": "pending",
                "file": os.path.join(
                    batch_folder_name,
                    f"l{r + 1:03d}_{safe_filename(floor_labels[r])}.json",
                ),
                "error": None,
            }
            for r in range(n_rows)
        ],
    }
    manifest_path = os.path.join(os.path.dirname(output_path), "level_manifest.json")
    atomic_write_json(manifest_path, manifest)

    def _set_batch_status(batch_id, status, error=None):
        for batch in manifest["batches"]:
            if batch["id"] == batch_id:
                batch["status"] = status
                batch["error"] = error
                return

    done = 0
    for r in range(n_rows):
        batch_id = f"l{r + 1:03d}"
        _set_batch_status(batch_id, "running")
        atomic_write_json(manifest_path, manifest)
        y0, y1 = row_bounds[r]
        flabel = floor_labels[r]

        row_cells: list[dict] = []
        for c in range(n_cols):
            x0, x1 = data_col_x[c], data_col_x[c + 1]
            clabel  = col_labels[c]
            cell    = _crop_upscale(img, x0, y0, x1, y1, upscale)
            done   += 1
            print(
                f"       [{done:>3}/{total}]  {flabel} / {clabel} …",
                end=" ", flush=True,
            )
            raw = extract_data_cell(client, cell, flabel, clabel)
            print("✓")
            row_cells.append(raw)

        # Combined-row detection: >50 % empty cells → full-width re-read
        empty_count = sum(1 for d in row_cells if _is_empty(d))
        if empty_count > n_cols // 2:
            print(
                f"       ↳ {empty_count}/{n_cols} empty — "
                f"COMBINED row detected; re-reading full width …",
                end=" ", flush=True,
            )
            full_row = _crop_upscale(
                img, data_col_x[0], y0, data_col_x[-1], y1, upscale
            )
            combined = extract_combined_row(client, full_row, flabel)
            print("✓")
            if not _is_empty(combined):
                print(f"       ↳ Combined read succeeded — broadcasting to all {n_cols} columns.")
                row_cells = [combined] * n_cols
            else:
                print(f"       ↳ Combined read also empty — keeping individual results.")

        floor_key = _to_key(flabel)
        row_entries = []
        for c, raw in enumerate(row_cells):
            clabel    = col_labels[c]
            entry_key = (floor_key, clabel)
            if entry_key in seen_entries:
                continue
            seen_entries.add(entry_key)
            row_entries.append(_build_entry(clabel, flabel, raw, _to_key))

        result["columns"].extend(row_entries)
        batch_path = os.path.join(
            batch_folder,
            f"{batch_id}_{safe_filename(flabel)}.json",
        )
        atomic_write_json(batch_path, {"level": flabel, "columns": row_entries})
        _set_batch_status(batch_id, "done")
        atomic_write_json(manifest_path, manifest)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        result = reshape_columns_to_levels(result.get("columns", []))
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Done!  {len(result['columns'])} entries extracted → {output_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §9  CLI  —  run as:  python src/main_11.py
# ══════════════════════════════════════════════════════════════════════════════

def _build_output_path(pdf_path: str, output_dir: str) -> str:
    """Return output JSON path: <output_dir>/<pdf_stem>/<pdf_stem>.json

    Example:
        input/pattern-11.pdf  →  output/pattern-11/pattern-11.json
    """
    stem = Path(pdf_path).stem
    return os.path.join(output_dir, stem, f"{stem}.json")


def process_pdf(pdf_path: str, dpi: int = 600, upscale: int = 2, debug: bool = False) -> dict:
    """auto_runner entry point. Wraps extract_column_schedule with the
    standard output path: <OUTPUT_DIR>/<pdf_stem>/<pdf_stem>.json."""
    output_path = _build_output_path(pdf_path, OUTPUT_DIR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return extract_column_schedule(
        pdf_path=pdf_path,
        output_path=output_path,
        dpi=dpi,
        upscale=upscale,
        debug=debug,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generalized column schedule PDF extractor.\n"
            "By default processes ALL PDFs in the configured input folder.\n"
            "Pass --pdf <name.pdf> to process a single file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pdf",
        default=None,
        help=(
            "Single PDF filename (or full path) to process. "
            "If omitted, every *.pdf in the input folder is processed."
        ),
    )
    ap.add_argument(
        "--input-dir",
        default=INPUT_DIR,
        help=f"Folder containing input PDFs  (default: {INPUT_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Folder where JSON results are written  (default: {OUTPUT_DIR})",
    )
    ap.add_argument("--dpi",     type=int, default=600,
                    help="PDF render resolution (600 recommended minimum)")
    ap.add_argument("--upscale", type=int, default=2,
                    help="Per-cell upscale factor (higher = better OCR, slower)")
    ap.add_argument("--debug",   action="store_true",
                    help="Save a colour-coded grid-overlay PNG for each PDF")
    args = ap.parse_args()

    # ── Resolve list of PDFs to process ──────────────────────────────────────
    if args.pdf:
        # Single file: accept bare name (looked up in input dir) or full path
        candidate = args.pdf if os.path.isabs(args.pdf) else \
                    os.path.join(args.input_dir, args.pdf)
        if not os.path.isfile(candidate):
            print(f"ERROR: PDF not found → {candidate}", file=sys.stderr)
            sys.exit(1)
        pdf_files = [candidate]
    else:
        # Batch mode: all PDFs in the input folder
        pdf_files = sorted(Path(args.input_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in input folder: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        pdf_files = [str(p) for p in pdf_files]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Column Schedule Extractor")
    print(f"  Input  : {args.input_dir}")
    print(f"  Output : {args.output_dir}")
    print(f"  Files  : {len(pdf_files)}")
    print(f"{'=' * 60}")

    # ── Process each PDF ──────────────────────────────────────────────────────
    success, failed = [], []

    for pdf_path in pdf_files:
        out_path = _build_output_path(pdf_path, args.output_dir)
        print(f"\n▶  Processing: {Path(pdf_path).name}  →  {Path(out_path).name}")
        try:
            extract_column_schedule(
                pdf_path    = pdf_path,
                output_path = out_path,
                dpi         = args.dpi,
                upscale     = args.upscale,
                debug       = args.debug,
            )
            success.append(Path(pdf_path).name)
        except Exception as exc:
            print(f"\n❌  FAILED: {Path(pdf_path).name}  —  {exc}", file=sys.stderr)
            failed.append(Path(pdf_path).name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Summary:  {len(success)} succeeded  |  {len(failed)} failed")
    if success:
        print(f"  ✅  {', '.join(success)}")
    if failed:
        print(f"  ❌  {', '.join(failed)}")
    print(f"{'=' * 60}\n")

    sys.exit(0 if not failed else 1)

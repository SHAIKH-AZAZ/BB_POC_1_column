"""
main_12.py  ─  Pattern-12 column schedule extractor
=====================================================
Generalized extraction — ZERO hard-coded pixel coordinates.

APPROACH  (adapted from pattern-11 architecture)
------------------------------------------------
1.  Render PDF at 300 DPI.
2.  Scan for bright-green [0,255,0] text to locate the footing section start
    (last green group = footing block).
3.  Detect data-column VERTICAL boundaries by scanning for full-height
    red lines: scipy find_peaks on red-pixel column density > 30 % over
    the floor-row y-range.
4.  Extract column marks (GPT-4o on marks strip at bottom).
5.  Reconcile V-lines with mark count: trim extra lines from LEFT
    (label column internal dividers).  Never resample.
6.  Detect floor-row HORIZONTAL boundaries:
      Primary  → red H-line density scan inside data columns.
      Fallback → evenly-spaced rows based on GPT-4o floor count.
7.  Extract ALL floor labels in one GPT-4o call on the full label column.
8.  Extract each data cell (GPT-4o with pixel pre-check).
9.  Expand combined marks: "C1,C18" → C1 + C18.
10. Stirrups always null.

DEPENDENCIES
------------
    pip install pdf2image Pillow scipy numpy openai
    sudo apt-get install poppler-utils   # or: brew install poppler
"""

import json
from extraction_guard import reshape_columns_to_levels
from pattern_batching import atomic_write_json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import base64
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from config import INPUT_DIR, OUTPUT_DIR, OPENAI_API_KEY   # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  §1  SCIPY LINE UTILITIES
# ═══════════════════════════════════════════════════════════════════

def cluster_lines(lines: list, gap: int = 10) -> list:
    """Merge groups of nearby positions into one representative (median)."""
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


# ═══════════════════════════════════════════════════════════════════
#  §2  FOOTING DETECTION  (green text groups)
# ═══════════════════════════════════════════════════════════════════

def detect_green_row_groups(img_arr: np.ndarray,
                             min_group_height: int = 15) -> list:
    """
    Find all contiguous bands of bright-green [0,255,0] pixels (floor labels
    and footing header).  Returns list of (y_start, y_end) tuples sorted top→bottom.

    Used ONLY to locate:
      • floor_data_y0  = y where first green text appears
      • footing_y0     = y where last green group starts
    """
    r = img_arr[:, :, 0].astype(int)
    g = img_arr[:, :, 1].astype(int)
    b = img_arr[:, :, 2].astype(int)
    green_mask = (g > 200) & (r < 60) & (b < 60)

    raw_rows = np.where(green_mask.any(axis=1))[0]
    if len(raw_rows) == 0:
        return []

    groups, cur = [], [int(raw_rows[0])]
    for y in raw_rows[1:]:
        if int(y) - cur[-1] < 20:          # tight clustering: single text lines
            cur.append(int(y))
        else:
            if cur[-1] - cur[0] >= min_group_height:
                groups.append((cur[0], cur[-1]))
            cur = [int(y)]
    if cur[-1] - cur[0] >= min_group_height:
        groups.append((cur[0], cur[-1]))
    return groups


# ═══════════════════════════════════════════════════════════════════
#  §3  COLUMN V-LINE DETECTION  (red pixel density)
# ═══════════════════════════════════════════════════════════════════

def detect_col_boundaries(img_arr: np.ndarray,
                           y0: int, y1: int,
                           x_min: int = 0,
                           threshold: float = 0.30,
                           min_dist: int = 15) -> list:
    """
    Vertical grid-separator lines: scipy find_peaks on red column density.
    Full-height separator ≈ 40-100 % red; single-cell box edge ≈ 3-5 % → filtered.
    Returns sorted list of absolute x positions.
    """
    region   = img_arr[y0:y1, x_min:]
    r = region[:, :, 0].astype(int)
    g = region[:, :, 1].astype(int)
    b = region[:, :, 2].astype(int)
    red_mask = (r > 150) & ((r - g) > 60) & ((r - b) > 60)
    density  = red_mask.mean(axis=0)

    peaks, _ = find_peaks(density, height=threshold, distance=min_dist)
    abs_x    = [int(p) + x_min for p in peaks]
    return cluster_lines(abs_x, gap=8)


# ═══════════════════════════════════════════════════════════════════
#  §4  FLOOR ROW DETECTION  (red H-line density)
# ═══════════════════════════════════════════════════════════════════

def detect_row_boundaries(img_arr: np.ndarray,
                           x0: int, x1: int,
                           y0: int = 0, y1: int = None,
                           threshold: float = 0.15,
                           min_dist: int = 80) -> list:
    """
    Horizontal floor-row separator lines: scipy find_peaks on red ROW density
    within the data-column x-range [x0, x1].

    A full-data-width H-line → density ≈ 1.0 (well above 0.15 threshold).
    A single-cell box edge → density ≈ 1/n_cols ≈ 0.09 (filtered out).

    Returns sorted list of absolute y positions (separator lines).
    """
    if y1 is None:
        y1 = img_arr.shape[0]
    region   = img_arr[y0:y1, x0:x1]
    r = region[:, :, 0].astype(int)
    g = region[:, :, 1].astype(int)
    b = region[:, :, 2].astype(int)
    red_mask = (r > 150) & ((r - g) > 60) & ((r - b) > 60)
    row_density = red_mask.mean(axis=1)

    peaks, _ = find_peaks(row_density, height=threshold, distance=min_dist)
    abs_y    = [int(p) + y0 for p in peaks]
    return cluster_lines(abs_y, gap=5)


# ═══════════════════════════════════════════════════════════════════
#  §5  COLUMN-MARKS STRIP LOCATOR
# ═══════════════════════════════════════════════════════════════════

def find_marks_strip_y(img_arr: np.ndarray,
                        x0: int, x1: int,
                        search_y0: int, search_y1: int) -> tuple:
    """
    Locate the narrow row with magenta (255,0,255) column-mark text in
    the footing section [search_y0, search_y1].
    Returns (y_start, y_end) padded generously.
    Falls back to the full search range if nothing found.
    """
    region   = img_arr[search_y0:search_y1, x0:x1]
    r = region[:, :, 0].astype(int)
    g = region[:, :, 1].astype(int)
    b = region[:, :, 2].astype(int)
    mag_mask = (r > 200) & (g < 60) & (b > 200)   # true magenta 255,0,255
    per_row  = mag_mask.sum(axis=1)

    text_rows = np.where(per_row > 8)[0]
    if len(text_rows) == 0:
        return (search_y0, search_y1)

    groups, cur = [], [int(text_rows[0])]
    for y in text_rows[1:]:
        if int(y) - cur[-1] <= 6:
            cur.append(int(y))
        else:
            groups.append(cur); cur = [int(y)]
    groups.append(cur)

    best   = max(groups, key=lambda gr: sum(int(per_row[y]) for y in gr))
    abs_y1 = best[0]  + search_y0
    abs_y2 = best[-1] + search_y0
    return (max(search_y0, abs_y1 - 60), min(search_y1, abs_y2 + 40))


# ═══════════════════════════════════════════════════════════════════
#  §6  VISION API  (OpenAI GPT-4o)
# ═══════════════════════════════════════════════════════════════════

def _get_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        raise RuntimeError("pip install openai")


def _img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop_upscale(img: Image.Image,
                  x0: int, y0: int, x1: int, y1: int,
                  upscale: int = 2) -> Image.Image:
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.width,  x1); y1 = min(img.height, y1)
    cell = img.crop((x0, y0, x1, y1))
    if upscale > 1:
        cell = cell.resize((cell.width * upscale, cell.height * upscale),
                           Image.LANCZOS)
    return cell


_RETRY_PHRASES = ("502", "503", "500", "504", "429",
                  "bad gateway", "service unavailable",
                  "timeout", "timed out", "connection")


def call_vision(client, img: Image.Image, prompt: str,
                max_tokens: int = 400, retries: int = 5) -> str:
    """GPT-4o call with exponential back-off on 5xx / 429 / timeout."""
    b64      = _img_to_b64(img)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}",
                                       "detail": "high"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content
        except Exception as exc:
            msg = str(exc).lower()
            if any(p in msg for p in _RETRY_PHRASES):
                wait = 2 ** attempt
                print(f"  ⚠ API error (attempt {attempt}/{retries}): "
                      f"{str(exc)[:80]} — retrying in {wait}s …")
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise RuntimeError(f"call_vision failed after {retries} retries: {last_exc}")


def _parse_json(text: str) -> dict | None:
    """Extract first JSON object from model response."""
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════
#  §7  PATTERN-12 EXTRACTION PROMPTS
# ═══════════════════════════════════════════════════════════════════

_PROMPT_COL_MARKS = """\
You are reading a strip cut from the BOTTOM of a structural column schedule drawing.
It shows a row labelled "COLUMN MARK" with individual column codes written in magenta
or pink text — one code per column, left to right.

Column codes look like:
  • Single marks:    C5, C8, C10, C12, C15
  • Combined marks:  C1,C18  or  C2,C9  or  C7,C16  (two columns sharing one schedule)

Rules:
  • Read every mark STRICTLY left to right.
  • Keep combined marks together exactly as written (e.g. "C1,C18").
  • Ignore any text that is NOT a column mark (headers, notes, dimension numbers).

Return ONLY valid raw JSON — no markdown fences, no explanation:
{"column_marks": ["C1,C18", "C2,C9", "C3,C17", "C4,C14", "C5", "C7,C16", "C8", "C10", "C11,C13", "C12", "C15"]}
"""

_PROMPT_ALL_FLOORS = """\
You are looking at the LEFT LABEL COLUMN of a structural column schedule drawing.
The text is printed in wide-spaced CAD font and may be rotated 90°.
The labels are stacked from TOP to BOTTOM — read ALL of them in order.

Each label contains:
  • Floor name — e.g. "10TH FLOOR COLUMN", "GROUND FLOOR COLUMN",
                      "BASEMENT COLUMN", "12TH FLOOR COLUMN",
                      "TERRACE FLOOR COLUMN", "1ST FLOOR COLUMN" etc.
  • Mix grade  — look for "MIX : M250" or "MIX:M300" → extract "M250" or "M300"
  • Steel grade — look for "STEEL:FE500" or "SEEL:FE500" → extract "FE500"

Important rules:
  • List EVERY distinct floor row from TOP to BOTTOM — do not skip any.
  • Do NOT deduplicate — if the same floor name appears twice in the schedule
    (e.g. a schedule covers two separate stacks), include it twice.
  • If MIX or steel grade is not visible for a row, use null.

Return ONLY valid raw JSON — no markdown, no explanation:
{"floors": [
    {"name": "10TH FLOOR COLUMN", "mix": "M250", "steel_grade": "FE500"},
    {"name": "GROUND FLOOR COLUMN", "mix": "M300", "steel_grade": "FE500"}
    {"name": "BASEMENT FLOOR COLUMN", "mix": "M300", "steel_grade": "FE500"}
]}
"""

_PROMPT_CELL = """\
You are reading ONE CELL from a structural column schedule grid.
The cell either contains a cross-section diagram OR is completely empty/blank.

If the cell contains a cross-section diagram:
  width         = the horizontal dimension number written near the TOP or SIDE of the box
                  (integer mm, e.g. 230, 300, 375, 450)
  length        = the vertical dimension number written to the LEFT of the box
                  (integer mm — often larger, e.g. 450, 750, 1000, 1200, 1950)
                  NOTE: this number is printed VERTICALLY (rotated 90°) — read it carefully.
  reinforcement = ALL bar annotations around the cross-section, each converted to "N-DT" format:
                    "4 φ 16"  → "4-16T"
                    "20 φ 12" → "20-12T"
                    "10 φ 20" → "10-20T"
                  Include ALL distinct groups; deduplicate identical ones.
                  φ may appear as ⌀ or a similar circle symbol.
                  If no bars visible → [].

If the cell is EMPTY (blank — no cross-section box, no annotations):
  → {"width": null, "length": null, "reinforcement": []}

Return ONLY valid raw JSON — no markdown, no explanation:
{"width": 300, "length": 750, "reinforcement": ["5-20T", "2-16T"]}
"""


# ═══════════════════════════════════════════════════════════════════
#  §8  NORMALIZERS
# ═══════════════════════════════════════════════════════════════════

def norm_reinforcement(raw: list) -> list:
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        s = str(item).strip().upper()
        if re.match(r'^\d+-\d+T$', s) and s not in seen:
            seen.add(s); out.append(s)
    return out


def norm_column_name(raw: str) -> str:
    if not raw:
        return ""
    c = re.sub(r'(\d+)\s*(TH|ST|ND|RD)\b', r'\1\2', str(raw).strip(), flags=re.I)
    return re.sub(r'\s+', ' ', c).strip().upper()


def norm_column_no(raw: str) -> str:
    if not raw:
        return ""
    return ','.join(p.strip().upper() for p in str(raw).split(',') if p.strip())


def norm_mix(raw) -> str | None:
    if not raw:
        return None
    m = re.search(r'M\s*(\d{2,3})', str(raw).upper())
    return f"M{m.group(1)}" if m else None


def norm_steel(raw) -> str:
    if not raw:
        return "FE500"
    m = re.search(r'FE\s*(\d{3})', str(raw).strip().upper())
    return f"FE{m.group(1)}" if m else "FE500"


# ═══════════════════════════════════════════════════════════════════
#  §9  EMPTY CELL DETECTOR  (pixel-based — avoids unnecessary API calls)
# ═══════════════════════════════════════════════════════════════════

def is_cell_empty(crop: Image.Image,
                  border: int = 8,
                  min_pixels: int = 60) -> bool:
    """
    True if the cell interior has fewer than `min_pixels` drawing pixels.
    Drawing pixels = dark text OR colored CAD annotation (red / pink).
    """
    arr = np.array(crop)
    h, w = arr.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        return True
    interior = arr[border: h - border, border: w - border]
    r = interior[:, :, 0].astype(int)
    g = interior[:, :, 1].astype(int)
    b = interior[:, :, 2].astype(int)
    dark    = (r < 110) & (g < 110) & (b < 110)
    colored = (r > 150) & ((r - g) > 60)
    return int((dark | colored).sum()) < min_pixels


# ═══════════════════════════════════════════════════════════════════
#  §10  PDF RENDERING
# ═══════════════════════════════════════════════════════════════════

def render_pdf(pdf_path: str, out_dir: str, dpi: int = 300) -> list:
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, page in enumerate(pages):
        p = os.path.join(out_dir, f"page_{i+1:01d}.png")
        page.save(p, "PNG")
        paths.append(p)
    print(f"  Rendered {len(paths)} page(s) at {dpi} DPI")
    return paths


# ═══════════════════════════════════════════════════════════════════
#  §11  PAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════════

def process_page(img_path: str, client) -> list:
    pil = Image.open(img_path).convert("RGB")
    arr = np.array(pil)
    H, W = arr.shape[:2]
    print(f"  Image: {W} × {H} px")

    # ── §11.1  Locate footing via green text groups ────────────────────────────
    all_groups = detect_green_row_groups(arr)
    print(f"  Green text groups: {len(all_groups)}")
    if not all_groups:
        print("  ❌ No green text found — skipping page")
        return []

    footing_y0    = all_groups[-1][0]   # last green group = footing / column-marks header
    floor_data_y0 = all_groups[0][0]    # first green group = start of floor data rows
    print(f"  Floor data y-range: {floor_data_y0}–{footing_y0}  "
          f"|  footing starts y={footing_y0}")

    # ── §11.2  Column V-line detection (robust multi-pass) ───────────────

    col_bounds = []

    # Pass 1: strict (original)
    col_bounds = detect_col_boundaries(arr, floor_data_y0, footing_y0,
                                    x_min=0, threshold=0.30, min_dist=15)

    # Pass 2: medium (for thinner lines)
    if len(col_bounds) < 3:
        col_bounds = detect_col_boundaries(arr, floor_data_y0, footing_y0,
                                        x_min=0, threshold=0.18, min_dist=12)

    # Pass 3: aggressive (pat-12 type drawings)
    if len(col_bounds) < 3:
        col_bounds = detect_col_boundaries(arr, floor_data_y0, footing_y0,
                                        x_min=0, threshold=0.10, min_dist=10)

    # Final fallback: use smoothed projection (VERY IMPORTANT)
    if len(col_bounds) < 3:
        print("  ⚠ Switching to fallback V-line detection")

        region = arr[floor_data_y0:footing_y0, :]
        r = region[:, :, 0].astype(int)
        g = region[:, :, 1].astype(int)
        b = region[:, :, 2].astype(int)

        red_mask = (r > 140) & ((r - g) > 40) & ((r - b) > 40)
        density = red_mask.mean(axis=0)

        # Smooth signal (THIS fixes broken lines)
        density = np.convolve(density, np.ones(15)/15, mode='same')

        peaks, _ = find_peaks(density, height=0.08, distance=8)
        col_bounds = cluster_lines([int(p) for p in peaks], gap=10)

    # Final check
    if len(col_bounds) < 3:
        print("  ❌ Column detection failed — skipping page")
        return []
    
    # ── §11.3  Column marks (GPT-4o) ─────────────────────────────────────────
    print("  [1/3] Extracting column marks …")

    my1, my2 = find_marks_strip_y(arr, x0=0, x1=W,
                                search_y0=footing_y0, search_y1=H)

    marks_img = _crop_upscale(pil, 0, my1, W, my2, upscale=3)
    print(f"  Column marks strip: y={my1}–{my2}")

    raw_marks = call_vision(client, marks_img, _PROMPT_COL_MARKS, max_tokens=200)
    parsed    = _parse_json(raw_marks)

    col_marks = []
    if parsed and isinstance(parsed.get("column_marks"), list):
        col_marks = [norm_column_no(m) for m in parsed["column_marks"] if m]

    # fallback (VERY IMPORTANT for pat-12)
    if not col_marks:
        print("  ⚠ Retry column marks using full footing section")
        marks_img2 = _crop_upscale(pil, 0, footing_y0, W, H, upscale=2)
        raw2       = call_vision(client, marks_img2, _PROMPT_COL_MARKS, max_tokens=200)
        parsed2    = _parse_json(raw2)

        if parsed2 and isinstance(parsed2.get("column_marks"), list):
            col_marks = [norm_column_no(m) for m in parsed2["column_marks"] if m]

    print(f"  → {len(col_marks)} marks: {col_marks}")

    if not col_marks:
        print("  ❌ No column marks found — skipping page")
        return []

    # ── §11.4  Reconcile V-lines with mark count ──────────────────────────────
    # Extra lines on the LEFT are label-column internal dividers — trim them.
    # NEVER resample: that loses the actual pixel boundaries.
    n_marks = len(col_marks)
    data_col_x = list(col_bounds)

    if len(data_col_x) - 1 > n_marks:
        excess = (len(data_col_x) - 1) - n_marks
        data_col_x = data_col_x[excess:]        # drop extra from left
        print(f"  Trimmed {excess} extra V-line(s) from left (label col dividers)")
    elif len(data_col_x) - 1 < n_marks:
        col_marks = col_marks[:len(data_col_x) - 1]
        n_marks   = len(col_marks)
        print(f"  ⚠ Truncated marks to {n_marks} to match detected columns")

    n_cols      = len(data_col_x) - 1
    label_x1    = data_col_x[0]          # label column right edge = data grid left edge
    label_x0    = 0
    print(f"  Grid x-start: {label_x1}  |  {n_cols} data columns  "
          f"X={data_col_x[0]}–{data_col_x[-1]}")

    # ── §11.5  Floor row boundary detection ───────────────────────────────────
    # Primary: red H-line scan inside the data columns
    # ── §11.5  Floor row boundary detection (robust) ─────────────

    h_lines = []

    # Pass 1: strict
    h_lines = detect_row_boundaries(arr,
        x0=data_col_x[0], x1=data_col_x[-1],
        y0=floor_data_y0, y1=footing_y0,
        threshold=0.15, min_dist=80)

    # Pass 2: medium
    if len(h_lines) < 5:
        h_lines = detect_row_boundaries(arr,
            x0=data_col_x[0], x1=data_col_x[-1],
            y0=floor_data_y0, y1=footing_y0,
            threshold=0.08, min_dist=60)

    # Pass 3: aggressive (pattern-12 fix)
    if len(h_lines) < 5:
        h_lines = detect_row_boundaries(arr,
            x0=data_col_x[0], x1=data_col_x[-1],
            y0=floor_data_y0, y1=footing_y0,
            threshold=0.04, min_dist=40)

    # Final fallback: smoothed projection
    if len(h_lines) < 5:
        print("  ⚠ Switching to fallback H-line detection")

        region = arr[floor_data_y0:footing_y0,
                    data_col_x[0]:data_col_x[-1]]

        r = region[:, :, 0].astype(int)
        g = region[:, :, 1].astype(int)
        b = region[:, :, 2].astype(int)

        red_mask = (r > 140) & ((r - g) > 40) & ((r - b) > 40)
        density = red_mask.mean(axis=1)

        # smooth (CRITICAL)
        density = np.convolve(density, np.ones(25)/25, mode='same')

        peaks, _ = find_peaks(density, height=0.03, distance=35)
        h_lines = cluster_lines([int(p) + floor_data_y0 for p in peaks], gap=8)

    # Build row bounds
    if h_lines:
        all_ys = [floor_data_y0] + h_lines + [footing_y0]
        row_bounds = [(all_ys[i], all_ys[i+1]) for i in range(len(all_ys)-1)]
        print(f"  Floor rows detected: {len(row_bounds)} (from H-lines)")
    else:
        row_bounds = None
        print("  ❌ H-line detection failed — fallback will be used")

    # ── §11.6  Floor labels (GPT-4o on full label column — one call) ──────────
    print("  [2/3] Extracting floor labels …")
    label_col_img  = _crop_upscale(pil, label_x0, floor_data_y0,
                                    label_x1, footing_y0, upscale=2)
    raw_all_floors = call_vision(client, label_col_img,
                                  _PROMPT_ALL_FLOORS, max_tokens=600)
    parsed_all     = _parse_json(raw_all_floors)

    floor_info_list = []
    if parsed_all and isinstance(parsed_all.get("floors"), list):
        for f in parsed_all["floors"]:
            floor_info_list.append({
                "name":        norm_column_name(f.get("name", "")),
                "mix":         norm_mix(f.get("mix")),
                "steel_grade": norm_steel(f.get("steel_grade")),
            })

    n_gpt_floors = len(floor_info_list)
    print(f"  GPT-4o identified {n_gpt_floors} floor rows")

    # If H-lines gave a very different count, prefer GPT-4o's count
    if row_bounds is None or abs(len(row_bounds) - n_gpt_floors) > 2:
        if n_gpt_floors > 0:
            step       = (footing_y0 - floor_data_y0) / n_gpt_floors
            row_bounds = [(int(floor_data_y0 + i * step),
                           int(floor_data_y0 + (i + 1) * step))
                          for i in range(n_gpt_floors)]
            print(f"  Using evenly-spaced row bounds ({n_gpt_floors} rows)")
        else:
            print("  ❌ Cannot determine floor rows — skipping page")
            return []

    n_rows = len(row_bounds)

    # Pad / trim floor_info_list to match detected row count
    while len(floor_info_list) < n_rows:
        floor_info_list.append({"name": "", "mix": None, "steel_grade": "FE500"})
    floor_info_list = floor_info_list[:n_rows]

    for idx, fi in enumerate(floor_info_list):
        print(f"    {idx + 1:02d}. '{fi['name'] or '(unnamed)'}'  "
              f"mix={fi['mix']}  grade={fi['steel_grade']}")

    # ── §11.7  Cell-by-cell extraction ────────────────────────────────────────
    total = n_rows * n_cols
    print(f"  [3/3] Extracting {total} cells ({n_rows} rows × {n_cols} cols) …")

    all_cols = []
    done     = 0
    stirrups = {"dia": None, "spacing": None}   # always null for column schedule

    for ri, (fi, (ry1, ry2)) in enumerate(zip(floor_info_list, row_bounds)):
        fname = fi["name"] or f"FLOOR_{ri + 1}"

        for ci in range(n_cols):
            cx1 = data_col_x[ci]
            cx2 = data_col_x[ci + 1]
            done += 1

            # Pixel pre-check — skip API call for blank cells
            raw_crop = pil.crop((max(0, cx1), max(0, ry1),
                                  min(W, cx2),  min(H, ry2)))
            if is_cell_empty(raw_crop):
                all_cols.append({
                    "column_no":    col_marks[ci],
                    "column_name":  norm_column_name(fname),
                    "size":         {"width": None, "depth": None, "length": None},
                    "reinforcement":[],
                    "stirrups":     stirrups,
                    "mix":          fi["mix"],
                    "steel_grade":  fi["steel_grade"],
                })
                print(f"       [{done:>3}/{total}]  {fname} / {col_marks[ci]}"
                      "  → empty (pixel check)")
                continue

            cell_img   = _crop_upscale(pil, cx1, ry1, cx2, ry2, upscale=3)
            print(f"       [{done:>3}/{total}]  {fname} / {col_marks[ci]} …",
                  end=" ", flush=True)
            raw_cell   = call_vision(client, cell_img, _PROMPT_CELL, max_tokens=150)
            parsed_cell = _parse_json(raw_cell)
            print("✓")

            if parsed_cell and isinstance(parsed_cell, dict):
                all_cols.append({
                    "column_no":    col_marks[ci],
                    "column_name":  norm_column_name(fname),
                    "size": {
                        "width":  parsed_cell.get("width"),
                        "depth":  None,
                        "length": parsed_cell.get("length"),
                    },
                    "reinforcement": norm_reinforcement(
                                        parsed_cell.get("reinforcement", [])),
                    "stirrups":     stirrups,
                    "mix":          fi["mix"],
                    "steel_grade":  fi["steel_grade"],
                })
            else:
                all_cols.append({
                    "column_no":    col_marks[ci],
                    "column_name":  norm_column_name(fname),
                    "size":         {"width": None, "depth": None, "length": None},
                    "reinforcement":[],
                    "stirrups":     stirrups,
                    "mix":          fi["mix"],
                    "steel_grade":  fi["steel_grade"],
                })

    print(f"  ✓ Page done: {len(all_cols)} raw entries")
    return all_cols


# ═══════════════════════════════════════════════════════════════════
#  §12  PDF PROCESSOR
# ═══════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: str) -> None:
    stem    = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}\n  PDF: {stem}.pdf\n{'═'*60}")

    client      = _get_client()
    image_paths = render_pdf(pdf_path, out_dir, dpi=300)

    batch_folder_name = "page_batches"
    batch_folder = os.path.join(out_dir, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    manifest = {
        "pattern": 12,
        "mode": "page_batches",
        "batches": [
            {
                "id": f"p{idx + 1:02d}",
                "page": idx + 1,
                "status": "pending",
                "file": os.path.join(batch_folder_name, f"p{idx + 1:02d}.json"),
                "error": None,
            }
            for idx in range(len(image_paths))
        ],
    }
    manifest_path = os.path.join(out_dir, "page_manifest.json")
    atomic_write_json(manifest_path, manifest)

    def _set_batch_status(batch_id, status, error=None):
        for batch in manifest["batches"]:
            if batch["id"] == batch_id:
                batch["status"] = status
                batch["error"] = error
                return

    raw_cols = []
    for page_index, img_path in enumerate(image_paths):
        batch_id = f"p{page_index + 1:02d}"
        _set_batch_status(batch_id, "running")
        atomic_write_json(manifest_path, manifest)
        print(f"\n  ── Page: {os.path.basename(img_path)} ──")
        page_cols = process_page(img_path, client)
        raw_cols.extend(page_cols)
        atomic_write_json(os.path.join(batch_folder, f"{batch_id}.json"), {"columns": page_cols})
        _set_batch_status(batch_id, "done")
        atomic_write_json(manifest_path, manifest)

    # ── Expand combined marks ("C1,C18" → C1 + C18 with identical data) ──────
    expanded = []
    for col in raw_cols:
        parts = [m.strip() for m in col["column_no"].split(",") if m.strip()]
        if len(parts) <= 1:
            expanded.append(col)
        else:
            for mark in parts:
                expanded.append({**col, "column_no": mark})

    # ── Deduplicate on (column_no, column_name) ───────────────────────────────
    seen, final = set(), []
    for col in expanded:
        key = (col["column_no"], col["column_name"])
        if key not in seen:
            seen.add(key); final.append(col)

    out_file = os.path.join(out_dir, f"{stem}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reshape_columns_to_levels(final), f, indent=2, ensure_ascii=False)

    print(f"\n✅  {len(final)} entries  →  {out_file}")
    if final:
        print("\n── First entry " + "─" * 45)
        print(json.dumps(final[0],  indent=2, ensure_ascii=False))
        print("\n── Last entry "  + "─" * 46)
        print(json.dumps(final[-1], indent=2, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════
#  §13  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdfs = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print("⚠  No PDF files found in input folder.")
        return
    for pdf in pdfs:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

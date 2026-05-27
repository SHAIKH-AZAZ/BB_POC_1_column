"""
pattern6_cell_verifier.py
=========================

Grid detection and level-label crop reader for Pattern 6 column schedules.

Pattern 6 layout (TRANSPOSED grid):

    ┌────────────┬─────────┬─────────┬─────────┬─────────┐
    │ <rotated   │ cell    │ cell    │ cell    │ cell    │   band 1
    │  level     │         │         │         │         │
    │  label>    │         │         │         │         │
    │            │  ring   │  ring   │  ring   │  ring   │   (sub-row)
    ├────────────┼─────────┼─────────┼─────────┼─────────┤
    │ ...        │ ...     │ ...     │ ...     │ ...     │   band 2..N
    ├────────────┼─────────┼─────────┼─────────┼─────────┤
    │  COLUMN    │ TA-C1   │ TA-C2   │ TA-C29  │ TA-C5   │   column-ID row
    └────────────┴─────────┴─────────┴─────────┴─────────┘

Differences from Pattern 1's verifier:
  * Levels live in ROWS (not a vertical bounded cell column).
  * Column IDs live in the BOTTOM row (not the top header).
  * Each data band contains an inner sub-row divider ("COL. RING/LINK")
    that must be IGNORED when listing band boundaries.

This module exposes the same surface API as `pattern1_cell_verifier`
so `pattern_batching.extract_levels_with_checkpoints` can wire it in
via the `detect_levels_fn` / `verify_cells_fn` / `job_decorator` hooks.

NOTE: OpenCV thresholds below are tuned conservatively. If a real
Pattern 6 PDF rasterises at a very different DPI, the
`MIN_BAND_HEIGHT_PX` / `MIN_STRIP_WIDTH_PX` constants may need
re-calibration.
"""

import json
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from extraction_guard import clean_json_string
from vision_extractor import extract_from_image


# ---------------------------------------------------------------------------
# Tunables — adjust if the rendered DPI of pattern-6 PDFs changes a lot.
# ---------------------------------------------------------------------------

# At ~700 DPI a band is ~ 350-450 px tall.
# Below this gap, two adjacent horizontal lines are treated as one
# (e.g. the COL. RING/LINK sub-row divider just under a band divider).
MIN_BAND_HEIGHT_PX = 200

# At ~700 DPI a data strip is ~ 250-350 px wide.
MIN_STRIP_WIDTH_PX = 120

# A "real" band divider must span at least this fraction of the page
# width (the COL. RING/LINK sub-row may not span the full width once
# the column drawings break the line).
MIN_HLINE_WIDTH_FRACTION = 0.70

# A "real" strip divider must span at least this fraction of the band
# height.
MIN_VLINE_HEIGHT_FRACTION = 0.50


# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern6Grid:
    """Detected grid geometry for a single Pattern 6 page image."""

    h_lines: list           # y-positions of major horizontal dividers, sorted
    v_lines: list           # x-positions of major vertical dividers, sorted
    # The leftmost column (between v_lines[0] and v_lines[1]) holds
    # rotated level labels. Data strips start at index 1.
    data_col_start_index: int = 1
    # The bottommost row (between h_lines[-2] and h_lines[-1]) holds
    # column-ID labels. Data bands live above that row.
    column_id_row_top_index_offset: int = 1

    @property
    def level_count(self):
        # h_lines = top + (level_count) band-dividers + col-ID-divider + bottom
        # so level_count = len(h_lines) - 2
        return max(0, len(self.h_lines) - 2)

    @property
    def data_col_count(self):
        # v_lines = left + label-divider + (data_col_count) strip-dividers + right
        # so data_col_count = len(v_lines) - 2
        return max(0, len(self.v_lines) - 2)

    @property
    def column_id_row_top(self):
        return self.h_lines[-2] if len(self.h_lines) >= 2 else None

    @property
    def column_id_row_bottom(self):
        return self.h_lines[-1] if self.h_lines else None


# ---------------------------------------------------------------------------
# Line detection helpers
# ---------------------------------------------------------------------------

def _cluster(positions, gap=8):
    if not positions:
        return []
    arr = sorted(set(int(p) for p in positions))
    groups = [[arr[0]]]
    for pos in arr[1:]:
        if pos - groups[-1][-1] <= gap:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return [int(np.median(group)) for group in groups]


def _filter_min_spacing(lines, min_spacing):
    if len(lines) < 2:
        return list(lines)
    kept = [lines[0]]
    for line in lines[1:]:
        if line - kept[-1] >= min_spacing:
            kept.append(line)
    return kept


def _line_widths(binary, axis, line_positions, band=6):
    """Measure how far each detected line actually extends along its axis.

    axis=0 -> horizontal lines (measure x-span via row projection).
    axis=1 -> vertical   lines (measure y-span via col projection).
    Returns parallel list of widths (in pixels).
    """
    widths = []
    if axis == 0:
        for y in line_positions:
            y0 = max(0, y - band)
            y1 = min(binary.shape[0], y + band + 1)
            strip = binary[y0:y1, :]
            col_hits = np.any(strip > 0, axis=0)
            widths.append(int(np.sum(col_hits)))
    else:
        for x in line_positions:
            x0 = max(0, x - band)
            x1 = min(binary.shape[1], x + band + 1)
            strip = binary[:, x0:x1]
            row_hits = np.any(strip > 0, axis=1)
            widths.append(int(np.sum(row_hits)))
    return widths


def _detect_pattern6_lines(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = gray.shape
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 12, 40), 1))
    h_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_morph, axis=1)
    raw_h = np.where(h_proj > width * 255 * 0.06)[0].tolist()
    h_lines_all = _cluster(raw_h, gap=8)

    # Keep only lines spanning a high fraction of the page width
    # (this drops the COL. RING/LINK sub-row dividers when they don't
    # span the full width).
    h_widths = _line_widths(h_morph, axis=0, line_positions=h_lines_all, band=6)
    h_lines = [
        pos for pos, w in zip(h_lines_all, h_widths)
        if w >= width * MIN_HLINE_WIDTH_FRACTION
    ]
    h_lines = _filter_min_spacing(h_lines, MIN_BAND_HEIGHT_PX)

    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 12, 40)))
    v_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_morph, axis=0)
    raw_v = np.where(v_proj > height * 255 * 0.04)[0].tolist()
    v_lines_all = _cluster(raw_v, gap=8)

    v_widths = _line_widths(v_morph, axis=1, line_positions=v_lines_all, band=6)
    v_lines = [
        pos for pos, w in zip(v_lines_all, v_widths)
        if w >= height * MIN_VLINE_HEIGHT_FRACTION
    ]
    v_lines = _filter_min_spacing(v_lines, MIN_STRIP_WIDTH_PX)

    return h_lines, v_lines


def detect_pattern6_grid(image_path):
    h_lines, v_lines = _detect_pattern6_lines(image_path)
    grid = Pattern6Grid(h_lines=h_lines, v_lines=v_lines)

    if grid.level_count < 1 or grid.data_col_count < 1:
        raise RuntimeError(
            f"Pattern 6 grid detection failed: h={len(h_lines)} v={len(v_lines)}"
        )
    return grid


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")[:80] or "CELL"


def _save_crop(img, box, path, min_longest=1200):
    left, top, right, bottom = box
    crop = img.crop((left, top, right, bottom))
    longest = max(crop.size)
    if longest and longest < min_longest:
        scale = min_longest / longest
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)),
            Image.LANCZOS,
        )
    crop.save(path, "PNG")
    return path


def crop_pattern6_level_strip(image_path, grid, output_dir, padding=6):
    """Crop the leftmost (rotated label) column for the WHOLE grid."""
    os.makedirs(output_dir, exist_ok=True)
    x1, x2 = grid.v_lines[0], grid.v_lines[1]
    y1 = grid.h_lines[0]
    y2 = grid.column_id_row_top  # stop above the column-ID row
    with Image.open(image_path).convert("RGB") as img:
        w, h = img.size
        box = (max(0, x1 + padding), max(0, y1 + padding),
               min(w, x2 - padding), min(h, y2 - padding))
        path = os.path.join(output_dir, "pattern6_level_strip.png")
        return _save_crop(img, box, path, min_longest=1400)


def crop_pattern6_column_id_row(image_path, grid, output_dir, padding=4):
    """Crop the bottom row that contains the column ID labels."""
    os.makedirs(output_dir, exist_ok=True)
    x1 = grid.v_lines[grid.data_col_start_index]
    x2 = grid.v_lines[-1]
    y1 = grid.column_id_row_top
    y2 = grid.column_id_row_bottom
    with Image.open(image_path).convert("RGB") as img:
        w, h = img.size
        box = (max(0, x1 + padding), max(0, y1 + padding),
               min(w, x2 - padding), min(h, y2 - padding))
        path = os.path.join(output_dir, "pattern6_column_id_row.png")
        return _save_crop(img, box, path, min_longest=1600)


def crop_pattern6_cell(image_path, grid, level_index, col_index, output_dir, label, padding=6):
    if level_index < 0 or level_index >= grid.level_count:
        raise IndexError(f"Level index {level_index} outside {grid.level_count}")
    if col_index < 0 or col_index >= grid.data_col_count:
        raise IndexError(f"Column index {col_index} outside {grid.data_col_count}")
    os.makedirs(output_dir, exist_ok=True)
    x1 = grid.v_lines[grid.data_col_start_index + col_index]
    x2 = grid.v_lines[grid.data_col_start_index + col_index + 1]
    y1 = grid.h_lines[level_index]
    y2 = grid.h_lines[level_index + 1]
    with Image.open(image_path).convert("RGB") as img:
        w, h = img.size
        box = (max(0, x1 + padding), max(0, y1 + padding),
               min(w, x2 - padding), min(h, y2 - padding))
        path = os.path.join(
            output_dir,
            f"l{level_index + 1:03d}_c{col_index + 1:03d}_{_safe_name(label)}.png",
        )
        return _save_crop(img, box, path, min_longest=1200)


# ---------------------------------------------------------------------------
# Level label reading (rotated text on the LEFT strip)
# ---------------------------------------------------------------------------

def _parse_levels_response(raw):
    data = json.loads(clean_json_string(raw))
    if isinstance(data, dict):
        levels = data.get("levels", [])
    elif isinstance(data, list):
        levels = data
    else:
        levels = []
    cleaned = []
    for level in levels:
        value = str(level).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def detect_levels_from_pattern6_label_crop(image_path, grid, output_dir):
    """Crop the leftmost label strip and ask the vision model to read
    the rotated 'COLUMN FROM X TO Y' labels top-to-bottom."""

    crop_path = crop_pattern6_level_strip(image_path, grid, output_dir)

    prompt = f"""
You are reading ONLY the cropped LEFT label strip from a Pattern 6 RCC
column schedule. The strip contains {grid.level_count} ROTATED level
labels stacked top-to-bottom. Each label is followed by a smaller
"CONCRETE MIX – Mxx" sub-line that is NOT part of the level name.

Return ONLY strict JSON in this exact shape:
{{
  "pattern": 6,
  "levels": []
}}

Rules:
- Read each rotated label exactly as printed (e.g. "COLUMN FROM THIRD
  FLOOR LVL. TO FOURTH FLOOR LVL.").
- Preserve top-to-bottom order.
- Include the leading word "COLUMN" if present, and keep "LVL." as
  "LVL." (do not expand to "LEVEL").
- Do NOT include the "CONCRETE MIX – Mxx" line.
- Do NOT add or invent levels.
- Expected level count: {grid.level_count}.
"""

    levels = _parse_levels_response(extract_from_image(crop_path, prompt))

    if len(levels) != grid.level_count:
        # One retry with a stricter instruction
        retry_prompt = prompt + (
            f"\nYour previous response had {len(levels)} levels but the grid "
            f"requires exactly {grid.level_count}. Re-read carefully and "
            "return exactly that many."
        )
        levels = _parse_levels_response(extract_from_image(crop_path, retry_prompt))

    if not levels:
        raise RuntimeError("Pattern 6 label-crop level detection returned no levels")

    return levels, crop_path


# ---------------------------------------------------------------------------
# Column-ID reading (bottom row of the grid)
# ---------------------------------------------------------------------------

def _parse_column_ids_response(raw):
    data = json.loads(clean_json_string(raw))
    if isinstance(data, dict):
        ids = data.get("column_ids", [])
    elif isinstance(data, list):
        ids = data
    else:
        ids = []
    cleaned = []
    for entry in ids:
        text = re.sub(r"\s*,\s*", ",", str(entry or "").strip())
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def detect_column_ids_from_pattern6_id_row(image_path, grid, output_dir):
    """Crop the bottom column-ID row and read the IDs left-to-right."""
    crop_path = crop_pattern6_column_id_row(image_path, grid, output_dir)
    prompt = f"""
You are reading ONLY the cropped BOTTOM row of a Pattern 6 RCC column
schedule. Each cell of this row contains one column-ID label such as
"TA-C1, TA-C4" or "TA-C29".

Return ONLY strict JSON in this exact shape:
{{
  "pattern": 6,
  "column_ids": []
}}

Rules:
- Read cells left-to-right, one column_ids[] item per cell.
- Preserve comma-separated IDs as one item (e.g. "TA-C1, TA-C4"
  becomes "TA-C1,TA-C4" with the space stripped).
- Keep the "TA-" prefix and the hyphen exactly as printed.
- Expected number of cells: {grid.data_col_count}.
"""
    ids = _parse_column_ids_response(extract_from_image(crop_path, prompt))
    return ids, crop_path


# ---------------------------------------------------------------------------
# Cell-level verification (re-asks the model per crop)
# ---------------------------------------------------------------------------

def _to_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None


def _parse_size(value):
    if isinstance(value, dict):
        return {
            "width": _to_int(value.get("width")),
            "depth": None,
            "length": _to_int(value.get("length")) or _to_int(value.get("depth")),
        }
    text = str(value or "")
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if len(nums) >= 2:
        return {"width": nums[0], "depth": None, "length": nums[1]}
    if len(nums) == 1:
        return {"width": nums[0], "depth": None, "length": None}
    return {"width": None, "depth": None, "length": None}


def _normalize_reinforcement(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = re.split(r"[,\n]", values)
    cleaned = []
    for v in values:
        text = str(v or "").strip().upper().replace(" ", "")
        if not text:
            continue
        # Pattern 6 format is "<N>-<dia>T"  e.g. "20-20T".
        # Allow simple variations and normalize to that.
        m = re.fullmatch(r"(\d+)[-]?(\d+)T", text)
        if m:
            text = f"{m.group(1)}-{m.group(2)}T"
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def verify_cell_crop(crop_path, level, column_no):
    """Re-extract size + reinforcement from ONE tight cell crop."""
    prompt = f"""
You are reading ONE cropped data cell from a Pattern 6 RCC column
schedule.

Context:
- Target level: {level}
- Target column group: {column_no}

Task:
- Read ONLY the visible text/dimensions inside this crop.
- The vertical dimension on the LEFT of the column drawing is LENGTH.
- The horizontal dimension at the BOTTOM is WIDTH.
- For L-shaped cells, sum the horizontal segments for width and the
  vertical segments for length.
- depth must always be null.
- Reinforcement sits at TOP-RIGHT and is formatted as "<N>-<dia>T",
  e.g. "20-20T", "4-16T". Return every bullet as a separate string.
- Ignore the "COL. RING/LINK" sub-row for this task.

Return ONLY strict JSON:
{{
  "size": {{"width": null, "depth": null, "length": null}},
  "reinforcement": [],
  "confidence": "high",
  "raw_text": ""
}}
"""
    raw = extract_from_image(crop_path, prompt)
    data = json.loads(clean_json_string(raw))
    return {
        "size": _parse_size(data.get("size")),
        "reinforcement": _normalize_reinforcement(data.get("reinforcement")),
        "confidence": str(data.get("confidence") or "").strip().lower(),
        "raw_text": str(data.get("raw_text") or "").strip(),
    }


def _has_verified_size(size):
    return bool(size and (size.get("width") is not None or size.get("length") is not None))


def verify_level_columns_with_crops(
    image_path,
    grid,
    level_index,
    level,
    columns,
    cell_crop_dir,
    column_grid_map=None,
):
    """Crop each column's cell for this level and re-extract size +
    reinforcement. Returns (updated_columns, verification_report)."""

    os.makedirs(cell_crop_dir, exist_ok=True)
    by_group = (column_grid_map or {}).get("by_group", {})
    report = {"level": level, "cells": []}

    for record_index, record in enumerate(columns):
        column_no = str(record.get("column_no") or "").strip()
        key = ",".join(re.findall(r"TA-?C\d+[A-Z]?", column_no.upper()))

        col_index = by_group.get(key, record_index)
        if col_index >= grid.data_col_count:
            report["cells"].append({
                "column_no": column_no,
                "skipped": True,
                "reason": f"col_index {col_index} out of range",
            })
            continue

        try:
            crop_path = crop_pattern6_cell(
                image_path, grid, level_index, col_index,
                cell_crop_dir, label=column_no,
            )
            verified = verify_cell_crop(crop_path, level, column_no)
        except Exception as exc:
            report["cells"].append({
                "column_no": column_no,
                "skipped": True,
                "reason": str(exc),
            })
            continue

        # Apply verified values when they look more confident.
        if _has_verified_size(verified["size"]):
            record["size"] = verified["size"]
        if verified["reinforcement"]:
            record["reinforcement"] = verified["reinforcement"]

        report["cells"].append({
            "column_no": column_no,
            "col_index": col_index,
            "crop": os.path.basename(crop_path),
            "size": verified["size"],
            "reinforcement": verified["reinforcement"],
            "confidence": verified["confidence"],
        })

    return columns, report

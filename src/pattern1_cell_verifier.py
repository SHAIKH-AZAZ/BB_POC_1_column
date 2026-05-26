import json
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pdfplumber
from PIL import Image

from extraction_guard import clean_json_string
from vision_extractor import extract_from_image


@dataclass(frozen=True)
class Pattern1Grid:
    h_lines: list
    v_lines: list
    header_bottom_index: int = 1
    data_col_start_index: int = 2

    @property
    def level_count(self):
        return max(0, (len(self.h_lines) - 3) // 2)

    @property
    def data_col_count(self):
        return max(0, len(self.v_lines) - self.data_col_start_index - 1)


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
        return lines

    kept = [lines[0]]
    for line in lines[1:]:
        if line - kept[-1] >= min_spacing:
            kept.append(line)
    return kept


def _detect_pattern1_lines(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = gray.shape
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 10, 40), 1))
    h_proj = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel), axis=1)
    h_lines = _cluster(
        np.where(h_proj > width * 255 * 0.08)[0].tolist(),
        gap=8,
    )
    h_lines = _filter_min_spacing(h_lines, 60)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 10, 40)))
    v_proj = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel), axis=0)
    v_lines = _cluster(
        np.where(v_proj > height * 255 * 0.03)[0].tolist(),
        gap=8,
    )
    v_lines = _filter_min_spacing(v_lines, 80)

    return h_lines, v_lines


def detect_pattern1_grid(image_path):
    h_lines, v_lines = _detect_pattern1_lines(image_path)
    grid = Pattern1Grid(h_lines=h_lines, v_lines=v_lines)

    if grid.level_count < 1 or grid.data_col_count < 1:
        raise RuntimeError(
            f"Pattern 1 grid detection failed: h={len(h_lines)} v={len(v_lines)}"
        )

    return grid


def _column_ids(value):
    return re.findall(r"\bC\d+[A-Z]?\b", str(value or "").upper())


def _column_group_key(value):
    ids = _column_ids(value)
    return ",".join(ids)


def extract_pattern1_header_grid_map(pdf_path, page_index, grid):
    """
    Map visible Pattern 1 header cells to detected image-grid column indices.

    The schedule body text is not extractable in the current PDFs, but the
    column header IDs usually are. This prevents verifier crops from relying on
    model output order when the model merges/splits header groups.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )

    c_words = [word for word in words if _column_ids(word.get("text", ""))]
    if not c_words:
        return {"labels": [], "by_group": {}, "by_id": {}}

    top_limit = min(float(word["top"]) for word in c_words) + 60
    header_words = [word for word in c_words if float(word["top"]) <= top_limit]

    groups = []
    for word in sorted(header_words, key=lambda item: (float(item["x0"]) + float(item["x1"])) / 2):
        center = (float(word["x0"]) + float(word["x1"])) / 2
        target = None
        for group in groups:
            if abs(center - group["center"]) <= 15:
                target = group
                break
        if target is None:
            target = {"center": center, "words": []}
            groups.append(target)
        target["words"].append(word)

    labels = []
    for group in sorted(groups, key=lambda item: item["center"]):
        ids = []
        for word in sorted(group["words"], key=lambda item: (float(item["top"]), float(item["x0"]))):
            for column_id in _column_ids(word.get("text", "")):
                if column_id not in ids:
                    ids.append(column_id)
        if ids:
            labels.append(",".join(ids))

    labels = labels[: grid.data_col_count]

    by_group = {}
    by_id = {}
    for index, label in enumerate(labels):
        key = _column_group_key(label)
        if key:
            by_group[key] = index
        for column_id in _column_ids(label):
            by_id[column_id] = index

    return {
        "labels": labels,
        "by_group": by_group,
        "by_id": by_id,
    }


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")[:80] or "CELL"


def crop_pattern1_cell(
    image_path,
    grid,
    level_index,
    col_index,
    output_dir,
    label,
    padding=6,
    min_longest=1200,
):
    if level_index < 0 or level_index >= grid.level_count:
        raise IndexError(f"Level index {level_index} outside grid level count {grid.level_count}")
    if col_index < 0 or col_index >= grid.data_col_count:
        raise IndexError(f"Column index {col_index} outside grid column count {grid.data_col_count}")

    os.makedirs(output_dir, exist_ok=True)

    x1 = grid.v_lines[grid.data_col_start_index + col_index]
    x2 = grid.v_lines[grid.data_col_start_index + col_index + 1]
    y1 = grid.h_lines[grid.header_bottom_index + (level_index * 2)]
    y2 = grid.h_lines[grid.header_bottom_index + (level_index * 2) + 2]

    with Image.open(image_path).convert("RGB") as img:
        width, height = img.size
        left = max(0, x1 + padding)
        top = max(0, y1 + padding)
        right = min(width, x2 - padding)
        bottom = min(height, y2 - padding)

        crop = img.crop((left, top, right, bottom))
        longest = max(crop.size)
        if longest and longest < min_longest:
            scale = min_longest / longest
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.LANCZOS,
            )

        path = os.path.join(
            output_dir,
            f"l{level_index + 1:03d}_c{col_index + 1:03d}_{_safe_name(label)}.png",
        )
        crop.save(path, "PNG")

    return path


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
        width = _to_int(value.get("width"))
        length = _to_int(value.get("length"))
        if length is None:
            length = _to_int(value.get("depth"))
        return {"width": width, "depth": None, "length": length}

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
        values = re.split(r"[+,]", values)

    cleaned = []
    for value in values:
        text = str(value or "").strip().upper().replace(" ", "")
        if not text:
            continue

        text = text.replace("TOR", "T")
        text = text.replace("TT", "T")
        text = text.replace("–", "-").replace("—", "-")

        parts = re.split(r"\+", text)
        for part in parts:
            if not part:
                continue
            match = re.fullmatch(r"(\d+)[-]?T(\d+)", part)
            if match:
                item = f"{match.group(1)}-T{match.group(2)}"
            else:
                item = part
            if item and item not in cleaned:
                cleaned.append(item)
    return cleaned


def _parse_verification(raw):
    data = json.loads(clean_json_string(raw))

    size = _parse_size(data.get("size"))
    reinforcement = _normalize_reinforcement(data.get("reinforcement"))

    return {
        "size": size,
        "reinforcement": reinforcement,
        "confidence": str(data.get("confidence") or "").strip().lower(),
        "raw_text": str(data.get("raw_text") or "").strip(),
    }


def verify_cell_crop(crop_path, level, column_no):
    prompt = f"""
You are reading ONE cropped data cell from a Pattern 1 RCC column schedule.

Context:
- Target level/floor: {level}
- Target column group: {column_no}

Task:
- Read ONLY the visible text inside this crop.
- The crop should contain one SIZE line and one REINF. line for this one column group and level.
- Do not borrow values from any other row or column.
- This schedule uses B x L size order.
- If the crop says "750 x 300", return width=750, depth=null, length=300.
- Keep the first visible number as width/B and the second visible number as length/L.
- Do not sort, rotate, or swap size dimensions.
- Normalize reinforcement as quantity-Tdiameter, e.g. "16-T16".
- If a value is not clearly visible, return null or [] for that field.

Return ONLY strict JSON:
{{
  "size": {{"width": null, "depth": null, "length": null}},
  "reinforcement": [],
  "confidence": "high",
  "raw_text": ""
}}
"""
    raw = extract_from_image(crop_path, prompt)
    return _parse_verification(raw)


def _has_verified_size(size):
    return bool(size and (size.get("width") is not None or size.get("length") is not None))


def verify_level_columns_with_crops(
    image_path,
    grid,
    level_index,
    level,
    columns,
    crop_dir,
    column_grid_map=None,
):
    corrected = []
    report = {
        "level": level,
        "level_index": level_index,
        "grid_level_count": grid.level_count,
        "grid_data_col_count": grid.data_col_count,
        "header_labels": (column_grid_map or {}).get("labels", []),
        "checked": 0,
        "corrected": [],
        "errors": [],
    }

    for col_index, col in enumerate(columns):
        item = dict(col)
        grid_col_index = _resolve_grid_col_index(
            item.get("column_no"),
            col_index,
            column_grid_map,
        )
        if grid_col_index >= grid.data_col_count:
            report["errors"].append(
                {
                    "column_no": item.get("column_no"),
                    "fallback_index": col_index,
                    "grid_col_index": grid_col_index,
                    "error": "column index outside detected grid",
                }
            )
            corrected.append(item)
            continue

        label = f"{level}_{item.get('column_no', '')}"
        try:
            crop_path = crop_pattern1_cell(
                image_path,
                grid,
                level_index,
                grid_col_index,
                crop_dir,
                label,
            )
            verified = verify_cell_crop(crop_path, level, item.get("column_no", ""))
            report["checked"] += 1

            before = {
                "size": item.get("size"),
                "reinforcement": item.get("reinforcement") or [],
            }

            changed = False
            if _has_verified_size(verified.get("size")):
                item["size"] = verified["size"]
                changed = item["size"] != before["size"]

            if verified.get("reinforcement"):
                item["reinforcement"] = verified["reinforcement"]
                changed = changed or item["reinforcement"] != before["reinforcement"]

            if changed:
                report["corrected"].append(
                    {
                        "column_no": item.get("column_no"),
                        "grid_col_index": grid_col_index,
                        "before": before,
                        "after": {
                            "size": item.get("size"),
                            "reinforcement": item.get("reinforcement") or [],
                        },
                        "confidence": verified.get("confidence"),
                        "raw_text": verified.get("raw_text"),
                        "crop": crop_path,
                    }
                )

        except Exception as exc:
            report["errors"].append(
                {
                    "column_no": item.get("column_no"),
                    "grid_col_index": grid_col_index,
                    "error": str(exc),
                }
            )

        corrected.append(item)

    return corrected, report


def _resolve_grid_col_index(column_no, fallback_index, column_grid_map):
    if not column_grid_map:
        return fallback_index

    by_group = column_grid_map.get("by_group") or {}
    by_id = column_grid_map.get("by_id") or {}

    group_key = _column_group_key(column_no)
    if group_key in by_group:
        return by_group[group_key]

    ids = _column_ids(column_no)
    mapped = [by_id[column_id] for column_id in ids if column_id in by_id]
    unique = []
    for index in mapped:
        if index not in unique:
            unique.append(index)

    if not unique:
        return fallback_index
    if fallback_index in unique:
        return fallback_index
    return unique[0]

"""
main_15.py — Pattern 15 (shear-wall column schedule).

Architecture (mirrors main_1's grid + crop approach):
  1. PDF -> high-res image render.
  2. pdfplumber reads the column IDs (SW1..SWn) and floor levels as VECTOR TEXT and
     builds a deterministic per-cell grid  (pattern15_pdf_grid). No vision, so the
     column labels can never be hallucinated.
  3. Each cell is cropped from the render at the pdfplumber coordinates, upscaled,
     and the vision model reads ONLY the size / reinforcement / stirrups / mix that
     are drawn as graphics (not in the text layer).
  4. Records are assembled level-wise into the canonical schema.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image
from pattern15_pdf_grid import build_pattern15_cells, crop_cell_image

MAX_WORKERS = 4

# Canonical top-to-bottom level order for this pattern.
_LEVEL_ORDER = [
    "THIRD FLOOR TO TERRACE LEVEL",
    "FIRST FLOOR TO THIRD FLOOR LEVEL",
    "FOUNDATION TO FIRST FLOOR LEVEL",
]


# ==============================
# CLEANERS
# ==============================

def clean_size(size):
    if not isinstance(size, dict):
        return {"width": None, "depth": None, "length": None}
    return {
        "width": size.get("width"),
        "depth": None,
        "length": size.get("length"),
    }


def clean_reinforcement(values):
    if not values:
        return []
    cleaned = []
    for v in values:
        for part in str(v).upper().split("+"):
            part = part.strip()
            if part and part not in cleaned:
                cleaned.append(part)
    return cleaned


def clean_stirrups(stirrups):
    if not isinstance(stirrups, dict):
        return {"dia": "", "spacing": ""}
    dia = str(stirrups.get("dia") or "").upper().strip()
    text = str(stirrups.get("spacing") or "").upper()
    spacing = sorted(
        {f"{n} C/C" for n in re.findall(r"(\d+)\s*C/?C", text)},
        key=lambda x: int(x.split()[0]),
    )
    return {"dia": dia, "spacing": ", ".join(spacing)}


def _sw_num(label):
    m = re.search(r"\d+", label or "")
    return int(m.group()) if m else 9999


# ==============================
# VISION READ (size / reinf / stirrups / mix only)
# ==============================

def _read_cells(crop_path, label, levels):
    n = len(levels)
    prompt = f"""This image is a cropped strip from a shear-wall (SW) column schedule,
for column "{label}". It contains {n} cross-section cell(s) stacked TOP to BOTTOM,
for these floor levels in order:
{json.dumps(levels, ensure_ascii=False)}

For EACH cell read ONLY the printed annotations (never invent values):
- size: two dimensions like "200 X 800" -> width=200 (the smaller), length=800
        (the larger), depth=null. If the cell shows "AS/PLAN" or a complex shape
        with no plain WxL, use null for width and length.
- reinforcement: like "4-T16 + 8-T12" -> ["4-T16", "8-T12"]. Split on "+".
- stirrups: link notes like "T8@75c/c" or "T8@200c/c" -> dia "T8", and spacing the
        number(s) before c/c (e.g. "75 C/C, 200 C/C").
- mix: a concrete grade like "M-30" if printed, else null.

Do NOT output the column id or the level name — those are already known.

Return STRICT JSON with EXACTLY {n} cell(s), in the SAME top-to-bottom order:
{{"cells": [{{"size": {{"width": null, "depth": null, "length": null}},
  "reinforcement": [], "stirrups": {{"dia": "", "spacing": ""}}, "mix": null}}]}}
"""
    try:
        data = json.loads(extract_from_image(crop_path, prompt))
    except Exception as exc:
        print(f"  [READ-FAIL] {label}: {exc}")
        return []
    return data.get("cells", []) if isinstance(data, dict) else []


# ==============================
# RESHAPE -> CANONICAL SCHEMA
# ==============================

def _reshape(records):
    from collections import OrderedDict

    by_level = OrderedDict((lv, []) for lv in _LEVEL_ORDER)
    for rec in records:
        by_level.setdefault(rec["column_name"], []).append(rec)

    levels = []
    for level, cols in by_level.items():
        cols = sorted(cols, key=lambda c: _sw_num(c["column_no"]))
        levels.append({
            "level": level,
            "columns": [
                {
                    "column_no": c["column_no"],
                    "size": c["size"],
                    "reinforcement": c["reinforcement"],
                    "stirrups": c["stirrups"],
                    "mix": c["mix"],
                }
                for c in cols
            ],
        })
    return {"levels": levels}


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=650)
    render_path = image_paths[0]

    print("  Reading SW labels + levels from PDF text (deterministic)...")
    jobs, (rw, rh) = build_pattern15_cells(pdf_path, render_path)
    labels = sorted({j["label"] for j in jobs}, key=_sw_num)
    total_cells = sum(len(j["levels"]) for j in jobs)
    print(f"  {len(labels)} column(s): {', '.join(labels)}")
    print(f"  {len(jobs)} crop(s), {total_cells} cell(s) to read")

    crop_dir = os.path.join(output_folder, "cell_crops")

    def run(job):
        crop_path = crop_cell_image(render_path, job["box"], crop_dir, job["label"])
        cells = _read_cells(crop_path, job["label"], job["levels"])
        out = []
        for i, level in enumerate(job["levels"]):
            cell = cells[i] if i < len(cells) and isinstance(cells[i], dict) else {}
            out.append({
                "column_no": job["label"],
                "column_name": level,
                "size": clean_size(cell.get("size")),
                "reinforcement": clean_reinforcement(cell.get("reinforcement")),
                "stirrups": clean_stirrups(cell.get("stirrups")),
                "mix": cell.get("mix"),
            })
        print(f"  ✓ {job['label']} ({len(job['levels'])} level(s))")
        return out

    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            records.extend(future.result())

    final_output = _reshape(records)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {output_file}")
    print(
        f"  {len(final_output['levels'])} level(s), "
        f"{sum(len(l['columns']) for l in final_output['levels'])} column entries total."
    )


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠ No PDF files found.")
        return
    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

"""
main_13.py  —  Pattern 13 column schedule extractor (model-driven flow)
=======================================================================

NEW FLOW (no scipy, no OpenCV, no upfront line detection):

  1. Pattern detection            (handled by auto_runner / pattern_detector)
  2. Render PDF page to image
  3. ★ Parallel level + column DETECTION  (one GPT call each)
       - LEVELS  : list of {name, y_top, y_bottom} top-to-bottom
       - COLUMNS : list of {column_no, x_left, x_right} left-to-right
       - The model returns the UPPER part of any "X TO Y" level name
         directly (e.g. "FOURTEENTH TO TERRACE" → "FOURTEENTH").
         No Python TO-splitting helper.
  4. Per-level horizontal-strip extraction  (parallel)
       - Crop the strip [0, y_top*H, W, y_bottom*H]
       - Send to extract_with_tools with the discovered column list
         baked into the prompt
       - Model uses zoom_region to drill into ambiguous cells
  5. Manifest + atomic level-batch JSONs (resumable)
  6. Final reshape → <pdf_stem>.json

Output schema = canonical project shape:
  {
    "levels": [
      {"level": "FOURTEENTH", "columns": [
         {"column_no": "SW1,SW2,SW3,SW4",
          "size": {"width": 230, "depth": null, "length": 1500},
          "reinforcement": [...],
          "stirrups": {"dia": [...], "spacing": [...]}}
      ]},
      ...
    ]
  }
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from extraction_guard import clean_json_string, reshape_columns_to_levels
from image_tools import crop_upscale_path
from pattern_batching import atomic_write_json, safe_filename, trace_key_for
from pattern_cleaners import standardize_records
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


# ════════════════════════════════════════════════════════════════════════════
# §0  TUNABLES
# ════════════════════════════════════════════════════════════════════════════

# Render DPI for Pattern 13. 300 is enough for the model to read SW labels;
# higher values mostly burn memory on A2 drawings.
DPI = int(os.getenv("PATTERN13_DPI", "300"))

# Parallel level workers — each level batch is one tool-protocol call.
MAX_LEVEL_WORKERS = max(1, int(os.getenv("PATTERN13_LEVEL_WORKERS", "3")))

# Padding (in normalised image units) added to the level strip so we
# don't slice through rotated text or thin borders.
LEVEL_Y_PADDING = float(os.getenv("PATTERN13_LEVEL_PADDING", "0.005"))


# ════════════════════════════════════════════════════════════════════════════
# §1  PROMPTS
# ════════════════════════════════════════════════════════════════════════════

# Level + bounds detection. The model returns ONLY the upper level name
# (e.g. "FOURTEENTH" for "FOURTEENTH TO TERRACE") — no Python TO-splitting
# happens anywhere downstream.
LEVEL_DETECTION_PROMPT = """\
You are reading a structural-engineering RCC COLUMN SCHEDULE drawing.

TASK
====
List EVERY visible level / floor / storey row in the schedule, top to
bottom, together with that row's vertical bounds in the image.

LEVEL NAME RULE — VERY IMPORTANT
================================
A level cell may show a single name (e.g. "TERRACE", "GROUND",
"BASEMENT") OR a range like "FOURTEENTH TO TERRACE" or "GF TO P01".

For ANY range, return ONLY the FIRST / UPPER name — the part BEFORE the
word "TO" — and never include "TO" or anything after it.
Apply this regardless of capitalisation or formatting.

Examples (return the right-hand side only):
  "FOURTEENTH TO TERRACE"          -> "FOURTEENTH"
  "GROUND TO FIRST"                -> "GROUND"
  "P05 TO P06"                     -> "P05"
  "BASE TO LG"                     -> "BASE"
  "BASEMENT TO GROUND"             -> "BASEMENT"
  "TERRACE"                        -> "TERRACE"  (no range, keep as-is)

Y BOUNDS
========
For each level, give the top and bottom of the horizontal strip that
holds that level's data, in NORMALISED coordinates (0.0 = top of image,
1.0 = bottom). Bounds may slightly overlap is fine; never invent extra
levels just to fill space.

OUTPUT
======
Return ONLY strict JSON, no markdown, no code fences:

{
  "levels": [
    {"name": "TERRACE",     "y_top": 0.06, "y_bottom": 0.10},
    {"name": "FOURTEENTH",  "y_top": 0.10, "y_bottom": 0.16},
    {"name": "THIRTEENTH",  "y_top": 0.16, "y_bottom": 0.22},
    {"name": "GROUND",      "y_top": 0.85, "y_bottom": 0.91},
    {"name": "BASEMENT",    "y_top": 0.91, "y_bottom": 0.97}
  ]
}
"""

# Column detection. Returns each visible column-ID group with its X bounds.
COLUMN_DETECTION_PROMPT = """\
You are reading a structural-engineering RCC COLUMN SCHEDULE drawing.

TASK
====
List EVERY visible column-ID group along the schedule's column-header
row (which may be at the TOP, BOTTOM, or both), together with each
group's horizontal bounds in NORMALISED coordinates (0.0 = left edge of
image, 1.0 = right edge).

A column-ID group may be a single ID (e.g. "SW17") or a comma-joined
group (e.g. "SW1,SW2,SW3,SW4" / "C1,C2" / "GC39" / "TA-C1,TA-C4").
Preserve the prefix exactly (SW, C, GC, PC, TA-C, …) and case.

Do NOT return:
  - the words COLUMN, COLUMNS, MARK, MARKED, NOS, SIZE, REINF., STIRRUPS
  - concrete grades like M30 / M-30
  - TYPE-XX size codes
  - row-label words like FLOOR, LEVEL, FOOTING, TERRACE, LMR

Skip anything that is clearly not a structural column / shear-wall ID.

OUTPUT
======
Return ONLY strict JSON, no markdown, no code fences:

{
  "columns": [
    {"column_no": "SW1,SW2,SW3,SW4",  "x_left": 0.18, "x_right": 0.32},
    {"column_no": "SW5,SW6,SW7,SW8",  "x_left": 0.32, "x_right": 0.45},
    {"column_no": "SW17,SW18",        "x_left": 0.45, "x_right": 0.55}
  ]
}
"""


def build_level_extraction_prompt(level_name, columns):
    """Per-level extraction prompt fed to extract_with_tools.

    The strip image already covers ONE level. We list the discovered
    column IDs in left-to-right order so the model knows what's in the
    strip and can call zoom_region for any cell that's unclear.
    """
    column_list = "\n".join(
        f"  - {c['column_no']}  (x_left={c['x_left']:.2f}, x_right={c['x_right']:.2f})"
        for c in columns
    )
    return f"""\
You are reading ONE horizontal strip from an RCC COLUMN SCHEDULE.

LEVEL CONTEXT
=============
This strip contains data for level "{level_name}" only.
Use storey_level = "{level_name}" for EVERY add_column call.
Never invent or rename the level.

COLUMNS IN THIS STRIP (left → right)
====================================
{column_list}

Each column above corresponds to one cell in the strip. If a cell is
visually empty (no text, no drawing data), still call add_column for
it with size = null and reinforcement = []; do NOT borrow values from
neighbouring cells.

PER-CELL EXTRACTION
===================
For each column listed above, extract:

  size            B x L format (project standard).
                  width = first number, length = second number, depth = null.
                  e.g. "230 X 1500" -> width=230, depth=null, length=1500.

  reinforcement   Quantity-Tdiameter list, split on "+".
                  e.g. "8-Y12 + 14-Y10" -> ["8-Y12", "14-Y10"].
                  e.g. "16-T16"         -> ["16-T16"].

  ties_dia        Stirrup bar diameter, e.g. "T8" or "Y8".
  ties_spacing    Stirrup spacing, e.g. "200 C/C".
                  If multiple zones, pick the most prominent; the model
                  may also use confirm_read after a zoom_region call to
                  capture additional zones.

  mix             Concrete grade if visible (e.g. "M30", "M-30").
                  Use null when not shown.

  steel_grade     e.g. "Fe500". Use null when not shown.

ZOOM USAGE
==========
The strip image is the FULL strip in normalised (0..1) coordinates.
If any column's text is small or ambiguous, call zoom_region with
normalised bounds INSIDE this strip to inspect it more closely, then
call confirm_read with the exact text you saw, then call add_column.

ENFORCED TOOL SEQUENCE
======================
1. think  (with column_headers = the column_no values listed above)
2. (optional) zoom_region + confirm_read for any unclear cell
3. add_column  — one call per column listed above, in left-to-right
                  order, with column_no copied EXACTLY from the list.
"""


# ════════════════════════════════════════════════════════════════════════════
# §2  Level / column detection
# ════════════════════════════════════════════════════════════════════════════


def _parse_json_object(raw):
    """Extract the first JSON object from a possibly-fenced model reply."""
    text = clean_json_string(raw or "")
    if not text:
        return {}
    # Strip any leading prose before the first {
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def detect_levels_with_bounds(image_path):
    """Return [{"name": str, "y_top": float, "y_bottom": float}] top-to-bottom.

    The model is instructed to return the upper part of any "X TO Y"
    range directly, so no further Python normalisation is required.
    """
    print(f"  [LEVELS] Detecting levels in {os.path.basename(image_path)} …")
    raw = extract_from_image(image_path, LEVEL_DETECTION_PROMPT)
    data = _parse_json_object(raw)

    raw_levels = data.get("levels") or data.get("storey_levels") or []
    cleaned = []
    seen = set()
    for entry in raw_levels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        # Defensive: model occasionally still leaves "TO ..." in the name.
        # Trim everything from " TO " onwards. NOT a split-helper function;
        # this is just a one-line guard at the seam.
        upper_idx = name.upper().find(" TO ")
        if upper_idx > 0:
            name = name[:upper_idx].strip()
        try:
            y_top = float(entry.get("y_top", 0.0))
            y_bottom = float(entry.get("y_bottom", 0.0))
        except (TypeError, ValueError):
            continue
        if y_bottom <= y_top:
            continue
        # Collapse duplicates with same name; keep the union of bounds.
        if name in seen:
            for prev in cleaned:
                if prev["name"] == name:
                    prev["y_top"] = min(prev["y_top"], y_top)
                    prev["y_bottom"] = max(prev["y_bottom"], y_bottom)
                    break
            continue
        seen.add(name)
        cleaned.append({
            "name": name,
            "y_top": max(0.0, y_top),
            "y_bottom": min(1.0, y_bottom),
        })

    cleaned.sort(key=lambda lv: lv["y_top"])
    if not cleaned:
        raise RuntimeError(
            f"Level detection returned no valid entries for {image_path}."
        )
    print(f"  [LEVELS] Found {len(cleaned)}: " + ", ".join(lv["name"] for lv in cleaned))
    return cleaned


def detect_columns_with_bounds(image_path):
    """Return [{"column_no": str, "x_left": float, "x_right": float}] left-to-right."""
    print(f"  [COLS]   Detecting columns in {os.path.basename(image_path)} …")
    raw = extract_from_image(image_path, COLUMN_DETECTION_PROMPT)
    data = _parse_json_object(raw)

    raw_cols = data.get("columns") or data.get("column_headers") or []
    cleaned = []
    for entry in raw_cols:
        if not isinstance(entry, dict):
            continue
        column_no = str(entry.get("column_no") or "").strip()
        if not column_no:
            continue
        try:
            x_left = float(entry.get("x_left", 0.0))
            x_right = float(entry.get("x_right", 0.0))
        except (TypeError, ValueError):
            continue
        if x_right <= x_left:
            continue
        cleaned.append({
            "column_no": column_no,
            "x_left": max(0.0, x_left),
            "x_right": min(1.0, x_right),
        })

    cleaned.sort(key=lambda c: c["x_left"])
    if not cleaned:
        raise RuntimeError(
            f"Column detection returned no valid entries for {image_path}."
        )
    print(f"  [COLS]   Found {len(cleaned)}: " + ", ".join(c["column_no"] for c in cleaned))
    return cleaned


# ════════════════════════════════════════════════════════════════════════════
# §3  Per-level extraction
# ════════════════════════════════════════════════════════════════════════════


def _crop_level_strip(image_path, level, output_dir, page_index, level_index):
    """Crop the horizontal strip for one level and return its file path."""
    y_top = max(0.0, level["y_top"] - LEVEL_Y_PADDING)
    y_bottom = min(1.0, level["y_bottom"] + LEVEL_Y_PADDING)
    out_path = os.path.join(
        output_dir,
        f"p{page_index + 1:02d}_l{level_index + 1:03d}_{safe_filename(level['name'])}.png",
    )
    crop_upscale_path(
        image_path,
        x1=0.0, y1=y_top,
        x2=1.0, y2=y_bottom,
        normalized=True,
        min_longest=2400,        # plenty of resolution for column reads
        out_path=out_path,
    )
    return out_path


def extract_level_strip(strip_path, level_name, columns, trace_key=None):
    """Run the tool-protocol extraction on one level strip.

    Returns the parsed list of column records (one per discovered column)
    after defensive normalisation.
    """
    prompt = build_level_extraction_prompt(level_name, columns)
    raw = extract_with_tools(strip_path, prompt, trace_key=trace_key)
    parsed = _parse_json_object(raw)
    columns_out = parsed.get("columns", [])
    if not isinstance(columns_out, list):
        return []

    expected_ids = [c["column_no"] for c in columns]
    expected_set = {cid for cid in expected_ids}
    fixed = []
    for entry in columns_out:
        if not isinstance(entry, dict):
            continue
        cno = str(entry.get("column_no") or "").strip()
        if not cno:
            continue
        # Force column_name to the level we asked for, regardless of what
        # the model wrote into storey_level.
        entry["column_name"] = level_name
        # If the model returned a column_no not in our discovered list,
        # keep it but flag it. Most prompts behave; this is defensive.
        if cno not in expected_set:
            print(
                f"    [WARN] '{level_name}': model returned unexpected "
                f"column_no={cno!r} (not in discovered list)"
            )
        fixed.append(entry)
    return fixed


# ════════════════════════════════════════════════════════════════════════════
# §4  Manifest scaffolding
# ════════════════════════════════════════════════════════════════════════════


def _build_manifest(file_name, jobs):
    return {
        "pattern": 13,
        "pdf": f"{file_name}.pdf",
        "mode": "level_batches",
        "levels": [job["level"] for job in jobs],
        "batches": [
            {
                "id": job["id"],
                "page": job["page"],
                "level": job["level"],
                "status": "pending",
                "file": job["relative_file"],
                "error": None,
            }
            for job in jobs
        ],
    }


def _set_batch_status(manifest, batch_id, status, error=None):
    for batch in manifest["batches"]:
        if batch["id"] == batch_id:
            batch["status"] = status
            batch["error"] = error
            return


def _merge_done_batches(manifest, output_folder):
    flat = []
    for batch in manifest["batches"]:
        if batch.get("status") != "done":
            continue
        path = os.path.join(output_folder, batch["file"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            print(f"  Could not read batch {path}: {exc}")
            continue
        cols = data.get("columns") or []
        if isinstance(cols, list):
            flat.extend(cols)
    return flat


# ════════════════════════════════════════════════════════════════════════════
# §5  Main pipeline
# ════════════════════════════════════════════════════════════════════════════


def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images at {DPI} DPI …")
    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=DPI)
    if not image_paths:
        print("⚠ No images rendered.")
        return

    batch_folder_name = "level_batches"
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    strip_folder = os.path.join(output_folder, "level_strips")
    os.makedirs(strip_folder, exist_ok=True)

    jobs = []

    # ── Phase 1: parallel level + column detection per page ──────────────────
    for page_index, img_path in enumerate(image_paths):
        print(f"\n──── Page {page_index + 1}/{len(image_paths)} ────")

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_levels = ex.submit(detect_levels_with_bounds, img_path)
            f_cols = ex.submit(detect_columns_with_bounds, img_path)
            try:
                levels = f_levels.result()
            except Exception as exc:
                print(f"  ❌ Level detection failed on page {page_index + 1}: {exc}")
                continue
            try:
                columns = f_cols.result()
            except Exception as exc:
                print(f"  ❌ Column detection failed on page {page_index + 1}: {exc}")
                continue

        # ── Phase 2: build per-level jobs ────────────────────────────────────
        for level_index, level in enumerate(levels):
            try:
                strip_path = _crop_level_strip(
                    img_path, level, strip_folder, page_index, level_index
                )
            except Exception as exc:
                print(f"  ❌ Could not crop level '{level['name']}': {exc}")
                continue

            batch_id = f"p{page_index + 1:02d}_l{level_index + 1:03d}"
            batch_file = f"{batch_id}_{safe_filename(level['name'])}.json"
            jobs.append({
                "id": batch_id,
                "page": page_index + 1,
                "page_index": page_index,
                "level": level["name"],
                "level_index": level_index,
                "columns": columns,
                "img_path": img_path,
                "strip_path": strip_path,
                "path": os.path.join(batch_folder, batch_file),
                "relative_file": os.path.join(batch_folder_name, batch_file),
                "trace_key": trace_key_for(img_path, batch_id, level["name"]),
            })

    if not jobs:
        print("⚠ No level batches to extract.")
        return

    # ── Phase 3: parallel per-level extraction ──────────────────────────────
    manifest = _build_manifest(file_name, jobs)
    manifest_path = os.path.join(output_folder, "level_manifest.json")
    atomic_write_json(manifest_path, manifest)

    def _run(job):
        records = extract_level_strip(
            job["strip_path"],
            job["level"],
            job["columns"],
            trace_key=job["trace_key"],
        )
        atomic_write_json(
            job["path"],
            {"level": job["level"], "columns": records},
        )
        return job

    with ThreadPoolExecutor(max_workers=MAX_LEVEL_WORKERS) as ex:
        future_to_job = {}
        for job in jobs:
            print(f"  Queueing level batch -> {job['level']}")
            _set_batch_status(manifest, job["id"], "running")
            future_to_job[ex.submit(_run, job)] = job
        atomic_write_json(manifest_path, manifest)

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job)):
            job = future_to_job[future]
            try:
                future.result()
                _set_batch_status(manifest, job["id"], "done", None)
                print(f"  ✓ {job['level']}")
            except Exception as exc:
                _set_batch_status(manifest, job["id"], "failed", str(exc))
                print(f"  ✗ {job['level']}: {exc}")
            atomic_write_json(manifest_path, manifest)

    # ── Phase 4: merge + reshape ────────────────────────────────────────────
    flat = _merge_done_batches(manifest, output_folder)

    # Standardise + reshape. apply_upper_level=False because the model
    # already returned the upper part; we never had a "X TO Y" string.
    cleaned = standardize_records(
        flat,
        stirrups_cleaner=None,
        apply_column_filter=False,
        apply_upper_level=False,
    )
    final = reshape_columns_to_levels(cleaned)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    total_cols = sum(len(lvl["columns"]) for lvl in final.get("levels", []))
    print(
        f"\n✅ Output saved to {output_file} "
        f"({len(final.get('levels', []))} level(s), {total_cols} column entries)"
    )


# ════════════════════════════════════════════════════════════════════════════
# §6  Entry point
# ════════════════════════════════════════════════════════════════════════════


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

import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pattern_cleaners import clean_size
from pattern_batching import load_prompt, atomic_write_json, safe_filename
from extraction_guard import _coerce_stirrups_to_strings
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pattern1_cell_verifier import (
    detect_pattern1_grid,
    detect_levels_from_pattern1_label_crop,
    extract_pattern1_header_grid_map,
    verify_level_columns_with_crops,
)
from pdf_to_images import convert_pdf_to_images
from vision_extractor import detect_levels_from_image, extract_with_tools

MAX_LEVEL_WORKERS = 3
VERIFY_PATTERN1_CELLS = (
    os.getenv("PATTERN1_VERIFY_CELLS", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
USE_PATTERN1_LEVEL_CROP = (
    os.getenv("PATTERN1_LEVEL_CROP", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
_COLUMN_ID_RE = re.compile(r"\b(?:C|CP|COL|COLUMN)\s*-?\s*\d+[A-Z]?\b", re.IGNORECASE)
_NON_COLUMN_LABELS = {
    "COLUMN NO",
    "COLUMN NOS",
    "COLUMN NUMBER",
    "COLUMN NUMBERS",
    "COLUMN MARK",
    "COLUMN MARKS",
}

# ==============================
# LOAD PROMPT
# ==============================


def build_level_batch_prompt(base_prompt, target_level, all_levels):
    levels_json = json.dumps(all_levels, ensure_ascii=False)
    return f"""
{base_prompt}

LEVEL-FIRST BATCH EXTRACTION:
- The visible floor/level list detected for this page is: {levels_json}
- Extract ONLY this target floor/level batch: "{target_level}".
- The target level has already been chosen from the visible level cell using the upper-level rule; do not reinterpret it as the lower level of a range.
- In think.storey_levels, return exactly one item: "{target_level}".
- Every add_column call in this batch must use storey_level exactly as "{target_level}".
- Do not extract, copy, borrow, or infer values from any other floor/level.
- If a column group has no visible values for "{target_level}", return null size and [] reinforcement for that column group.
- Keep only actual column identifier headers/groups from the visible table, such as C1 or C1,C7,C8.
- Do not call add_column for table labels such as "Column Nos.", "SIZE", "REINF.", or "STIRRUPS".
- Limit all detail extraction to "{target_level}" only.
"""


def build_level_discovery_context():
    return """
Pattern 1 context:
- The level/floor labels are in the bounded level column beside the SIZE/REINF. rows.
- A level cell can span the SIZE row and REINF. row, and its text may be stacked across multiple lines.
- For a stacked range cell, read the whole bounded cell before deciding the level name.
- Copy LEVEL and FLOOR exactly as visible; do not swap those words.
- Use the upper level from each range cell:
  "TERRACE FLOOR LEVEL / To / 4TH FLOOR LEVEL" -> "TERRACE FLOOR LEVEL"
  "GROUND FLOOR LEVEL / To / BASEMENT LEVEL" -> "GROUND FLOOR LEVEL"
  "BASEMENT LEVEL / To / FOUNDATION LEVEL" -> "BASEMENT LEVEL"
- Do not return "FOUNDATION LEVEL" when it appears only as the lower line of "BASEMENT LEVEL / To / FOUNDATION LEVEL".
- Do not return "1ST BASEMENT LEVEL" unless the ordinal "1ST" is visibly present in that same bounded cell.
"""


# ==============================
# CLEAN SIZE
# ==============================


# ==============================
# CLEAN REINFORCEMENT
# ==============================


def clean_reinforcement(values):
    if not values:
        return []
    cleaned = []
    for v in values:
        v = str(v).strip().upper()
        for part in v.split("+"):
            part = part.strip()
            if part and part not in cleaned:
                cleaned.append(part)
    return cleaned


# ==============================
# CLEAN STIRRUPS
# ==============================


def clean_stirrups(stirrups):
    if not stirrups:
        return {"dia": [], "spacing": []}

    if isinstance(stirrups, dict):
        dia = stirrups.get("dia", [])
        spacing = stirrups.get("spacing", [])
        if isinstance(dia, str):
            dia = [dia] if dia else []
        if isinstance(spacing, str):
            spacing = [spacing] if spacing else []
        return {
            "dia": sorted(set(d for d in dia if d)),
            "spacing": sorted(set(s for s in spacing if s)),
        }

    text = str(stirrups).upper()
    dia = []
    spacing = []

    dm = re.search(r"T\d+", text)
    if dm:
        dia = [dm.group()]

    for s in re.findall(r"\d+\s*C/?C", text):
        s = re.sub(r"C/?C", "C/C", s.strip())
        spacing.append(s)

    return {"dia": dia, "spacing": sorted(set(spacing))}


def is_valid_column_no(value):
    text = str(value or "").strip()
    if not text:
        return False

    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    if normalized in _NON_COLUMN_LABELS:
        return False

    return bool(_COLUMN_ID_RE.search(text))


# ==============================
# RESHAPE TO LEVEL-CENTRIC JSON
# ==============================


def reshape_to_levels(flat_records):
    """
    Output shape (canonical project-wide):
    {
      "levels": [
        {
          "level": "ABOVE TERRACE LEVEL",   // "UNKNOWN" if no floor data
          "columns": [
            {
              "column_no": "C1,C2",
              "size": {"width": 700, "depth": null, "length": 700},
              "reinforcement": ["8-T16"],
              "stirrups": {"dia": "T8", "spacing": "100 C/C, 200 C/C"}
            }
          ]
        }
      ]
    }
    """
    level_map = OrderedDict()

    for record in flat_records:
        level = str(record.get("column_name") or "").strip() or "UNKNOWN"

        col_entry = {
            "column_no": record.get("column_no", ""),
            "size": record.get("size")
            or {"width": None, "depth": None, "length": None},
            "reinforcement": record.get("reinforcement") or [],
            "stirrups": _coerce_stirrups_to_strings(record.get("stirrups")),
        }

        if level not in level_map:
            level_map[level] = []
        level_map[level].append(col_entry)

    return {
        "levels": [
            {"level": level, "columns": cols} for level, cols in level_map.items()
        ]
    }


def reshape_batches_to_levels(level_batches):
    return {
        "levels": [
            {"level": level, "columns": columns}
            for level, columns in level_batches
        ]
    }


def column_entry_from_record(record):
    return {
        "column_no": record.get("column_no", ""),
        "size": record.get("size") or {"width": None, "depth": None, "length": None},
        "reinforcement": record.get("reinforcement") or [],
        "stirrups": _coerce_stirrups_to_strings(record.get("stirrups")),
    }


def update_batch_status(manifest, batch_id, status, error=None):
    for batch in manifest["batches"]:
        if batch["id"] == batch_id:
            batch["status"] = status
            batch["error"] = error
            break


def build_manifest(file_name, batch_jobs):
    return {
        "pattern": 1,
        "pdf": f"{file_name}.pdf",
        "levels": [job["level"] for job in batch_jobs],
        "batches": [
            {
                "id": job["id"],
                "page": job["page"],
                "level": job["level"],
                "status": "pending",
                "file": job["relative_file"],
                "error": None,
            }
            for job in batch_jobs
        ],
    }


def merge_done_level_batches(manifest, output_folder):
    levels = []
    for batch in manifest["batches"]:
        if batch.get("status") != "done":
            continue

        batch_path = os.path.join(output_folder, batch["file"])
        try:
            with open(batch_path, "r", encoding="utf-8") as f:
                level_data = json.load(f)
                level_data["columns"] = [
                    col
                    for col in level_data.get("columns", [])
                    if is_valid_column_no(col.get("column_no"))
                ]
                levels.append(level_data)
        except Exception as exc:
            print(f"  Could not merge level batch {batch_path}: {exc}")

    return {"levels": levels}


def extract_level_columns(img_path, prompt, levels, level, trace_key=None):
    level_prompt = build_level_batch_prompt(prompt, level, levels)
    result = extract_with_tools(img_path, level_prompt, trace_key=trace_key)
    parsed = json.loads(result)

    level_columns = []
    for col in parsed.get("columns", []):
        if not is_valid_column_no(col.get("column_no")):
            print(f"  Skipping non-column header: {col.get('column_no')}")
            continue

        col["column_name"] = level
        col["size"] = clean_size(col.get("size"))
        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))
        col["stirrups"] = clean_stirrups(col.get("stirrups"))
        col.setdefault("mix", None)
        col.setdefault("steel_grade", None)
        level_columns.append(column_entry_from_record(col))

    return level_columns


def run_level_job(job, prompt):
    level_columns = extract_level_columns(
        job["img_path"],
        prompt,
        job["levels"],
        job["level"],
        trace_key=job["trace_key"],
    )

    if VERIFY_PATTERN1_CELLS and job.get("grid") is not None:
        print(f"  Verifying cell crops -> {job['level']}")
        level_columns, verify_report = verify_level_columns_with_crops(
            job["img_path"],
            job["grid"],
            job["level_index"],
            job["level"],
            level_columns,
            job["cell_crop_dir"],
            column_grid_map=job.get("column_grid_map"),
        )
        os.makedirs(job["cell_crop_dir"], exist_ok=True)
        atomic_write_json(job["cell_verify_report_path"], verify_report)

    atomic_write_json(
        job["path"],
        {
            "level": job["level"],
            "columns": level_columns,
        },
    )
    return {
        "id": job["id"],
        "level": job["level"],
        "status": "done",
        "error": None,
    }


# ==============================
# PROCESS PDF
# ==============================


def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder)

    prompt = load_prompt(1)
    batch_folder_name = "level_batches"
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    cell_crop_folder_name = "cell_crops"
    cell_crop_folder = os.path.join(output_folder, cell_crop_folder_name)
    os.makedirs(cell_crop_folder, exist_ok=True)

    batch_jobs = []

    for page_index, img_path in enumerate(tqdm(image_paths)):
        page_grid = None
        column_grid_map = None
        if VERIFY_PATTERN1_CELLS or USE_PATTERN1_LEVEL_CROP:
            try:
                page_grid = detect_pattern1_grid(img_path)
                print(
                    "  Pattern 1 grid detected: "
                    f"{page_grid.level_count} level row(s), "
                    f"{page_grid.data_col_count} data column(s)"
                )
            except Exception as exc:
                print(f"  Pattern 1 grid detection unavailable: {exc}")
                page_grid = None

        print(f"  Detecting levels -> {img_path}")
        levels = []
        if USE_PATTERN1_LEVEL_CROP and page_grid is not None:
            try:
                level_crop_dir = os.path.join(
                    cell_crop_folder,
                    f"page_{page_index + 1:02d}_level_labels",
                )
                levels, level_crop_path = detect_levels_from_pattern1_label_crop(
                    img_path,
                    page_grid,
                    level_crop_dir,
                )
                print(f"  Level labels read from crop -> {level_crop_path}")
            except Exception as exc:
                print(f"  Level-label crop detection failed; falling back to full page: {exc}")

        if not levels:
            levels = detect_levels_from_image(
                img_path,
                pattern_number=1,
                prompt_context=build_level_discovery_context(),
            )

        if not levels:
            print("  No levels detected; falling back to whole-page extraction.")
            levels = ["UNKNOWN"]

        print(f"  Detected {len(levels)} level(s): {', '.join(levels)}")

        if VERIFY_PATTERN1_CELLS and page_grid is not None:
            if page_grid.level_count != len(levels):
                print(
                    "  Cell verification disabled for this page: "
                    f"detected {len(levels)} level name(s) but grid has "
                    f"{page_grid.level_count} level row(s)."
                )
                page_grid = None
                column_grid_map = None
            else:
                try:
                    column_grid_map = extract_pattern1_header_grid_map(
                        pdf_path,
                        page_index,
                        page_grid,
                    )
                    header_count = len(column_grid_map.get("labels") or [])
                    print(f"  Header grid map detected: {header_count} column header(s)")
                except Exception as exc:
                    print(f"  Cell verification disabled for this page: {exc}")
                    page_grid = None
                    column_grid_map = None

        for level_index, level in enumerate(levels):
            batch_id = f"p{page_index + 1:02d}_l{level_index + 1:03d}"
            batch_file = f"{batch_id}_{safe_filename(level)}.json"
            cell_batch_folder_name = os.path.join(
                cell_crop_folder_name,
                f"{batch_id}_{safe_filename(level)}",
            )
            cell_batch_folder = os.path.join(output_folder, cell_batch_folder_name)
            batch_jobs.append(
                {
                    "id": batch_id,
                    "page": page_index + 1,
                    "img_path": img_path,
                    "level": level,
                    "level_index": level_index,
                    "levels": levels,
                    "grid": page_grid,
                    "column_grid_map": column_grid_map,
                    "cell_crop_dir": cell_batch_folder,
                    "cell_verify_report_path": os.path.join(
                        cell_batch_folder,
                        "verification_report.json",
                    ),
                    "path": os.path.join(batch_folder, batch_file),
                    "relative_file": os.path.join(batch_folder_name, batch_file),
                    "trace_key": f"{os.path.basename(img_path)}__{batch_id}_{safe_filename(level)}",
                }
            )

    manifest_path = os.path.join(output_folder, "level_manifest.json")
    manifest = build_manifest(file_name, batch_jobs)
    atomic_write_json(manifest_path, manifest)

    with ThreadPoolExecutor(max_workers=MAX_LEVEL_WORKERS) as executor:
        future_to_job = {}
        for job in batch_jobs:
            level = job["level"]
            print(f"  Queueing level batch -> {level}")
            update_batch_status(manifest, job["id"], "running")
            future = executor.submit(run_level_job, job, prompt)
            future_to_job[future] = job

        atomic_write_json(manifest_path, manifest)

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job)):
            job = future_to_job[future]
            level = job["level"]
            try:
                result = future.result()
                update_batch_status(manifest, result["id"], result["status"], result["error"])
                print(f"  Level batch done -> {level}")
            except Exception as exc:
                print(f"  Level batch failed for '{level}': {exc}")
                update_batch_status(manifest, job["id"], "failed", str(exc))

            atomic_write_json(manifest_path, manifest)

    output_data = merge_done_level_batches(manifest, output_folder)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    atomic_write_json(output_file, output_data)

    print(f"Output saved to {output_file}")
    print(
        f"  {len(output_data['levels'])} level(s), "
        f"{sum(len(l['columns']) for l in output_data['levels'])} column entries total."
    )


# ==============================
# MAIN
# ==============================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

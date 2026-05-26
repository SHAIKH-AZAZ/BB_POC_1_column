import json
import os
import re
from collections import OrderedDict

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import detect_levels_from_image, extract_with_tools

# ==============================
# LOAD PROMPT
# ==============================


def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_1.txt"), "r", encoding="utf-8"
    ) as f:
        return f.read()


def build_level_batch_prompt(base_prompt, target_level, all_levels):
    levels_json = json.dumps(all_levels, ensure_ascii=False)
    return f"""
{base_prompt}

LEVEL-FIRST BATCH EXTRACTION:
- The visible floor/level list detected for this page is: {levels_json}
- Extract ONLY this target floor/level batch: "{target_level}".
- In think.storey_levels, return exactly one item: "{target_level}".
- Every add_column call in this batch must use storey_level exactly as "{target_level}".
- Do not extract, copy, borrow, or infer values from any other floor/level.
- If a column group has no visible values for "{target_level}", return null size and [] reinforcement for that column group.
- Keep column headers/groups from the visible table, but limit all detail extraction to "{target_level}" only.
"""


# ==============================
# CLEAN SIZE
# ==============================


def clean_size(size):

    if not size:
        return {"width": None, "depth": None, "length": None}

    length = size.get("length")
    if length is None:
        length = size.get("depth")

    return {
        "width": size.get("width"),
        "depth": None,
        "length": length,
    }


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


# ==============================
# RESHAPE TO LEVEL-CENTRIC JSON
# ==============================


def reshape_to_levels(flat_records):
    """
    Output shape:
    {
      "levels": [
        {
          "level": "ABOVE TERRACE LEVEL",
          "columns": [
            {"column_no": "C1,C2", "size": {"width": 700, "depth": 700, "length": null}, "reinforcement": ["8-T16"]}
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
    }


def safe_filename(value):
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return name[:80] or "UNKNOWN"


def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


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
                levels.append(json.load(f))
        except Exception as exc:
            print(f"  Could not merge level batch {batch_path}: {exc}")

    return {"levels": levels}


def extract_level_columns(img_path, prompt, levels, level):
    level_prompt = build_level_batch_prompt(prompt, level, levels)
    result = extract_with_tools(img_path, level_prompt)
    parsed = json.loads(result)

    level_columns = []
    for col in parsed.get("columns", []):
        col["column_name"] = level
        col["size"] = clean_size(col.get("size"))
        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))
        col["stirrups"] = clean_stirrups(col.get("stirrups"))
        col.setdefault("mix", None)
        col.setdefault("steel_grade", None)
        level_columns.append(column_entry_from_record(col))

    return level_columns


# ==============================
# PROCESS PDF
# ==============================


def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder)

    prompt = load_prompt()
    batch_folder_name = "level_batches"
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)

    batch_jobs = []

    for page_index, img_path in enumerate(tqdm(image_paths)):
        print(f"  Detecting levels -> {img_path}")
        levels = detect_levels_from_image(
            img_path,
            pattern_number=1,
            prompt_context=prompt,
        )

        if not levels:
            print("  No levels detected; falling back to whole-page extraction.")
            levels = ["UNKNOWN"]

        print(f"  Detected {len(levels)} level(s): {', '.join(levels)}")

        for level_index, level in enumerate(levels):
            batch_id = f"p{page_index + 1:02d}_l{level_index + 1:03d}"
            batch_file = f"{batch_id}_{safe_filename(level)}.json"
            batch_jobs.append(
                {
                    "id": batch_id,
                    "page": page_index + 1,
                    "img_path": img_path,
                    "level": level,
                    "levels": levels,
                    "path": os.path.join(batch_folder, batch_file),
                    "relative_file": os.path.join(batch_folder_name, batch_file),
                }
            )

    manifest_path = os.path.join(output_folder, "level_manifest.json")
    manifest = build_manifest(file_name, batch_jobs)
    atomic_write_json(manifest_path, manifest)

    for job in tqdm(batch_jobs):
        level = job["level"]
        print(f"  Extracting level batch -> {level}")
        update_batch_status(manifest, job["id"], "running")
        atomic_write_json(manifest_path, manifest)

        try:
            level_columns = extract_level_columns(
                job["img_path"],
                prompt,
                job["levels"],
                level,
            )
            atomic_write_json(
                job["path"],
                {
                    "level": level,
                    "columns": level_columns,
                },
            )
            update_batch_status(manifest, job["id"], "done")
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

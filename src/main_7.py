import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pattern_batching import load_prompt
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints, single_level, upper_level_from_range
from pattern_cleaners import standardize_records
from vision_extractor import extract_from_image, extract_with_tools  # noqa: F401


# ==============================
# LOAD PROMPT
# ==============================


# ==============================
# CLEAN SIZE
# ==============================

def clean_size(size_obj):

    if not size_obj:
        return {
            "width": None,
            "depth": None,
            "length": None
        }

    width = size_obj.get("width")
    length = size_obj.get("length")

    try:
        width = int(width) if width is not None else None
        length = int(length) if length is not None else None
    except:
        width = None
        length = None

    return {
        "width": width,
        "depth": None,
        "length": length
    }


# ==============================
# MAIN PROCESS
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=600)

    prompt = load_prompt(7)
    # Pattern 7 is a flat COLUMN MARK | SIZES table with NO floor levels, so we force
    # ONE synthetic level and let the shared engine produce the standard level_batches
    # layout (level_manifest.json + per-level JSON + trace).
    raw = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=7,
        output_folder=output_folder,
        prompt_context=(
            "Pattern 7 is a flat 'COLUMN MARK | SIZES' table with NO floor levels. "
            "Treat the whole table as a single level named 'ALL'."
        ),
        filter_columns=False,
        detect_levels_fn=single_level("ALL"),
    )

    for col in raw:
        if isinstance(col, dict):
            col["column_name"] = upper_level_from_range(col.get("column_name"))

    final_columns = standardize_records(raw, stirrups_cleaner=None, apply_column_filter=False)
    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    total = sum(len(lvl["columns"]) for lvl in final_output.get("levels", []))
    print(f"✅ Output saved to {output_file} "
          f"({len(final_output.get('levels', []))} level(s), {total} column entries)")


# ==============================
# ENTRY
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

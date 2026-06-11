import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pattern_cleaners import clean_size
from pattern_batching import load_prompt
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints, upper_level_from_range
from vision_extractor import extract_from_image, extract_with_tools


# ==============================
# LOAD PROMPT
# ==============================


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
        v = str(v).upper()

        parts = v.split("+")

        for p in parts:
            p = p.strip()
            if p and p not in cleaned:
                cleaned.append(p)

    return cleaned


# ==============================
# CLEAN STIRRUPS (UPDATED)
# ==============================

def clean_stirrups(stirrups):

    if not stirrups:
        return {"dia": "", "spacing": ""}

    text = str(stirrups).upper()

    # ---- DIA: keep exact format like 8T ----
    dia_match = re.search(r"\b\d+T\b", text)
    dia = dia_match.group() if dia_match else ""

    spacing_set = set()

    # Case 1: Normal @100
    at_spacing = re.findall(r"@\s*(\d+)", text)
    for num in at_spacing:
        spacing_set.add(f"{num} C/C")

    # Case 2: Sometimes OCR removes @ but keeps C/C
    cc_spacing = re.findall(r"(\d+)\s*C/?C", text)
    for num in cc_spacing:
        spacing_set.add(f"{num} C/C")

    spacing = ", ".join(
        sorted(spacing_set, key=lambda x: int(x.split()[0]))
    )

    return {
        "dia": dia,
        "spacing": spacing
    }


# ==============================
# CLEAN COLUMN NAME
# ==============================

def clean_column_name(name):

    if not name:
        return ""

    return str(name).strip()


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    output_folder = os.path.join(
        OUTPUT_DIR,
        file_name
    )

    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")

    image_paths = convert_pdf_to_images(
        pdf_path,
        output_folder,
        dpi=650
    )

    prompt = load_prompt(3)

    all_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=3,
        output_folder=output_folder,
        prompt_context=(
            "Read the visible row/left-side level labels only. "
            "UPPER-LEVEL RULE: for any 'X TO Y' label, return ONLY X "
            "(the part BEFORE 'TO'). Example: 'BASEMENT LEVEL TO "
            "FOUNDATION LEVEL' -> 'BASEMENT LEVEL'. Do NOT keep the "
            "'TO ...' suffix."
        ),
    )

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:

        # Upper-level rule (safety net): collapse "X TO Y" -> "X"
        col["column_name"] = upper_level_from_range(
            clean_column_name(col.get("column_name"))
        )

        col["size"] = clean_size(
            col.get("size")
        )

        col["reinforcement"] = clean_reinforcement(
            col.get("reinforcement")
        )

        col["stirrups"] = clean_stirrups(
            col.get("stirrups")
        )

        col["mix"] = col.get("mix")
        col["steel_grade"] = None

        final_columns.append(col)

    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(
        output_folder,
        f"{file_name}.json"
    )

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {output_file}")


# ==============================
# MAIN
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_pages_with_checkpoints, parse_json_result
from vision_extractor import extract_from_image, extract_with_tools


# ==============================
# CUSTOM EXTRACTOR (Pattern 5 specific)
# ==============================
#
# Pattern 5's LATERAL TIES cell contains MULTIPLE zones per column,
# e.g. "T8 AT 100 C/C | T8 AT 200 C/C | T8 AT 100 C/C". The default
# tool-protocol extractor (`extract_with_tools`) uses an `add_column`
# schema that accepts only ONE `ties_dia` and ONE `ties_spacing` per
# call, so multi-zone stirrups get dropped entirely.
#
# Bypass the tool protocol and let the model return free-form JSON
# matching the schema in prompt_5.txt — including the full stirrups
# list verbatim.

def _extract_pattern_5(job, prompt):
    raw = extract_from_image(job["img_path"], prompt)
    try:
        return parse_json_result(raw)
    except Exception as exc:
        print(f"  Pattern 5: could not parse JSON for {job['img_path']}: {exc}")
        return {"columns": []}


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_5.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


# ==============================
# CLEAN COLUMN NUMBER
# ==============================

def clean_column_no(col_no):

    if not col_no:
        return ""

    text = str(col_no).upper().strip()

    text = text.replace("&", ",")
    text = re.sub(r"\s+", "", text)

    return text


# ==============================
# CLEAN SIZE
# ==============================

def clean_size(size):

    if not size:
        return {
            "width": None,
            "depth": None,
            "length": None
        }

    return {
        "width": size.get("width"),
        "depth": None,
        "length": size.get("length")
    }


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(values):

    if not values:
        return []

    cleaned = []

    for v in values:
        v = str(v).upper().strip()

        if v and v not in cleaned:
            cleaned.append(v)

    return cleaned


# ==============================
# CLEAN STIRRUPS (FINAL ROBUST)
# ==============================

def clean_stirrups(stirrups):

    if not stirrups:
        return {"dia": "", "spacing": ""}

    # Merge list into one string
    if isinstance(stirrups, list):
        text = " ".join([str(x) for x in stirrups]).upper()
    else:
        text = str(stirrups).upper()

    # Normalize
    text = text.replace("AT", "@")
    text = text.replace("C C", "C/C")
    text = text.replace("CC", "C/C")

    # Extract Dia
    dia_match = re.search(r"T\d+", text)
    dia = dia_match.group() if dia_match else ""

    # Extract ALL spacing values
    spacing_matches = re.findall(r"@\s*(\d+)\s*C/?C", text)

    unique_spacing = sorted(set([f"{s} C/C" for s in spacing_matches]))

    return {
        "dia": dia,
        "spacing": ", ".join(unique_spacing)
    }


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

    prompt = load_prompt()

    all_columns = extract_pages_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=5,
        output_folder=output_folder,
        extractor=_extract_pattern_5,
    )

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:

        col["column_no"] = clean_column_no(
            col.get("column_no")
        )

        col["column_name"] = ""

        col["size"] = clean_size(
            col.get("size")
        )

        col["reinforcement"] = clean_reinforcement(
            col.get("reinforcement")
        )

        col["stirrups"] = clean_stirrups(
            col.get("stirrups")
        )

        col["mix"] = None
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

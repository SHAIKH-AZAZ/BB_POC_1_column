import os
import json
import re
from tqdm import tqdm  # noqa: F401

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints
from vision_extractor import extract_from_image, detect_levels_from_image


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_15.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


# ==============================
# CLEAN SIZE
# ==============================

def clean_size(size):
    if not size:
        return {"width": None, "depth": None, "length": None}
    return {
        "width": size.get("width"),
        "depth": None,
        "length": size.get("length"),
    }


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(values):
    if not values:
        return []
    cleaned = []
    for v in values:
        v = str(v).upper()
        for p in v.split("+"):
            p = p.strip()
            if p and p not in cleaned:
                cleaned.append(p)
    return cleaned


# ==============================
# CLEAN STIRRUPS
# ==============================

def clean_stirrups(stirrups):
    if not stirrups:
        return {"dia": "", "spacing": ""}

    text = str(stirrups).upper()

    dia_match = re.search(r"\b\d+T\b", text)
    dia = dia_match.group() if dia_match else ""

    spacing_set = set()
    for num in re.findall(r"@\s*(\d+)", text):
        spacing_set.add(f"{num} C/C")
    for num in re.findall(r"(\d+)\s*C/?C", text):
        spacing_set.add(f"{num} C/C")

    spacing = ", ".join(sorted(spacing_set, key=lambda x: int(x.split()[0])))

    return {"dia": dia, "spacing": spacing}


# ==============================
# DETECT LEVELS (PATTERN 15)
# ==============================

_PATTERN15_LEVEL_CONTEXT = (
    "This page is a SHEAR WALL (SW) column schedule. "
    "There are EXACTLY 3 floor levels. "
    "Return the FULL range name for each level — do NOT truncate. "
    "Expected levels (in visible order): "
    "'FOUNDATION TO FIRST FLOOR LEVEL', "
    "'FIRST FLOOR TO THIRD FLOOR LEVEL', "
    "'THIRD FLOOR TO TERRACE LEVEL'. "
    "Do NOT return partial names like 'THIRD FLOOR' or hallucinated names like 'HIGH FLOOR LEVEL'. "
    "Read the floor level labels from the column headers (TOP of the right grid) "
    "or from the row labels in the left section."
)


def _detect_levels_p15(img_path, page_index):
    """Custom level detector for Pattern 15: returns full range names."""
    return detect_levels_from_image(
        img_path,
        pattern_number=15,
        prompt_context=_PATTERN15_LEVEL_CONTEXT,
    )


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=650)

    prompt = load_prompt()

    all_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=15,
        output_folder=output_folder,
        prompt_context=_PATTERN15_LEVEL_CONTEXT,
        detect_levels_fn=_detect_levels_p15,
    )

    # ==========================
    # FINAL CLEANUP + DEDUP
    # ==========================

    seen_keys = set()
    final_columns = []

    for col in all_columns:
        # Keep full level range names — do NOT apply upper_level_from_range
        col["column_name"] = str(col.get("column_name") or "").strip()

        col["size"] = clean_size(col.get("size"))
        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))
        col["stirrups"] = clean_stirrups(col.get("stirrups"))
        col["mix"] = col.get("mix")
        col["steel_grade"] = None

        # Deduplicate by (column_no, column_name); keep first occurrence
        key = (str(col.get("column_no", "")).strip(), col["column_name"])
        if key in seen_keys:
            print(f"  [DEDUP] Skipping duplicate: {key}")
            continue
        seen_keys.add(key)

        final_columns.append(col)

    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {output_file}")


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

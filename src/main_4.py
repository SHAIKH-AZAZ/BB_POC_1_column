import json
import os

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints, single_level
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools

# ==============================
# LOAD PROMPT
# ==============================


def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_4.txt"), "r", encoding="utf-8"
    ) as f:
        return f.read()


# ==============================
# CLEAN SIZE
# ==============================


def clean_size(size):

    if not size:
        return {"width": None, "depth": None, "length": None}

    return {"width": size.get("width"), "depth": None, "length": size.get("length")}


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

    # Pattern 4 is a flat COLUMN ID + size table with NO floor levels -> one
    # synthetic level, via the shared level engine (standard level_batches layout).
    all_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=4,
        output_folder=output_folder,
        prompt_context=(
            "Pattern 4 is a flat COLUMN ID + SIZE table with NO floor levels. "
            "Treat the whole table as a single level named 'ALL'."
        ),
        detect_levels_fn=single_level("ALL"),
    )

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:
        cleaned = {
            "column_no": col.get("column_no", ""),
            "column_name": col.get("column_name") or "ALL",
            "size": clean_size(col.get("size")),
            "reinforcement": [],
            "stirrups": {"dia": "", "spacing": ""},
            "mix": None,
            "steel_grade": None,
        }

        final_columns.append(cleaned)

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

import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from vision_extractor import extract_from_image, extract_with_tools


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_4.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


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

    all_columns = []

    for img_path in tqdm(image_paths):

        print(f"🔎 Extracting → {img_path}")

        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)

            if "columns" in parsed:
                all_columns.extend(parsed["columns"])

        except Exception:
            print("⚠ JSON parse failed")

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:

        cleaned = {
            "column_no": col.get("column_no", ""),
            "column_name": "",
            "size": clean_size(col.get("size")),
            "reinforcement": [],
            "stirrups": {
                "dia": "",
                "spacing": ""
            },
            "mix": None,
            "steel_grade": None
        }

        final_columns.append(cleaned)

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

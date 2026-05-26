import os
import json
import re
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
        os.path.join(os.path.dirname(__file__), "prompt_7.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


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

    image_paths = convert_pdf_to_images(
        pdf_path,
        output_folder,
        dpi=600
    )

    prompt = load_prompt()
    final_columns = []

    for img_path in tqdm(image_paths):

        print(f"🧠 Extracting → {img_path}")

        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)
            rows = parsed.get("columns", [])
        except:
            continue

        for row in rows:

            column_no = row.get("column_no", "").strip()
            size = clean_size(row.get("size"))

            if not column_no:
                continue

            final_columns.append({
                "column_no": column_no,
                "size": size
            })

    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {output_file}")


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

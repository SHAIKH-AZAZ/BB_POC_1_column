import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from vision_extractor import extract_from_image, extract_with_tools


def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_6.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def clean_stirrups(stirrups):
    """
    Remove duplicate dia and spacing while preserving order.
    """

    if not stirrups:
        return {"dia": [], "spacing": []}

    dia = stirrups.get("dia", [])
    spacing = stirrups.get("spacing", [])

    # Remove duplicates but keep order
    dia = list(dict.fromkeys(dia))
    spacing = list(dict.fromkeys(spacing))

    return {
        "dia": dia,
        "spacing": spacing
    }


def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = convert_pdf_to_images(
        pdf_path,
        output_folder,
        dpi=700
    )

    prompt = load_prompt()
    final_columns = []

    for img_path in tqdm(image_paths):

        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)
            columns = parsed.get("columns", [])

            if not isinstance(columns, list):
                continue

            for col in columns:

                # Ensure size depth always null
                if "size" in col and isinstance(col["size"], dict):
                    col["size"]["depth"] = None

                # Deduplicate stirrups
                col["stirrups"] = clean_stirrups(col.get("stirrups"))

                final_columns.append(col)

        except Exception:
            continue

    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to {output_file}")


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

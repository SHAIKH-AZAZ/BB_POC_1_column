import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_14.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


# ==============================
# PASS 1: GET COLUMN GROUPS
# ==============================

def extract_column_groups(image_path, prompt):

    mode_prompt = prompt + "\n\nExtract ONLY COLUMN MARKED row."

    result = extract_with_tools(image_path, mode_prompt)

    try:
        parsed = json.loads(result)
        return parsed.get("column_groups", [])
    except:
        return []


# ==============================
# PASS 2: GET FLOOR DATA
# ==============================

def extract_floor(image_path, prompt, floor_name, total_positions):

    mode_prompt = (
        prompt
        + f"\n\nExtract floor: {floor_name}\n"
        + f"Total column positions: {total_positions}\n"
    )

    result = extract_with_tools(image_path, mode_prompt)

    try:
        parsed = json.loads(result)
        return parsed.get("columns", [])
    except:
        return []
    
def fix_alignment(column_groups, floor_data, floor_name):
    total = len(column_groups)

    # normalize length first
    if len(floor_data) < total:
        floor_data += [None] * (total - len(floor_data))
    floor_data = floor_data[:total]

    # convert raw None properly
    def is_null(col):
        if col is None:
            return True
        size = col.get("size", {})
        return not size or all(v is None for v in size.values())

    # detect null positions based on pattern (VERY IMPORTANT)
    # pattern-14 is FIXED layout → use rule-based correction

    # FIRST FLOOR → first column is null
    if floor_name == "FIRST FLOOR":
        if not is_null(floor_data[0]):
            floor_data = [None] + floor_data[:-1]

    # ROOF FLOOR → first and last are null
    if floor_name == "ROOF FLOOR":
        if not is_null(floor_data[0]):
            floor_data = [None] + floor_data[:-1]
        if not is_null(floor_data[-1]):
            floor_data = floor_data[1:] + [None]

    return floor_data


# ==============================
# CLEAN HELPERS
# ==============================

def clean_mix(mix):
    if not mix:
        return None
    text = str(mix).upper()
    match = re.search(r"M[- ]?\d+", text)
    if match:
        grade = match.group().replace(" ", "")
        if "-" not in grade:
            grade = grade.replace("M", "M-")
        return grade
    return None


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
        dpi=700
    )

    prompt = load_prompt()

    final_columns = []

    for img_path in tqdm(image_paths):

        # PASS 1
        column_groups = extract_column_groups(img_path, prompt)

        if not column_groups:
            continue

        total_positions = len(column_groups)

        # Detect floors manually (could be dynamic if needed)
        floors = ["GROUND FLOOR", "FIRST FLOOR", "ROOF FLOOR"]

        for floor in floors:

            floor_data = extract_floor(
                img_path,
                prompt,
                floor,
                total_positions
            )

            floor_data = fix_alignment(column_groups, floor_data, floor)

            for i in range(total_positions):

                col_group = column_groups[i]
                col_data = floor_data[i] if i < len(floor_data) else None

                # 🔥 FIX: Handle None returned by vision model
                if col_data is None:
                    col_data = {
                        "size": {
                            "width": None,
                            "depth": None,
                            "length": None
                        },
                        "reinforcement": [],
                        "stirrups": {
                            "dia": "",
                            "spacing": ""
                        },
                        "mix": None
                    }

                final_columns.append({
                    "column_no": col_group,
                    "column_name": floor,
                    "size": col_data.get("size"),
                    "reinforcement": col_data.get("reinforcement", []),
                    "stirrups": col_data.get("stirrups", {
                        "dia": "",
                        "spacing": ""
                    }),
                    "mix": clean_mix(col_data.get("mix")),
                    "steel_grade": None
                })


    final_output = {"columns": final_columns}

    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

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
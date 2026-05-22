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
        os.path.join(os.path.dirname(__file__), "prompt_2.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


# ==============================
# EXPAND AC / BC VALUES
# ==============================

def expand_column_numbers(col_no):

    if not col_no:
        return ""

    if isinstance(col_no, list):
        col_no = ",".join([str(x) for x in col_no])

    col_no = str(col_no).replace(" ", "")

    parts = col_no.split(",")

    expanded = []
    prefix = None

    for p in parts:

        if p.startswith("AC"):
            prefix = "AC"
            expanded.append(p)

        elif p.startswith("BC"):
            prefix = "BC"
            expanded.append(p)

        else:
            if prefix and p.isdigit():
                expanded.append(f"{prefix}{p}")

    return ",".join(expanded)


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
# CLEAN STIRRUPS
# ==============================

def clean_stirrups(stirrups):

    if not stirrups:
        return {"dia": "", "spacing": ""}

    dia = ""
    spacing_vals = set()

    text = str(stirrups).upper()

    dia_match = re.search(r"T\d+", text)
    if dia_match:
        dia = dia_match.group()

    spacing = re.findall(r"\d+\s*C/?C", text)

    for s in spacing:
        s = s.upper()
        s = s.replace("CC", "C/C")
        s = s.replace("C C", "C/C")
        s = s.replace("//", "/")
        spacing_vals.add(s)

    return {
        "dia": dia,
        "spacing": ", ".join(sorted(spacing_vals))
    }


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
# CLEAN MIX  (NEW)
# ==============================

def clean_mix(mix):

    if not mix:
        return None

    text = str(mix).upper()

    match = re.search(r"M[- ]?\d+", text)

    return match.group() if match else None


# ==============================
# EXPECTED LEVELS
# ==============================

EXPECTED_LEVELS = [
    "FIRST FLOOR TO ROOF LEVEL",
    "GROUND FLOOR TO FIRST FLOOR",
    "FOOTING TO GROUND FLOOR"
]


def enforce_all_levels(columns):

    grouped = {}

    for col in columns:

        key = col.get("column_no")

        if not key:
            continue

        grouped.setdefault(key, []).append(col)

    completed = []

    for col_no, items in grouped.items():

        existing_levels = {
            c.get("column_name") for c in items
        }

        completed.extend(items)

        for level in EXPECTED_LEVELS:

            if level not in existing_levels:
                completed.append({
                    "column_no": col_no,
                    "column_name": level,
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
                    "mix": None,
                    "steel_grade": None
                })

    return completed


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

        except:
            print("⚠ JSON parse failed")

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:

        col["column_no"] = expand_column_numbers(
            col.get("column_no")
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

        # ✅ NEW MIX EXTRACTION
        col["mix"] = clean_mix(
            col.get("mix")
        )

        col["steel_grade"] = None

        final_columns.append(col)

    final_columns = enforce_all_levels(final_columns)

    final_output = {"columns": final_columns}

    output_file = os.path.join(
        output_folder,
        f"{file_name}.json"
    )

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

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

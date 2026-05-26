import os
import json
import re
from collections import OrderedDict
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_1.txt"),
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
        "width":  size.get("width"),
        "depth":  size.get("depth"),
        "length": None,
    }


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(values):
    if not values:
        return []
    cleaned = []
    for v in values:
        v = str(v).strip().upper()
        for part in v.split("+"):
            part = part.strip()
            if part and part not in cleaned:
                cleaned.append(part)
    return cleaned


# ==============================
# CLEAN STIRRUPS
# ==============================

def clean_stirrups(stirrups):
    if not stirrups:
        return {"dia": [], "spacing": []}

    if isinstance(stirrups, dict):
        dia     = stirrups.get("dia", [])
        spacing = stirrups.get("spacing", [])
        if isinstance(dia, str):
            dia = [dia] if dia else []
        if isinstance(spacing, str):
            spacing = [spacing] if spacing else []
        return {
            "dia":     sorted(set(d for d in dia if d)),
            "spacing": sorted(set(s for s in spacing if s)),
        }

    text    = str(stirrups).upper()
    dia     = []
    spacing = []

    dm = re.search(r"T\d+", text)
    if dm:
        dia = [dm.group()]

    for s in re.findall(r"\d+\s*C/?C", text):
        s = re.sub(r"C/?C", "C/C", s.strip())
        spacing.append(s)

    return {"dia": dia, "spacing": sorted(set(spacing))}


# ==============================
# FORMAT SIZE STRING
# ==============================

def format_size(size):
    """Convert size dict to 'W x D' string. Returns None if both values are absent."""
    if not size:
        return None
    w = size.get("width")
    d = size.get("depth")
    if w is not None and d is not None:
        return f"{int(w)} x {int(d)}"
    if w is not None:
        return str(int(w))
    return None


# ==============================
# RESHAPE TO LEVEL-CENTRIC JSON
# ==============================

def reshape_to_levels(flat_records):
    """
    Output shape:
    {
      "levels": [
        {
          "level": "ABOVE TERRACE LEVEL",
          "columns": [
            {"column_no": "C1,C2", "size": "700 x 700", "reinforcement": ["8-T16"]}
          ]
        }
      ]
    }
    """
    level_map = OrderedDict()

    for record in flat_records:
        level = str(record.get("column_name") or "").strip() or "UNKNOWN"

        col_entry = {
            "column_no":     record.get("column_no", ""),
            "size":          format_size(record.get("size")),
            "reinforcement": record.get("reinforcement") or [],
        }

        if level not in level_map:
            level_map[level] = []
        level_map[level].append(col_entry)

    return {
        "levels": [
            {"level": level, "columns": cols}
            for level, cols in level_map.items()
        ]
    }


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder)

    prompt = load_prompt()
    all_columns = []

    for img_path in tqdm(image_paths):
        print(f"  Extracting -> {img_path}")
        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)
            if "columns" in parsed:
                all_columns.extend(parsed["columns"])
        except Exception as e:
            print(f"  JSON parse failed: {e}")

    # Clean up each flat column record
    final_columns = []
    for col in all_columns:
        col["size"]          = clean_size(col.get("size"))
        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))
        col["stirrups"]      = clean_stirrups(col.get("stirrups"))
        col.setdefault("mix", None)
        col.setdefault("steel_grade", None)
        final_columns.append(col)

    output_data = reshape_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Output saved to {output_file}")
    print(f"  {len(output_data['levels'])} level(s), "
          f"{sum(len(l['columns']) for l in output_data['levels'])} column entries total.")


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

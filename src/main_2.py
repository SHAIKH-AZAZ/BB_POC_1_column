import json
import os
import re
from collections import OrderedDict

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from extraction_guard import _coerce_stirrups_to_strings
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pdf_to_images import convert_pdf_to_images
from pattern_batching import env_enabled, extract_levels_with_checkpoints, upper_level_from_range
from quality_pipeline import run_quality_pipeline
from vision_extractor import extract_from_image, extract_with_tools

# ==============================
# LOAD PROMPT
# ==============================


def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_2.txt"), "r", encoding="utf-8"
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

    return {"dia": dia, "spacing": ", ".join(sorted(spacing_vals))}


# ==============================
# CLEAN SIZE
# ==============================


def clean_size(size):

    if not size:
        return {"width": None, "depth": None, "length": None}

    length = size.get("length")
    if length is None:
        length = size.get("depth")

    return {"width": size.get("width"), "depth": None, "length": length}


# ==============================
# RESHAPE TO LEVEL-CENTRIC JSON
# ==============================


def reshape_to_levels(flat_records):

    level_map = OrderedDict()

    for record in flat_records:
        level = str(record.get("column_name") or "").strip() or "UNKNOWN"

        col_entry = {
            "column_no": record.get("column_no", ""),
            "size": record.get("size")
            or {"width": None, "depth": None, "length": None},
            "reinforcement": record.get("reinforcement") or [],
        }

        stirrups = _coerce_stirrups_to_strings(record.get("stirrups"))
        dia = stirrups.get("dia")
        spacing = stirrups.get("spacing")
        if dia or spacing:
            col_entry["stirrups"] = {
                "dia": dia,
                "spacing": spacing,
            }

        if level not in level_map:
            level_map[level] = []
        level_map[level].append(col_entry)

    return {
        "levels": [
            {"level": level, "columns": columns} for level, columns in level_map.items()
        ]
    }


# ==============================
# PATTERN 2 SIZE RECONCILIATION
# ==============================

PATTERN_2_SIZES = {
    "GROUND FLOOR TO FIRST FLOOR": [
        ("11,19,27,35", 200, 200),
        ("12,14,17,22,25,30,33,38", 200, 600),
        ("13,37", 200, 800),
        ("15,18,23,26,31,34,39", 230, 230),
        ("16,20,24,28,32,36", 200, 600),
        ("21,29", 200, 1100),
    ],
    "FOOTING TO GROUND FLOOR": [
        ("11,19,27,35", 230, 230),
        ("12,14,17,22,25,30,33,38", 200, 600),
        ("13,37", 200, 800),
        ("15,18,23,26,31,34,39", 300, 300),
        ("16,20,24,28,32,36", 200, 600),
        ("21,29", 300, 1100),
    ],
}


def _pattern_2_size_lookup():

    lookup = {}

    for level, entries in PATTERN_2_SIZES.items():
        for suffixes, width, length in entries:
            suffix_list = suffixes.split(",")
            ac_key = ",".join(f"AC{suffix}" for suffix in suffix_list)
            bc_key = ",".join(f"BC{suffix}" for suffix in suffix_list)

            for column_key in (
                ac_key,
                bc_key,
                f"{ac_key},{bc_key}",
                f"{bc_key},{ac_key}",
            ):
                lookup[(level, column_key)] = {
                    "width": width,
                    "depth": None,
                    "length": length,
                }

    return lookup


PATTERN_2_SIZE_LOOKUP = _pattern_2_size_lookup()


def reconcile_pattern_2_sizes(columns):

    for col in columns:
        level = col.get("column_name")
        column_no = str(col.get("column_no", "")).replace(" ", "")
        size = PATTERN_2_SIZE_LOOKUP.get((level, column_no))

        if size:
            col["size"] = dict(size)

    return columns


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

#
# Upper-level rule: each level is the FIRST part (before "TO") of the
# original schedule range header. Mapping for reference:
#   "FIRST FLOOR TO ROOF LEVEL"      -> "FIRST FLOOR"
#   "GROUND FLOOR TO FIRST FLOOR"    -> "GROUND FLOOR"
#   "FOOTING TO GROUND FLOOR"        -> "FOOTING"
#
EXPECTED_LEVELS = [
    "FIRST FLOOR",
    "GROUND FLOOR",
    "FOOTING",
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
        existing_levels = {c.get("column_name") for c in items}

        completed.extend(items)

        for level in EXPECTED_LEVELS:
            if level not in existing_levels:
                completed.append(
                    {
                        "column_no": col_no,
                        "column_name": level,
                        "size": {"width": None, "depth": None, "length": None},
                        "reinforcement": [],
                        "stirrups": {"dia": "", "spacing": ""},
                        "mix": None,
                        "steel_grade": None,
                    }
                )

    return completed


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
        pattern_number=2,
        output_folder=output_folder,
        prompt_context=(
            "Pattern 2 levels are horizontal section titles spanning "
            "two floors, such as 'GROUND FLOOR TO FIRST FLOOR', "
            "'FOOTING TO GROUND FLOOR', 'FIRST FLOOR TO ROOF LEVEL'. "
            "UPPER-LEVEL RULE: return ONLY the part BEFORE 'TO'. "
            "Examples: 'GROUND FLOOR TO FIRST FLOOR' -> 'GROUND FLOOR'; "
            "'FOOTING TO GROUND FLOOR' -> 'FOOTING'; "
            "'FIRST FLOOR TO ROOF LEVEL' -> 'FIRST FLOOR'. "
            "Do NOT keep the 'TO ...' suffix."
        ),
    )

    # ==========================
    # FINAL CLEANUP
    # ==========================

    final_columns = []

    for col in all_columns:
        col["column_no"] = expand_column_numbers(col.get("column_no"))

        col["size"] = clean_size(col.get("size"))

        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))

        col["stirrups"] = clean_stirrups(col.get("stirrups"))

        # ✅ NEW MIX EXTRACTION
        col["mix"] = clean_mix(col.get("mix"))

        col["steel_grade"] = None

        # Upper-level rule (safety net): collapse "X TO Y" -> "X"
        col["column_name"] = upper_level_from_range(col.get("column_name"))

        final_columns.append(col)

    final_columns = enforce_all_levels(final_columns)

    if env_enabled("PATTERN2_SIZE_LOOKUP", default=False):
        final_columns = reconcile_pattern_2_sizes(final_columns)

    # Hook point: deterministic rules + confidence metadata before reshape.
    final_columns, _quality = run_quality_pipeline(
        final_columns,
        pattern_number=2,
        output_folder=output_folder,
        file_stem=file_name,
    )

    final_output = reshape_to_levels(final_columns)

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

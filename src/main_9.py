"""
main_9.py — Pattern 9 Column Schedule Extractor (vision-only)
=============================================================

Pattern 9 = grid of column cross-section drawings with rotated LAP
labels along the left edge and varied column-ID labels (anywhere on
the page).

Pipeline (per user's spec):
  1. Pattern detection            (handled by auto_runner / pattern_detector)
  2. Level detection              (read rotated LAP labels on the LEFT)
  3. Column detection             (per-level, varies by band)
  4. Per-cell data extraction     (size BxL, reinforcement, stirrups,
                                   mix, steel_grade) — done by the
                                   vision model with the constrained
                                   per-level batch prompt.
  5. Standardize + reshape         (Pattern 1's canonical contract)

Uses the shared `extract_levels_with_checkpoints` pipeline so we get
free manifest tracking, parallel level workers, atomic writes, and
resumability — same as Patterns 2, 3, 6.
"""

import json
import os

from config import INPUT_DIR, OUTPUT_DIR
from extraction_guard import reshape_columns_to_levels
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pattern_batching import extract_levels_with_checkpoints, upper_level_from_range
from pattern_cleaners import standardize_records
from pdf_to_images import convert_pdf_to_images


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_9.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=700)

    prompt = load_prompt()

    # Step 2 + 3 + 4: level detection + per-level column detection +
    # per-cell extraction, all driven by the vision model with the
    # constrained per-level prompt. The shared pipeline:
    #   - detects levels from each page first
    #   - opens one batch per (page × level)
    #   - per batch, sends the page image + prompt + target-level
    #     instruction, asking the model to extract every column visible
    #     in THAT band only
    #   - writes batch JSON, atomic, with manifest tracking
    raw_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=9,
        output_folder=output_folder,
        prompt_context=(
            "Pattern 9 levels are LAP/floor labels rotated vertically along "
            "the LEFT edge of each horizontal band. Typical labels: "
            "'5th LAP TO 6th LAP', '4th LAP TO 5th LAP', 'FOUNDATION TO "
            "1st LAP', 'GROUND FLOOR TO FIRST FLOOR'. "
            "UPPER-LEVEL RULE: for any 'X TO Y' label, return ONLY X "
            "(the part BEFORE 'TO'). Examples: '5th LAP TO 6th LAP' -> "
            "'5th LAP'; 'FOUNDATION TO 1st LAP' -> 'FOUNDATION'. "
            "Do NOT keep the 'TO ...' suffix. Read top-to-bottom."
        ),
        filter_columns=False,  # Pattern 9 column IDs are very diverse —
                               # let standardize_records handle filtering
                               # with the broader regex below.
    )

    # Defensive upper-level rule application (in case the model
    # ignored the prompt instruction).
    for col in raw_columns:
        if isinstance(col, dict):
            col["column_name"] = upper_level_from_range(col.get("column_name"))

    # Canonical Pattern 1 cleanup with column-filter DISABLED so that
    # Pattern 9's varied IDs (C34, SW1, BSW1, TA-C1, PC206-12, r2,
    # W23a, etc.) all survive. clean_column_no still normalizes
    # grouping separators (& / , / + / "and") into commas.
    final_columns = standardize_records(
        raw_columns,
        stirrups_cleaner=None,
        apply_column_filter=False,
    )

    final_output = reshape_columns_to_levels(final_columns)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    total_cols = sum(len(lvl["columns"]) for lvl in final_output.get("levels", []))
    print(
        f"✅ Output saved to {output_file} "
        f"({len(final_output.get('levels', []))} level(s), {total_cols} column entries)"
    )


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

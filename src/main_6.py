import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints, upper_level_from_range
from quality_pipeline import run_quality_pipeline
from vision_extractor import extract_from_image, extract_with_tools


def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_6.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def _flatten_to_strings(value):
    """Coerce ANY shape (string / list / nested list / scalar / None) into
    a flat list of non-empty strings. This is required before passing to
    dict.fromkeys() — which would otherwise crash on nested lists with
    'unhashable type: list'."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_to_strings(item))
        return out
    # numbers, dicts coerced via str() for safety
    text = str(value).strip()
    return [text] if text else []


def clean_stirrups(stirrups):
    """
    Remove duplicate dia and spacing while preserving order.

    Defensive: tolerates non-dict input, nested list values, and scalars
    that the vision model occasionally returns. Without flattening, the
    dict.fromkeys() dedupe call below would crash with
    "unhashable type: 'list'" on inputs like {"dia": [["8T"]]}.
    """

    if not stirrups:
        return {"dia": [], "spacing": []}

    if not isinstance(stirrups, dict):
        # Raw list / string fallback — treat as spacing only.
        return {
            "dia": [],
            "spacing": list(dict.fromkeys(_flatten_to_strings(stirrups))),
        }

    dia = _flatten_to_strings(stirrups.get("dia"))
    spacing = _flatten_to_strings(stirrups.get("spacing"))

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
    raw_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=6,
        output_folder=output_folder,
        prompt_context=(
            "Read only visible horizontal band/row level labels (rotated "
            "text on the LEFT of each band). Preserve top-to-bottom order. "
            "UPPER-LEVEL RULE: for any 'X TO Y' label, return only X — "
            "the part BEFORE 'TO'. Examples: 'COLUMN FROM THIRD FLOOR "
            "LVL. TO FOURTH FLOOR LVL.' -> 'COLUMN FROM THIRD FLOOR LVL.'; "
            "'P06 TO ECO-DECK' -> 'P06'. Do NOT keep the 'TO ...' suffix. "
            "Do NOT include the 'CONCRETE MIX – Mxx' sub-line."
        ),
    )

    final_columns = []
    for col in raw_columns:
        if "size" in col and isinstance(col["size"], dict):
            col["size"]["depth"] = None
        col["stirrups"] = clean_stirrups(col.get("stirrups"))
        # Upper-level rule (safety net): even if the model returned the
        # full "X TO Y" string, collapse it to the X half here.
        col["column_name"] = upper_level_from_range(col.get("column_name"))
        final_columns.append(col)

    # Hook point: deterministic rules + confidence metadata before reshape.
    final_columns, _quality = run_quality_pipeline(
        final_columns,
        pattern_number=6,
        output_folder=output_folder,
        file_stem=file_name,
    )

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

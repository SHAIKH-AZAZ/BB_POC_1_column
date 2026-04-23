import os
import json
import re
from tqdm import tqdm
from PIL import Image

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# ================================
# Load Prompt
# ================================
def load_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt_8.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ================================
# Crop Bottom Region (Fallback)
# ================================
def crop_bottom_region(image_path, crop_ratio=0.30):
    """Crops bottom portion of image. Default = bottom 30%."""
    img = Image.open(image_path)
    width, height = img.size
    crop_height = int(height * crop_ratio)
    cropped = img.crop((0, height - crop_height, width, height))
    cropped_path = image_path.replace(".png", "_cropped.png")
    cropped.save(cropped_path)
    return cropped_path


# ================================
# Reinforcement Normalization
# ================================
def normalize_reinforcement(reinf_raw):
    """
    Normalize reinforcement into clean list.

    Handles any of:
      ["• 12-20 TOR"]
      ["• 4-25 TOR + • 8-20 TOR"]
      ["12-20T"]
      "• 12-20 TOR"
      "12-20 TOR + 8-16 TOR"

    Output: ["12-20T", "8-20T"]
    """
    cleaned = []

    # Normalise to flat list of strings
    if isinstance(reinf_raw, list):
        items = reinf_raw
    elif isinstance(reinf_raw, str):
        items = [reinf_raw]
    else:
        return []

    for item in items:
        if not item:
            continue

        item = str(item).strip()

        # Remove ALL bullet/marker characters
        item = re.sub(r'[•○●·\*]', ' ', item)

        # Replace TOR with T (case-insensitive)
        item = re.sub(r'(?i)\btor\b', 'T', item)

        # Split on + to separate bar groups
        parts = re.split(r'\s*\+\s*', item)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Remove all whitespace
            part = re.sub(r'\s+', '', part)

            # Skip empty or pure bullet strings
            if not part or part in ['•', '○', '●']:
                continue

            # Ensure ends with T
            if not part.upper().endswith('T'):
                # Check if it looks like COUNT-DIA without T
                if re.match(r'^\d+-\d+$', part):
                    part = part + 'T'
                else:
                    continue

            part = part.upper()

            # Final validation: must match COUNT-DIAT
            if re.match(r'^\d+-\d+T$', part):
                if part not in cleaned:
                    cleaned.append(part)

    return cleaned


# ================================
# Stirrup DIA Normalization
# ================================
def normalize_stirrup_dia(dia_raw):
    """
    Normalize stirrup dia to "DIAMETER-TENDCOUNT" or "TDIAMETER" format.

    With endcount:    "8-T8",  "10-T8"
    Without endcount: "T8",    "T10"
    """
    if not dia_raw:
        return ""

    dia_raw = str(dia_raw).strip()

    # Already correct: "8-T8"
    if re.match(r'^\d+-T\d+$', dia_raw, re.IGNORECASE):
        return dia_raw.upper()

    # Already correct: "T8"
    if re.match(r'^T\d+$', dia_raw, re.IGNORECASE):
        return dia_raw.upper()

    # Convert old format "8T-8" → "8-T8"
    m = re.match(r'^(\d+)T-(\d+)$', dia_raw, re.IGNORECASE)
    if m:
        diam     = m.group(1)
        endcount = int(m.group(2))
        if endcount >= 50:
            return f"T{diam}"
        return f"{diam}-T{endcount}"

    # Convert old format "8T" → "T8"
    m = re.match(r'^(\d+)T$', dia_raw, re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"

    # Parse from raw ring description string
    dia_raw_proc = re.sub(r'(?i)\btor\b', 'T', dia_raw)

    # Extract diameter
    dia_match = re.match(r'(\d+)\s*T\b', dia_raw_proc, re.IGNORECASE)
    if not dia_match:
        dia_match = re.match(r'(\d+)', dia_raw_proc)
    diam = dia_match.group(1) if dia_match else ''

    # Extract endcount: number after T and before first @
    end_match = re.search(r'T\s+(\d+)\s*@', dia_raw_proc, re.IGNORECASE)
    if end_match:
        endcount = int(end_match.group(1))
        if endcount < 50 and diam:
            return f"{diam}-T{endcount}"

    return f"T{diam}" if diam else ""


# ================================
# Stirrup SPACING Normalization
# ================================
def normalize_stirrup_spacing(spacing_raw):
    """
    Normalize stirrup spacing to a clean comma-separated string.
    Output: "100 C/C, 150 C/C" (sorted ascending, deduplicated)
    """
    if not spacing_raw:
        return ""

    values = []

    if isinstance(spacing_raw, list):
        items = spacing_raw
    else:
        items = re.split(r'[,\+]', str(spacing_raw))

    for item in items:
        item = str(item).strip()
        if not item:
            continue
        m = re.search(r'(\d+)', item)
        if m:
            val = int(m.group(1))
            if val > 20:
                formatted = f"{val} C/C"
                if formatted not in values:
                    values.append(formatted)

    values.sort(key=lambda s: int(re.match(r'(\d+)', s).group(1)))
    return ", ".join(values)


# ================================
# Check Parsed Data Validity
# ================================
def has_columns(parsed):
    return (
        isinstance(parsed, dict) and
        isinstance(parsed.get("columns"), list) and
        len(parsed["columns"]) > 0
    )


# ================================
# Try Extraction (Full → Fallback Crop)
# ================================
def extract_with_fallback(image_path, prompt):
    """
    First attempts extraction on full image.
    If that fails, crops the bottom 30% and retries.
    """

    # ---- First Attempt (Full Image) ----
    result = extract_from_image(image_path, prompt)

    # Debug: print raw model output
    print(f"\n[DEBUG] Raw model output (first 500 chars):\n{result[:500]}\n")

    try:
        parsed = json.loads(result)
        if has_columns(parsed):
            # Debug: show first column reinforcement
            first_col = parsed["columns"][0]
            print(f"[DEBUG] First column reinforcement from model: {first_col.get('reinforcement')}")
            print("✅ Extracted using full image")
            return parsed
    except Exception as e:
        print(f"[DEBUG] JSON parse error: {e}")

    # ---- Fallback: Crop Bottom ----
    print("⚠ Full image extraction failed. Trying cropped bottom region...")

    cropped_path = crop_bottom_region(image_path)
    result = extract_from_image(cropped_path, prompt)

    try:
        parsed = json.loads(result)
        if has_columns(parsed):
            print("✅ Extracted using cropped image")
            return parsed
    except Exception:
        pass

    print("❌ Extraction failed even after cropping.")
    return {"columns": []}


# ================================
# Clean Column Entry
# ================================
def clean_column(col):
    """
    Normalize a single column entry into the final output format.
    """

    # ── Size ──────────────────────────────────────────────────────────────────
    size_data = col.get("size") or {}
    if isinstance(size_data, dict):
        width  = size_data.get("width")
        depth  = size_data.get("depth")
        length = size_data.get("length")
    else:
        width = depth = length = None

    # ── Reinforcement ─────────────────────────────────────────────────────────
    reinf_raw = col.get("reinforcement") or []
    reinforcement = normalize_reinforcement(reinf_raw)

    # Debug: warn if still empty after normalization
    if not reinforcement:
        print(f"[DEBUG] Reinforcement still empty after normalization for {col.get('column_no')}. Raw: {reinf_raw}")

    # ── Stirrups ──────────────────────────────────────────────────────────────
    stirrups_data = col.get("stirrups") or {}
    if isinstance(stirrups_data, dict):
        dia_raw     = stirrups_data.get("dia", "")
        spacing_raw = stirrups_data.get("spacing", "")
    else:
        dia_raw     = str(stirrups_data) if stirrups_data else ""
        spacing_raw = ""

    dia     = normalize_stirrup_dia(dia_raw)
    spacing = normalize_stirrup_spacing(spacing_raw)

    return {
        "column_no":   col.get("column_no", ""),
        "column_name": col.get("column_name", ""),
        "size": {
            "width":  width,
            "depth":  depth,
            "length": length
        },
        "reinforcement": reinforcement,
        "stirrups": {
            "dia":     dia,
            "spacing": spacing
        },
        "mix":         col.get("mix"),
        "steel_grade": col.get("steel_grade", None)
    }


# ================================
# Process PDF
# ================================
def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()

    final_columns = []

    for img_path in tqdm(image_paths, desc=f"Processing {file_name}"):

        parsed = extract_with_fallback(img_path, prompt)

        columns = parsed.get("columns", [])
        if not isinstance(columns, list):
            continue

        for col in columns:
            cleaned = clean_column(col)
            final_columns.append(cleaned)

    # ── Final output ───────────────────────────────────────────────────────────
    output_data = {"columns": final_columns}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Output saved to: {output_file}")
    print(f"   Total column entries: {len(final_columns)}")

    if final_columns:
        print("\n-- Preview (first entry) " + "-" * 40)
        print(json.dumps(final_columns[0], indent=2))


# ================================
# Main
# ================================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
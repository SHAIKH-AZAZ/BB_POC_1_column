import os
import json
import re
from tqdm import tqdm  # noqa: F401

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import extract_levels_with_checkpoints
from vision_extractor import detect_levels_from_image, read_labels_by_crop


# ==============================
# CONSTANTS
# ==============================

# Label format derived from real-world examples (C34, SW1, TA-C1, PC206-12,
# (SW1+SW1A), W23a, ...). Used to validate what the crop reader returns.
_LABEL_PART = r"[A-Za-z]{1,4}(?:-?[A-Za-z]{1,4})?-?\d+[A-Za-z]?(?:-\d+[A-Za-z]?)?"
_LABEL_RE = re.compile(rf"^\(?{_LABEL_PART}(?:\+{_LABEL_PART})?\)?$")

# Words that mean the model put a level/heading where the label should be.
_NOT_A_LABEL = re.compile(r"LEVEL|FLOOR|FOUNDATION|TERRACE|MARKED|SCHEDULE", re.I)


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_15.txt"),
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
        "width": size.get("width"),
        "depth": None,
        "length": size.get("length"),
    }


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(values):
    if not values:
        return []
    cleaned = []
    for v in values:
        v = str(v).upper()
        for p in v.split("+"):
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

    text = str(stirrups).upper()

    dia_match = re.search(r"\b\d+T\b", text)
    dia = dia_match.group() if dia_match else ""

    spacing_set = set()
    for num in re.findall(r"@\s*(\d+)", text):
        spacing_set.add(f"{num} C/C")
    for num in re.findall(r"(\d+)\s*C/?C", text):
        spacing_set.add(f"{num} C/C")

    spacing = ", ".join(sorted(spacing_set, key=lambda x: int(x.split()[0])))

    return {"dia": dia, "spacing": spacing}


# ==============================
# DETECT LEVELS (PATTERN 15)
# ==============================

_PATTERN15_LEVEL_CONTEXT = (
    "This page is a column/shear-wall schedule. "
    "Read the floor level labels from the column headers (TOP of the right grid) "
    "or from the row labels in the left section, EXACTLY as printed. "
    "Levels are usually RANGE names like 'FOUNDATION TO FIRST FLOOR LEVEL' or "
    "'THIRD FLOOR TO TERRACE LEVEL'. "
    "Return the FULL range name for each level — do NOT truncate "
    "(e.g. 'THIRD FLOOR TO TERRACE LEVEL', NOT 'THIRD FLOOR'). "
    "Do NOT invent levels that are not printed in the drawing."
)


def _detect_levels_p15(img_path, page_index):
    """Custom level detector for Pattern 15: returns full range names."""
    return detect_levels_from_image(
        img_path,
        pattern_number=15,
        prompt_context=_PATTERN15_LEVEL_CONTEXT,
    )


# ==============================
# DETECT COLUMN LABELS (CROP + UPSCALE READER — read from the image, not hardcoded)
# ==============================

def _clean_label_list(labels):
    """Validate/dedup raw label strings read from the cropped ID cells.

    Dedup is case-insensitive (so 'SW7' and 'sw7' collapse) but the first-seen
    casing is preserved (so mixed-case IDs like 'W23a' survive intact).
    """
    cleaned = []
    seen = set()
    for lab in labels or []:
        lab = re.sub(r"\s+", "", str(lab))
        if not lab or not re.search(r"\d", lab) or _NOT_A_LABEL.search(lab):
            continue
        # Explicit guard against the "COLUMN MARKED" confabulation.
        if re.match(r"^COLUMN\d+$", lab, re.IGNORECASE):
            print(f"  [LABEL] rejecting confabulated '{lab}'")
            continue
        if not _LABEL_RE.match(lab):
            print(f"  [ODD-LABEL] '{lab}' kept as-is")
        key = lab.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(lab)
    return cleaned


def _looks_like_invented_sequence(labels):
    """True when labels are one prefix numbered exactly 1..n (n>=6) — the
    classic hallucination signature (C1, C2, C3, ...)."""
    if len(labels) < 6:
        return False
    parsed = []
    for lab in labels:
        m = re.match(r"^([A-Za-z]+)-?(\d+)$", lab)
        if not m:
            return False
        parsed.append((m.group(1).upper(), int(m.group(2))))
    prefixes = {p for p, _ in parsed}
    nums = sorted(n for _, n in parsed)
    return len(prefixes) == 1 and nums == list(range(1, len(nums) + 1))


def _detect_labels(img_path):
    """Read the real column IDs via the locate -> crop+upscale -> read reader."""
    labels = _clean_label_list(read_labels_by_crop(img_path))
    if _looks_like_invented_sequence(labels):
        print(f"  [LABEL] WARNING result still looks invented: {labels[:4]}...")
    return labels


# ==============================
# VERIFY COLUMN LABELS
# ==============================

def _verify_labels(job, columns):
    """
    Post-extraction guardrail (per batch / per floor level), driven by the
    labels detected from the image itself (job["allowed_labels"]):
      1. Entries whose label is in the detected set → kept as-is.
      2. Entries with any other label (e.g. hallucinated C1, C2...) → remapped
         positionally to the first MISSING detected label (extraction order
         matches visual order, so position is reliable).
      3. Extras beyond the detected labels are dropped.
    If no labels were detected for the page, falls back to format validation
    (drop empty/heading-like labels, dedup).
    """
    if not columns:
        return columns

    level = job.get("level", "?")
    allowed = job.get("allowed_labels") or []

    if allowed:
        canon = {lab.lower(): lab for lab in allowed}  # case-insensitive match
        correct = {}
        wrong = []
        for col in columns:
            cno = re.sub(r"\s+", "", str(col.get("column_no", "")))
            key = canon.get(cno.lower())
            if key is not None and key not in correct:
                col = dict(col)
                col["column_no"] = key  # canonicalize to the verified casing
                correct[key] = col
            else:
                wrong.append(col)

        missing = [lab for lab in allowed if lab not in correct]
        for col, lab in zip(wrong, missing):
            print(f"  [FIX-ID] '{col.get('column_no')}' → '{lab}' (level: {level})")
            col = dict(col)
            col["column_no"] = lab
            correct[lab] = col

        dropped = len(wrong) - len(missing)
        if dropped > 0:
            print(f"  [DROP] {dropped} extra entry(ies) beyond detected labels (level: {level})")

        return [correct[lab] for lab in allowed if lab in correct]

    # Fallback: no detected labels — keep what looks like a real label.
    seen = set()
    result = []
    for col in columns:
        cno = re.sub(r"\s+", "", str(col.get("column_no", "")))
        if not cno or not re.search(r"\d", cno) or _NOT_A_LABEL.search(cno):
            print(f"  [DROP-LABEL] '{col.get('column_no')}' (level: {level})")
            continue
        if cno in seen:
            continue
        seen.add(cno)
        col = dict(col)
        col["column_no"] = cno
        result.append(col)
    return result


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

    # --------------------------
    # PASS 1: detect the column labels actually printed on each page
    # --------------------------
    page_labels = {}
    for page_index, img_path in enumerate(image_paths):
        print(f"  Detecting column labels -> {img_path}")
        labels = _detect_labels(img_path)
        page_labels[page_index] = labels
        print(f"  Detected {len(labels)} label(s): {', '.join(labels) or '(none)'}")

    all_labels = []
    for labels in page_labels.values():
        for lab in labels:
            if lab not in all_labels:
                all_labels.append(lab)

    if all_labels:
        prompt += (
            "\n\n=====================================================\n"
            "LABELS VERIFIED ON THIS PAGE (from a prior reading pass)\n"
            "=====================================================\n\n"
            f"The column labels printed on this page are EXACTLY:\n"
            f"  {', '.join(all_labels)}\n\n"
            "Every add_column() column_no MUST be one of these exact labels.\n"
            "think() column_headers MUST be exactly this list.\n"
            f"Expected entries per floor level: {len(all_labels)} "
            "(one per label). Do NOT add more.\n"
        )

    def _attach_labels(job):
        return {"allowed_labels": page_labels.get(job["page_index"], [])}

    all_columns = extract_levels_with_checkpoints(
        image_paths,
        prompt,
        pattern_number=15,
        output_folder=output_folder,
        prompt_context=_PATTERN15_LEVEL_CONTEXT,
        detect_levels_fn=_detect_levels_p15,
        verify_cells_fn=_verify_labels,   # <-- label validation hook
        job_decorator=_attach_labels,     # <-- per-page detected labels
    )

    # ==========================
    # FINAL CLEANUP + DEDUP
    # ==========================

    seen_keys = set()
    final_columns = []

    for col in all_columns:
        # Keep full level range names — do NOT apply upper_level_from_range
        col["column_name"] = str(col.get("column_name") or "").strip()

        col["size"] = clean_size(col.get("size"))
        col["reinforcement"] = clean_reinforcement(col.get("reinforcement"))
        col["stirrups"] = clean_stirrups(col.get("stirrups"))
        col["mix"] = col.get("mix")
        col["steel_grade"] = None

        # Deduplicate by (column_no, column_name); keep first occurrence
        key = (str(col.get("column_no", "")).strip(), col["column_name"])
        if key in seen_keys:
            print(f"  [DEDUP] Skipping duplicate: {key}")
            continue
        seen_keys.add(key)

        final_columns.append(col)

    final_output = reshape_columns_to_levels(final_columns)

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

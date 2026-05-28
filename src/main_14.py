import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from image_tools import crop_upscale, crop_upscale_path, zoom_and_extract  # noqa: F401
from pattern_batching import MAX_LEVEL_WORKERS, atomic_write_json, safe_filename, trace_key_for
from pattern_cleaners import standardize_records
from vision_extractor import detect_levels_from_image, extract_from_image, extract_with_tools


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

def extract_column_groups(image_path, prompt, trace_key=None):

    mode_prompt = prompt + "\n\nExtract ONLY COLUMN MARKED row."

    result = extract_with_tools(image_path, mode_prompt, trace_key=trace_key)

    try:
        parsed = json.loads(result)
        return parsed.get("column_groups", [])
    except:
        return []


# ==============================
# PASS 2: GET FLOOR DATA
# ==============================

def extract_floor(image_path, prompt, floor_name, total_positions, trace_key=None):

    mode_prompt = (
        prompt
        + f"\n\nExtract floor: {floor_name}\n"
        + f"Total column positions: {total_positions}\n"
    )

    result = extract_with_tools(image_path, mode_prompt, trace_key=trace_key)

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


def discover_floors(image_path, pattern_number=14):
    context = (
        "Pattern 14 floors are visible floor row labels such as GROUND FLOOR, "
        "FIRST FLOOR, and ROOF FLOOR. Return only floor names, top to bottom. "
        "Do not return column groups, SIZE, STEEL, RING, MIX, or notes."
    )
    try:
        floors = detect_levels_from_image(
            image_path,
            pattern_number=pattern_number,
            prompt_context=context,
        )
    except Exception as exc:
        print(f"  Floor discovery failed; using fallback floors: {exc}")
        floors = []

    cleaned = []
    for floor in floors:
        name = str(floor or "").strip().upper()
        if name and name not in cleaned:
            cleaned.append(name)

    return cleaned or ["GROUND FLOOR", "FIRST FLOOR", "ROOF FLOOR"]


def update_batch_status(manifest, batch_id, status, error=None):
    for batch in manifest["batches"]:
        if batch["id"] == batch_id:
            batch["status"] = status
            batch["error"] = error
            return


def merge_done_batches(manifest, output_folder):
    final_columns = []
    for batch in manifest["batches"]:
        if batch.get("status") != "done":
            continue
        path = os.path.join(output_folder, batch["file"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            columns = data.get("columns", [])
            if isinstance(columns, list):
                final_columns.extend(columns)
        except Exception as exc:
            print(f"  Could not merge floor batch {path}: {exc}")
    return final_columns


def run_floor_job(job, prompt):
    floor_data = extract_floor(
        job["img_path"],
        prompt,
        job["floor"],
        job["total_positions"],
        trace_key=job["trace_key"],
    )
    floor_data = fix_alignment(job["column_groups"], floor_data, job["floor"])

    columns = []
    for i in range(job["total_positions"]):
        col_group = job["column_groups"][i]
        col_data = floor_data[i] if i < len(floor_data) else None

        if col_data is None:
            col_data = {
                "size": {"width": None, "depth": None, "length": None},
                "reinforcement": [],
                "stirrups": {"dia": "", "spacing": ""},
                "mix": None,
            }

        columns.append({
            "column_no": col_group,
            "column_name": job["floor"],
            "size": col_data.get("size"),
            "reinforcement": col_data.get("reinforcement", []),
            "stirrups": col_data.get("stirrups", {
                "dia": "",
                "spacing": ""
            }),
            "mix": clean_mix(col_data.get("mix")),
            "steel_grade": None
        })

    atomic_write_json(job["path"], {"level": job["floor"], "columns": columns})
    return {"id": job["id"]}


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

    batch_folder_name = "level_batches"
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    batch_jobs = []

    for page_index, img_path in enumerate(tqdm(image_paths)):

        # PASS 1
        column_groups = extract_column_groups(
            img_path,
            prompt,
            trace_key=trace_key_for(
                img_path,
                f"p{page_index + 1:02d}_groups",
                "column_groups",
            ),
        )

        if not column_groups:
            continue

        total_positions = len(column_groups)

        floors = discover_floors(img_path)
        print(f"  Detected {len(floors)} floor(s): {', '.join(floors)}")

        for floor_index, floor in enumerate(floors):
            batch_id = f"p{page_index + 1:02d}_l{floor_index + 1:03d}"
            batch_file = f"{batch_id}_{safe_filename(floor)}.json"
            batch_jobs.append({
                "id": batch_id,
                "page": page_index + 1,
                "floor": floor,
                "img_path": img_path,
                "column_groups": column_groups,
                "total_positions": total_positions,
                "path": os.path.join(batch_folder, batch_file),
                "relative_file": os.path.join(batch_folder_name, batch_file),
                "trace_key": trace_key_for(img_path, batch_id, floor),
            })

    manifest = {
        "pattern": 14,
        "mode": "level_batches",
        "batches": [
            {
                "id": job["id"],
                "page": job["page"],
                "level": job["floor"],
                "status": "pending",
                "file": job["relative_file"],
                "error": None,
            }
            for job in batch_jobs
        ],
    }
    manifest_path = os.path.join(output_folder, "level_manifest.json")
    atomic_write_json(manifest_path, manifest)

    with ThreadPoolExecutor(max_workers=MAX_LEVEL_WORKERS) as executor:
        future_to_job = {}
        for job in batch_jobs:
            print(f"  Queueing floor batch -> {job['floor']}")
            update_batch_status(manifest, job["id"], "running")
            future_to_job[executor.submit(run_floor_job, job, prompt)] = job

        atomic_write_json(manifest_path, manifest)
        for future in tqdm(as_completed(future_to_job), total=len(future_to_job)):
            job = future_to_job[future]
            try:
                result = future.result()
                update_batch_status(manifest, result["id"], "done", None)
                print(f"  Floor batch done -> {job['floor']}")
            except Exception as exc:
                print(f"  Floor batch failed for '{job['floor']}': {exc}")
                update_batch_status(manifest, job["id"], "failed", str(exc))
            atomic_write_json(manifest_path, manifest)

    final_columns = merge_done_batches(manifest, output_folder)

    # Pattern 1 standardization: filter non-column rows + apply canonical
    # size/reinforcement/mix cleaners. Stirrups left untouched so
    # Pattern 14's own dict shape is preserved.
    final_columns = standardize_records(final_columns, stirrups_cleaner=None)

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

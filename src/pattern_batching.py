import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from vision_extractor import detect_levels_from_image, extract_with_tools


MAX_LEVEL_WORKERS = int(os.getenv("MAX_LEVEL_WORKERS", os.getenv("PATTERN_MAX_WORKERS", "3")))


def load_prompt(pattern_number):
    """Read the extraction prompt for a pattern from `prompt_<n>.txt` (same dir)."""
    with open(
        os.path.join(os.path.dirname(__file__), f"prompt_{pattern_number}.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()

# 1-6 letter prefix (C, AC, PC, GC, SW, RC, BC, CP, COL, COLUMN, ...) directly
# followed by digits and an optional trailing letter. No whitespace between
# the prefix and the digits — that prevents "300 X 1150" from matching as
# "X 1150".
_COLUMN_ID_RE = re.compile(r"\b[A-Z]{1,6}-?\d+[A-Z]?\b", re.IGNORECASE)

_NON_COLUMN_LABELS = {
    "COLUMN NO",
    "COLUMN NOS",
    "COLUMN NUMBER",
    "COLUMN NUMBERS",
    "COLUMN MARK",
    "COLUMN MARKS",
    "SIZE",
    "REINF",
    "REINFORCEMENT",
    "STIRRUPS",
    "STIRRUP",
    "STEEL",
    "LINKS",
    "MIX",
    "GRADE",
    "TYPE",
}

# Whole-string patterns that look column-like but aren't column IDs:
#   TYPE-26 / TYPE 31 — size-category label inside a SIZE cell
#   M40 / M250        — concrete grade
_NON_COLUMN_PATTERNS = (
    re.compile(r"^TYPE[\s\-]*\d+[A-Z]?$", re.IGNORECASE),
    re.compile(r"^M\d+$", re.IGNORECASE),
)


_LEVEL_RANGE_SPLIT_RE = re.compile(r"\s*/?\s*\bTO\b\s*/?\s*", re.IGNORECASE)


def upper_level_from_range(label):
    """Apply Pattern-1's upper-level rule to a level-range label.

    A bounded level cell may carry a range such as "P06 TO ECO-DECK" or
    "TERRACE FLOOR LEVEL / To / 4TH FLOOR LEVEL". Across every column
    schedule we have seen, the cell's storey reference is the FIRST
    (upper) name -- the second name is the floor below it.

    Returns the upper portion. Strings without " TO " are returned as-is.
    """
    if label is None:
        return label
    text = str(label).strip()
    if not text:
        return text
    parts = [p.strip() for p in _LEVEL_RANGE_SPLIT_RE.split(text) if p and p.strip()]
    return parts[0] if parts else text


def upper_levels(labels):
    """Vectorized form of `upper_level_from_range`."""
    return [upper_level_from_range(lbl) for lbl in (labels or [])]


def single_level(label="ALL"):
    """A `detect_levels_fn` that forces exactly ONE synthetic level.

    Use for floor-less schedules (COLUMN MARK | SIZES, COLUMN ID tables, ...) so the
    shared level engine still emits the standard `level_batches/` layout with a single
    batch instead of letting vision invent spurious floor levels."""
    return lambda img_path, page_index: [label]


def is_valid_column_no(value):
    text = str(value or "").strip()
    if not text:
        return False
    if any(p.match(text) for p in _NON_COLUMN_PATTERNS):
        return False
    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    if normalized in _NON_COLUMN_LABELS:
        return False
    return bool(_COLUMN_ID_RE.search(text))


def filter_valid_columns(columns):
    if not isinstance(columns, list):
        return []
    kept = []
    for col in columns:
        if not isinstance(col, dict):
            continue
        if not is_valid_column_no(col.get("column_no")):
            print(f"  Skipping non-column row: {col.get('column_no')!r}")
            continue
        kept.append(col)
    return kept


def env_enabled(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def safe_filename(value):
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return name[:80] or "UNKNOWN"


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def parse_json_result(raw):
    text = str(raw or "").strip()
    if "```" in text:
        parts = text.split("```")
        block = parts[1] if len(parts) >= 3 else parts[0]
        if block.lower().startswith("json"):
            block = block[4:]
        text = block.strip()
    return json.loads(text)


def trace_key_for(image_path, batch_id, label):
    return f"{os.path.basename(image_path)}__{batch_id}_{safe_filename(label)}"


def build_level_batch_prompt(base_prompt, target_level, all_levels, pattern_number):
    levels_json = json.dumps(all_levels, ensure_ascii=False)
    return f"""
{base_prompt}

LEVEL-FIRST BATCH EXTRACTION:
- Pattern number: {pattern_number}
- The visible floor/level/section list detected for this page is: {levels_json}
- Extract ONLY this target floor/level/section batch: "{target_level}".
- Use "{target_level}" exactly as column_name / storey_level for every record in this batch.
- Do not process any other level, floor, section, or row in this batch.
- Do not borrow values from neighboring levels or sections.
- If a target cell has no visible values, return null size and [] reinforcement for that column only.
- Keep only actual column identifiers/groups as column_no values.
- Return only strict JSON in the usual {{"columns": [...]}} shape.
"""


def _update_status(manifest, batch_id, status, error=None):
    for batch in manifest["batches"]:
        if batch["id"] == batch_id:
            batch["status"] = status
            batch["error"] = error
            return


def _read_done_columns(manifest, output_folder):
    columns = []
    for batch in manifest["batches"]:
        if batch.get("status") != "done":
            continue
        path = os.path.join(output_folder, batch["file"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("columns", [])
            if isinstance(items, list):
                columns.extend(items)
        except Exception as exc:
            print(f"  Could not merge batch {path}: {exc}")
    return columns


def _run_batch_jobs(output_folder, manifest_path, manifest, jobs, worker):
    if not jobs:
        atomic_write_json(manifest_path, manifest)
        return []

    atomic_write_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=MAX_LEVEL_WORKERS) as executor:
        future_to_job = {}
        for job in jobs:
            print(f"  Queueing batch -> {job['label']}")
            _update_status(manifest, job["id"], "running")
            future_to_job[executor.submit(worker, job)] = job

        atomic_write_json(manifest_path, manifest)

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job)):
            job = future_to_job[future]
            try:
                result = future.result()
                _update_status(manifest, result["id"], "done", None)
                print(f"  Batch done -> {job['label']}")
            except Exception as exc:
                print(f"  Batch failed -> {job['label']}: {exc}")
                _update_status(manifest, job["id"], "failed", str(exc))
            atomic_write_json(manifest_path, manifest)

    return _read_done_columns(manifest, output_folder)


def extract_pages_with_checkpoints(
    image_paths,
    prompt,
    pattern_number,
    output_folder,
    batch_folder_name="page_batches",
    extractor=None,
    filter_columns=True,
):
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)

    jobs = []
    for page_index, img_path in enumerate(image_paths):
        batch_id = f"p{page_index + 1:02d}"
        batch_file = f"{batch_id}_page_{page_index + 1:02d}.json"
        jobs.append(
            {
                "id": batch_id,
                "page": page_index + 1,
                "img_path": img_path,
                "label": f"page {page_index + 1}",
                "path": os.path.join(batch_folder, batch_file),
                "relative_file": os.path.join(batch_folder_name, batch_file),
                "trace_key": trace_key_for(img_path, batch_id, "page"),
            }
        )

    manifest = {
        "pattern": pattern_number,
        "mode": "page_batches",
        "batches": [
            {
                "id": job["id"],
                "page": job["page"],
                "label": job["label"],
                "status": "pending",
                "file": job["relative_file"],
                "error": None,
            }
            for job in jobs
        ],
    }
    manifest_path = os.path.join(output_folder, "page_manifest.json")

    def worker(job):
        if extractor is None:
            raw = extract_with_tools(job["img_path"], prompt, trace_key=job["trace_key"])
            parsed = parse_json_result(raw)
        else:
            parsed = extractor(job, prompt)
        columns = parsed.get("columns", []) if isinstance(parsed, dict) else []
        if not isinstance(columns, list):
            columns = []
        if filter_columns:
            columns = filter_valid_columns(columns)
        atomic_write_json(job["path"], {"columns": columns})
        return {"id": job["id"]}

    return _run_batch_jobs(output_folder, manifest_path, manifest, jobs, worker)


def extract_levels_with_checkpoints(
    image_paths,
    prompt,
    pattern_number,
    output_folder,
    prompt_context="",
    batch_folder_name="level_batches",
    filter_columns=True,
    detect_levels_fn=None,
    verify_cells_fn=None,
    job_decorator=None,
    extractor=None,
):
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)

    jobs = []
    all_page_levels = []
    for page_index, img_path in enumerate(image_paths):
        print(f"  Detecting levels -> {img_path}")
        levels = None
        if detect_levels_fn is not None:
            try:
                levels = detect_levels_fn(img_path, page_index)
            except Exception as exc:
                print(f"  Custom level detector failed; falling back to default: {exc}")
                levels = None
        if not levels:
            levels = detect_levels_from_image(
                img_path,
                pattern_number=pattern_number,
                prompt_context=prompt_context,
            )
        if not levels:
            print("  No levels detected; falling back to one whole-page batch.")
            levels = ["UNKNOWN"]
        all_page_levels.append({"page": page_index + 1, "levels": levels})
        print(f"  Detected {len(levels)} level(s): {', '.join(levels)}")

        for level_index, level in enumerate(levels):
            batch_id = f"p{page_index + 1:02d}_l{level_index + 1:03d}"
            batch_file = f"{batch_id}_{safe_filename(level)}.json"
            job = {
                "id": batch_id,
                "page": page_index + 1,
                "page_index": page_index,
                "level": level,
                "level_index": level_index,
                "levels": levels,
                "img_path": img_path,
                "label": level,
                "path": os.path.join(batch_folder, batch_file),
                "relative_file": os.path.join(batch_folder_name, batch_file),
                "trace_key": trace_key_for(img_path, batch_id, level),
            }
            if job_decorator is not None:
                try:
                    extras = job_decorator(job) or {}
                    if isinstance(extras, dict):
                        job.update(extras)
                except Exception as exc:
                    print(f"  job_decorator failed for {level!r}: {exc}")
            jobs.append(job)

    manifest = {
        "pattern": pattern_number,
        "mode": "level_batches",
        "pages": all_page_levels,
        "levels": [job["level"] for job in jobs],
        "batches": [
            {
                "id": job["id"],
                "page": job["page"],
                "level": job["level"],
                "status": "pending",
                "file": job["relative_file"],
                "error": None,
            }
            for job in jobs
        ],
    }
    manifest_path = os.path.join(output_folder, "level_manifest.json")

    def worker(job):
        batch_prompt = build_level_batch_prompt(
            prompt,
            job["level"],
            job["levels"],
            pattern_number,
        )
        if extractor is not None:
            parsed = extractor(job, batch_prompt)
        else:
            raw = extract_with_tools(job["img_path"], batch_prompt, trace_key=job["trace_key"])
            parsed = parse_json_result(raw)
        columns = parsed.get("columns", []) if isinstance(parsed, dict) else []
        if not isinstance(columns, list):
            columns = []
        if filter_columns:
            columns = filter_valid_columns(columns)
        for col in columns:
            if isinstance(col, dict):
                col["column_name"] = job["level"]
        if verify_cells_fn is not None:
            try:
                columns = verify_cells_fn(job, columns) or columns
            except Exception as exc:
                print(f"  verify_cells_fn failed for {job['level']!r}: {exc}")
        atomic_write_json(
            job["path"],
            {
                "level": job["level"],
                "columns": columns,
            },
        )
        return {"id": job["id"]}

    return _run_batch_jobs(output_folder, manifest_path, manifest, jobs, worker)

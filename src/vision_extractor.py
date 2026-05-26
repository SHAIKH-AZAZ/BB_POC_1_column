"""
vision_extractor.py - Column project

Tool-augmented extraction with enforced sequence:
think -> (optional zoom_region -> confirm_read) -> add_column.
"""

import base64
import io
import json
import re
import time

from PIL import Image
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_IMAGE_DETAIL, OPENAI_MODEL
from extraction_guard import (
    ExtractionState,
    build_column_record,
    clean_json_string,
)


client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def _image_content(base64_image, detail=OPENAI_IMAGE_DETAIL):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": detail,
        },
    }


def _crop_image_b64(image_path, x1, y1, x2, y2):
    """Crop to normalized coords (0.0-1.0), upscale to >=1200px, return base64 PNG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))

    if x2 <= x1:
        x2 = min(1.0, x1 + 0.05)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.05)

    left = int(x1 * w)
    top = int(y1 * h)
    right = max(int(x2 * w), left + 20)
    bottom = max(int(y2 * h), top + 20)

    cropped = img.crop((left, top, right, bottom))
    longest = max(cropped.size)
    if longest < 1200:
        scale = 1200 / longest
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_REGION_PURPOSE_ENUM = [
    "header",
    "row_label",
    "data_cell",
    "global_note",
    "ambiguous_text",
]


def _with_tool_protocol(prompt_text):
    return (
        "ENFORCED TOOL PROTOCOL:\n"
        "You MUST use tools - do NOT return raw JSON text.\n"
        "Step 1: call think() with your full extraction plan.\n"
        "Step 2: call add_column() once for EVERY (column_group x storey_level) combination.\n"
        "  - A valid column_group must contain actual column IDs such as C1 or C1,C7,C8.\n"
        "  - Never call add_column for table labels like 'Column Nos.', 'SIZE', 'REINF.', or 'STIRRUPS'.\n"
        "  - For B x L sizes, pass width=B, length=L, and depth=null.\n"
        "  - For W x D sizes, pass width=W, depth=D, and length=null.\n"
        "  - Work left-to-right across columns, then move to the next storey level.\n"
        "  - Do NOT stop early. Every row in every column group must get its own add_column call.\n"
        "Optional: call zoom_region() + confirm_read() for any cell that is blurry or ambiguous.\n\n"
        f"{prompt_text}"
    )


COLUMN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Mandatory first call. Return structured observable extraction planning data, "
                "not private reasoning. Count headers, storey rows, expected records, and zoom targets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_quality": {
                        "type": "string",
                        "description": "Resolution/sharpness/skew/rotation notes.",
                    },
                    "table_bounds": {
                        "type": "object",
                        "description": "Normalized table/header/data-grid bounds.",
                    },
                    "column_headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Actual visible column identifier header groups left to right. "
                            "Exclude table labels such as Column Nos., SIZE, REINF., and STIRRUPS."
                        ),
                    },
                    "storey_levels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All visible level labels top to bottom.",
                    },
                    "expected_count": {
                        "type": "integer",
                        "description": "Expected add_column calls, usually headers x storey levels.",
                    },
                    "zoom_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "region_id": {"type": "string"},
                                "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                                "target": {"type": "string"},
                                "x1": {"type": "number"},
                                "y1": {"type": "number"},
                                "x2": {"type": "number"},
                                "y2": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["region_id", "purpose", "target", "reason"],
                        },
                    },
                    "normalization_rules": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "extraction_order": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "image_quality",
                    "table_bounds",
                    "column_headers",
                    "storey_levels",
                    "expected_count",
                    "zoom_plan",
                    "normalization_rules",
                    "extraction_order",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_region",
            "description": (
                "Crop and zoom a region for closer inspection. Optional - use only when a cell "
                "is blurry, ambiguous, or has a two-digit quantity that needs confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                    "x2": {"type": "number"},
                    "y2": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["region_id", "purpose", "x1", "y1", "x2", "y2", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_read",
            "description": (
                "Record exact text read from a zoomed region. Call after zoom_region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "string"},
                    "applies_to": {"type": "string"},
                },
                "required": ["region_id", "text", "confidence", "applies_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_column",
            "description": (
                "Record ONE column schedule entry (one column-group x one storey-level). "
                "Call this once per combination. Do not batch multiple levels or groups into one call. "
                "zoom_region/confirm_read are optional - use them only for ambiguous or blurry cells."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column_no": {
                        "type": "string",
                        "description": (
                            "Full multi-line header joined with commas. "
                            "Read EVERY line and EVERY digit carefully (C67 != C7, C14 != C4). "
                            "Must be an actual column identifier group, never 'Column Nos.'."
                        ),
                    },
                    "storey_level": {
                        "type": "string",
                        "description": (
                            "Storey/floor label copied exactly from the table row "
                            "(e.g. 'TERRACE FLOOR LEVEL To 4TH FLOOR LEVEL')."
                        ),
                    },
                    "width": {
                        "type": ["number", "null"],
                        "description": "First size value. For B x L schedules, this is B/breadth.",
                    },
                    "depth": {
                        "type": ["number", "null"],
                        "description": "Depth value only when the drawing uses W x D or W x D x L. Use null for B x L.",
                    },
                    "length": {
                        "type": ["number", "null"],
                        "description": "Length value. For B x L schedules, this is the second value L.",
                    },
                    "reinforcement": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Bars as quantity-Tdiameter. Split '+' notation: "
                            "'4-T20+16-T16' becomes ['4-T20','16-T16']. "
                            "Read 2-digit quantities carefully: 20-T20 not 2-T20."
                        ),
                    },
                    "ties_dia": {
                        "type": ["string", "null"],
                        "description": "Stirrup/tie bar diameter, e.g. 'T8'. Read from STIRRUPS row.",
                    },
                    "ties_spacing": {
                        "type": ["string", "null"],
                        "description": "Stirrup spacing, e.g. '200 C/C'. Read from STIRRUPS row.",
                    },
                    "mix": {"type": ["string", "null"]},
                    "steel_grade": {"type": ["string", "null"]},
                    "source_region_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: zoom region IDs used for this record.",
                    },
                },
                "required": [
                    "column_no",
                    "storey_level",
                    "reinforcement",
                ],
            },
        },
    },
]


def extract_with_tools(image_path, prompt_text, max_iterations=300, trace_key=None):
    """
    Tool extraction with enforced state. Returns JSON string: {"columns": [...]}.
    A trace is written beside the rendered image as <pdf-stem>_trace.json.
    """
    base64_image = encode_image(image_path)
    state = ExtractionState(
        project="column",
        image_path=image_path,
        output_key="columns",
        duplicate_key_fields=["column_no", "column_name"],
        trace_key=trace_key,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _with_tool_protocol(prompt_text)},
                _image_content(base64_image),
            ],
        }
    ]

    collected_columns = []

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=COLUMN_TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            break

        tool_results = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "think":
                state.handle_think(args)
                headers = args.get("column_headers") or []
                levels = args.get("storey_levels") or []
                expected = state.expected_count()
                print("\n============================================================")
                print("  [THINK] Structured extraction plan accepted")
                print(f"  headers={len(headers)} levels={len(levels)} expected={expected}")
                print("============================================================\n")
                result_content = (
                    f"Plan accepted. {len(headers)} column groups x {len(levels)} storey levels "
                    f"= {expected} expected add_column calls.\n"
                    "NOW start calling add_column immediately - one call per (column_group, storey_level) combination.\n"
                    f"Column groups in order: {', '.join(headers)}.\n"
                    f"Storey levels in order: {', '.join(levels)}.\n"
                    "Work through ALL column groups for LEVEL 1, then ALL for LEVEL 2, etc. "
                    "Do NOT stop until every combination has been recorded. "
                    "Use zoom_region only if a cell is genuinely unclear."
                )

            elif fn_name == "zoom_region":
                region, message = state.handle_zoom(args)
                if not region:
                    result_content = message
                else:
                    print(
                        f"  zoom_region {region['region_id']} "
                        f"({region['x1']:.2f},{region['y1']:.2f})->"
                        f"({region['x2']:.2f},{region['y2']:.2f})"
                    )
                    cropped_b64 = _crop_image_b64(
                        image_path,
                        region["x1"],
                        region["y1"],
                        region["x2"],
                        region["y2"],
                    )
                    result_content = [
                        {
                            "type": "text",
                            "text": (
                                f"{message} For headers, capture every line and every digit. "
                                "For reinforcement, confirm the full quantity, especially 20/24/28/32."
                            ),
                        },
                        _image_content(cropped_b64),
                    ]

            elif fn_name == "confirm_read":
                _, result_content = state.handle_confirm_read(args)
                print(f"  confirm_read {args.get('region_id', '')}: {str(args.get('text', ''))[:80]}")

            elif fn_name == "add_column":
                ok, message = state.can_add_record(args)
                if not ok:
                    result_content = message
                else:
                    col = build_column_record(args)
                    collected_columns.append(col)
                    state.add_record(col, args)
                    expected = state.expected_count() or "?"
                    remaining = (state.expected_count() or 0) - len(collected_columns)
                    result_content = (
                        f"Recorded {len(collected_columns)}/{expected}: "
                        f"'{col['column_no']}' @ '{col['column_name']}'. "
                        f"{remaining} more to go - keep calling add_column for remaining combinations."
                    )
                    print(f"  add_column: {col['column_no']} | {col['column_name']}")

            else:
                state.warn(f"unknown tool ignored: {fn_name}")
                result_content = f"Unknown tool ignored: {fn_name}"

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                }
            )

        messages.extend(tool_results)

    state.validate(collected_columns)
    trace_path = state.write_trace()
    print(f"  Trace saved to {trace_path}")

    if collected_columns:
        print(f"  {len(collected_columns)} column entry(ies).")
        return json.dumps({"columns": collected_columns})

    print("  Tool extraction returned 0 columns -- falling back.")
    return extract_from_image(image_path, prompt_text)


def extract_from_image(image_path, prompt_text, retries=3):
    """Single-pass fallback with retry and JSON fence cleanup."""
    base64_image = encode_image(image_path)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            _image_content(base64_image),
                        ],
                    }
                ],
                temperature=0,
            )
            return clean_json_string(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"  Fallback extraction failed, retrying ({attempt}/{retries})...")
                time.sleep(5)

    raise RuntimeError(f"Extraction failed after {retries} retries: {last_error}")


def _parse_levels_response(raw):
    data = json.loads(clean_json_string(raw))

    if isinstance(data, dict):
        levels = data.get("levels", [])
    elif isinstance(data, list):
        levels = data
    else:
        levels = []

    cleaned = []
    for level in levels:
        value = str(level).strip()
        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


def _has_level_range(levels):
    return any(re.search(r"\bTO\b", str(level), flags=re.IGNORECASE) for level in levels)


def _build_pattern1_level_retry_prompt(pattern_number, previous_levels, prompt_context):
    previous_json = json.dumps(previous_levels, ensure_ascii=False)
    return f"""
You are correcting ONLY the visible Pattern 1 floor/level list from a column schedule image.

Your previous level list was invalid because it included full range labels:
{previous_json}

Return ONLY strict JSON in this exact shape:
{{
  "pattern": {pattern_number},
  "levels": []
}}

Correction rules:
- Re-read the bounded left-side level/floor cells from top to bottom.
- Each levels[] value must be the upper visible level of that bounded cell.
- No levels[] value may contain the word "To".
- Do not invent or rename level/floor names.
- Copy LEVEL and FLOOR exactly as visible; they are not interchangeable words.
- Treat stacked text in one bounded cell as one label.
- "GROUND FLOOR LEVEL / To / BASEMENT LEVEL" must return "GROUND FLOOR LEVEL".
- "BASEMENT LEVEL / To / FOUNDATION LEVEL" must return "BASEMENT LEVEL".
- Do not return "FOUNDATION LEVEL" when it appears only as the lower part of the basement-to-foundation range.
- Do not add "1ST" unless it is visibly printed in that same bounded cell.

Additional pattern context:
{prompt_context}
"""


def detect_levels_from_image(image_path, pattern_number, prompt_context=""):
    """Return visible floor/level names for one page using a focused vision prompt."""

    prompt = f"""
You are extracting ONLY the visible floor/level names from a column schedule image.

Pattern number: {pattern_number}

Return ONLY strict JSON in this exact shape:
{{
  "pattern": {pattern_number},
  "levels": []
}}

Rules:
- Read only the visible floor/level cells from the drawing.
- Do not extract column numbers, sizes, reinforcement, stirrups, notes, or any other details.
- Do not invent, rename, reword, or assume level/floor names.
- Copy LEVEL and FLOOR exactly as visible; they are not interchangeable words.
- Preserve the visible order from top to bottom.
- For Pattern 1, read the left-side level/floor column.
- For Pattern 1, treat each bounded level/floor cell as one label even when the text is stacked across multiple lines.
- For Pattern 1, if a visible cell is a range written like "UPPER LEVEL To LOWER LEVEL", return only the upper visible level exactly as written.
- Never return the lower part of a range as its own level unless it appears in another visible cell as the upper level or as a standalone label.
- Do not add ordinal words such as "1ST" unless they are visibly present in that same level/floor cell.
- Example: "1ST FLOOR LEVEL To GROUND FLOOR LEVEL" becomes "1ST FLOOR LEVEL".
- Example: "GROUND FLOOR LEVEL To BASEMENT LEVEL" becomes "GROUND FLOOR LEVEL".
- Example: "BASEMENT LEVEL To FOUNDATION LEVEL" becomes "BASEMENT LEVEL".
- If the cell is stacked as three lines "BASEMENT LEVEL" / "To" / "FOUNDATION LEVEL", return only "BASEMENT LEVEL"; do not return "FOUNDATION LEVEL".
- If a visible cell is not a range, keep it exactly as visible.

Additional pattern context:
{prompt_context}
"""

    raw = extract_from_image(image_path, prompt)
    levels = _parse_levels_response(raw)

    if pattern_number == 1 and _has_level_range(levels):
        print("  Level detection returned range labels; retrying with upper-level correction prompt.")
        retry_prompt = _build_pattern1_level_retry_prompt(
            pattern_number,
            levels,
            prompt_context,
        )
        retry_raw = extract_from_image(image_path, retry_prompt)
        retry_levels = _parse_levels_response(retry_raw)
        if retry_levels:
            levels = retry_levels

    if pattern_number == 1 and _has_level_range(levels):
        raise RuntimeError(
            "Pattern 1 level detection returned range labels after retry: "
            + ", ".join(levels)
        )

    return levels

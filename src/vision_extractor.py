"""
vision_extractor.py - Column project

Tool-augmented extraction with enforced sequence:
think -> zoom_region -> confirm_read -> add_column.
"""

import base64
import io
import json
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
                        "description": "All visible header groups left to right.",
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
                "Crop and zoom a planned region. Must be called after think and before confirm_read. "
                "Use for every important header, row label, ambiguous cell, and two-digit quantity."
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
                "Record exact text read from a zoomed region. Required before add_column can "
                "reference that region_id in source_region_ids."
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
                "Record one column schedule entry. Requires source_region_ids that have already "
                "been zoomed and confirmed with confirm_read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column_no": {
                        "type": "string",
                        "description": "Full multi-line header joined with commas, e.g. C1,C7,C8,C14,C67,C75.",
                    },
                    "storey_level": {
                        "type": "string",
                        "description": "Storey/floor label copied exactly from the table.",
                    },
                    "width": {"type": ["number", "null"]},
                    "depth": {"type": ["number", "null"]},
                    "reinforcement": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bars as quantity-Tdiameter, splitting + into separate entries.",
                    },
                    "ties_dia": {"type": ["string", "null"]},
                    "ties_spacing": {"type": ["string", "null"]},
                    "mix": {"type": ["string", "null"]},
                    "steel_grade": {"type": ["string", "null"]},
                    "source_region_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Confirmed region IDs supporting this record.",
                    },
                },
                "required": [
                    "column_no",
                    "storey_level",
                    "reinforcement",
                    "source_region_ids",
                ],
            },
        },
    },
]


def extract_with_tools(image_path, prompt_text, max_iterations=100):
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
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
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
                result_content = state.handle_think(args)
                print("\n============================================================")
                print("  [THINK] Structured extraction plan accepted")
                print(f"  headers={len(args.get('column_headers', []) or [])} "
                      f"levels={len(args.get('storey_levels', []) or [])} "
                      f"expected={state.expected_count()}")
                print("============================================================\n")

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
                    result_content = (
                        f"Column '{col['column_no']}' @ '{col['column_name']}' recorded "
                        f"({len(collected_columns)} total). Continue with the next planned record."
                    )
                    print(f"  add_column: {col['column_no']} | {col['column_name']}")

            else:
                state.warn(f"unknown tool ignored: {fn_name}")
                result_content = f"Unknown tool '{fn_name}' ignored."

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

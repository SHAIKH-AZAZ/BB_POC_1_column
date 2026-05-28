import json
import os
import re
import threading
from datetime import datetime, timezone

ALLOWED_REGION_PURPOSES = {
    "header",
    "row_label",
    "data_cell",
    "global_note",
    "ambiguous_text",
}

_TRACE_WRITE_LOCK = threading.Lock()
_NON_COLUMN_LABELS = {
    "COLUMN NO",
    "COLUMN NOS",
    "COLUMN NUMBER",
    "COLUMN NUMBERS",
    "SIZE",
    "REINF",
    "REINFORCEMENT",
    "STIRRUPS",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


def _as_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _label_key(value):
    return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()


def _spacing(value):
    text = str(value).upper().strip()
    if not text:
        return None
    if re.fullmatch(r"\d{2,4}", text):
        return f"{text} C/C"
    text = text.replace("CC", "C/C")
    text = re.sub(r"\s*C\s*/?\s*C\s*", " C/C", text)
    m = re.search(r"(\d{2,4})\s*C/C", text)
    if m:
        return f"{m.group(1)} C/C"
    return None


def extract_rebar_parts(*texts):
    dia = []
    spacing = []
    for text in texts:
        for item in _as_list(text):
            clean = str(item).upper().replace("PHI", "T").replace("Ø", "T")
            for dm in re.finditer(r"\bT?\s*(\d{1,2})\b(?=\s*(?:@|C/C|MM|$))", clean):
                value = f"T{dm.group(1)}"
                if value not in dia:
                    dia.append(value)
            for dm in re.finditer(r"\bT\s*(\d{1,2})\b", clean):
                value = f"T{dm.group(1)}"
                if value not in dia:
                    dia.append(value)
            for sm in re.finditer(r"(?:@|-)?\s*(\d{2,4})\s*C\s*/?\s*C", clean):
                value = f"{sm.group(1)} C/C"
                if value not in spacing:
                    spacing.append(value)
            for sm in re.finditer(r"@\s*(\d{2,4})", clean):
                value = f"{sm.group(1)} C/C"
                if value not in spacing:
                    spacing.append(value)
    return {"dia": dia, "spacing": spacing}


def build_column_record(args):
    return {
        "column_no": args.get("column_no", ""),
        "column_name": args.get("storey_level", ""),
        "size": {
            "width": args.get("width"),
            "depth": args.get("depth"),
            "length": args.get("length"),
        },
        "reinforcement": args.get("reinforcement", []) or [],
        "stirrups": {
            "dia": [args["ties_dia"]] if args.get("ties_dia") else [],
            "spacing": [args["ties_spacing"]] if args.get("ties_spacing") else [],
        },
        "mix": args.get("mix"),
        "steel_grade": args.get("steel_grade"),
    }


def build_slab_record(args):
    along = args.get("steel_along_span", []) or []
    across = args.get("steel_across_span", []) or []
    reinf = extract_rebar_parts(along, across)
    record = {
        "slab_id": args.get("slab_id") or args.get("slab_type", ""),
        "thickness": args.get("thickness"),
        "type": args.get("type", "") or "",
        "mix": args.get("mix", "") or "",
        "reinforcement": reinf,
    }
    if args.get("remarks") is not None:
        record["remarks"] = args.get("remarks")
    return record


def build_footing_record(args):
    short_span = args.get("short_span_reinf")
    long_span = args.get("long_span_reinf")
    reinf = extract_rebar_parts(short_span, long_span)
    column_id = args.get("column_id", "")
    return {
        "footing_id": args.get("footing_id") or column_id,
        "column_id": column_id,
        "size": {
            "width": args.get("plan_width"),
            "depth": args.get("depth_bottom")
            if args.get("depth_bottom") is not None
            else args.get("depth_top"),
            "length": args.get("plan_length"),
        },
        "reinforcement": reinf,
        "nos": args.get("nos"),
        "mix": args.get("mix") or args.get("concrete_mix"),
        "steel_grade": args.get("steel_grade"),
    }


class ExtractionState:
    def __init__(
        self,
        project,
        image_path,
        output_key,
        duplicate_key_fields,
        trace_key=None,
    ):
        self.project = project
        self.image_path = image_path
        self.output_key = output_key
        self.duplicate_key_fields = duplicate_key_fields
        self.trace_key = trace_key
        self.think_seen = False
        self.think = None
        self.zoom_regions = {}
        self.confirmed_reads = {}
        self.records = []
        self.record_sources = []
        self.warnings = []

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
        print(f"  [WARN] {message}")

    def handle_think(self, args):
        self.think_seen = True
        self.think = dict(args)
        missing = [
            key
            for key in (
                "image_quality",
                "table_bounds",
                "expected_count",
                "zoom_plan",
                "normalization_rules",
                "extraction_order",
            )
            if key not in args
        ]
        if missing:
            self.warn(f"think missing structured field(s): {', '.join(missing)}")
        expected = self.expected_count()
        return (
            "Structured extraction plan accepted. "
            f"Expected record count: {expected if expected is not None else 'unknown'}. "
            "Now call zoom_region for planned regions, then confirm_read for each important read, "
            f"then call add_{self.project} with source_region_ids."
        )

    def handle_zoom(self, args):
        if not self.think_seen:
            self.warn("zoom_region rejected before think")
            return (
                None,
                "Call think first; zoom_region is only available after the structured plan.",
            )

        region_id = str(
            args.get("region_id") or f"region_{len(self.zoom_regions) + 1}"
        ).strip()
        purpose = str(args.get("purpose") or "ambiguous_text").strip()
        if purpose not in ALLOWED_REGION_PURPOSES:
            self.warn(f"zoom_region purpose '{purpose}' is not recognized")
            purpose = "ambiguous_text"

        region = {
            "region_id": region_id,
            "purpose": purpose,
            "x1": float(args.get("x1", 0.0)),
            "y1": float(args.get("y1", 0.0)),
            "x2": float(args.get("x2", 1.0)),
            "y2": float(args.get("y2", 1.0)),
            "reason": args.get("reason", ""),
        }
        self.zoom_regions[region_id] = region
        return region, (
            f"Zoomed region '{region_id}' accepted. "
            "Read it carefully and call confirm_read with the exact text before adding records."
        )

    def handle_confirm_read(self, args):
        if not self.think_seen:
            self.warn("confirm_read rejected before think")
            return False, "Call think first, then zoom_region, then confirm_read."

        region_id = str(args.get("region_id") or "").strip()
        if region_id not in self.zoom_regions:
            self.warn(f"confirm_read rejected for unzoomed region '{region_id}'")
            return (
                False,
                f"Region '{region_id}' has not been zoomed. Call zoom_region first.",
            )

        text = str(args.get("text") or "").strip()
        if not text:
            self.warn(f"confirm_read for region '{region_id}' is empty")

        read = {
            "region_id": region_id,
            "text": text,
            "confidence": args.get("confidence"),
            "applies_to": args.get("applies_to"),
        }
        self.confirmed_reads[region_id] = read
        return True, f"Confirmed read for region '{region_id}' recorded."

    def can_add_record(self, args):
        if not self.think_seen:
            self.warn(f"add_{self.project} rejected before think")
            return False, f"Call think before add_{self.project}."
        if self.project == "column":
            column_no = args.get("column_no")
            if _label_key(column_no) in _NON_COLUMN_LABELS:
                self.warn(f"add_column rejected non-column label '{column_no}'")
                return (
                    False,
                    (
                        f"'{column_no}' is a table label, not a column identifier. "
                        "Call add_column only for actual column IDs such as C1 or C1,C7,C8."
                    ),
                )
        # zoom_region / confirm_read are optional accuracy aids, not blocking gates.
        # source_region_ids are recorded in the trace for auditability but not required.
        return True, "Record accepted."

    def add_record(self, record, args):
        self.records.append(record)
        self.record_sources.append(
            {
                "record": record,
                "source_region_ids": _as_list(args.get("source_region_ids")),
            }
        )

    def expected_count(self):
        if not self.think:
            return None
        explicit = _as_int(self.think.get("expected_count"))
        if explicit is not None:
            return explicit
        if self.project == "column":
            headers = self.think.get("column_headers") or []
            levels = self.think.get("storey_levels") or []
            if headers and levels:
                return len(headers) * len(levels)
        if self.project == "slab":
            slab_ids = self.think.get("slab_ids") or []
            if slab_ids:
                return len(slab_ids)
        if self.project == "footing":
            groups = self.think.get("column_groups") or []
            if groups:
                return len(groups)
        return None

    def validate(self, records=None):
        records = records if records is not None else self.records
        expected = self.expected_count()
        if expected is not None and expected != len(records):
            self.warn(f"expected {expected} record(s), collected {len(records)}")

        seen = {}
        for record in records:
            key = tuple(
                str(record.get(field, "")) for field in self.duplicate_key_fields
            )
            if key in seen:
                self.warn(f"duplicate record key detected: {key}")
            seen[key] = True

        if self.project == "column":
            for record in records:
                column_no = str(record.get("column_no", "")).replace(" ", "")
                if "C7,C75" in column_no and "C67" not in column_no:
                    self.warn(
                        f"suspicious column header '{record.get('column_no')}' may have lost C67"
                    )

        trace_text = (
            json.dumps(self.think or {})
            + " "
            + json.dumps(list(self.confirmed_reads.values()))
        )
        record_text = json.dumps(records)
        for match in re.findall(r"\b(?:1[0-9]|2[0-9]|3[0-9])-T\d+\b", trace_text):
            if match not in record_text:
                self.warn(
                    f"confirmed two-digit reinforcement '{match}' not found in records"
                )

    def trace_payload(self):
        return {
            "project": self.project,
            "image_path": self.image_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "think": self.think,
            "zoom_regions": list(self.zoom_regions.values()),
            "confirmed_reads": list(self.confirmed_reads.values()),
            "records": self.records,
            "added_records": self.record_sources,
            "expected_count": self.expected_count(),
            "actual_count": len(self.records),
            "warnings": self.warnings,
        }

    def write_trace(self):
        folder = os.path.dirname(os.path.abspath(self.image_path))
        stem = (
            os.path.basename(folder)
            or os.path.splitext(os.path.basename(self.image_path))[0]
        )
        path = os.path.join(folder, f"{stem}_trace.json")
        page_key = self.trace_key or os.path.basename(self.image_path)

        with _TRACE_WRITE_LOCK:
            existing = {
                "project": self.project,
                "pdf_stem": stem,
                "pages": {},
                "summary": {},
            }
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing.update(loaded)
                        existing.setdefault("pages", {})
                except (OSError, json.JSONDecodeError):
                    self.warn(f"could not read existing trace file: {path}")

            existing["pages"][page_key] = self.trace_payload()
            all_warnings = []
            total_records = 0
            for page in existing["pages"].values():
                total_records += int(page.get("actual_count") or 0)
                all_warnings.extend(page.get("warnings") or [])
            existing["summary"] = {
                "page_count": len(existing["pages"]),
                "record_count": total_records,
                "warning_count": len(all_warnings),
                "warnings": _dedupe(all_warnings),
            }

            tmp_path = f"{path}.{page_key}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        return path


def clean_json_string(text):
    text = str(text or "").strip()
    if "```" in text:
        parts = text.split("```")
        block = parts[1] if len(parts) >= 3 else parts[0]
        if block.lower().startswith("json"):
            block = block[4:]
        text = block.strip()
    return text


def _format_size(size):
    """Return the common B x L size object used by Pattern 1."""
    empty = {"width": None, "depth": None, "length": None}
    if not size:
        return dict(empty)
    if not isinstance(size, dict):
        nums = [int(n) for n in re.findall(r"\d+", str(size))]
        if len(nums) >= 2:
            return {"width": nums[0], "depth": None, "length": nums[1]}
        if len(nums) == 1:
            return {"width": nums[0], "depth": None, "length": None}
        return dict(empty)
    length = size.get("length")
    if length is None:
        length = size.get("depth")
    return {
        "width": _as_int(size.get("width")),
        "depth": None,
        "length": _as_int(length),
    }


def _coerce_stirrups_to_strings(stirrups):
    """Normalize ANY stirrups input shape to Pattern 2 scalar-string shape:
        {"dia": "T8", "spacing": "100 C/C, 200 C/C"}

    Always returns the dict, even when stirrups is missing/empty
    (defaults to {"dia": "", "spacing": ""}).

    Also normalizes the dia notation:
        "8T"   -> "T8"
        "10T"  -> "T10"
        "T8"   -> "T8"
    so the output is consistent across patterns whose drawings use
    either prefix order.
    """
    if not stirrups:
        return {"dia": "", "spacing": ""}

    # Dict input ({"dia": ..., "spacing": ...})
    if isinstance(stirrups, dict):
        raw_dia = stirrups.get("dia") or ""
        raw_sp = stirrups.get("spacing") or ""
    # List input ([raw_zone_strings, ...]) - join into spacing
    elif isinstance(stirrups, list):
        raw_dia = ""
        # try to pull a dia token from the first element
        for item in stirrups:
            m = re.search(r"\b([TYHRD#])\s*-?\s*(\d+)|\b(\d+)\s*([TYHRD#])\b", str(item).upper())
            if m:
                raw_dia = m.group(0).replace(" ", "").replace("-", "")
                break
        raw_sp = ", ".join(str(s) for s in stirrups if s)
    else:
        raw_dia = ""
        raw_sp = str(stirrups)

    # Flatten list-valued dia/spacing. Keep multi-zone diameters; Pattern 10
    # links can legitimately be ["T10", "T8"] for one cell.
    if isinstance(raw_dia, list):
        raw_dia = ", ".join(str(d).strip() for d in raw_dia if d)
    if isinstance(raw_sp, list):
        raw_sp = ", ".join(str(s).strip() for s in raw_sp if s)

    dia_parts = []
    for part in re.split(r"\s*,\s*", str(raw_dia).strip().upper()):
        part = part.replace(" ", "")
        # Normalize "8T" -> "T8" (Pattern 6 notation -> Pattern 2 notation)
        m = re.match(r"^(\d+)([TYHRD#])$", part)
        if m:
            part = f"{m.group(2)}{m.group(1)}"
        if part and part not in dia_parts:
            dia_parts.append(part)

    return {
        "dia": ", ".join(dia_parts),
        "spacing": str(raw_sp).strip(),
    }


def reshape_columns_to_levels(flat_records):
    """
    Convert flat column records into level-centric (Option B) structure
    with the project-wide canonical shape:

      {
        "levels": [
          {
            "level": "<NAME or UNKNOWN if no floor data>",
            "columns": [
              {
                "column_no": "...",
                "size": {"width": ..., "depth": null, "length": ...},
                "reinforcement": [...],
                "stirrups": {"dia": "T8", "spacing": "100 C/C, 200 C/C"},
                "mix": ...,           # only included when present
                "steel_grade": ...    # only included when present
              }
            ]
          }
        ],
        "columns": [ ...flat back-compat list... ]
      }

    Key contracts (apply to every pattern):
      * stirrups is ALWAYS present, even when empty
        ({"dia": "", "spacing": ""}). Pattern 2 scalar-string shape.
      * "8T"-style dia (Pattern 6) is rewritten to "T8" style.
      * If column_name (level) is missing, "UNKNOWN" is used.
    """
    from collections import OrderedDict

    level_map = OrderedDict()

    for record in flat_records:
        level = str(record.get("column_name") or "").strip()
        if not level:
            level = "UNKNOWN"

        col_entry = {
            "column_no": record.get("column_no", ""),
            "size": _format_size(record.get("size")),
            "reinforcement": record.get("reinforcement") or [],
            "stirrups": _coerce_stirrups_to_strings(record.get("stirrups")),
        }

        if record.get("mix"):
            col_entry["mix"] = record["mix"]
        if record.get("steel_grade"):
            col_entry["steel_grade"] = record["steel_grade"]

        if level not in level_map:
            level_map[level] = []
        level_map[level].append(col_entry)

    return {
        "levels": [
            {"level": level, "columns": cols} for level, cols in level_map.items()
        ],
        "columns": [col for cols in level_map.values() for col in cols],
    }

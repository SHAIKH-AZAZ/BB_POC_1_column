import math
import re

from knowledge_loader import get_knowledge_bundle
from pattern_batching import is_valid_column_no, upper_level_from_range


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_rebar_area(reinforcement):
    area = 0.0
    for token in reinforcement or []:
        text = str(token).upper().replace(" ", "")
        m = re.match(r"(\d+)-T(\d+)", text)
        if not m:
            continue
        count = int(m.group(1))
        dia = int(m.group(2))
        area += count * (math.pi * (dia ** 2) / 4.0)
    return area


def _extract_spacing_values(stirrups):
    values = []
    if isinstance(stirrups, dict):
        spacing = stirrups.get("spacing")
        if isinstance(spacing, list):
            text = ",".join(str(x) for x in spacing)
        else:
            text = str(spacing or "")
    else:
        text = str(stirrups or "")
    for m in re.findall(r"(\d+)", text):
        values.append(int(m))
    return values


def evaluate_records(records, *, pattern_number=None, context=None):
    kb = get_knowledge_bundle()
    standards = kb.get("codes", {}).get("standards", {})
    is_456 = standards.get("IS_456", {})
    min_ratio = float(is_456.get("column_reinforcement_ratio_min", 0.008))
    max_ratio = float(is_456.get("column_reinforcement_ratio_max", 0.06))
    max_tie_spacing = int(is_456.get("max_tie_spacing_mm", 300))
    results = []

    for idx, record in enumerate(records or []):
        if not isinstance(record, dict):
            continue
        violations = []
        auto_fix_candidates = []
        column_no = str(record.get("column_no") or "").strip()
        column_name = str(record.get("column_name") or "").strip()

        if not column_no:
            violations.append(
                {"rule_id": "missing_column_id", "severity": "high", "message": "column_no is empty", "field": "column_no"}
            )
        elif not is_valid_column_no(column_no):
            violations.append(
                {
                    "rule_id": "invalid_column_id",
                    "severity": "medium",
                    "message": f"column_no '{column_no}' does not match known naming grammar",
                    "field": "column_no",
                }
            )

        if " TO " in column_name.upper():
            fixed = upper_level_from_range(column_name)
            violations.append(
                {
                    "rule_id": "range_level_label",
                    "severity": "medium",
                    "message": f"column_name '{column_name}' still contains a range label",
                    "field": "column_name",
                }
            )
            auto_fix_candidates.append({"kind": "upper_level_from_range", "field": "column_name", "value": fixed})

        size = record.get("size") if isinstance(record.get("size"), dict) else {}
        width = _to_int(size.get("width"))
        length = _to_int(size.get("length") if size.get("length") is not None else size.get("depth"))
        if width is None or length is None:
            violations.append(
                {
                    "rule_id": "missing_size",
                    "severity": "medium",
                    "message": "size.width or size.length is missing",
                    "field": "size",
                }
            )
        elif width <= 0 or length <= 0:
            violations.append(
                {
                    "rule_id": "invalid_size",
                    "severity": "high",
                    "message": "size dimensions must be positive",
                    "field": "size",
                }
            )

        reinforcement = record.get("reinforcement") or []
        if not reinforcement:
            violations.append(
                {
                    "rule_id": "missing_reinforcement",
                    "severity": "medium",
                    "message": "reinforcement list is empty",
                    "field": "reinforcement",
                }
            )
        elif width and length:
            steel_area = _extract_rebar_area(reinforcement)
            gross_area = width * length
            if gross_area > 0 and steel_area > 0:
                ratio = steel_area / gross_area
                if ratio < min_ratio:
                    violations.append(
                        {
                            "rule_id": "low_rebar_ratio",
                            "severity": "medium",
                            "message": f"reinforcement ratio {ratio:.4f} below IS 456 min {min_ratio}",
                            "field": "reinforcement",
                        }
                    )
                elif ratio > max_ratio:
                    violations.append(
                        {
                            "rule_id": "high_rebar_ratio",
                            "severity": "high",
                            "message": f"reinforcement ratio {ratio:.4f} above IS 456 max {max_ratio}",
                            "field": "reinforcement",
                        }
                    )

        spacing_values = _extract_spacing_values(record.get("stirrups"))
        if spacing_values and max(spacing_values) > max_tie_spacing:
            violations.append(
                {
                    "rule_id": "tie_spacing_exceeds_max",
                    "severity": "medium",
                    "message": f"tie spacing exceeds {max_tie_spacing} mm",
                    "field": "stirrups.spacing",
                }
            )

        results.append(
            {
                "index": idx,
                "pattern_number": pattern_number,
                "column_no": column_no,
                "column_name": column_name,
                "violations": violations,
                "auto_fix_candidates": auto_fix_candidates,
            }
        )

    return {
        "records": results,
        "summary": {
            "total_records": len(results),
            "total_violations": sum(len(x["violations"]) for x in results),
            "high_severity": sum(
                1 for x in results for v in x["violations"] if v.get("severity") == "high"
            ),
        },
    }

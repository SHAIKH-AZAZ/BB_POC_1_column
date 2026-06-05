import json
import os

from confidence import score_records
from knowledge_loader import apply_ocr_normalization_to_record
from pattern_batching import upper_level_from_range
from rules_engine import evaluate_records
from uncertainty_review import build_uncertainty_review


def _clean_runtime_flags(records):
    out = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        item = dict(rec)
        item.pop("__fallback_used", None)
        out.append(item)
    return out


def run_quality_pipeline(records, *, pattern_number, output_folder, file_stem, context=None):
    context = context or {}
    normalized = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        item = apply_ocr_normalization_to_record(record)
        if "column_name" in item:
            item["column_name"] = upper_level_from_range(item.get("column_name"))
        normalized.append(item)

    rules_result = evaluate_records(normalized, pattern_number=pattern_number, context=context)
    confidence_rows = score_records(normalized, rules_result, context=context)
    review_mode = "interactive" if context.get("interactive_review") else "batch"
    review = build_uncertainty_review(
        normalized,
        confidence_rows,
        rules_result,
        mode=review_mode,
        context=context,
    )

    quality_payload = {
        "pattern": pattern_number,
        "context": context,
        "rules_engine": rules_result,
        "confidence": confidence_rows,
        "uncertainty_review": review,
    }
    os.makedirs(output_folder, exist_ok=True)
    quality_path = os.path.join(output_folder, f"{file_stem}_quality.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_payload, f, indent=2, ensure_ascii=False)
    print(f"  Quality metadata saved to {quality_path}")
    return _clean_runtime_flags(normalized), quality_payload

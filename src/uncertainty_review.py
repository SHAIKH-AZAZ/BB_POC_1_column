def build_uncertainty_review(records, confidence_rows, rules_result, *, mode="batch", context=None):
    context = context or {}
    confidence_by_index = {row["index"]: row for row in confidence_rows or []}
    rules_by_index = {row["index"]: row for row in (rules_result or {}).get("records", [])}
    needs_review = []
    questions = []

    for idx, record in enumerate(records or []):
        conf = confidence_by_index.get(idx, {})
        rules = rules_by_index.get(idx, {})
        score = conf.get("confidence_score", 1.0)
        high_violation = any(v.get("severity") == "high" for v in rules.get("violations", []))
        unresolved_range = " TO " in str((record or {}).get("column_name", "")).upper()
        invalid_id = not str((record or {}).get("column_no", "")).strip()
        if score >= 0.7 and not high_violation and not unresolved_range and not invalid_id:
            continue

        item = {
            "index": idx,
            "column_no": (record or {}).get("column_no", ""),
            "column_name": (record or {}).get("column_name", ""),
            "confidence_score": score,
            "violations": rules.get("violations", []),
            "needs_review": True,
            "suggested_crop_link": f"trace://record/{idx}",
        }
        needs_review.append(item)

        if mode == "interactive":
            questions.append(
                {
                    "index": idx,
                    "question": (
                        f"Please re-check column '{item['column_no']}' at level '{item['column_name']}' "
                        "for reinforcement and stirrup clarity."
                    ),
                }
            )

    if context.get("missing_expected_records"):
        questions.append(
            {
                "index": None,
                "question": "Detected missing expected records. Please review skipped/empty rows.",
            }
        )
    return {
        "mode": mode,
        "review_count": len(needs_review),
        "needs_review_records": needs_review,
        "clarifying_questions": questions,
    }

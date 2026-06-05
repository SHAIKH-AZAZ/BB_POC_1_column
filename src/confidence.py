def score_records(records, rules_result, *, context=None):
    context = context or {}
    fallback_penalty = 0.15 if context.get("fallback_used") else 0.0
    retry_penalty = min(float(context.get("retry_count", 0)) * 0.03, 0.2)
    expected_missing_penalty = 0.1 if context.get("missing_expected_records") else 0.0
    rule_by_index = {r["index"]: r for r in (rules_result or {}).get("records", [])}
    scored = []

    for idx, record in enumerate(records or []):
        reasons = []
        score = 1.0
        if not isinstance(record, dict):
            continue

        item_fallback = bool(record.get("__fallback_used"))
        if fallback_penalty or item_fallback:
            applied = 0.15 if item_fallback else fallback_penalty
            score -= applied
            reasons.append(f"fallback extraction penalty -{applied:.2f}")
        if retry_penalty:
            score -= retry_penalty
            reasons.append(f"retry penalty -{retry_penalty:.2f}")
        if expected_missing_penalty:
            score -= expected_missing_penalty
            reasons.append(f"expected record mismatch penalty -{expected_missing_penalty:.2f}")

        rules = rule_by_index.get(idx, {})
        for violation in rules.get("violations", []):
            sev = violation.get("severity")
            if sev == "high":
                score -= 0.35
                reasons.append(f"high violation: {violation.get('rule_id')}")
            elif sev == "medium":
                score -= 0.2
                reasons.append(f"medium violation: {violation.get('rule_id')}")
            else:
                score -= 0.1
                reasons.append(f"low violation: {violation.get('rule_id')}")

        field_conf = {
            "column_no": 1.0 if record.get("column_no") else 0.4,
            "column_name": 1.0 if record.get("column_name") else 0.6,
            "size": 1.0 if (record.get("size") or {}).get("width") and (record.get("size") or {}).get("length") else 0.5,
            "reinforcement": 1.0 if record.get("reinforcement") else 0.5,
        }

        score = max(0.0, min(1.0, round(score, 3)))
        scored.append(
            {
                "index": idx,
                "column_no": record.get("column_no", ""),
                "column_name": record.get("column_name", ""),
                "confidence_score": score,
                "confidence_reasons": reasons or ["no penalties detected"],
                "field_confidence": field_conf,
            }
        )
    return scored

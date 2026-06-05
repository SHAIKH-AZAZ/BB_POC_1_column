import json
import os
import re
from functools import lru_cache


_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")


def _load_json_like(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@lru_cache(maxsize=1)
def get_knowledge_bundle():
    return {
        "codes": _load_json_like(os.path.join(_BASE_DIR, "codes.json"), {}),
        "naming_patterns": _load_json_like(os.path.join(_BASE_DIR, "naming_patterns.yml"), {}),
        "ocr_confusions": _load_json_like(os.path.join(_BASE_DIR, "ocr_confusions.yml"), {}),
    }


def build_prompt_knowledge_context(pattern_number=None):
    kb = get_knowledge_bundle()
    codes = kb.get("codes", {}).get("standards", {})
    is_456 = codes.get("IS_456", {})
    is_13920 = codes.get("IS_13920", {})
    families = kb.get("naming_patterns", {}).get("families", [])
    return (
        "Domain constraints:\n"
        f"- Reinforcement ratio target bounds: {is_456.get('column_reinforcement_ratio_min', 0.008)}"
        f" to {is_456.get('column_reinforcement_ratio_max', 0.06)}.\n"
        f"- Typical tie spacing max: {is_456.get('max_tie_spacing_mm', 300)} mm"
        f" (use {is_13920.get('seismic_max_tie_spacing_mm', 150)} mm in ductile/seismic zones).\n"
        f"- Typical column ID families: {', '.join(families)}.\n"
        "- Resolve OCR confusions conservatively: TOR->T, CC/C C->C/C, and O/0 or I/1 only inside alphanumeric IDs.\n"
        f"- Pattern context: {pattern_number if pattern_number is not None else 'unknown'}."
    )


def _normalize_id_text(text):
    value = str(text or "")
    value = re.sub(r"(?<=\w)O(?=\d)|(?<=\d)O(?=\w)", "0", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\w)I(?=\d)|(?<=\d)I(?=\w)", "1", value, flags=re.IGNORECASE)
    return value


def normalize_text(text):
    value = str(text or "")
    conf = get_knowledge_bundle().get("ocr_confusions", {})
    for old, new in (conf.get("token_replacements") or {}).items():
        value = re.sub(rf"\b{re.escape(old)}\b", new, value, flags=re.IGNORECASE)
    return value


def apply_ocr_normalization_to_record(record):
    if not isinstance(record, dict):
        return record
    item = dict(record)
    if "column_no" in item:
        item["column_no"] = _normalize_id_text(normalize_text(item.get("column_no")))
    if "column_name" in item:
        item["column_name"] = normalize_text(item.get("column_name"))
    if "reinforcement" in item and isinstance(item["reinforcement"], list):
        item["reinforcement"] = [normalize_text(v).upper() for v in item["reinforcement"]]
    stirrups = item.get("stirrups")
    if isinstance(stirrups, dict):
        dia = stirrups.get("dia")
        spacing = stirrups.get("spacing")
        if isinstance(dia, list):
            dia = [normalize_text(v).upper() for v in dia]
        elif dia is not None:
            dia = normalize_text(dia).upper()
        if isinstance(spacing, list):
            spacing = [normalize_text(v).upper() for v in spacing]
        elif spacing is not None:
            spacing = normalize_text(spacing).upper()
        item["stirrups"] = {"dia": dia, "spacing": spacing}
    return item

"""
pattern_cleaners.py
===================

Canonical per-column field cleaners shared by every main_N.py.

The contracts here match Pattern 1's standard:

  clean_size(size)          -> {"width": int|None, "depth": None, "length": int|None}
  clean_reinforcement(list) -> list[str]    (uppercased, split-on-+, dedupe preserve order)
  clean_mix(value)          -> "M-XX" | None

`standardize_records()` is the convenience wrapper most callers want:

    from pattern_cleaners import standardize_records

    final_columns = standardize_records(
        raw_columns,
        # Per-pattern stirrups cleaner. Pass None to leave stirrups
        # untouched (some patterns store their own shape).
        stirrups_cleaner=my_clean_stirrups,
        # Set False to skip the is_valid_column_no filter — useful when
        # a pattern's column IDs don't match the default regex.
        apply_column_filter=True,
        # Optional default for fields the model may omit.
        defaults={"steel_grade": None},
    )

That covers what every bespoke main_N.py needs at the OUTPUT side
without touching its bespoke extraction flow (fitz / OpenCV / 2-pass).
"""

import re

from pattern_batching import filter_valid_columns, upper_level_from_range


# ---------------------------------------------------------------------------
# SIZE
# ---------------------------------------------------------------------------

def clean_size(size):
    """Pattern 1's size contract.

    * Keeps width as-is.
    * Forces depth = None.
    * If length is missing but depth was set, falls back to depth as length
      (handles model outputs that used the wrong key).
    """
    if not size:
        return {"width": None, "depth": None, "length": None}

    if not isinstance(size, dict):
        # Tolerate single-number / string inputs from less-strict patterns.
        nums = re.findall(r"\d+", str(size))
        if len(nums) >= 2:
            return {"width": int(nums[0]), "depth": None, "length": int(nums[1])}
        if len(nums) == 1:
            return {"width": int(nums[0]), "depth": None, "length": int(nums[0])}
        return {"width": None, "depth": None, "length": None}

    length = size.get("length")
    if length is None:
        length = size.get("depth")
    return {
        "width": size.get("width"),
        "depth": None,
        "length": length,
    }


# ---------------------------------------------------------------------------
# REINFORCEMENT
# ---------------------------------------------------------------------------

def clean_reinforcement(values):
    """Split on '+', uppercase, strip, dedupe preserving order."""
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    cleaned = []
    for v in values:
        text = str(v or "").strip().upper()
        if not text:
            continue
        for part in text.split("+"):
            part = part.strip()
            if part and part not in cleaned:
                cleaned.append(part)
    return cleaned


# ---------------------------------------------------------------------------
# MIX
# ---------------------------------------------------------------------------

def clean_mix(mix):
    """Normalize mix grade.

    Examples:
      "M30"   -> "M-30"
      "M 30"  -> "M-30"
      "M-30"  -> "M-30"
      "m30"   -> "M-30"
      None    -> None
      ""      -> None
    """
    if mix is None:
        return None
    text = str(mix).strip().upper()
    if not text:
        return None
    match = re.search(r"M\s*[-]?\s*(\d+)", text)
    return f"M-{match.group(1)}" if match else None


# ---------------------------------------------------------------------------
# STIRRUPS — two common shapes
#
# Most patterns store stirrups in one of two shapes:
#
#   "dict-of-lists"   {"dia": ["8T"], "spacing": ["100 C/C (Z1)", ...]}
#   "dict-of-strings" {"dia": "T8",   "spacing": "100 C/C, 200 C/C"}
#
# Each cleaner returns its own shape and accepts EITHER input shape so
# patterns can swap freely. Patterns with bespoke stirrup formats
# (e.g. Pattern 5's BBSTEEL regex output) should keep their own
# cleaner and just pass it as `stirrups_cleaner` to standardize_records.
# ---------------------------------------------------------------------------

def clean_stirrups_lists(stirrups):
    """Pattern 6 shape: lists of dia + spacing, preserving zone annotations.

    Does NOT rewrite "8T" to "T8" — keep the source notation order.
    """
    if not stirrups:
        return {"dia": [], "spacing": []}

    dia_out, sp_out = [], []

    def _push(target, value):
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)

    if isinstance(stirrups, dict):
        raw_dia = stirrups.get("dia") or []
        raw_sp = stirrups.get("spacing") or []
        if isinstance(raw_dia, (list, tuple)):
            for d in raw_dia:
                _push(dia_out, str(d).upper().replace(" ", ""))
        elif raw_dia:
            _push(dia_out, str(raw_dia).upper().replace(" ", ""))
        if isinstance(raw_sp, (list, tuple)):
            for s in raw_sp:
                _push(sp_out, s)
        elif raw_sp:
            for piece in re.split(r",(?![^()]*\))", str(raw_sp)):
                _push(sp_out, piece.strip())
        return {"dia": dia_out, "spacing": sp_out}

    items = stirrups if isinstance(stirrups, list) else [stirrups]
    label_re = re.compile(
        r"^\s*(?P<dia>\d+\s*T|T\s*\d+|Y\s*\d+|H\s*\d+|R\s*\d+|D\s*\d+|#\s*\d+)\s*"
        r"[-@\s]\s*(?P<sp>\d+\s*(?:MM)?\s*C\s*/?\s*C\s*(?:\([^)]*\))?)\s*$",
        re.IGNORECASE,
    )
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        m = label_re.match(text.upper())
        if m:
            _push(dia_out, m.group("dia").upper().replace(" ", ""))
            _push(sp_out, m.group("sp"))
        else:
            _push(sp_out, text)
    return {"dia": dia_out, "spacing": sp_out}


def clean_stirrups_strings(stirrups):
    """Pattern 2 shape: scalar dia + comma-joined spacing string.

    First diameter wins. Spacing is comma-joined in encounter order
    (no sort, no dedupe — zone order is meaningful).
    """
    if not stirrups:
        return {"dia": "", "spacing": ""}

    lists = clean_stirrups_lists(stirrups)
    dia = lists["dia"][0] if lists["dia"] else ""
    # Normalize "8T" notation -> "T8" for the strings shape (Pattern 2
    # convention).
    m = re.match(r"^(\d+)([TYHRD#])$", dia)
    if m:
        dia = f"{m.group(2)}{m.group(1)}"
    return {
        "dia": dia,
        "spacing": ", ".join(lists["spacing"]),
    }


# ---------------------------------------------------------------------------
# COLUMN NO normalization
# ---------------------------------------------------------------------------

# Column-ID prefixes we see across the 14 patterns. Used by
# `clean_column_no()` to decide which inner tokens are real IDs vs
# noise (counts, zone numbers, etc.) when splitting a grouped label.
#
# Real-world variety we must handle (Pattern 9 + others):
#   C34, C1A                            (C-family)
#   P1, P11A                            (P-family)
#   r2, W23a                            (lowercase variants — case kept)
#   AC1, BC1, CP1, GC1, NC1, PC1,       (2-letter prefixes)
#   SC1, RW1, LW1, SW1, TW33
#   BSW1                                (3-letter prefixes)
#   SW-1, CP-01, CL-12                  (letter prefix + hyphen + digits)
#   TA-C1, TB-C1, TC-C1                 (T-letter compound prefix)
#   PC206-12, PC197A-8                  (ID with trailing -digits sub-id)
#
# The regex below is greedy enough to catch all of the above while
# still rejecting M-grade, TYPE-XX, and lone header words.
_COLUMN_ID_TOKEN_RE = re.compile(
    r"\b("
    # ── (1) T<X>- compound prefix : TA-C1, TB-C1, TC-C1 ─────────────
    r"T[A-Z]-[A-Z]\d+[A-Z]?(?:-\d+)?"
    r"|"
    # ── (2) base form : 1-4 letter prefix + optional hyphen + digits
    #        + optional letter suffix + optional -digits sub-id ──────
    r"[A-Z]{1,4}-?\d+[A-Z]?(?:-\d+)?"
    r"|"
    # ── (3) rare leading-digit form : 1A, 12B2 ──────────────────────
    r"\d+[A-Z]+\d*"
    r")\b",
    re.IGNORECASE,
)

# Words/tokens that are NEVER an ID and should be stripped before
# parsing a grouped label.
_NON_ID_TOKEN_RE = re.compile(
    r"\b(?:"
    r"NOS?\.?|NUMBER|NUMBERS|MARK|MARKS|TYPE|"
    r"AND|OR|"
    r"COLUMN|COLUMNS|"
    r"GROUP|GROUPS|"        # Pattern 9 organizational labels — never column IDs
    r"LAP|LAPS|"            # Pattern 9 storey labels — never column IDs
    r"SR\.?|S\.?\s*NO\.?"
    r")\b",
    re.IGNORECASE,
)


# Words that look like "GROUP 1" / "GROUP 2" should also be filtered
# OUT of the final tokens list. The token regex would accidentally
# match "GROUP1" / "GROUP 1" otherwise (4-letter prefix + digits).
_FORBIDDEN_TOKEN_RE = re.compile(
    r"^(?:GROUP|LAP|TYPE|NOS)\d+[A-Z]?$",
    re.IGNORECASE,
)


def clean_column_no(value):
    """Normalize ANY shape of column-ID label into a comma-joined
    canonical form with no spaces.

    Accepts grouped IDs separated by ANY of:
        ","   "&"   "/"   "  and  "   "  AND  "   "-" between two IDs
    Drops:
        leading counts like "2 NOS."
        words like "NOS", "AND", "OR", "MARK", "TYPE"
        empty fragments

    Preserves the original prefix (TA-, SW, P, AC, BC, PC, GC, CP, CL,
    M, etc.) and the original case of letters. Tightens whitespace.

    Examples:
        "C1 & C3"          -> "C1,C3"
        "TA-C1, TA-C4"     -> "TA-C1,TA-C4"
        "C70/C72"          -> "C70,C72"
        "C1 and C4"        -> "C1,C4"
        "C1, C2 & C3"      -> "C1,C2,C3"
        "2 NOS. C1"        -> "C1"
        "C1 (2 NOS.)"      -> "C1"
        "C1-C4"            -> "C1,C4"   (range hyphen between two IDs)
        "CP-01"            -> "CP-01"   (kept — hyphen is part of one ID)
        "TA-C1"            -> "TA-C1"   (kept — hyphen is part of one ID)
        ""                 -> ""
        None               -> ""
    """
    if value is None:
        return ""

    # Already-list input: join with commas first, then re-clean.
    if isinstance(value, (list, tuple)):
        joined = ",".join(str(v).strip() for v in value if v is not None and str(v).strip())
        return clean_column_no(joined)

    text = str(value).strip()
    if not text:
        return ""

    # Strip trailing/leading parenthetical noise like "(2 NOS.)" /
    # "(TYP)" that some schedules append next to a column ID.
    text = re.sub(r"\((?:[^)]*?(?:NOS?\.?|TYP|TYPICAL|EACH)[^)]*?)\)", " ", text, flags=re.IGNORECASE)

    # Strip "<digits> NOS." count prefixes/suffixes.
    text = re.sub(r"\b\d+\s*NOS?\.?\s*", " ", text, flags=re.IGNORECASE)

    # Strip the explicit non-ID words.
    text = _NON_ID_TOKEN_RE.sub(" ", text)

    # Pull every recognizable ID token out of the remaining text. This
    # is the robust path — we don't rely on a particular separator,
    # we just extract every token that LOOKS like a column ID.
    tokens = []
    for match in _COLUMN_ID_TOKEN_RE.finditer(text):
        token = match.group(1).strip()
        if not token:
            continue
        # Tighten internal spaces (e.g. "TA C1" -> "TA-C1" when the
        # original had a space instead of a hyphen).
        token = re.sub(r"\s+", "-", token)
        # Avoid emitting "M30" / "M-30" / "TYPE-26" as a column ID.
        if re.fullmatch(r"M[-\s]?\d+", token, re.IGNORECASE):
            continue
        if re.fullmatch(r"TYPE[-\s]?\d+[A-Z]?", token, re.IGNORECASE):
            continue
        # Skip Pattern 9-style "GROUP 1" / "LAP 1" organizational
        # labels that aren't real column IDs.
        if _FORBIDDEN_TOKEN_RE.match(token):
            continue
        if token not in tokens:
            tokens.append(token)

    if tokens:
        return ",".join(tokens)

    # Fallback: nothing recognized — fall back to legacy normalization
    # so we don't lose unusual labels entirely.
    fallback = str(value).strip()
    fallback = re.sub(r"\s*&\s*", ",", fallback)
    fallback = re.sub(r"\s+/\s+", "/", fallback)
    fallback = re.sub(r"\s+(?:and|AND)\s+", ",", fallback)
    fallback = re.sub(r"\s*,\s*", ",", fallback)
    fallback = re.sub(r"\s{2,}", " ", fallback)
    return fallback


# ---------------------------------------------------------------------------
# Combined post-processing helper
# ---------------------------------------------------------------------------

def standardize_records(
    records,
    *,
    stirrups_cleaner=None,
    apply_column_filter=True,
    apply_size=True,
    apply_reinforcement=True,
    apply_mix=True,
    apply_column_no=True,
    apply_upper_level=True,
    defaults=None,
):
    """Run the canonical cleanup over a flat list of column-record dicts.

    Returns a new list (does not mutate input).

    Each step is opt-out so bespoke patterns can keep their custom logic
    for fields where the canonical cleaner would lose information.

    apply_upper_level (default True):
        For any column_name shaped like "X TO Y", keep only the X
        (upper / first-before-TO) portion. Single-name labels like
        "GROUND FLOOR" pass through unchanged, so this is safe to
        leave enabled for every pattern.
    """
    if not isinstance(records, list):
        return []

    defaults = dict(defaults or {})

    if apply_column_filter:
        records = filter_valid_columns(records)

    out = []
    for record in records:
        if not isinstance(record, dict):
            continue
        # shallow copy so we never mutate the caller's dict
        item = dict(record)

        if apply_column_no:
            item["column_no"] = clean_column_no(item.get("column_no"))
        if apply_size:
            item["size"] = clean_size(item.get("size"))
        if apply_reinforcement:
            item["reinforcement"] = clean_reinforcement(item.get("reinforcement"))
        if apply_mix and "mix" in item:
            item["mix"] = clean_mix(item.get("mix"))
        if apply_upper_level and "column_name" in item:
            item["column_name"] = upper_level_from_range(item.get("column_name"))
        if stirrups_cleaner is not None:
            item["stirrups"] = stirrups_cleaner(item.get("stirrups"))

        for key, default_value in defaults.items():
            item.setdefault(key, default_value)

        out.append(item)

    return out

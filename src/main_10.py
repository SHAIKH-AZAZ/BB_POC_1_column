"""
main_10.py  —  Pattern-10 Column Schedule Extractor
====================================================
Mirrors pattern-9 logic exactly:

  PATTERN-9                          PATTERN-10
  ─────────────────────────────────  ──────────────────────────────────────
  SIZE token = "300x950" (1 word)    SIZE span  = "300 X 1150" (3 tokens)
  anchor = SIZE token midpoint (X,Y) anchor = span midpoint (X), SIZE-row Y
  row_centres = Y of SIZE token      row_centres = Y of SIZE row (top of cell)
  nearest_index(y - shift, …)        nearest_index(y - y_shift, …)
  shift = half * 0.5                 y_shift = sub_spacing  (1 sub-row down)

The y_shift converts "SIZE row Y" anchors into effective "STEEL row" anchors
so the symmetric window (±y_half) covers all three sub-rows:
   SIZE  at  anchor - sub_spacing   (distance = sub_spacing ≈ 33 pt)
   STEEL at  anchor + 0             (distance = 0)
   LINKS at  anchor + sub_spacing   (distance = sub_spacing ≈ 33 pt)
Both are < y_half ≈ 47 pt → all captured.  Next floor SIZE at +99 pt → NOT captured.

Column/floor labels come from GPT-4.1-mini vision (they are rasterised images).
"""

import os
import re
import json
import tempfile
from collections import defaultdict
from tqdm import tqdm

import fitz  # PyMuPDF

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from extraction_guard import reshape_columns_to_levels
from pattern_batching import atomic_write_json, upper_levels
from vision_extractor import extract_from_image, extract_with_tools


# ─────────────────────────────────────────────────────────────────────────────
# Regexes
# ─────────────────────────────────────────────────────────────────────────────

_STEEL_RE = re.compile(r"\d+-T\d+",    re.IGNORECASE)   # "4-T16"
_TYPE_RE  = re.compile(r"\(TYPE-\d+\)", re.IGNORECASE)   # "(TYPE-26)"
_MIX_RE   = re.compile(r"^M\d+$",      re.IGNORECASE)   # "M40"
X_SHIFT = 9

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities  (identical to pattern-9)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_1d(positions: list) -> list:
    """Ratio-jump clustering — same algorithm as pattern-9."""
    if not positions:
        return []
    positions = sorted(positions)
    if len(positions) < 2:
        return [float(positions[0])]
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    ds    = sorted(diffs)
    threshold = 20.0
    if len(ds) >= 4:
        max_r, best = 1.0, 0
        for i in range(len(ds) - 1):
            if ds[i] > 0:
                r = ds[i + 1] / ds[i]
                if r > max_r:
                    max_r, best = r, i
        if max_r > 2.0:
            threshold = max((ds[best] + ds[best + 1]) / 2, threshold)
    else:
        threshold = max(ds[len(ds) // 2] * 2, threshold)
    clusters = [[positions[0]]]
    for i, d in enumerate(diffs):
        if d < threshold:
            clusters[-1].append(positions[i + 1])
        else:
            clusters.append([positions[i + 1]])
    return [sum(c) / len(c) for c in clusters]


def _median_spacing(centers: list) -> float:
    if len(centers) < 2:
        return 200.0
    diffs = sorted(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    return diffs[len(diffs) // 2]


def nearest_index(value: float, centers: list, half: float) -> int:
    """Return index of the nearest centre within ±half; -1 if none qualify."""
    best_i, best_d = -1, float("inf")
    for i, c in enumerate(centers):
        d = abs(value - c)
        if d < best_d:
            best_d, best_i = d, i
    return best_i if best_d <= half else -1


# ─────────────────────────────────────────────────────────────────────────────
# Sub-row reconciliation  (pattern-10 specific)
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile_rows(raw_clusters: list, n_floors: int):
    """
    Pattern-10 cells have 3 sub-rows: SIZE / STEEL / LINKS (~33 pt apart).
    If SIZE detection accidentally fires on multiple sub-rows, cluster_1d may
    return 27 clusters instead of 9.  Downsample by taking every k-th cluster.

    Returns (size_row_centres, sub_spacing, was_reconciled).
      was_reconciled = True  → sub-rows were detected and downsampled;
                               caller should apply y_shift = sub_spacing.
      was_reconciled = False → clusters already match floor count;
                               caller should use y_shift = 0.
    """
    sub_spacing = _median_spacing(raw_clusters)
    n = len(raw_clusters)
    if n_floors <= 0 or n <= n_floors:
        return raw_clusters, sub_spacing, False      # no downsampling needed
    step = round(n / n_floors)
    if step < 2:
        return raw_clusters, sub_spacing, False
    row_centres = raw_clusters[::step][:n_floors]
    print(f"  Reconciled {n} raw Y clusters → {len(row_centres)} floor rows "
          f"(step={step}, sub_spacing={sub_spacing:.1f} pt)")
    return row_centres, sub_spacing, True             # downsampled: sub-rows found


# ─────────────────────────────────────────────────────────────────────────────
# Data parsers
# ─────────────────────────────────────────────────────────────────────────────

def split_size_steel(line: str):
    """
    '300 X 1150 (TYPE-26) 4-T16 + 18-T12'  →  (size_part, steel_part)
    Split at TYPE token, or at first N-TNN if no TYPE token.
    """
    m = _TYPE_RE.search(line)
    if m:
        return line[:m.end()].strip(), line[m.end():].strip()
    m2 = _STEEL_RE.search(line)
    if m2:
        return line[:m2.start()].strip(), line[m2.start():].strip()
    return line.strip(), ""


def parse_size_10(txt: str) -> dict:
    """'300 X 1150'  →  {width:300, depth:None, length:1150}"""
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)(?:\s*[xX]\s*(\d+))?", txt)
    if not m:
        return {"width": None, "depth": None, "length": None}
    nums = [int(g) for g in m.groups() if g is not None]
    if len(nums) == 2:
        return {"width": nums[0], "depth": None,    "length": nums[1]}
    return  {"width": nums[0], "depth": nums[1], "length": nums[2]}


def parse_reinforcement(txt: str) -> list:
    """'4-T16 + 18-T12'  →  ['4-T16', '18-T12']"""
    result = []
    for m in re.finditer(r"(\d+)-T(\d+)", txt, re.IGNORECASE):
        entry = f"{m.group(1)}-T{m.group(2)}"
        if entry not in result:
            result.append(entry)
    return result


def parse_links(txt: str) -> dict:
    """
    Parse LINKS text in any of these formats:
      'LOC 1:-T10 -100 AND LOC 2:-T8 -150'    (spaces)
      'LOC:-T12-125 AND LOC:-T8-150'           (no spaces)
      'LOC 1:-T10-100 AND LOC 2:-T8-150'       (mixed)
      'T10@100 AND T8@150'                     (@ separator)

    →  {dia: ['T10','T8'], spacing: ['100 C/C','150 C/C']}

    Strategy:
      • Diameter  — first "T<digits>" in each AND-segment
      • Spacing   — the LAST "[-@]<2-4 digits>" in each AND-segment
                    that is NOT immediately followed by another digit
                    and falls in the range 50–500
                    (the LAST match avoids catching T12 in "-T12-125")
    """
    segments  = re.split(r"\bAND\b", txt, flags=re.IGNORECASE)
    dias: list     = []
    spacings: list = []
    for seg in segments:
        # ── Diameter ─────────────────────────────────────────────────────────
        dm = re.search(r"T(\d+)", seg, re.IGNORECASE)
        if dm:
            t = f"T{dm.group(1)}"
            if t not in dias:
                dias.append(t)

        # ── Spacing — find ALL candidate matches, take the LAST one ──────────
        # Pattern: dash or @, optional spaces, 2-4 digits NOT followed by digit
        # extract ALL valid spacings (robust)
        for m in re.finditer(r"(?<!T)(?:-|@)\s*(\d{2,4})", seg):
            val = int(m.group(1))
            if 50 <= val <= 500:
                sp = f"{val} C/C"
                if sp not in spacings:
                    spacings.append(sp)
    return {"dia": dias, "spacing": spacings}


# ─────────────────────────────────────────────────────────────────────────────
# Vision label detection  (GPT-4.1-mini)
# ─────────────────────────────────────────────────────────────────────────────

_VISION_PROMPT = """\
This image shows a structural COLUMN SCHEDULE table from a construction drawing.

Read two things:

1. COLUMN HEADERS — labels at the TOP or BOTTOM of each data column.
   Format: two letters + digits, e.g. "GC39", "PC174", "GC38", "PC171".
   List them LEFT to RIGHT as they appear.

2. FLOOR RANGE LABELS — labels on the LEFT side of the table, one per row.
   Each bounded row cell carries a range like "FLOOR_A TO FLOOR_B".
   USE THE UPPER (FIRST) LEVEL ONLY as the storey reference:
       "P06 TO ECO-DECK"  -> "P06"
       "P05 TO P06"       -> "P05"
       "GF TO P01"        -> "GF"
       "BASE TO LG"       -> "BASE"
   If a row cell shows only a single name (e.g. "GF"), keep that name as-is.
   List them strictly from TOP to BOTTOM as they appear.

Rules:
  • Only report labels that are CLEARLY VISIBLE — do not guess.
  • Include ALL column headers (there may be 10+ of them).
  • Include ALL floor range rows.

Return ONLY raw JSON, no markdown, no code fences:
{
  "col_headers":  ["GC39", "PC174", "GC38"],
  "floor_ranges": ["P06", "P05", "P04", "P03", "P02", "P01", "GF", "LG", "BASE"]
}
"""


def _vision_detect_labels(pdf_path: str) -> dict:
    print("  🔭 Running vision model to detect column/floor labels …")

    # 🔥 FIX: use output folder instead of temp directory
    output_folder = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(pdf_path))[0])
    os.makedirs(output_folder, exist_ok=True)
    imgs = convert_pdf_to_images(pdf_path, output_folder, dpi=200)
    if not imgs:
        print("  ⚠️  PDF→image conversion failed.")
        return {"col_headers": [], "floor_ranges": []}

    # Label discovery is single-shot JSON, not tool-protocol extraction.
    # extract_with_tools enforces add_column tool calls and only returns
    # {"columns": [...]} — it would discard col_headers/floor_ranges even
    # when the model produced them in the `think` step.
    raw = extract_from_image(imgs[0], _VISION_PROMPT)

    try:
        text = (raw or "").strip()
        if "```" in text:
            parts = text.split("```")
            block = parts[1] if len(parts) >= 3 else parts[0]
            if block.lower().startswith("json"):
                block = block[4:]
            text = block.strip()
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
        result = json.loads(text)
        col_headers = [
            str(h).strip().upper()
            for h in (result.get("col_headers") or result.get("column_headers") or [])
        ]
        raw_floors = [
            str(f).strip().upper()
            for f in (
                result.get("floor_ranges")
                or result.get("storey_levels")
                or result.get("level_ranges")
                or []
            )
        ]
        # Defensive: apply Pattern-1's upper-level rule even if the model
        # returned full "X TO Y" ranges. Idempotent for single-name labels.
        floor_ranges = upper_levels(raw_floors)
        if raw_floors != floor_ranges:
            print(f"  Normalised level ranges via upper-level rule: {raw_floors} -> {floor_ranges}")
        print(f"  Detected {len(col_headers)} column header(s), {len(floor_ranges)} floor range(s).")
        return {"col_headers": col_headers, "floor_ranges": floor_ranges}
    except Exception as e:
        print(f"  ⚠️  Vision parse failed: {e}\n  Raw: {(raw or '')[:300]}")
        return {"col_headers": [], "floor_ranges": []}


# ─────────────────────────────────────────────────────────────────────────────
# Grid detection  —  pattern-10 equivalent of pattern-9's _detect_grid_centres
# ─────────────────────────────────────────────────────────────────────────────

def _get_all_words(doc: fitz.Document) -> list:
    out = []
    for page in doc:
        out.extend((w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words"))
    return out


def _detect_grid_centres_10(words: list):
    """
    Pattern-10 equivalent of pattern-9's _detect_grid_centres.

    Pattern-9 finds single SIZE tokens "300x950" and uses their midpoints as
    cell anchors.  Pattern-10 has three tokens: "NNN" "X" "MMM".

    For each complete "NNN X MMM" triplet on the same text line:
      x_anchor = midpoint of the full span  =  (x0_of_NNN + x1_of_MMM) / 2
      y_anchor = Y midpoint of the SIZE line

    These (x, y) anchors are clustered exactly as in pattern-9 to find
    col_centres (X) and SIZE row Y centres.

    Returns (col_centres, raw_row_ys).
    """
    # Group tokens by snapped Y (4-pt grid) to find triplets on the same line
    snap_groups: dict = defaultdict(list)
    for x0, y0, x1, y1, txt in words:
        snap_y = round((y0 + y1) / 2 / 4) * 4
        snap_groups[snap_y].append((x0, x1, (y0 + y1) / 2, txt.strip()))

    xs: list = []
    ys: list = []

    for snap_y, tokens in snap_groups.items():
        tokens_s = sorted(tokens, key=lambda t: t[0])   # left → right
        n = len(tokens_s)
        for i in range(n - 2):
            a = tokens_s[i]        # "NNN"
            b = tokens_s[i + 1]    # "X"
            c = tokens_s[i + 2]    # "MMM"
            if (re.match(r"^\d{2,4}$", a[3]) and
                    b[3].upper() == "X" and
                    re.match(r"^\d{3,4}$", c[3])):
                # Span midpoint — same principle as pattern-9 SIZE token centre
                x_anchor = (a[0] + c[1]) / 2
                y_anchor = (a[2] + c[2]) / 2   # mean Y of "NNN" and "MMM"
                xs.append(x_anchor)
                ys.append(y_anchor)

    if not xs:
        return [], []

    col_centres  = cluster_1d(xs)
    raw_row_ys   = cluster_1d(ys)

    return col_centres, raw_row_ys


# ─────────────────────────────────────────────────────────────────────────────
# Layout detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_layout_10(doc: fitz.Document, pdf_path: str) -> dict:
    """
    Build the full grid layout.

    Mirrors detect_layout() from pattern-9:
      • X/Y centres from SIZE triplet span midpoints (cluster_1d)
      • Human-readable labels from GPT-4.1-mini vision
      • half = spacing * 0.48  (same as pattern-9)
      • y_shift = sub_spacing  (shifts Y coordinate before comparison so the
                                symmetric window covers all three sub-rows)
    """
    words = _get_all_words(doc)

    # ── Vision: actual label text ─────────────────────────────────────────────
    vision       = _vision_detect_labels(pdf_path)
    col_headers  = vision["col_headers"]    # left → right
    floor_ranges = vision["floor_ranges"]   # top  → bottom
    n_floors     = len(floor_ranges)

    # ── Grid centres from SIZE span midpoints ─────────────────────────────────
    col_centres, raw_row_ys = _detect_grid_centres_10(words)

    if not col_centres or not raw_row_ys:
        print("  ❌  No 'NNN X MMM' patterns found — page may be fully rasterised.")
        return {}

    # ── Reconcile sub-row clusters → floor-level row centres ─────────────────
    size_row_centres, sub_spacing, was_reconciled = _reconcile_rows(raw_row_ys, n_floors)

    # ── Spacing + window — SAME FORMULA as pattern-9 ─────────────────────────
    col_spacing  = _median_spacing(col_centres)
    row_spacing  = _median_spacing(size_row_centres)

    col_half = col_spacing * 0.48      # pattern-9: lap_half = spacing * 0.48
    y_half   = row_spacing * 0.48      # pattern-9: half     = spacing * 0.48

    # y_shift: only shift if sub-rows were actually detected and downsampled.
    #   was_reconciled=True  → SIZE is at the TOP of a 3-sub-row cell;
    #                          shift by sub_spacing to land window on STEEL (mid).
    #   was_reconciled=False → clusters already represent floor centres directly;
    #                          no shift needed (SIZE IS the row centre).
    # Pattern-9 equivalent: gi = nearest_index(grp_coord - shift, grp_centres, half)
    if was_reconciled:
        y_shift = sub_spacing   # one sub-row down ≈ 33 pt
    else:
        y_shift = 0.0           # SIZE is already at the row centre
        

    # ── Assign names by rank order ────────────────────────────────────────────
    col_names = [
        col_headers[i] if i < len(col_headers) else f"COL {i + 1}"
        for i in range(len(col_centres))
    ]
    row_names = [
        floor_ranges[i] if i < len(floor_ranges) else f"ROW {i + 1}"
        for i in range(len(size_row_centres))
    ]

    return {
        "col_centres":      col_centres,
        "col_names":        col_names,
        "col_half":         col_half,
        "row_centres":      size_row_centres,  # SIZE row Y anchors
        "row_names":        row_names,
        "y_half":           y_half,
        "y_shift":          y_shift,
        "row_spacing":      row_spacing,
        "col_spacing":      col_spacing,
        "sub_spacing":      sub_spacing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-page extraction  —  mirrors pattern-9's extract_from_pdf_page
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_pdf_page_10(page: fitz.Page, layout: dict) -> list:
    words = page.get_text("words")
    if not words:
        return []

    col_centres = layout["col_centres"]
    col_names   = layout["col_names"]
    row_centres = layout["row_centres"]   # SIZE row Y
    row_names   = layout["row_names"]
    col_half    = layout["col_half"]
    y_half      = layout["y_half"]
    y_shift     = layout["y_shift"]
    n_cols      = len(col_names)
    n_rows      = len(row_names)

    # ── Step 1: assign each word to (snap_y, ri, ci) ─────────────────────────
    # Pattern-9: gi = nearest_index(grp_coord - shift, grp_centres, half)
    # Pattern-10: ri = nearest_index(yc - y_shift,    row_centres, y_half)
    cell_lines: dict = defaultdict(list)

    for entry in words:
        x0, y0, x1, y1, txt = entry[0], entry[1], entry[2], entry[3], entry[4]
        txt = txt.strip()
        if not txt:
            continue
        xc = (x0 + x1) / 2
        yc = (y0 + y1) / 2

        ci = nearest_index(xc - X_SHIFT, col_centres, col_half)
        ri = nearest_index(yc - y_shift, row_centres, y_half * 2.2)
        if ci < 0 or ri < 0:
            continue

        snap_y = round(yc / 4) * 4
        cell_lines[(snap_y, ri, ci)].append((x0, txt))

    # ── Step 2: classify lines per cell ──────────────────────────────────────
    size_acc:  dict = {}
    reinf_acc: dict = defaultdict(list)
    links_acc: dict = defaultdict(list)
    mix_acc:   dict = {}

    for (snap_y, ri, ci), tokens in cell_lines.items():
        key       = (ri, ci)
        line_text = " ".join(t[1] for t in sorted(tokens, key=lambda t: t[0]))
        lu        = line_text.upper()

        # SIZE  (may carry STEEL on the same line)
        if re.search(r"\d+\s*[xX]\s*\d+", line_text):
            size_part, steel_part = split_size_steel(line_text)
            if key not in size_acc:
                size_acc[key] = parse_size_10(size_part)
            if steel_part and not reinf_acc[key]:
                r = parse_reinforcement(steel_part)
                if r:
                    reinf_acc[key] = r

        # LINKS  (check before standalone STEEL to avoid "1:-T10" mis-match)
        elif "LOC" in lu or "AND" in lu:
            links_acc[key].append(line_text)

        # Standalone STEEL
        elif _STEEL_RE.search(line_text) and not reinf_acc[key]:
            r = parse_reinforcement(line_text)
            if r:
                reinf_acc[key] = r

        # Concrete mix
        elif _MIX_RE.match(line_text.strip()):
            mix_acc[key] = line_text.strip().upper()

    # ── Step 3: assemble records ──────────────────────────────────────────────
    # Every (row, col) pair is always emitted — empty cells get null values so
    # the output preserves the full grid sequence.
    records = []
    _null_size = {"width": None, "depth": None, "length": None}
    for ri in range(n_rows):
        for ci in range(n_cols):
            key        = (ri, ci)
            links_text = " ".join(links_acc.get(key, []))
            stirrups   = parse_links(links_text) if links_text else {"dia": "", "spacing": ""}
            records.append({
                "column_no":     col_names[ci],
                "column_name":   row_names[ri],
                "size":          size_acc.get(key, _null_size),
                "reinforcement": reinf_acc.get(key, []),
                "stirrups":      stirrups,
                "mix":           mix_acc.get(key),
                "steel_grade":   None,
            })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Merge / sort / report
# ─────────────────────────────────────────────────────────────────────────────

def _merge_pages(records: list) -> list:
    seen: dict = {}
    for rec in records:
        seen[(rec["column_no"], rec["column_name"])] = rec
    return list(seen.values())


def _sort_records(records: list, layout: dict) -> list:
    col_order = {c: i for i, c in enumerate(layout["col_names"])}
    row_order = {r: i for i, r in enumerate(layout["row_names"])}
    return sorted(
        records,
        key=lambda r: (
            row_order.get(r["column_name"], 99),
            col_order.get(r["column_no"],   99),
        ),
    )


def _report(records: list, layout: dict):
    col_names = layout.get("col_names", [])
    row_names = layout.get("row_names", [])
    found: dict = defaultdict(set)
    for r in records:
        found[r["column_name"]].add(r["column_no"])
    print(f"\n📊 Extraction summary:")
    print(f"   Layout  : {len(row_names)} rows × {len(col_names)} cols"
          f" = {len(row_names) * len(col_names)} expected")
    print(f"   Records : {len(records)}")
    for row in row_names:
        missing = set(col_names) - found.get(row, set())
        if missing:
            print(f"   ⚠️  '{row}' missing: {sorted(missing)}")
    if not any(set(col_names) - found.get(r, set()) for r in row_names):
        print("   ✅  All cells extracted.")


def _update_manifest_status(manifest, batch_id, status, error=None):
    for batch in manifest["batches"]:
        if batch["id"] == batch_id:
            batch["status"] = status
            batch["error"] = error
            return

# def draw_debug_cells(page, layout, page_no, output_folder):

#     import fitz

#     pix = page.get_pixmap(dpi=150)
#     img = fitz.Pixmap(pix, 0)  # copy

#     page_rect = page.rect

#     col_centres = layout["col_centres"]
#     row_centres = layout["row_centres"]
#     col_half    = layout["col_half"]
#     y_half      = layout["y_half"]
#     y_shift     = layout["y_shift"]

#     shape = page.new_shape()

#     for ci, cx in enumerate(col_centres):
#         for ri, cy in enumerate(row_centres):

#             # 🔥 THIS IS YOUR ACTUAL CELL WINDOW
#             x_shift = X_SHIFT
#             top    = (cy - y_half * 0.5) + y_shift
#             bottom = (cy + y_half * 2.5) + y_shift
#             left   = cx - col_half + x_shift
#             right  = cx + col_half + x_shift

#             rect = fitz.Rect(left, top, right, bottom)

#             # draw rectangle
#             shape.draw_rect(rect)

#     shape.finish(color=(1, 0, 0), width=0.7)  # red boxes
#     shape.commit()

#     out_path = os.path.join(output_folder, f"debug_page_{page_no+1}.png")
#     pix = page.get_pixmap(dpi=150)
#     pix.save(out_path)

#     print(f"🟥 Debug image saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str):
    file_name     = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)

    print(f"\n🔍 Detecting layout from '{file_name}' …")
    layout = detect_layout_10(doc, pdf_path)

    if not layout:
        print("  ❌  Layout detection failed.")
        return

    print(f"  Columns ({len(layout['col_names'])}): {layout['col_names']}")
    print(f"  Rows    ({len(layout['row_names'])}): {layout['row_names']}")

    batch_folder_name = "page_batches"
    batch_folder = os.path.join(output_folder, batch_folder_name)
    os.makedirs(batch_folder, exist_ok=True)
    manifest = {
        "pattern": 10,
        "mode": "page_batches",
        "batches": [
            {
                "id": f"p{page_no + 1:02d}",
                "page": page_no + 1,
                "status": "pending",
                "file": os.path.join(batch_folder_name, f"p{page_no + 1:02d}.json"),
                "error": None,
            }
            for page_no in range(len(doc))
        ],
    }
    manifest_path = os.path.join(output_folder, "page_manifest.json")
    atomic_write_json(manifest_path, manifest)

    all_records: list = []
    for page_no in tqdm(range(len(doc)), desc=f"Processing {file_name}"):
        batch_id = f"p{page_no + 1:02d}"
        _update_manifest_status(manifest, batch_id, "running")
        atomic_write_json(manifest_path, manifest)
        page    = doc[page_no]
        # draw_debug_cells(page, layout, page_no, output_folder)
        records = extract_from_pdf_page_10(page, layout)
        status  = f"✅ {len(records)} records" if records else "⚠️  no records"
        tqdm.write(f"  Page {page_no + 1}: {status}")
        all_records.extend(records)
        atomic_write_json(os.path.join(batch_folder, f"{batch_id}.json"), {"columns": records})
        _update_manifest_status(manifest, batch_id, "done")
        atomic_write_json(manifest_path, manifest)

    final = _sort_records(_merge_pages(all_records), layout)
    _report(final, layout)

    output_file = os.path.join(output_folder, f"{file_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reshape_columns_to_levels(final), f, indent=2, ensure_ascii=False)
    print(f"✅ Saved → {output_file}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in INPUT_DIR.")
        return
    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()

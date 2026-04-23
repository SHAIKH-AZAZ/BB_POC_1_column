"""
visualize_cells.py — Draw extraction rectangles on PDF pages
============================================================
For each detected (lap_column × group_row) cell, draws the exact
rectangle that the extractor uses to capture words — so you can
visually verify the grid alignment before running the full extract.

Usage:
    python src/visualize_cells.py

Output:
    OUTPUT_DIR/<file_name>/<file_name>_cells.pdf
"""

import os
import fitz  # PyMuPDF

from config import INPUT_DIR, OUTPUT_DIR
from main_9 import detect_layout   # reuse your existing layout detection


# ── Colours (R, G, B) in 0-1 range ──────────────────────────────────────────
RECT_STROKE  = (0.0, 0.4, 1.0)   # blue border
RECT_FILL    = (0.7, 0.85, 1.0)  # light-blue fill
CENTRE_DOT   = (1.0, 0.0, 0.0)   # red dot at SIZE centre
LABEL_COLOUR = (0.0, 0.0, 0.0)   # black label text
RECT_OPACITY = 0.25               # fill transparency


def draw_cells(pdf_path: str):
    file_name     = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)
    doc    = fitz.open(pdf_path)
    layout = detect_layout(doc)

    if not layout:
        print(f"  ⚠️  No layout detected for '{file_name}' — skipping.")
        return

    lap_centres = layout["lap_centres"]
    grp_centres = layout["grp_centres"]
    lap_half    = layout["lap_half"]
    lap_names   = layout["lap_names"]
    grp_labels  = layout["group_labels"]
    grp_axis    = layout.get("grp_axis", "y")
    grp_above   = layout.get("grp_above", layout["half"])
    grp_below   = layout.get("grp_below", layout["half"])

    print(f"\n🎨 Drawing cells for '{file_name}' …")
    print(f"   {len(grp_labels)} groups × {len(lap_names)} LAPs"
          f" = {len(grp_labels) * len(lap_names)} cells per page")

    for page_no, page in enumerate(doc):
        page_h = page.rect.height
        page_w = page.rect.width
        shape  = page.new_shape()

        for gi, gc in enumerate(grp_centres):

            # ── Midpoint boundaries — prevents any overlap ───────────────────
            prev_gc = grp_centres[gi - 1] if gi > 0               else gc - 2 * grp_above
            next_gc = grp_centres[gi + 1] if gi < len(grp_centres) - 1 else gc + 2 * grp_below

            mid_above = (prev_gc + gc) / 2   # midpoint to cell above
            mid_below = (gc + next_gc) / 2   # midpoint to cell below

            for li, lc in enumerate(lap_centres):

                # ── Midpoint boundaries for LAP columns (X axis) ─────────────
                prev_lc = lap_centres[li - 1] if li > 0                  else lc - 2 * lap_half
                next_lc = lap_centres[li + 1] if li < len(lap_centres)-1 else lc + 2 * lap_half

                mid_left  = (prev_lc + lc) / 2
                mid_right = (lc + next_lc) / 2

                if grp_axis == "y":
                    x0 = lc - lap_half
                    x1 = lc + lap_half
                    y0 = gc - grp_above
                    y1 = gc + grp_below
                    cx, cy = lc, gc
                else:
                    x0 = gc - layout["half"]
                    x1 = gc + layout["half"]
                    y0 = lc - layout["lap_above"]
                    y1 = lc + layout["lap_below"]
                    cx, cy = gc, lc

                rect = fitz.Rect(x0, y0, x1, y1)

                # ── Filled rectangle ─────────────────────────────────────────
                shape.draw_rect(rect)
                shape.finish(
                    color=RECT_STROKE,
                    fill=RECT_FILL,
                    fill_opacity=RECT_OPACITY,
                    width=0.8,
                )

                # ── Red dot at SIZE centre ────────────────────────────────────
                shape.draw_circle(fitz.Point(cx, cy), 3)
                shape.finish(color=CENTRE_DOT, fill=CENTRE_DOT, width=0)

                # ── Horizontal line at centre — makes asymmetry obvious ───────
                shape.draw_line(fitz.Point(x0, cy), fitz.Point(x1, cy))
                shape.finish(color=(1.0, 0.0, 0.0), width=0.4, dashes="[2 2] 0")

                # ── Cell label ───────────────────────────────────────────────
                label = f"{grp_labels[gi]}\n{lap_names[li]}"
                page.insert_textbox(
                    fitz.Rect(x0 + 2, y0 + 2, x1 - 2, y0 + 28),
                    label,
                    fontsize=5,
                    color=LABEL_COLOUR,
                    align=fitz.TEXT_ALIGN_LEFT,
                )

        shape.commit()
        print(f"   ✅ Page {page_no + 1} annotated")

    out_path = os.path.join(output_folder, f"{file_name}_cells.pdf")
    doc.save(out_path)
    print(f"\n✅ Saved → {out_path}")


def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in INPUT_DIR.")
        return
    for pdf in pdf_files:
        draw_cells(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
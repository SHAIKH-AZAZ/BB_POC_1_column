"""Quick diagnostic for pattern-10 LINKS row attribution."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import fitz
from main_10 import _detect_grid_centres_10, _reconcile_rows, _get_all_words, _median_spacing, nearest_index, X_SHIFT

PDF = os.path.join(os.path.dirname(__file__), "..", "input", "pattern-10.pdf")
doc = fitz.open(PDF)
words = _get_all_words(doc)

col_centres, raw_row_ys = _detect_grid_centres_10(words)
print(f"col_centres: {[round(c,1) for c in col_centres]}")
print(f"raw_row_ys ({len(raw_row_ys)}): {[round(y,1) for y in raw_row_ys]}")

n_floors = 9
size_row_centres, sub_spacing, was_reconciled = _reconcile_rows(raw_row_ys, n_floors)
print(f"was_reconciled={was_reconciled}, sub_spacing={sub_spacing:.2f}")
print(f"size_row_centres ({len(size_row_centres)}): {[round(y,1) for y in size_row_centres]}")

row_spacing = _median_spacing(size_row_centres)
y_half = row_spacing * 0.48
y_shift = sub_spacing if was_reconciled else 0.0
print(f"row_spacing={row_spacing:.2f}, y_half={y_half:.2f}, y_shift={y_shift:.2f}")
print(f"window (y_half*2.2)={y_half*2.2:.2f}")

# Now inspect every word that contains LOC or AND
page = doc[0]
print("\nLINKS-like words (contain LOC / AND / T10 / T8 / T12):")
for w in page.get_text("words"):
    x0,y0,x1,y1,txt = w[0],w[1],w[2],w[3],w[4]
    tu = txt.upper().strip()
    if not tu:
        continue
    if "LOC" in tu or "AND" in tu:
        yc = (y0+y1)/2
        ri = nearest_index(yc - y_shift, size_row_centres, y_half*2.2)
        # also test no shift
        ri_noshift = nearest_index(yc, size_row_centres, y_half*2.2)
        # distance to each row centre after shift
        dists = [round(abs((yc-y_shift) - c),1) for c in size_row_centres]
        print(f"  y={yc:6.1f} txt='{txt}' -> ri={ri}  (no-shift ri={ri_noshift})  dists={dists}")

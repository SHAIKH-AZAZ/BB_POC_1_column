"""
cell_cropper.py
---------------
Detects the table grid in a column schedule image and crops
individual cells so each cell can be sent independently to
a vision model.

Requirements:
    pip install opencv-python-headless Pillow numpy --break-system-packages
"""

import cv2
import numpy as np
from PIL import Image
import os


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _cluster(positions, gap=14):
    """Merge nearby pixel positions into single median values."""
    if not positions:
        return []
    arr = sorted(set(int(p) for p in positions))
    groups = [[arr[0]]]
    for p in arr[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.median(g)) for g in groups]


def _filter_min_spacing(lines, min_spacing):
    """
    Remove lines that are closer than min_spacing to the previous
    kept line.  This eliminates spurious lines from drawing details
    (stirrup lines, dimension arrows, text) that appear INSIDE cells.
    """
    if len(lines) < 2:
        return lines
    kept = [lines[0]]
    for line in lines[1:]:
        if line - kept[-1] >= min_spacing:
            kept.append(line)
    return kept


def _try_detect_lines(gray, thresh_val,
                      h_kernel_w, h_fill_frac,
                      v_kernel_h, v_fill_frac,
                      min_row_h, min_col_w,
                      gap=14):
    """One detection attempt. Returns (h_lines, v_lines)."""
    h, w = gray.shape

    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Horizontal lines
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    h_proj = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk), axis=1)
    h_lines = _cluster(
        np.where(h_proj > w * 255 * h_fill_frac)[0].tolist(), gap=gap
    )
    h_lines = _filter_min_spacing(h_lines, min_row_h)

    # Vertical lines
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    v_proj = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk), axis=0)
    v_lines = _cluster(
        np.where(v_proj > h * 255 * v_fill_frac)[0].tolist(), gap=gap
    )
    v_lines = _filter_min_spacing(v_lines, min_col_w)

    return h_lines, v_lines


# ──────────────────────────────────────────────
# PUBLIC: DETECT GRID
# ──────────────────────────────────────────────

def detect_grid(image_path, debug=True):
    """
    Detect horizontal and vertical table-border lines.

    Key behaviour
    -------------
    • Spurious lines from drawing internals (stirrup symbols, dimension
      lines, text leaders) are removed with a minimum-spacing filter so
      that the GROUND FLOOR row is never split in two.
    • Falls back through progressively relaxed parameters until a valid
      grid (≥ 2 h-lines and ≥ 2 v-lines) is found.

    Returns (h_lines, v_lines).
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = gray.shape
    if debug:
        print(f"    🖼  Image size : {w} × {h} px")

    # Minimum spacing between real grid lines.
    # A column schedule has at most ~10 rows and ~15 cols, so real cells
    # are at least 1/15 of the image dimension.
    min_row_h = max(60,  h // 15)   # e.g. 3500px/15 ≈ 233 px
    min_col_w = max(30,  w // 20)   # e.g. 2500px/20 ≈ 125 px

    if debug:
        print(f"    📏  Min row gap : {min_row_h} px  |  min col gap : {min_col_w} px")

    # Parameter sets: (thresh, h_kernel_w, h_fill, v_kernel_h, v_fill)
    param_sets = [
        (200, max(w // 10, 40), 0.15, max(h // 10, 40), 0.05),
        (200, max(w // 10, 40), 0.08, max(h // 10, 40), 0.03),
        (200, max(w // 15, 25), 0.08, max(h // 15, 25), 0.03),
        (160, max(w // 15, 25), 0.06, max(h // 15, 25), 0.02),
        (160, max(w // 20, 20), 0.05, max(h // 20, 20), 0.02),
        (127, max(w // 30, 15), 0.04, max(h // 30, 15), 0.01),
    ]

    best_h, best_v = [], []

    for idx, (tv, hkw, hff, vkh, vff) in enumerate(param_sets):
        h_lines, v_lines = _try_detect_lines(
            gray, tv, hkw, hff, vkh, vff, min_row_h, min_col_w
        )
        if debug:
            print(
                f"    [pass {idx}] tv={tv} hk={hkw} hf={hff:.2f} "
                f"→ {len(h_lines)} h  |  "
                f"vk={vkh} vf={vff:.2f} → {len(v_lines)} v"
            )

        if len(h_lines) >= 2 and len(v_lines) >= 2:
            if debug:
                print(f"    ✅  Grid found  h={h_lines}  v={v_lines}")
            return h_lines, v_lines

        if len(h_lines) + len(v_lines) > len(best_h) + len(best_v):
            best_h, best_v = h_lines, v_lines

    # Otsu fallback
    if debug:
        print("    ⚠️   Trying Otsu binarisation …")

    _, binary_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    for hkw, vkh, fill in [
        (max(w // 15, 25), max(h // 15, 25), 0.04),
        (max(w // 25, 15), max(h // 25, 15), 0.02),
    ]:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hkw, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vkh))

        ho = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, hk)
        vo = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, vk)

        h_lines = _filter_min_spacing(
            _cluster(np.where(np.sum(ho, axis=1) > w * 255 * fill)[0].tolist()),
            min_row_h
        )
        v_lines = _filter_min_spacing(
            _cluster(np.where(np.sum(vo, axis=0) > h * 255 * fill)[0].tolist()),
            min_col_w
        )

        if debug:
            print(f"    [Otsu hk={hkw} vk={vkh} f={fill}] "
                  f"→ {len(h_lines)} h  {len(v_lines)} v")

        if len(h_lines) >= 2 and len(v_lines) >= 2:
            if debug:
                print(f"    ✅  Grid found via Otsu  h={h_lines}  v={v_lines}")
            return h_lines, v_lines

    if debug:
        print(f"    ❌  Detection failed. Best: h={best_h}  v={best_v}")

    return best_h, best_v


# ──────────────────────────────────────────────
# PUBLIC: CROP ALL CELLS
# ──────────────────────────────────────────────

def crop_all_cells(image_path, crop_dir, padding=8, debug=True):
    """
    Detect the grid and save every cell as a separate PNG.

    Returns
    -------
    cells      : list[list[str]]  cells[row][col] = path to crop PNG
    h_lines    : list[int]
    v_lines    : list[int]
    col_widths : list[int]
    """
    os.makedirs(crop_dir, exist_ok=True)

    h_lines, v_lines = detect_grid(image_path, debug=debug)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return [], h_lines, v_lines, []

    img_pil = Image.open(image_path).convert("RGB")
    iw, ih  = img_pil.size

    col_widths = [v_lines[c + 1] - v_lines[c] for c in range(len(v_lines) - 1)]

    cells = []
    for r in range(len(h_lines) - 1):
        y1, y2 = h_lines[r], h_lines[r + 1]
        row_cells = []

        for c in range(len(v_lines) - 1):
            x1, x2 = v_lines[c], v_lines[c + 1]

            left   = max(0, x1 + padding)
            top    = max(0, y1 + padding)
            right  = min(iw, x2 - padding)
            bottom = min(ih, y2 - padding)

            crop = (img_pil.crop((left, top, right, bottom))
                    if right > left and bottom > top
                    else Image.new("RGB", (20, 20), (255, 255, 255)))

            path = os.path.join(crop_dir, f"cell_r{r:02d}_c{c:02d}.png")
            crop.save(path, "PNG")
            row_cells.append(path)

        cells.append(row_cells)

    if debug:
        print(f"    📦  Cropped {len(cells)} rows × "
              f"{len(cells[0]) if cells else 0} cols")
        if cells:
            print(f"    Row heights : "
                  f"{[h_lines[r+1]-h_lines[r] for r in range(len(h_lines)-1)]}")
            print(f"    Col widths  : {col_widths}")

    return cells, h_lines, v_lines, col_widths


# ──────────────────────────────────────────────
# PUBLIC: COLUMN / LABEL HELPERS
# ──────────────────────────────────────────────

def get_data_col_indices(col_widths, image_width, min_fraction=0.08):
    """
    Return indices of columns wide enough to be data columns.
    Narrow label / mix columns are excluded.
    """
    min_w = image_width * min_fraction
    return [i for i, w in enumerate(col_widths) if w >= min_w]


def get_label_col_index(_col_widths):
    """First (leftmost) column is always the row-label column."""
    return 0
import sys
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

def cluster_lines(lines, gap=12):
    if not lines: return []
    lines = sorted(lines)
    clusters, group = [], [lines[0]]
    for x in lines[1:]:
        if x - group[-1] <= gap:
            group.append(x)
        else:
            clusters.append(int(round(np.mean(group))))
            group = [x]
    clusters.append(int(round(np.mean(group))))
    return clusters

def _gradient_peaks(arr2d, threshold, min_distance):
    from scipy.signal import find_peaks
    grad = np.mean(np.abs(np.diff(arr2d.astype(float), axis=0)), axis=1)
    if grad.max() < 1e-6: return []
    norm_grad = grad / (grad.max() + 1e-9)
    abs_thr = max(threshold, norm_grad.mean() + norm_grad.std())
    peaks, _ = find_peaks(norm_grad, height=abs_thr, distance=min_distance)
    return peaks.tolist()

def _darkness_peaks(arr2d, threshold, min_distance):
    from scipy.signal import find_peaks
    darkness = np.mean(arr2d < 100, axis=1)
    peaks, _ = find_peaks(darkness, height=threshold, distance=min_distance)
    return peaks.tolist()

img = convert_from_path("../input/pattern-13.pdf", dpi=300)[0]
arr = np.array(img, dtype=float)

# Mock h_grid bounds
gray = np.mean(arr[:, :, :3], axis=2)
h_raw = _darkness_peaks(gray, 0.15, 60)
h_grid = cluster_lines(h_raw)
if len(h_grid) < 2: h_grid = [0, img.height]
y0, y1 = h_grid[0], h_grid[-1]

_arr_v = arr[y0:y1, :, :3]
_gray_v = np.mean(_arr_v, axis=2)

v_dark = _darkness_peaks(_gray_v.T, 0.15, 30)
print(f"v_dark (0.15): {len(v_dark)}")

for thr in [0.15, 0.30, 0.40, 0.50, 0.60]:
    v_grad = _gradient_peaks(_gray_v.T, thr, 30)
    print(f"v_grad ({thr}): {len(v_grad)}")

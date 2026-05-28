"""
image_tools.py
==============

Shared image-cropping and zooming utilities available to EVERY
main_N.py module.

Two complementary "tools" are exposed here:

  1. crop_upscale_*()          — CODE-DRIVEN pre-crop. Python decides
                                  the bbox, crops the image, optionally
                                  upscales to a minimum size, and hands
                                  the cropped bytes back to the caller
                                  for further use (save to disk, send to
                                  vision model, etc).

  2. zoom_and_extract()        — A higher-level helper that crops a
                                  region AND sends it to the vision
                                  model in one call. Returns parsed
                                  JSON. Useful when a main_N.py wants
                                  the same effect as the model-driven
                                  zoom_region tool but with FULL
                                  Python control over which region and
                                  which prompt.

  3. (model-driven zoom_region) The on-the-fly zoom_region tool inside
                                  `vision_extractor.extract_with_tools`
                                  is unchanged. Any pattern that routes
                                  through extract_with_tools already
                                  has this; this module documents how
                                  to invoke it indirectly.

Coordinate conventions
----------------------
  - Functions accept EITHER an absolute pixel bbox OR a normalized
    (0.0–1.0) bbox via the `normalized=True` flag.
  - Y origin = top-left (PIL convention).

Typical usage in any main_N.py
------------------------------
    from image_tools import crop_upscale_path, zoom_and_extract

    # Code-driven crop -> save to disk
    crop_upscale_path(
        image_path, x1=0.10, y1=0.20, x2=0.40, y2=0.45,
        normalized=True,
        out_path="output/foo/crop.png",
    )

    # Code-driven zoom + vision extraction
    result = zoom_and_extract(
        image_path,
        bbox=(0.10, 0.20, 0.40, 0.45),
        prompt="Read the column ID from this strip and return JSON.",
        normalized=True,
        use_tools=False,            # single-shot vision call
    )
"""

import base64
import io
import json
import os
from typing import Optional, Tuple, Union

from PIL import Image


# ---------------------------------------------------------------------------
# 1. Crop + upscale primitives
# ---------------------------------------------------------------------------

def _open_image(image_or_path: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    return Image.open(image_or_path).convert("RGB")


def _resolve_bbox(
    img: Image.Image,
    x1: float, y1: float, x2: float, y2: float,
    normalized: bool,
) -> Tuple[int, int, int, int]:
    w, h = img.size
    if normalized:
        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y2 = max(0.0, min(1.0, float(y2)))
        left   = int(x1 * w)
        top    = int(y1 * h)
        right  = int(x2 * w)
        bottom = int(y2 * h)
    else:
        left   = int(x1)
        top    = int(y1)
        right  = int(x2)
        bottom = int(y2)

    # Defensive: ensure ordering and minimum 20×20 px
    if right <= left:
        right = min(w, left + 20)
    if bottom <= top:
        bottom = min(h, top + 20)
    left   = max(0, left)
    top    = max(0, top)
    right  = min(w, right)
    bottom = min(h, bottom)
    return left, top, right, bottom


def crop_image(
    image_or_path: Union[str, Image.Image],
    x1: float, y1: float, x2: float, y2: float,
    *,
    normalized: bool = False,
    padding: int = 0,
) -> Image.Image:
    """Plain crop — no upscaling. Returns a PIL Image."""
    img = _open_image(image_or_path)
    left, top, right, bottom = _resolve_bbox(img, x1, y1, x2, y2, normalized)
    if padding:
        w, h = img.size
        left   = max(0, left - padding)
        top    = max(0, top - padding)
        right  = min(w, right + padding)
        bottom = min(h, bottom + padding)
    return img.crop((left, top, right, bottom))


def crop_upscale(
    image_or_path: Union[str, Image.Image],
    x1: float, y1: float, x2: float, y2: float,
    *,
    normalized: bool = False,
    min_longest: int = 1200,
    upscale: Optional[int] = None,
    padding: int = 0,
) -> Image.Image:
    """Crop a region and upscale so the longest side is at least
    `min_longest` px (or by a fixed integer factor if `upscale` given).

    Returns a PIL Image.
    """
    cropped = crop_image(
        image_or_path, x1, y1, x2, y2,
        normalized=normalized,
        padding=padding,
    )
    if upscale and upscale > 1:
        cropped = cropped.resize(
            (cropped.width * upscale, cropped.height * upscale),
            Image.LANCZOS,
        )
    else:
        longest = max(cropped.size)
        if longest and longest < min_longest:
            scale = min_longest / longest
            cropped = cropped.resize(
                (int(cropped.width * scale), int(cropped.height * scale)),
                Image.LANCZOS,
            )
    return cropped


def crop_upscale_path(
    image_path: str,
    x1: float, y1: float, x2: float, y2: float,
    *,
    normalized: bool = False,
    min_longest: int = 1200,
    upscale: Optional[int] = None,
    padding: int = 0,
    out_path: Optional[str] = None,
) -> str:
    """Crop+upscale a region of a file-on-disk image and (optionally)
    save the result to `out_path`. Returns the saved path (or a
    temp-path-like default if `out_path` is None — currently raises so
    callers must specify out_path)."""
    if not out_path:
        raise ValueError("crop_upscale_path requires out_path")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = crop_upscale(
        image_path, x1, y1, x2, y2,
        normalized=normalized,
        min_longest=min_longest,
        upscale=upscale,
        padding=padding,
    )
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# 2. PIL Image <-> base64 (for sending crops to the vision API)
# ---------------------------------------------------------------------------

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to base64 (raw, no data: URL prefix)."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_upscale_b64(
    image_or_path: Union[str, Image.Image],
    x1: float, y1: float, x2: float, y2: float,
    *,
    normalized: bool = False,
    min_longest: int = 1200,
    upscale: Optional[int] = None,
    padding: int = 0,
) -> str:
    """Crop + upscale + base64-encode in one call."""
    return image_to_base64(crop_upscale(
        image_or_path, x1, y1, x2, y2,
        normalized=normalized,
        min_longest=min_longest,
        upscale=upscale,
        padding=padding,
    ))


# ---------------------------------------------------------------------------
# 3. Code-driven "zoom + extract" — high-level wrapper
# ---------------------------------------------------------------------------
#
# This is the Python equivalent of the model-driven zoom_region tool.
# Use it when a main_N.py wants to PRE-CROP a region and send it to the
# vision model with full Python control over the prompt and region.
#
# `use_tools=False` (default)  →  single-shot call via extract_from_image
# `use_tools=True`             →  full tool-protocol call via extract_with_tools
#                                  (the model can further zoom inside)

def zoom_and_extract(
    image_path: str,
    bbox: Tuple[float, float, float, float],
    prompt: str,
    *,
    normalized: bool = False,
    min_longest: int = 1200,
    upscale: Optional[int] = None,
    padding: int = 0,
    use_tools: bool = False,
    trace_key: Optional[str] = None,
    crop_out_path: Optional[str] = None,
):
    """Crop a region from `image_path`, optionally save it for audit
    (`crop_out_path`), then send it to the vision model with `prompt`.

    Returns:
      - dict / list (parsed JSON) if the model returned valid JSON
      - str (raw text)            otherwise

    `use_tools=False` (default) is a single-shot call → faster, simpler.
    `use_tools=True` enables the full tool protocol → model can call
    `zoom_region` to drill deeper inside the cropped region.
    """
    # Lazy import to avoid circular deps with vision_extractor
    from vision_extractor import extract_from_image, extract_with_tools

    # Crop + (optional) persist for inspection.
    if crop_out_path:
        crop_upscale_path(
            image_path, *bbox,
            normalized=normalized,
            min_longest=min_longest,
            upscale=upscale,
            padding=padding,
            out_path=crop_out_path,
        )
        crop_path = crop_out_path
    else:
        # Write to a sibling temp file so vision can read by path.
        cropped = crop_upscale(
            image_path, *bbox,
            normalized=normalized,
            min_longest=min_longest,
            upscale=upscale,
            padding=padding,
        )
        base, _ = os.path.splitext(image_path)
        crop_path = f"{base}_zoom_tmp.png"
        cropped.save(crop_path, "PNG")

    # Call vision.
    if use_tools:
        raw = extract_with_tools(crop_path, prompt, trace_key=trace_key)
    else:
        raw = extract_from_image(crop_path, prompt)

    # Try to parse JSON; fall back to raw text.
    text = str(raw or "").strip()
    if "```" in text:
        parts = text.split("```")
        block = parts[1] if len(parts) >= 3 else parts[0]
        if block.lower().startswith("json"):
            block = block[4:]
        text = block.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return raw


# ---------------------------------------------------------------------------
# 4. Back-compat aliases (let main_11 / main_12 / main_13 keep their
#    local _crop_upscale calls working with the same signature).
# ---------------------------------------------------------------------------

def _crop_upscale_compat(
    img: Image.Image,
    x0: int, y0: int, x1: int, y1: int,
    upscale: int = 2,
) -> Image.Image:
    """Drop-in replacement for the local `_crop_upscale` defined in
    main_11.py / main_12.py / main_13.py. Keep the same call signature
    so they can swap with a one-line import change."""
    return crop_upscale(
        img,
        x0, y0, x1, y1,
        normalized=False,
        upscale=upscale,
        min_longest=0,  # rely on upscale factor
    )

"""Photo-quality checks and the "where we found the spots" overlay.

The thresholds below are starting values that were checked against synthetic
photos only. Before relying on them, run scripts/calibrate_quality.py on a
folder of real good and bad photos taken with the phones your farmers use, and
adjust THRESHOLDS.

Issue codes (the web page turns these into Kannada / Hindi / English text):
    blur             leaf is out of focus or the image is too small to be sharp
    dark             the whole photo is too dark
    glare            bright reflections on the leaf (can look like powdery mildew)
    small_leaf       the leaf fills too little of the frame
    multiple_leaves  more than one leaf or object found
    low_res          the photo has too few pixels
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage

# For every metric except glare_frac, LOWER is worse.
THRESHOLDS = {
    "sharpness": {"minor": 15.0, "severe": 6.0},      # variance of the Laplacian inside the leaf
    "image_mean_v": {"minor": 0.28, "severe": 0.18},  # average brightness of the photo (0..1)
    "glare_frac": {"minor": 0.05, "severe": 0.15},    # share of the leaf that is blown out (higher is worse)
    "leaf_fraction": {"minor": 0.15, "severe": 0.06}, # share of the frame covered by the leaf
    "long_side_px": {"minor": 480, "severe": 256},    # longest side of the photo as received
}

# A second object counts as "another leaf" when it is at least this share of the biggest one
# (and at least 2% of the frame).
SECOND_LEAF_RATIO = 0.15

SEVERITY_ORDER = {"none": 0, "minor": 1, "severe": 2}


def padded_box(w: int, h: int, target: int) -> tuple[int, int, int, int]:
    """Where the resized photo sits inside the square canvas used for analysis.

    Returns (x0, y0, width, height). main._resize_padded uses the same function,
    so the overlay lines up with the photo.
    """
    if w <= 0 or h <= 0:
        return 0, 0, target, target
    scale = target / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return (target - new_w) // 2, (target - new_h) // 2, new_w, new_h


def laplacian_variance(gray255: np.ndarray, mask: np.ndarray) -> float:
    """Sharpness: variance of the Laplacian of the grey image, measured inside the mask.

    The mask is shrunk first so the leaf edge against the background does not
    count as sharpness.
    """
    lap = ndimage.laplace(gray255.astype(np.float32))
    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=2)
    if int(inner.sum()) < 200:
        inner = mask
    if int(inner.sum()) < 50:
        inner = np.ones_like(mask, dtype=bool)
    return float(lap[inner].var())


def _grade(value: float, limits: dict, higher_is_worse: bool = False) -> str:
    if higher_is_worse:
        if value > limits["severe"]:
            return "severe"
        if value > limits["minor"]:
            return "minor"
        return "none"
    if value < limits["severe"]:
        return "severe"
    if value < limits["minor"]:
        return "minor"
    return "none"


def assess(*, orig_w: int, orig_h: int, rgb: np.ndarray, v: np.ndarray, s: np.ndarray,
           leaf_mask: np.ndarray, leaf_fraction: float, component_sizes: list,
           pad_box: tuple[int, int, int, int], frame_px: int) -> dict:
    """Grade the photo. Returns {"severity", "issues", "metrics"}."""
    x0, y0, bw, bh = pad_box
    gray255 = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * 255.0

    image_mean_v = float(v[y0:y0 + bh, x0:x0 + bw].mean()) if bw > 0 and bh > 0 else float(v.mean())
    sharpness = laplacian_variance(gray255, leaf_mask)

    region = ndimage.binary_dilation(leaf_mask, iterations=3)
    # A reflection is neutral white. Pale pink, peach or straw has a tint (saturation 0.10 or more) and is dead
    # tissue, which main._pale_dead_tissue counts as damage, so it must not also be called glare.
    blown = (v > 0.93) & (s < 0.10) & region
    glare_frac = float(blown.sum()) / max(int(region.sum()), 1)

    long_side = max(int(orig_w), int(orig_h))

    metrics = {
        "sharpness": round(sharpness, 2),
        "image_mean_v": round(image_mean_v, 3),
        "glare_frac": round(glare_frac, 3),
        "leaf_fraction": round(float(leaf_fraction), 3),
        "long_side_px": long_side,
    }

    issues = []

    def add(code: str, severity: str):
        if severity != "none":
            issues.append({"code": code, "severity": severity})

    add("blur", _grade(sharpness, THRESHOLDS["sharpness"]))
    add("dark", _grade(image_mean_v, THRESHOLDS["image_mean_v"]))
    add("glare", _grade(glare_frac, THRESHOLDS["glare_frac"], higher_is_worse=True))
    add("small_leaf", _grade(leaf_fraction, THRESHOLDS["leaf_fraction"]))
    add("low_res", _grade(long_side, THRESHOLDS["long_side_px"]))

    # Several leaves is only ever a warning: a spotted leaf can split into two pieces.
    sizes = sorted((int(x) for x in component_sizes), reverse=True)
    if len(sizes) >= 2 and sizes[1] >= max(SECOND_LEAF_RATIO * sizes[0], 0.02 * frame_px * frame_px):
        add("multiple_leaves", "minor")

    severity = "none"
    for issue in issues:
        if SEVERITY_ORDER[issue["severity"]] > SEVERITY_ORDER[severity]:
            severity = issue["severity"]
    return {"severity": severity, "issues": issues, "metrics": metrics}


def make_overlay_png(lesion: np.ndarray, chlor: np.ndarray,
                     pad_box: tuple[int, int, int, int]) -> str:
    """A transparent PNG (data URL) with the spots in red and the yellowing in yellow.

    It is cropped to the photo area, so the page can draw it straight over the
    photo at any size.
    """
    h, w = lesion.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[chlor] = (255, 214, 0, 110)
    rgba[lesion] = (230, 40, 30, 170)
    ring = ndimage.binary_dilation(lesion, iterations=1) & ~lesion
    rgba[ring] = (255, 255, 255, 230)
    x0, y0, bw, bh = pad_box
    crop = np.ascontiguousarray(rgba[y0:y0 + bh, x0:x0 + bw])
    buf = io.BytesIO()
    Image.fromarray(crop, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def cap_confidence(confidence: str, severity: str, is_healthy: bool) -> str:
    """A bad photo can lower the confidence but never raise it."""
    if is_healthy:
        return confidence
    if severity == "severe":
        return "low"
    if severity == "minor" and confidence == "high":
        return "moderate"
    return confidence
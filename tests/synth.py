"""Synthetic leaf photos for tests. They only exercise the code paths: they say nothing about
how the app does on real field photos (see README, "Measuring accuracy")."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

RNG_SEED = 7


def _leaf_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, angle: float = 0.3) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ca, sa = np.cos(angle), np.sin(angle)
    x = (xx - cx) * ca + (yy - cy) * sa
    y = -(xx - cx) * sa + (yy - cy) * ca
    return (x / rx) ** 2 + (y / ry) ** 2 <= 1.0


def make_leaf(size=(800, 600), spots: int = 0, leaf_scale: float = 0.8, second_leaf: bool = False,
              seed: int = RNG_SEED) -> Image.Image:
    """A textured green leaf on a light background, optionally with brown spots."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), (232, 228, 214), np.float32)           # light soil/paper background
    mask = _leaf_mask(h, w, w * 0.5, h * 0.5, w * 0.5 * leaf_scale, h * 0.5 * leaf_scale)
    if second_leaf:
        mask |= _leaf_mask(h, w, w * 0.17, h * 0.2, w * 0.16, h * 0.16, 0.8)
        mask |= _leaf_mask(h, w, w * 0.85, h * 0.82, w * 0.14, h * 0.14, 0.1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    veins = 0.5 + 0.5 * np.sin(xx / 9.0 + yy / 23.0)
    texture = rng.normal(0, 7, (h, w)).astype(np.float32)
    green = np.stack([60 + 14 * veins, 135 + 22 * veins, 52 + 8 * veins], axis=-1) + texture[..., None]
    img[mask] = green[mask]
    for _ in range(spots):
        sx = rng.uniform(w * 0.25, w * 0.75)
        sy = rng.uniform(h * 0.25, h * 0.75)
        r = rng.uniform(w * 0.012, w * 0.03)
        spot = _leaf_mask(h, w, sx, sy, r * 1.6, r, rng.uniform(0, 3)) & mask
        img[spot] = np.array([118, 74, 38], np.float32) + texture[spot][:, None] * 0.5
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def make_spotted_leaf(size=(1200, 900), n: int = 30, rx: float = 12, ry: float = 7, kind: str = "solid",
                      seed: int = RNG_SEED):
    """A leaf with `n` small spots of half-size rx x ry pixels. Returns (image, true_spot_share), where the share
    is the spot area divided by the leaf area, so a test can compare it with what the app measured.
    kind="target" gives a pale centre inside a brown margin, like a rice brown spot lesion."""
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.asarray(make_leaf(size=size, spots=0, leaf_scale=0.85, seed=seed)).astype(np.float32)
    leaf = arr[..., 1] > arr[..., 0] + 30
    truth = np.zeros((h, w), bool)
    centres = []
    for _ in range(n):
        for _try in range(200):                                  # keep spots apart so they do not merge
            cx, cy = rng.uniform(w * 0.25, w * 0.75), rng.uniform(h * 0.25, h * 0.75)
            if all((cx - x) ** 2 + (cy - y) ** 2 >= (3 * max(rx, ry)) ** 2 for x, y in centres):
                break
        centres.append((cx, cy))
        ang = rng.uniform(0, 3)
        r = int(max(rx, ry)) + 3
        x0, x1, y0, y1 = max(int(cx) - r, 0), min(int(cx) + r, w), max(int(cy) - r, 0), min(int(cy) + r, h)
        sub = (slice(y0, y1), slice(x0, x1))
        m = _leaf_mask(y1 - y0, x1 - x0, cx - x0, cy - y0, rx, ry, ang) & leaf[sub]
        patch = arr[sub]
        patch[m] = (112, 70, 36) if kind == "target" else (118, 74, 38)
        if kind == "target":
            inner = _leaf_mask(y1 - y0, x1 - x0, cx - x0, cy - y0, rx * 0.55, ry * 0.55, ang) & leaf[sub]
            patch[inner] = (176, 160, 128)
        truth[sub] |= m
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return im, float(truth.sum()) / max(int(leaf.sum()), 1)


def blurred(im: Image.Image, radius: float = 6.0) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius))


def darkened(im: Image.Image, factor: float = 0.22) -> Image.Image:
    return Image.fromarray(np.clip(np.asarray(im, np.float32) * factor, 0, 255).astype(np.uint8))


def with_glare(im: Image.Image, blobs: int = 6) -> Image.Image:
    arr = np.asarray(im).copy()
    h, w, _ = arr.shape
    rng = np.random.default_rng(3)
    for _ in range(blobs):
        cx, cy = rng.uniform(w * 0.3, w * 0.7), rng.uniform(h * 0.3, h * 0.7)
        yy, xx = np.mgrid[0:h, 0:w]
        arr[(xx - cx) ** 2 + (yy - cy) ** 2 <= (w * 0.05) ** 2] = (252, 252, 250)
    return Image.fromarray(arr)


def blank(size=(640, 480), colour=(240, 240, 240)) -> Image.Image:
    return Image.new("RGB", size, colour)


def to_jpeg(im: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()

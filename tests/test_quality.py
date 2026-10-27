import base64
import io

import numpy as np
from PIL import Image

import main
import quality
import synth


def codes(im):
    q = main.analyze_leaf(im, True)["_extras"]["quality"]
    return q, {i["code"]: i["severity"] for i in q["issues"]}


def test_good_photo_has_no_issues():
    q, c = codes(synth.make_leaf(size=(1200, 900), spots=8))
    assert q["severity"] == "none" and c == {}


def test_blurry_photo_is_flagged():
    _, c = codes(synth.blurred(synth.make_leaf(size=(1200, 900), spots=8), 6))
    assert "blur" in c


def test_dark_photo_is_flagged():
    _, c = codes(synth.darkened(synth.make_leaf(spots=6), 0.22))
    assert c.get("dark") == "severe"


def test_glare_is_flagged():
    _, c = codes(synth.with_glare(synth.make_leaf(size=(1200, 900), spots=6)))
    assert "glare" in c


def test_tiny_photo_is_flagged_low_res():
    _, c = codes(synth.make_leaf(size=(200, 150), spots=2))
    assert c.get("low_res") == "severe"


def test_small_leaf_is_flagged():
    _, c = codes(synth.make_leaf(size=(1200, 900), spots=2, leaf_scale=0.22))
    assert c.get("small_leaf") == "severe"


def test_two_separate_leaves_are_flagged():
    _, c = codes(synth.make_leaf(size=(1200, 900), spots=4, leaf_scale=0.5, second_leaf=True))
    assert c.get("multiple_leaves") == "minor"


def test_one_leaf_is_not_flagged_as_several():
    _, c = codes(synth.make_leaf(size=(1200, 900), spots=4, leaf_scale=0.6))
    assert "multiple_leaves" not in c


def test_a_photo_can_only_lower_confidence():
    assert quality.cap_confidence("high", "none", False) == "high"
    assert quality.cap_confidence("high", "minor", False) == "moderate"
    assert quality.cap_confidence("moderate", "minor", False) == "moderate"
    assert quality.cap_confidence("high", "severe", False) == "low"
    assert quality.cap_confidence("low", "none", False) == "low"
    # A healthy-looking leaf keeps its result here: the caller turns it into "inconclusive".
    assert quality.cap_confidence("high", "severe", True) == "high"


def test_padded_box_matches_the_resize_used_for_analysis():
    for size in [(1200, 900), (900, 1200), (288, 288), (5, 3000), (4000, 3)]:
        im = Image.new("RGB", size, (10, 200, 10))
        canvas = main._resize_padded(im, main.IMAGE_SIZE)
        x0, y0, w, h = quality.padded_box(size[0], size[1], main.IMAGE_SIZE)
        arr = np.asarray(canvas)
        grey = np.all(arr == 128, axis=-1)
        assert not grey[y0:y0 + h, x0:x0 + w].all() or w * h <= 1
        assert 0 <= x0 and 0 <= y0 and x0 + w <= main.IMAGE_SIZE and y0 + h <= main.IMAGE_SIZE


def test_overlay_is_a_transparent_png_the_size_of_the_photo_area():
    im = synth.make_leaf(size=(1200, 900), spots=8)
    ex = main.analyze_leaf(im, True)["_extras"]
    url = ex["overlay"]
    assert url.startswith("data:image/png;base64,")
    png = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    _, _, w, h = quality.padded_box(1200, 900, main.IMAGE_SIZE)
    assert png.size == (w, h) and png.mode == "RGBA"
    alpha = np.asarray(png)[..., 3]
    assert alpha.max() > 0, "spots should be marked"
    assert (alpha == 0).mean() > 0.5, "most of the photo should stay uncovered"


def test_clean_leaf_overlay_is_nearly_empty():
    ex = main.analyze_leaf(synth.make_leaf(size=(1200, 900), spots=0), True)["_extras"]
    png = Image.open(io.BytesIO(base64.b64decode(ex["overlay"].split(",", 1)[1])))
    assert (np.asarray(png)[..., 3] > 0).mean() < 0.02


def test_analyze_without_extras_is_unchanged():
    f = main.analyze_leaf(synth.make_leaf(spots=3))
    assert "_extras" not in f and "lesion_ratio" in f


def test_segment_stats_keep_component_sizes():
    im = synth.make_leaf(size=(1200, 900), spots=2, leaf_scale=0.5, second_leaf=True)
    f = main.analyze_leaf(im)
    sizes = f["_mask_stats"]["component_sizes"]
    assert len(sizes) >= 2 and sizes == sorted(sizes, reverse=True)

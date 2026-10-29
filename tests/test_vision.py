"""What the app measures on the leaf, and the leaf damage estimate built from it.

These use made-up leaves. They show the code does what it says (a dark gap is not a spot, a pale dead patch is
damage, a shadow is not damage). They do not show how well it does on field photos."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

import main
import synth
from conftest import analyze

W, H = 1200, 900


def healthy():
    return synth.make_leaf(size=(W, H), spots=0)


def paint(im, box, colour):
    im = im.copy()
    ImageDraw.Draw(im).rectangle(box, fill=colour)
    return im


# ------------------------------------------------------------ black gaps between leaves are not spots

def test_a_black_gap_inside_the_leaf_outline_is_not_a_spot():
    gap = paint(healthy(), (540, 380, 660, 460), (18, 16, 14))                  # ~3% of the leaf, enclosed by it
    f = main.analyze_leaf(gap)
    assert f["spot_count"] == 0 and f["lesion_ratio"] < 0.005, f


def test_a_small_black_speck_is_still_a_spot():
    speck = paint(healthy(), (540, 380, 570, 410), (18, 16, 14))                # a lesion-sized dark spot
    f = main.analyze_leaf(speck)
    assert f["spot_count"] >= 1 and f["lesion_ratio"] > 0


def test_the_gap_rule_leaves_the_leaf_itself_alone():
    plain, gap = main.analyze_leaf(healthy()), main.analyze_leaf(paint(healthy(), (540, 380, 660, 460), (18, 16, 14)))
    assert abs(plain["leaf_fraction"] - gap["leaf_fraction"]) < 0.02


# ------------------------------------------------------------ pale dead tissue next to green is damage

PEACH = (236, 200, 180)          # dried leaf, pink or peach


def test_a_pale_dead_patch_on_the_leaf_is_counted():
    streak = paint(healthy(), (700, 250, 820, 650), PEACH)
    f = main.analyze_leaf(streak)
    assert f["lesion_ratio"] > 0.05 and f["spot_count"] >= 1
    assert f["damage_ratio"] >= f["lesion_ratio"] * 0.9


def test_pale_dead_tissue_is_damage_and_is_not_also_called_glare():
    very_pale = (252, 230, 222)                                                 # bright, but with a pink tint
    f = main.analyze_leaf(paint(healthy(), (700, 250, 820, 650), very_pale), with_extras=True)
    codes = [i["code"] for i in f["_extras"]["quality"]["issues"]]
    assert "glare" not in codes and f["lesion_ratio"] > 0.05


def test_a_pale_patch_that_hangs_off_the_end_of_the_leaf_is_counted_too():
    tip = paint(healthy(), (880, 330, 960, 570), PEACH)                         # beyond the green, touching it
    assert main.analyze_leaf(tip)["lesion_ratio"] > 0.02


@pytest.mark.parametrize("name, colour", [("cream paper", (226, 214, 184)), ("pale card", (222, 218, 200))])
def test_a_pale_background_is_not_damage(name, colour):
    im = healthy()
    a = np.asarray(im).copy()
    bg = (abs(a[..., 0].astype(int) - 232) < 6) & (abs(a[..., 1].astype(int) - 228) < 6) & (abs(a[..., 2].astype(int) - 214) < 8)
    a[bg] = colour
    f = main.analyze_leaf(Image.fromarray(a))
    assert f["lesion_ratio"] < 0.02, (name, f["lesion_ratio"])


def test_glare_is_not_damage():
    f = main.analyze_leaf(synth.with_glare(healthy()))
    assert f["lesion_ratio"] < 0.01 and f["damage_ratio"] < 0.01


def test_a_finger_on_the_leaf_is_mostly_not_damage():
    im = healthy().copy()
    ImageDraw.Draw(im).ellipse((700, 560, 960, 720), fill=(224, 172, 140))
    assert main.analyze_leaf(im)["lesion_ratio"] < 0.01


def test_a_thin_brown_fringe_at_the_edge_is_not_a_spot():
    im = healthy().copy()
    ImageDraw.Draw(im).line((300, 450, 900, 450), fill=(110, 80, 40), width=2)      # two pixels wide
    assert main.analyze_leaf(im)["spot_count"] == 0


# ------------------------------------------------------------ small spots are not lost

@pytest.mark.parametrize("kind", ["solid", "target"])
def test_small_spots_inside_the_leaf_are_found(kind):
    """30 spots, 24 x 14 px in a 1200 px photo: about 6 x 3 px once the photo is shrunk for measuring. An earlier
    3x3 opening erased almost all of them (4 found of 30, 0.2 of the real area)."""
    im, truth = synth.make_spotted_leaf(size=(W, H), n=30, rx=12, ry=7, kind=kind)
    f = main.analyze_leaf(im)
    assert f["spot_count"] >= 20, f["spot_count"]
    assert 0.8 * truth <= f["lesion_ratio"] <= 1.3 * truth, (truth, f["lesion_ratio"])


def test_bigger_spots_are_measured_about_right():
    im, truth = synth.make_spotted_leaf(size=(W, H), n=30, rx=20, ry=12)
    f = main.analyze_leaf(im)
    assert f["spot_count"] >= 20 and 0.9 * truth <= f["lesion_ratio"] <= 1.15 * truth


def test_the_same_spots_are_found_in_a_bigger_photo():
    im, truth = synth.make_spotted_leaf(size=(2 * W, 2 * H), n=30, rx=24, ry=14)
    f = main.analyze_leaf(im)
    assert f["spot_count"] >= 20 and 0.8 * truth <= f["lesion_ratio"] <= 1.3 * truth


@pytest.mark.parametrize("raw, expected", [(None, 288), ("", 288), ("abc", 288), ("512", 512), (" 384 ", 384),
                                           ("50", 160), ("99999", 1024), ("-5", 160), ("288.5", 288)])
def test_analysis_size_setting_is_safe(raw, expected):
    assert main._analysis_size(raw) == expected


def test_a_brown_fringe_along_a_dark_gap_is_still_not_a_spot():
    """Two overlapping leaves: a dark gap, with a brownish rim (about 2 px when shrunk) where the leaf meets it."""
    im = healthy().copy()
    d = ImageDraw.Draw(im)
    d.rectangle((900, 300, 1000, 600), fill=(110, 80, 40))      # the rim
    d.rectangle((909, 300, 1000, 600), fill=(18, 16, 14))       # the gap itself, leaving a rim 9 px wide
    f = main.analyze_leaf(im)
    assert f["spot_count"] == 0 and f["lesion_ratio"] < 0.005, f


def test_a_spot_just_inside_the_edge_is_kept_but_a_sliver_on_the_edge_is_not():
    spots = np.zeros((60, 60), bool)
    leaf = np.zeros((60, 60), bool)
    leaf[10:50, 10:50] = True
    spots[10:12, 20:40] = True                      # a 2 px sliver on the outline
    spots[30:33, 30:33] = True                      # a 3 x 3 spot well inside
    out = main._drop_edge_fringe(spots, leaf)
    assert not out[10:12, 20:40].any() and out[30:33, 30:33].all()


def test_a_leaf_that_fills_the_frame_keeps_its_spots_at_the_frame_edge():
    spots = np.zeros((60, 60), bool)
    leaf = np.ones((60, 60), bool)
    spots[0:4, 20:24] = True                        # touches the frame, not a leaf outline
    assert main._drop_edge_fringe(spots, leaf)[0:4, 20:24].all()


# ------------------------------------------------------------ a hand holding the leaf is not leaf

SKIN = (224, 172, 140)           # a light hand
SHADED_SKIN = (170, 120, 92)     # fingers in shadow
DARK_SKIN = (150, 100, 75)       # a medium-brown hand: saturation about 0.5


def with_hand(im, colour=SKIN):
    """Three fingers lying across the leaf and a palm beside it."""
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 330, H), fill=colour)                                   # palm, left of the leaf
    for y in (250, 420, 590):
        d.rectangle((200, y, 760, y + 70), fill=colour)                        # fingers across it
    return im


@pytest.mark.parametrize("colour", [SKIN, SHADED_SKIN, DARK_SKIN])
def test_a_hand_is_not_counted_as_leaf_or_damage(colour):
    f = main.analyze_leaf(with_hand(healthy(), colour))
    assert f["lesion_ratio"] < 0.03 and f["damage_ratio"] < 0.05, (f["lesion_ratio"], f["damage_ratio"])
    assert f["_mask_stats"]["skin_share"] > 0.08, "the hand should have been found"


def test_no_skin_is_found_on_a_leaf_without_a_hand():
    assert main.analyze_leaf(healthy())["_mask_stats"]["skin_share"] == 0.0
    assert main.analyze_leaf(synth.make_spotted_leaf(size=(W, H), n=30, rx=12, ry=7)[0])["_mask_stats"]["skin_share"] == 0.0


def test_small_lesions_in_a_skin_like_colour_are_still_lesions():
    """Colour alone cannot tell skin from a brown lesion; size does. These spots are far smaller than a hand."""
    im = healthy().copy()
    d = ImageDraw.Draw(im)
    for i in range(20):
        x, y = 330 + (i % 5) * 90, 280 + (i // 5) * 90
        d.ellipse((x, y, x + 26, y + 16), fill=(170, 112, 70))                  # hue ~24, saturation ~0.59
    f = main.analyze_leaf(im)
    assert f["spot_count"] >= 14 and f["lesion_ratio"] > 0.01 and f["_mask_stats"]["skin_share"] == 0.0


def test_skin_like_unit_rules():
    h = np.full((200, 200), 100.0)                  # green everywhere
    s = np.full((200, 200), 0.6)
    v = np.full((200, 200), 0.6)
    assert not main._skin_like(h, s, v).any()
    h2, s2, v2 = h.copy(), s.copy(), v.copy()
    h2[20:100, 20:100], s2[20:100, 20:100], v2[20:100, 20:100] = 22.0, 0.38, 0.8     # a big skin-coloured patch
    assert main._skin_like(h2, s2, v2)[50, 50] and not main._skin_like(h2, s2, v2)[150, 150]
    h3, s3, v3 = h.copy(), s.copy(), v.copy()
    h3[20:35, 20:35], s3[20:35, 20:35], v3[20:35, 20:35] = 22.0, 0.38, 0.8           # 0.6 % of the frame: a lesion
    assert not main._skin_like(h3, s3, v3).any()
    s4 = s2.copy()
    s4[20:100, 20:100] = 0.62                                                         # strongly coloured brown
    assert not main._skin_like(h2, s4, v2).any()


# ------------------------------------------------------------ "nothing fits" is not "healthy" when the leaf is damaged

def _nothing_fits(monkeypatch):
    original = main.rank_candidates

    def wrapped(*a, **k):
        scored = original(*a, **k)
        scored[0] = {**scored[0], "vision_score": 0.05, "score": 0.10}
        return scored
    monkeypatch.setattr(main, "rank_candidates", wrapped)


def test_a_badly_damaged_leaf_that_matches_nothing_is_not_called_healthy(client, monkeypatch):
    """A blast photo with a hand in it used to come out as 'healthy, high' because no profile fitted its 63 % 'damage'."""
    _nothing_fits(monkeypatch)
    streak = paint(healthy(), (700, 250, 820, 650), PEACH)
    body = analyze(client, streak).json()
    assert body["is_healthy"] is False and body["inconclusive"] is False
    assert body["confidence"] == "low" and body["spray_withheld"] is True
    assert body["candidates"], "the best guesses are still shown, with advice to ask an expert"


def test_an_undamaged_leaf_that_matches_nothing_is_still_healthy(client, monkeypatch):
    _nothing_fits(monkeypatch)
    body = analyze(client, healthy()).json()
    assert body["is_healthy"] is True and body["spray_withheld"] is False


def test_the_healthy_damage_limit_is_the_damage_cards_lowest_band():
    assert main.HEALTHY_MAX_DAMAGE == 0.05


# ------------------------------------------------------------ disease profiles: progression, and what a deficiency is not

def _vision_scores(im, crop="rice"):
    f = main.analyze_leaf(im)
    scored = main.rank_candidates(f, crop, dict(main.WEATHER_FALLBACK), "", None, 0.0, None, None, 0.0)
    return f, {d["id"]: d["vision_score"] for d in scored}


def test_progression_features_fit_fully_once_the_profile_mean_is_reached():
    items = (("lesion_ratio", (0.10, 0.05, 0.5)), ("spot_count", (5, 2, 0.5)))
    exact = main._score_profile({"lesion_ratio": 0.10, "spot_count": 5}, items, ("lesion_ratio",))
    assert exact == pytest.approx(1.0)
    more = main._score_profile({"lesion_ratio": 0.60, "spot_count": 5}, items, ("lesion_ratio",))
    assert more == pytest.approx(1.0), "a more advanced infection of the same disease is not a worse match"
    plain = main._score_profile({"lesion_ratio": 0.60, "spot_count": 5}, items)
    assert plain < 0.6, "without the flag, far too much damage is a mismatch"
    less = main._score_profile({"lesion_ratio": 0.0, "spot_count": 5}, items, ("lesion_ratio",))
    assert less < 0.6, "too little damage is still a mismatch"


def test_only_rice_blast_and_bacterial_blight_are_marked_as_progressing():
    marked = {d["id"]: d["at_least"] for d in main.DISEASES if d.get("at_least")}
    assert set(marked) == {"rice_blast", "rice_blb"}
    for did, keys in marked.items():
        assert set(keys) <= set(main.DISEASES_BY_ID[did]["profile"]), (did, keys)


def test_a_mostly_dead_leaf_is_blight_or_blast_not_a_nutrient_deficiency():
    """A photo of dried rice leaves used to come out 'Nitrogen Deficiency' because the yellowing alone carried it."""
    dead = paint(healthy(), (600, 150, 860, 750), PEACH)
    f, v = _vision_scores(dead)
    assert f["lesion_ratio"] > 0.2
    assert v["nitrogen_deficiency"] < min(v["rice_blb"], v["rice_blast"]), v


@pytest.mark.parametrize("crop", ["rice", "tomato", "ragi"])
def test_an_evenly_yellow_leaf_is_still_a_nutrient_deficiency(crop):
    yellow = paint(healthy(), (300, 200, 900, 500), (205, 195, 70))
    f, v = _vision_scores(yellow, crop)
    assert f["lesion_ratio"] < 0.01 and f["chlorosis_ratio"] > 0.15
    assert max(v, key=v.get) == "nitrogen_deficiency", v


# ------------------------------------------------------------ the damage estimate

def damage(im):
    return main.analyze_leaf(im)["damage_ratio"]


def test_damage_is_zero_for_a_clean_leaf_and_grows_with_spots():
    assert damage(healthy()) == 0
    few, many = damage(synth.make_leaf(size=(W, H), spots=4)), damage(synth.make_leaf(size=(W, H), spots=14))
    assert 0 < few < many < 0.2


def test_a_shadow_on_a_healthy_leaf_is_not_damage():
    a = np.asarray(healthy()).astype(np.float32).copy()
    a[:, : int(W * 0.2)] *= 0.55                                                # a hand's shadow over a fifth
    assert damage(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))) < 0.01


def test_yellowing_counts_as_damage():
    a = np.asarray(synth.make_leaf(size=(W, H), spots=0, leaf_scale=1.2)).copy()
    patch = a[:, : int(W * 0.45)]
    leafish = (patch[..., 1] > patch[..., 0] + 20) & (patch[..., 1] > patch[..., 2] + 20)
    patch[leafish] = (205, 190, 40)
    f = main.analyze_leaf(Image.fromarray(a))
    assert f["damage_ratio"] > 0.35, f["damage_ratio"]


@pytest.mark.parametrize("ratio, healthy_flag, level", [
    # the band follows the whole percentage shown on screen, so the label never disagrees with the number
    (0.00, False, "good"), (0.044, False, "good"), (0.049, False, "warn"), (0.05, False, "warn"),
    (0.244, False, "warn"), (0.249, False, "bad"), (0.25, False, "bad"), (0.90, False, "bad"),
    (0.02, True, "good"),
])
def test_damage_bands(ratio, healthy_flag, level):
    hs = main.build_health_score({}, {}, {"damage_ratio": ratio}, healthy_flag)
    assert hs["level"] == level and hs["score"] == round(ratio * 100)
    assert hs["type"] == "leaf_damage_percent" and hs["is_demo"] is True


def test_healthy_leaf_keeps_the_healthy_label_in_every_language():
    for lang in ("en", "hi", "kn"):
        hs = main.build_health_score({}, {}, {"damage_ratio": 0.0}, True, lang)
        assert hs["label"] == main.HEALTH_LABELS["healthy"][lang]


def test_damage_is_clamped_and_survives_a_missing_value():
    assert main.build_health_score({}, {}, {"damage_ratio": 3.0}, False)["score"] == 100
    assert main.build_health_score({}, {}, {}, False)["score"] == 0


# ------------------------------------------------------------ through the API

def test_the_estimate_comes_from_the_photo_not_from_the_weather(client, monkeypatch, force_diseased):
    im = synth.make_leaf(size=(W, H), spots=10)
    first = analyze(client, im).json()["health_score"]

    async def stormy(lat, lon):
        return {**main.WEATHER_FALLBACK, "temp": 27.0, "rh": 96, "rain_48h": 40.0, "rain_7d": 90.0}
    monkeypatch.setattr(main, "fetch_weather_cached", stormy)
    second = analyze(client, im).json()["health_score"]
    assert first["score"] == second["score"] and first["level"] == second["level"]
    assert first["score"] == round(100 * main.analyze_leaf(im)["damage_ratio"])


def test_the_estimate_does_not_follow_the_confidence(client, force_confidence, force_diseased):
    im = synth.make_leaf(size=(W, H), spots=10)
    force_confidence("high")
    high = analyze(client, im).json()["health_score"]["score"]
    force_confidence("low")
    low = analyze(client, im).json()["health_score"]["score"]
    assert high == low

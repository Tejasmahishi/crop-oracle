"""When the app is unsure it must not suggest anything the farmer would have to buy and spray."""
import pytest

import main
import synth
from conftest import analyze

SPRAY_KEYS = {"fungicide", "bactericide", "fertilize"}


def all_plan_keys(card):
    return [s["key"] for items in card["plan"]["buckets"].values() for s in items]


# ------------------------------------------------------------ pure functions

@pytest.mark.parametrize("kw, expected", [
    (dict(is_healthy=True, model_agrees=False, model_disagrees=False, top_score=0.1, gap=0.0), "high"),
    (dict(is_healthy=False, model_agrees=True, model_disagrees=False, top_score=0.3, gap=0.0), "high"),
    (dict(is_healthy=False, model_agrees=False, model_disagrees=True, top_score=0.9, gap=0.5), "low"),
    (dict(is_healthy=False, model_agrees=False, model_disagrees=False, top_score=0.65, gap=0.2), "high"),
    (dict(is_healthy=False, model_agrees=False, model_disagrees=False, top_score=0.45, gap=0.1), "moderate"),
    (dict(is_healthy=False, model_agrees=False, model_disagrees=False, top_score=0.45, gap=0.02), "low"),
    (dict(is_healthy=False, model_agrees=False, model_disagrees=False, top_score=0.2, gap=0.3), "low"),
])
def test_decide_confidence_keeps_the_original_rules(kw, expected):
    assert main.decide_confidence(**kw) == expected


@pytest.mark.parametrize("disease_id", sorted(main.DISEASES_BY_ID))
def test_gate_removes_every_input_step_for_every_disease(disease_id):
    d = main.DISEASES_BY_ID[disease_id]
    for region in main.REGION_NAMES:
        reg = main.check_regulation(d, region)
        gated = main.apply_input_gate(reg, True)
        kinds = [s["action"] if s["type"] == "spray" else s["action"]["t"] for s in gated["steps"]]
        assert not (set(kinds) & (SPRAY_KEYS | {"fungicide", "bactericide"}))
        assert all(s["type"] != "spray" for s in gated["steps"])
        assert kinds[:2] == ["retake_photos", "contact_kvk"]
        assert gated["substitutions"] == [] and gated["flags"] == []
        assert gated["spray_withheld"] is True
        # harmless steps survive
        harmless = [s["action"]["t"] for s in reg["steps"]
                    if s["type"] == "cultural" and s["action"]["t"] not in main.INPUT_ACTIONS]
        assert [k for k in kinds[2:]] == harmless


def test_gate_off_changes_nothing():
    d = main.DISEASES_BY_ID["rice_blast"]
    reg = main.check_regulation(d, "ka")
    assert main.apply_input_gate(reg, False) is reg


def test_sanity_some_diseases_really_do_have_spray_steps():
    d = main.DISEASES_BY_ID["rice_blast"]
    reg = main.check_regulation(d, "ka")
    assert any(s["type"] == "spray" for s in reg["steps"])


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_new_steps_exist_in_every_language_and_bucket(lang):
    for key in ("retake_photos", "contact_kvk"):
        assert main.ACTIONS[key][lang].strip()
        assert key in main.ACTION_BUCKETS
    assert main._SPOKEN_UNCERTAIN[lang].format(name="X")
    assert main._SPOKEN_INCONCLUSIVE[lang].strip()
    assert main.STAGE_TEXT["regulatory_withheld"][lang].strip()


def test_plan_for_gated_result_has_no_source_links_on_the_new_steps():
    d = main.DISEASES_BY_ID["rice_blast"]
    reg = main.apply_input_gate(main.check_regulation(d, "ka"), True)
    plan = main.build_plan(d, reg, d["refs"], main.CORPUS, "en")
    keys = [s["key"] for items in plan.values() for s in items]
    assert keys[0] == "retake_photos" and "contact_kvk" in keys
    for items in plan.values():
        for s in items:
            if s["key"] in ("retake_photos", "contact_kvk"):
                assert s["refs"] == []
    assert not (set(keys) & SPRAY_KEYS)


def test_spoken_text_when_unsure_never_gives_a_diagnosis_score():
    top = {"score": 0.31, "name": {"en": "Rice Blast"}}
    text = main.build_spoken(top, False, {"right_now": [{"text": "Do the thing"}]}, "en", "Rice Blast", uncertain=True)
    assert "not sure" in text and "do not spray" in text.lower()
    assert "31" not in text and "score" not in text.lower()


# ------------------------------------------------------------ through the API

def test_low_confidence_response_has_no_spray_advice(client, force_confidence, force_diseased):
    force_confidence("low")
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=10))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["confidence"] == "low" and body["spray_withheld"] is True
    card = body["candidates"][0]
    keys = all_plan_keys(card)
    assert not (set(keys) & SPRAY_KEYS)
    assert keys[0] == "retake_photos" and "contact_kvk" in keys
    assert all(s["type"] != "spray" for s in card["regulatory"]["steps"])
    assert card["regulatory"]["substitutions"] == []
    assert card["csv_remedy"] is None
    assert "not sure" in body["spoken"] and "spray" in body["spoken"].lower()
    reg_stage = next(s for s in body["stages"] if s["id"] == "regulatory")
    assert "withheld" in reg_stage["detail"]


@pytest.mark.parametrize("crop", list(main.SUPPORTED_CROPS))
def test_low_confidence_never_sprays_for_any_crop(client, force_confidence, crop, force_diseased):
    force_confidence("low")
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=12), crop=crop)
    assert r.status_code == 200, r.text
    card = r.json()["candidates"]
    assert card, "a spotted leaf should produce a candidate"
    assert not (set(all_plan_keys(card[0])) & SPRAY_KEYS)


@pytest.mark.parametrize("lang", ["hi", "kn"])
def test_low_confidence_texts_are_localised(client, force_confidence, lang, force_diseased):
    force_confidence("low")
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=10), language=lang)
    body = r.json()
    texts = [s["text"] for items in body["candidates"][0]["plan"]["buckets"].values() for s in items]
    assert main.ACTIONS["retake_photos"][lang] in texts
    assert body["spoken"] != main._SPOKEN_UNCERTAIN["en"].format(name="x")


def test_high_confidence_still_gives_the_full_plan(client, force_confidence, force_diseased):
    force_confidence("high")
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=10))
    body = r.json()
    assert body["spray_withheld"] is False
    card = body["candidates"][0]
    keys = all_plan_keys(card)
    assert "retake_photos" not in keys and "contact_kvk" not in keys
    top_id = card["diagnosis"]["id"]
    wants_spray = any(a["t"] in ("fungicide", "bactericide") for a in main.DISEASES_BY_ID[top_id]["actions"])
    assert wants_spray == any(k in ("fungicide", "bactericide") for k in keys)


def test_poor_photo_lowers_confidence_and_withholds_spray(client, force_confidence, force_diseased):
    force_confidence("high")           # the scoring is sure, but the photo is a 200 px thumbnail
    r = analyze(client, synth.make_leaf(size=(200, 150), spots=6))
    body = r.json()
    assert r.status_code == 200
    assert body["photo_quality"]["severity"] == "severe"
    assert body["confidence"] == "low" and body["spray_withheld"] is True


def test_minor_photo_problem_lowers_high_to_moderate(client, force_confidence, force_diseased):
    force_confidence("high")
    r = analyze(client, synth.with_glare(synth.make_leaf(size=(1200, 900), spots=10)))
    body = r.json()
    assert body["photo_quality"]["severity"] == "minor"
    assert body["confidence"] == "moderate" and body["spray_withheld"] is False


def _force_healthy_looking(monkeypatch):
    original = main.rank_candidates

    def wrapped(*a, **k):
        scored = original(*a, **k)
        scored[0] = {**scored[0], "vision_score": 0.05, "score": 0.10}
        return scored
    monkeypatch.setattr(main, "rank_candidates", wrapped)


def test_healthy_looking_leaf_in_an_unreadable_photo_is_inconclusive_not_healthy(client, monkeypatch):
    _force_healthy_looking(monkeypatch)
    r = analyze(client, synth.make_leaf(size=(200, 150), spots=0))
    body = r.json()
    assert r.status_code == 200
    assert body["inconclusive"] is True and body["is_healthy"] is False
    assert body["confidence"] == "low"
    assert body["candidates"] == []
    assert body["health_score"] is None
    assert body["spoken"] == main._SPOKEN_INCONCLUSIVE["en"]


def test_healthy_leaf_in_a_good_photo_is_still_healthy(client, monkeypatch):
    _force_healthy_looking(monkeypatch)
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=0))
    body = r.json()
    assert body["is_healthy"] is True and body["inconclusive"] is False
    assert body["spray_withheld"] is False
    assert body["health_score"] is not None


def test_dark_photo_error_says_why(client):
    r = analyze(client, synth.darkened(synth.make_leaf(spots=6), 0.22))
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], str) and body["code"] == "no_leaf"
    assert any(i["code"] == "dark" for i in body["photo_quality"]["issues"])


def test_blank_photo_is_never_healthy(client):
    r = analyze(client, synth.blank())
    assert r.status_code == 422 and r.json()["code"] == "no_leaf"

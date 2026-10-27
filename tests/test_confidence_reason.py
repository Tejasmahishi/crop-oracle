"""Why a result is "low", said truthfully.

A good photo must never be blamed for a result that is low because two diseases look alike. The server names
the real reason and the page words it; these tests pin the reasons down."""
import pytest

import main
import synth
from conftest import analyze


# ------------------------------------------------------------ the reason itself

@pytest.mark.parametrize("kw, expected", [
    # a photo too poor to read always wins: the fix is a new photo
    (dict(inconclusive=True, severity="severe", model_disagrees=False, top_score=0.7), "photo"),
    (dict(inconclusive=False, severity="severe", model_disagrees=False, top_score=0.7), "photo"),
    (dict(inconclusive=False, severity="severe", model_disagrees=True, top_score=0.2), "photo"),
    # the trained model argues with the colour check
    (dict(inconclusive=False, severity="ok", model_disagrees=True, top_score=0.7), "model"),
    (dict(inconclusive=False, severity="minor", model_disagrees=True, top_score=0.2), "model"),
    # nothing fits well
    (dict(inconclusive=False, severity="ok", model_disagrees=False, top_score=0.30), "weak"),
    (dict(inconclusive=False, severity="minor", model_disagrees=False, top_score=0.399), "weak"),
    # a decent top score that two diseases share: the common case
    (dict(inconclusive=False, severity="ok", model_disagrees=False, top_score=0.40), "alike"),
    (dict(inconclusive=False, severity="minor", model_disagrees=False, top_score=0.75), "alike"),
])
def test_reason_for_a_low_result(kw, expected):
    assert main.low_confidence_reason(**kw) == expected


def test_a_minor_photo_problem_is_not_blamed_when_diseases_look_alike():
    """Minor problems only lower high to moderate, so they can never be why a result is low."""
    assert main.low_confidence_reason(inconclusive=False, severity="minor", model_disagrees=False,
                                      top_score=0.6) == "alike"


def test_the_cut_offs_are_the_original_ones():
    """They were set by hand and are not measured. Changing them must be a decision, not an accident."""
    assert (main.HIGH_MIN_SCORE, main.HIGH_MIN_GAP) == (0.60, 0.15)
    assert (main.MODERATE_MIN_SCORE, main.MODERATE_MIN_GAP) == (0.40, 0.08)


# ------------------------------------------------------------ through the API

def test_two_close_diseases_are_low_and_name_the_runner_up(client, scores):
    scores(0.62, 0.58)
    body = analyze(client, synth.make_leaf(size=(1200, 900), spots=10)).json()
    assert body["confidence"] == "low" and body["spray_withheld"] is True
    assert body["confidence_reason"] == "alike"
    runner = body["runner_up"]
    assert runner and runner["id"] != body["candidates"][0]["diagnosis"]["id"]
    assert runner["name"]["en"] and runner["name"]["hi"] and runner["name"]["kn"]


def test_a_weak_best_match_is_low_for_that_reason(client, scores):
    scores(0.30, 0.10)
    body = analyze(client, synth.make_leaf(size=(1200, 900), spots=10)).json()
    assert body["confidence"] == "low" and body["confidence_reason"] == "weak"
    assert body["runner_up"] is None


def test_a_clear_winner_has_no_reason_and_no_runner_up(client, scores):
    scores(0.72, 0.30)
    body = analyze(client, synth.make_leaf(size=(1200, 900), spots=10)).json()
    assert body["confidence"] == "high" and body["spray_withheld"] is False
    assert body["confidence_reason"] is None and body["runner_up"] is None


def test_a_badly_blurred_photo_is_blamed_on_the_photo(client, scores):
    scores(0.62, 0.58)
    blurred = synth.blurred(synth.make_leaf(size=(1200, 900), spots=10), 14)
    body = analyze(client, blurred).json()
    assert body["photo_quality"]["severity"] == "severe"
    assert body["confidence"] == "low" and body["confidence_reason"] == "photo"
    assert body["runner_up"] is None


def test_the_reason_is_in_every_language(client, scores):
    for lang in ("en", "hi", "kn"):
        scores(0.62, 0.58)
        body = analyze(client, synth.make_leaf(size=(1200, 900), spots=10), language=lang).json()
        assert body["confidence_reason"] == "alike" and body["runner_up"]["name"][lang]


# ------------------------------------------------------------ the model loader

def test_no_trained_model_is_said_out_loud_at_start_up(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RICE_MODEL_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(main, "RICE_INTERPRETER", None)
    main._load_rice_model()
    assert "rice model not found" in capsys.readouterr().out


def test_loader_tries_the_newer_runtime_and_names_all_three_when_none_is_installed(capsys, monkeypatch, tmp_path):
    fake = tmp_path / "rice_model.tflite"
    fake.write_bytes(b"not a real model")
    monkeypatch.setattr(main, "RICE_MODEL_PATH", fake)
    monkeypatch.setattr(main, "RICE_INTERPRETER", None)
    tried = []

    def refuse(name):
        tried.append(name)
        raise ImportError(name)
    monkeypatch.setattr(main.importlib, "import_module", refuse)
    main._load_rice_model()
    assert tried == ["tflite_runtime.interpreter", "ai_edge_litert.interpreter", "tensorflow.lite.python.interpreter"]
    out = capsys.readouterr().out
    assert "tflite-runtime" in out and "ai-edge-litert" in out and "tensorflow" in out
    assert main.RICE_INTERPRETER is None

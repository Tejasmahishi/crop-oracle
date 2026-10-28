"""The trained rice model: that it is given pixels the way it expects, and that its classes mean what the app thinks.

Most of these use a small stand-in for the model, so they run on any machine. The last group uses the real
rice_model.tflite and skips itself when that file or a TFLite runtime is missing. None of this measures how
accurate the model is on field photos (see README, "The rice model")."""
import numpy as np
import pytest
from PIL import Image

import main
import synth
from conftest import analyze

pytestmark = pytest.mark.filterwarnings("ignore")


class FakeModel:
    """Stands in for a TFLite interpreter. Like the supplied MobileNetV3 model it rescales 0..255 to -1..1
    itself (internal_rescale=True), then answers from the average colour of the picture it is handed."""

    def __init__(self, internal_rescale=True):
        self.internal_rescale = internal_rescale
        self.seen = None
        self._out = None
        rng = np.random.default_rng(1)
        self._w = rng.normal(0, 1, (4, 4))

    def set_tensor(self, index, arr):
        self.seen = np.array(arr)

    def invoke(self):
        x = self.seen[0].astype(np.float64)
        if self.internal_rescale:
            x = x / 127.5 - 1.0
        feats = np.array([x[..., 0].mean(), x[..., 1].mean(), x[..., 2].mean(), x.std()])
        logits = (self._w @ feats) * 6.0
        e = np.exp(logits - logits.max())
        self._out = (e / e.sum()).astype(np.float32)[None]

    def get_tensor(self, index):
        return self._out


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(main, "RICE_INTERPRETER", model)
    monkeypatch.setattr(main, "RICE_INPUT_INDEX", 0)
    monkeypatch.setattr(main, "RICE_OUTPUT_INDEX", 0)
    monkeypatch.setattr(main, "RICE_CLASS_NAMES", ["Bacterial leaf blight", "Brown spot", "Health", "Leaf smut"])
    monkeypatch.setattr(main, "RICE_INPUT_SCALE", "raw")
    return model


# ------------------------------------------------------------ how the pixels are given to the model

def test_the_model_is_given_raw_0_to_255_pixels_by_default(fake):
    """The supplied model rescales by itself. Rescaling here as well made every photo look the same to it."""
    main._run_rice_model(Image.new("RGB", (300, 200), (200, 100, 50)))
    assert fake.seen.shape == (1, 224, 224, 3) and fake.seen.dtype == np.float32
    assert np.allclose(fake.seen[0, 100, 100], [200, 100, 50], atol=1)
    assert fake.seen.max() > 1.5, "values must not be squeezed into -1..1 or 0..1 before the model"


@pytest.mark.parametrize("scale, expected", [
    ("raw", [200, 100, 50]),
    ("unit", [200 / 255, 100 / 255, 50 / 255]),
    ("signed", [200 / 127.5 - 1, 100 / 127.5 - 1, 50 / 127.5 - 1]),
])
def test_the_scaling_can_be_changed_for_a_model_without_its_own_rescaling(fake, monkeypatch, scale, expected):
    monkeypatch.setattr(main, "RICE_INPUT_SCALE", scale)
    main._run_rice_model(Image.new("RGB", (224, 224), (200, 100, 50)))
    assert np.allclose(fake.seen[0, 10, 10], expected, atol=0.01)


def test_two_different_photos_give_two_different_answers(fake):
    dark = main._run_rice_model(Image.new("RGB", (224, 224), (20, 40, 20)))
    bright = main._run_rice_model(Image.new("RGB", (224, 224), (230, 215, 190)))
    assert dark and bright and dark != bright
    assert abs(dark[0]["confidence"] - bright[0]["confidence"]) > 0.01 or dark[0]["class"] != bright[0]["class"]


def test_answer_is_the_top_three_named_classes_in_order(fake):
    top3 = main._run_rice_model(Image.new("RGB", (224, 224), (60, 135, 52)))
    assert len(top3) == 3
    assert [t["confidence"] for t in top3] == sorted((t["confidence"] for t in top3), reverse=True)
    assert all(t["class"] in main.RICE_CLASS_NAMES for t in top3)


def test_no_model_means_no_answer(monkeypatch):
    monkeypatch.setattr(main, "RICE_INTERPRETER", None)
    assert main._run_rice_model(Image.new("RGB", (224, 224), (60, 135, 52))) == []


# ------------------------------------------------------------ the start-up check

def test_the_start_up_check_passes_when_the_scaling_is_right(fake):
    assert main._rice_input_spread() >= main.RICE_MIN_PROBE_SPREAD


def test_the_start_up_check_catches_scaling_twice(fake, monkeypatch):
    """This is the bug that made every photo come out as 'leaf smut': the model rescales, and so did the app."""
    monkeypatch.setattr(main, "RICE_INPUT_SCALE", "signed")
    assert main._rice_input_spread() < main.RICE_MIN_PROBE_SPREAD


def test_the_start_up_check_returns_nothing_without_a_model(monkeypatch):
    monkeypatch.setattr(main, "RICE_INTERPRETER", None)
    assert main._rice_input_spread() is None


def test_a_model_that_cannot_run_does_not_break_the_check(fake, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no")
    monkeypatch.setattr(fake, "invoke", boom)
    assert main._rice_input_spread() is None


def test_unknown_scale_setting_falls_back_to_raw():
    assert main.RICE_INPUT_SCALE in main._RICE_SCALERS


# ------------------------------------------------------------ the model's classes mean what the app thinks

def test_every_class_the_model_can_name_is_mapped_to_a_known_disease_or_to_healthy():
    names = main.RICE_CLASS_NAMES
    assert names, "no class names"
    for name in names:
        meta = main.TFLITE_CLASS_TO_META.get(name)
        assert meta is not None, f"{name!r} is not in class_indices_mapping.json"
        if meta.get("healthy"):
            assert meta.get("id") is None
        else:
            assert meta.get("id") in {d["id"] for d in main.DISEASES_BY_CROP["rice"]}, (name, meta)


def test_a_confident_answer_reaches_the_ranking_as_the_right_disease():
    best, conf, evidence = main.tflite_hint_to_internal([{"class": "Brown spot", "confidence": 0.93}])
    assert (best, round(conf, 2)) == ("rice_brown_spot", 0.93) and evidence[0]["internal_id"] == "rice_brown_spot"
    best, conf, _ = main.tflite_hint_to_internal([{"class": "Health", "confidence": 0.9}])
    assert best is None and conf == 0.0, "'Health' is not a disease and must not be boosted as one"


def test_the_rice_model_is_asked_only_for_rice(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run_rice_model", lambda im: calls.append(im.size) or [])
    r = client.post("/api/analyze", data={"crop": "tomato", "language": "en", "region": "ka"},
                    files={"image": ("l.jpg", synth.to_jpeg(synth.make_leaf(spots=4)), "image/jpeg")})
    assert r.status_code in (200, 400) and calls == []
    r = client.post("/api/analyze", data={"crop": "rice", "language": "en", "region": "ka"},
                    files={"image": ("l.jpg", synth.to_jpeg(synth.make_leaf(spots=4)), "image/jpeg")})
    assert r.status_code == 200 and len(calls) == 1


# ------------------------------------------------------------ the model only advises

def _says(monkeypatch, cls, conf=1.0):
    """Make the server's rice model answer `cls` with `conf` for every photo."""
    others = [c for c in main.RICE_CLASS_NAMES if c != cls][:2]
    monkeypatch.setattr(main, "_run_rice_model", lambda im: (
        [{"class": cls, "confidence": conf}] + [{"class": o, "confidence": (1 - conf) / 2} for o in others]))


def _run(client, im):
    r = analyze(client, im)
    assert r.status_code == 200, r.text
    return r.json()


def _ranking(body):
    return [(c["diagnosis"]["id"], round(c["score"], 4)) if "score" in c else c["diagnosis"]["id"]
            for c in body["candidates"]]


def test_the_default_role_is_advisory():
    assert main.RICE_MODEL_ROLE == "advisory"


def test_a_confident_wrong_model_answer_cannot_change_the_result_in_advisory_mode(client, monkeypatch):
    """This happened on a real photo of bacterial leaf blight: 'leaf smut, 100 %', and smut came out first."""
    im = synth.make_leaf(size=(1200, 900), spots=10)
    monkeypatch.setattr(main, "_run_rice_model", lambda i: [])
    without = _run(client, im)
    _says(monkeypatch, "Leaf smut", 1.0)
    with_model = _run(client, im)
    assert _ranking(with_model) == _ranking(without)
    assert with_model["confidence"] == without["confidence"]
    assert with_model["confidence_reason"] == without["confidence_reason"]
    assert with_model["spray_withheld"] == without["spray_withheld"]


def test_advisory_answer_is_returned_for_the_operator_but_not_shown_as_evidence(client, monkeypatch):
    _says(monkeypatch, "Leaf smut", 1.0)
    ev = _run(client, synth.make_leaf(size=(1200, 900), spots=10))["evidence_sources"]
    assert ev["rice_model_role"] == "advisory" and ev["rice_model_used"] is False
    assert ev["tflite"] == [], "the page lists these as evidence for the result"
    assert ev["rice_model_advisory"] == {"class": "Leaf smut", "confidence": 1.0}


def test_no_model_answer_means_no_advisory_entry(client, monkeypatch):
    monkeypatch.setattr(main, "_run_rice_model", lambda i: [])
    ev = _run(client, synth.make_leaf(size=(1200, 900), spots=10))["evidence_sources"]
    assert ev["rice_model_advisory"] is None and ev["tflite"] == []


def test_score_role_gives_the_model_a_vote_again(client, monkeypatch):
    im = synth.make_leaf(size=(1200, 900), spots=10)
    monkeypatch.setattr(main, "_run_rice_model", lambda i: [])
    baseline = {c["diagnosis"]["id"]: c for c in _run(client, im)["candidates"]}
    monkeypatch.setattr(main, "RICE_MODEL_ROLE", "score")
    _says(monkeypatch, "Leaf smut", 1.0)
    body = _run(client, im)
    ev = body["evidence_sources"]
    assert ev["rice_model_used"] is True and ev["tflite"][0]["class"] == "Leaf smut"
    assert ev["rice_model_advisory"] is None
    assert body["candidates"][0]["diagnosis"]["id"] == "rice_leaf_smut"
    assert body["confidence"] == "high", "model_agrees raises the confidence in score mode (the original rule)"
    assert baseline, "sanity: the run without a model returned candidates"


def test_advisory_mode_does_not_touch_the_client_prediction_path(client, monkeypatch):
    """A browser-sent prediction is only read when the operator turns ALLOW_CLIENT_ML on; that is a deliberate choice."""
    monkeypatch.setattr(main, "ALLOW_CLIENT_ML", True)
    monkeypatch.setattr(main, "_run_rice_model", lambda i: pytest.fail("the server model must not run"))
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=10),
                tflite_top3='[{"class": "Brown spot", "confidence": 0.9}]')
    ev = r.json()["evidence_sources"]
    assert ev["rice_model_used"] is True and ev["tflite"][0]["class"] == "Brown spot"


# ------------------------------------------------------------ the real model (skipped when it is not there)

def _real_model(monkeypatch):
    if not main.RICE_MODEL_PATH.exists():
        pytest.skip("rice_model.tflite is not next to main.py")
    for name, value in (("RICE_INTERPRETER", None), ("RICE_INPUT_INDEX", None), ("RICE_OUTPUT_INDEX", None)):
        monkeypatch.setattr(main, name, value)
    monkeypatch.setattr(main, "RICE_INPUT_SCALE", "raw")
    main._load_rice_model()
    if main.RICE_INTERPRETER is None:
        pytest.skip("no TFLite runtime (tflite-runtime, ai-edge-litert or tensorflow) is installed")


def test_real_model_answers_differently_for_different_pictures(monkeypatch):
    _real_model(monkeypatch)
    assert main._rice_input_spread() >= main.RICE_MIN_PROBE_SPREAD


def test_real_model_is_not_stuck_on_one_answer(monkeypatch):
    """With the pixels scaled twice, the supplied model said about 68% leaf smut and 30% brown spot for any photo."""
    _real_model(monkeypatch)
    rng = np.random.default_rng(5)
    answers = []
    for im in (Image.new("RGB", (224, 224), (255, 255, 255)),
               Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)),
               synth.make_leaf(size=(600, 450), spots=8),
               synth.make_leaf(size=(600, 450), spots=0)):
        top3 = main._run_rice_model(im)
        assert len(top3) == 3 and abs(sum(t["confidence"] for t in top3) - 1.0) < 0.01
        answers.append({t["class"]: t["confidence"] for t in top3})
    smut = [a.get("Leaf smut", 0.0) for a in answers]
    assert max(smut) - min(smut) > 0.2, smut


def test_real_model_class_order_matches_the_class_file(monkeypatch):
    _real_model(monkeypatch)
    n_out = int(main.RICE_INTERPRETER.get_output_details()[0]["shape"][-1])
    assert n_out == len(main.RICE_CLASS_NAMES)

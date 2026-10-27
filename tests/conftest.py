"""Shared test set-up. The tests never touch the network: weather, geocoding and the
outside plant-disease service are replaced with fixed answers."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# These must be set before main is imported, because main reads them at import time.
_TMP = tempfile.mkdtemp(prefix="croporacle-test-")
os.environ["DATA_DIR"] = _TMP
os.environ["ADMIN_TOKEN"] = "test-admin-token"
for _name in ("KINDWISE_API_KEY", "KINDWISE_KEY", "SARVAM_API_KEY", "SARVAM_KEY",
              "PRIVACY_CONTACT", "PRIVACY_OPERATOR", "ALLOW_CLIENT_ML", "CORS_ALLOWED_ORIGINS",
              "ENABLE_DOCS", "SLM_ENABLED", "DEBUG_MASKS"):
    os.environ.pop(_name, None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import feedback as fb  # noqa: E402
import guard  # noqa: E402
import main  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fresh limits, a fresh feedback store and no network for every test."""
    monkeypatch.setattr(guard, "SETTINGS", guard.Settings(admin_token="test-admin-token"))
    monkeypatch.setattr(guard, "LIMITER", guard.RateLimiter())
    store = fb.Store(tmp_path / "feedback", secret=b"test-secret")
    monkeypatch.setattr(main, "STORE", store)
    monkeypatch.setattr(main, "PRIVACY_CONTACT", "")
    monkeypatch.setattr(main, "KINDWISE_KEY", "")
    monkeypatch.setattr(main, "ALLOW_CLIENT_ML", False)
    # The trained rice model is switched off, so the results do not depend on whether rice_model.tflite sits
    # next to main.py on the machine running the tests. tests/test_rice_model.py switches it on when it needs it.
    monkeypatch.setattr(main, "RICE_INTERPRETER", None)

    async def fake_weather(lat, lon):
        return dict(main.WEATHER_FALLBACK)

    async def fake_place(lat, lon):
        return {"label": "Test place", "country_code": "in", "region": "ka"}

    async def fake_kindwise(raw, crop, lang="en"):
        return None

    async def fake_pubmed(term, retmax=4):
        return []

    main._pubmed_cache.clear()
    monkeypatch.setattr(main, "pubmed_search", fake_pubmed)
    monkeypatch.setattr(main, "fetch_weather_cached", fake_weather)
    monkeypatch.setattr(main, "reverse_geocode_cached", fake_place)
    monkeypatch.setattr(main, "kindwise_analyze", fake_kindwise)
    return store


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def store(isolated):
    return isolated


@pytest.fixture
def with_contact(monkeypatch):
    monkeypatch.setattr(main, "PRIVACY_CONTACT", "help@example.org")


@pytest.fixture
def force_confidence(monkeypatch):
    """force_confidence("low") makes every analysis come out with that confidence."""
    def _set(level):
        monkeypatch.setattr(main, "decide_confidence", lambda **kw: level)
    return _set


@pytest.fixture
def force_diseased(monkeypatch):
    """Make the top candidate look clearly diseased, whatever the picture, so a test about
    advice does not depend on the hand-set scoring profiles."""
    original = main.rank_candidates

    def wrapped(*a, **k):
        scored = original(*a, **k)
        if scored:
            scored[0] = {**scored[0], "vision_score": 0.8, "score": 0.7}
        return scored
    monkeypatch.setattr(main, "rank_candidates", wrapped)


@pytest.fixture
def scores(monkeypatch):
    """scores(top, second) pins the two best scores; vision stays high so the leaf counts as diseased."""
    original = main.rank_candidates

    def _set(top, second):
        def wrapped(*a, **k):
            scored = original(*a, **k)
            scored[0] = {**scored[0], "vision_score": 0.8, "score": top}
            scored[1] = {**scored[1], "vision_score": 0.5, "score": second}
            return scored
        monkeypatch.setattr(main, "rank_candidates", wrapped)
    return _set


def analyze(client, im=None, crop="rice", **extra):
    """POST a photo to /api/analyze. Returns the response."""
    im = im if im is not None else synth.make_leaf(spots=6)
    data = {"crop": crop, "language": "en", "region": "ka"}
    data.update(extra)
    return client.post("/api/analyze", data=data,
                       files={"image": ("leaf.jpg", synth.to_jpeg(im), "image/jpeg")})

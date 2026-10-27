import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import guard
import main
import synth
from conftest import analyze


# ------------------------------------------------------------ the limiter itself

def test_sliding_window_blocks_then_recovers():
    lim = guard.RateLimiter()
    assert [lim.check("b", "ip", 3, 60, now=t)[0] for t in (0, 1, 2)] == [True, True, True]
    ok, wait = lim.check("b", "ip", 3, 60, now=3)
    assert ok is False and 1 <= wait <= 60
    assert lim.check("b", "ip", 3, 60, now=59)[0] is False
    assert lim.check("b", "ip", 3, 60, now=60.5)[0] is True


def test_limits_are_per_address_and_per_bucket():
    lim = guard.RateLimiter()
    assert lim.check("analyze", "1.1.1.1", 1, now=0)[0]
    assert not lim.check("analyze", "1.1.1.1", 1, now=1)[0]
    assert lim.check("analyze", "2.2.2.2", 1, now=1)[0]
    assert lim.check("general", "1.1.1.1", 1, now=1)[0]


def test_limit_of_zero_means_off():
    lim = guard.RateLimiter()
    assert all(lim.check("b", "ip", 0, now=i)[0] for i in range(100))


def test_idle_keys_are_swept():
    lim = guard.RateLimiter()
    for i in range(100):
        lim.check("b", f"ip{i}", 5, 60, now=0)
    lim.check("b", "later", 5, 60, now=500)
    assert len(lim._hits) == 1


def test_forwarded_header_is_ignored_unless_proxies_are_declared():
    h = {"x-forwarded-for": "6.6.6.6, 9.9.9.9"}
    assert guard.client_ip(h, "10.0.0.1", 0) == "10.0.0.1"
    assert guard.client_ip(h, "10.0.0.1", 1) == "9.9.9.9"        # the address our own proxy saw
    assert guard.client_ip(h, "10.0.0.1", 2) == "6.6.6.6"
    assert guard.client_ip({}, "10.0.0.1", 1) == "10.0.0.1"      # header missing: fall back
    assert guard.client_ip({"x-forwarded-for": "6.6.6.6"}, "10.0.0.1", 2) == "10.0.0.1"
    assert guard.client_ip({}, None, 0) == "unknown"


def test_admin_token_comparison():
    guard.SETTINGS.admin_token = "s3cret"
    assert guard.admin_allowed("s3cret")
    assert not guard.admin_allowed("s3cre") and not guard.admin_allowed("") and not guard.admin_allowed(None)
    guard.SETTINGS.admin_token = ""
    assert not guard.admin_allowed("") and not guard.admin_allowed("anything")


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("RATE_ANALYZE_PER_MIN", "7")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", " https://a.example , https://b.example ")
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "oops")
    s = guard.Settings.from_env()
    assert s.analyze_per_min == 7 and s.trusted_proxies == 0
    assert s.cors_origins == ["https://a.example", "https://b.example"]


# ------------------------------------------------------------ rate limiting on the real app

def test_analyze_is_rate_limited_and_other_routes_are_not_affected(client):
    guard.SETTINGS.analyze_per_min = 2
    assert analyze(client).status_code == 200
    assert analyze(client).status_code == 200
    r = analyze(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert "Too many" in r.json()["detail"]
    assert client.get("/api/health").status_code == 200


def test_feedback_has_its_own_limit(client):
    guard.SETTINGS.feedback_per_min = 2
    codes = [client.post("/api/feedback", data={"analysis_token": "x", "verdict": "match"}).status_code
             for _ in range(3)]
    assert codes == [400, 400, 429]
    assert client.delete("/api/feedback/abc", headers={"X-Delete-Token": "x"}).status_code == 429


def test_spoofed_forwarded_header_does_not_dodge_the_limit(client):
    guard.SETTINGS.general_per_min = 2
    codes = [client.get("/api/health", headers={"X-Forwarded-For": f"8.8.8.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_forwarded_header_is_used_when_a_proxy_is_declared(client):
    guard.SETTINGS.general_per_min = 1
    guard.SETTINGS.trusted_proxies = 1
    a = client.get("/api/health", headers={"X-Forwarded-For": "5.5.5.5"}).status_code
    b = client.get("/api/health", headers={"X-Forwarded-For": "5.5.5.5"}).status_code
    c = client.get("/api/health", headers={"X-Forwarded-For": "7.7.7.7"}).status_code
    assert (a, b, c) == (200, 429, 200)


# ------------------------------------------------------------ sizes

def test_declared_body_size_is_refused_before_reading(client):
    guard.SETTINGS.max_body_bytes = 1000
    r = client.post("/api/analyze", data={"crop": "rice"},
                    files={"image": ("a.jpg", b"x" * 5000, "image/jpeg")})
    assert r.status_code == 413


def test_oversized_photo_is_refused_while_streaming(client):
    guard.SETTINGS.max_body_bytes = 50 * 1024 * 1024
    big = b"\xff\xd8\xff" + b"\0" * (main.MAX_IMAGE_BYTES + 1024)
    r = client.post("/api/analyze", data={"crop": "rice"}, files={"image": ("a.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and "10 MB" in r.json()["detail"]


def test_garbage_and_empty_uploads_are_clean_400s(client):
    r = client.post("/api/analyze", data={"crop": "rice"}, files={"image": ("a.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/api/analyze", data={"crop": "rice"}, files={"image": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_busy_server_answers_503_with_retry_after(client, monkeypatch):
    import asyncio
    guard.SETTINGS.analysis_queue_seconds = 1
    monkeypatch.setattr(main, "ANALYSIS_SEM", asyncio.Semaphore(0))
    r = analyze(client)
    assert r.status_code == 503 and r.headers["retry-after"] == "10"


def test_a_slot_is_released_after_a_failed_analysis(client, monkeypatch):
    import asyncio
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(main, "ANALYSIS_SEM", sem)
    assert client.post("/api/analyze", data={"crop": "rice"},
                       files={"image": ("a.jpg", b"junk", "image/jpeg")}).status_code == 400
    assert analyze(client).status_code == 200            # would hang or 503 if the slot leaked
    assert sem._value == 1


# ------------------------------------------------------------ private routes

@pytest.mark.parametrize("path", ["/api/coverage", "/api/dataset/stats", "/api/dataset/class/x",
                                  "/api/dataset/sample-image/x"])
def test_debug_routes_need_the_admin_token(client, path):
    assert client.get(path).status_code == 404
    assert client.get(path, headers={"X-Admin-Token": "wrong"}).status_code == 404
    assert client.get(path, headers={"X-Admin-Token": "test-admin-token"}).status_code in (200, 404)


def test_coverage_works_with_the_token(client):
    r = client.get("/api/coverage", headers={"X-Admin-Token": "test-admin-token"})
    assert r.status_code == 200 and "csv_remedies" in r.json()


def test_debug_routes_are_off_when_no_admin_token_is_configured(client):
    guard.SETTINGS.admin_token = ""
    assert client.get("/api/coverage", headers={"X-Admin-Token": ""}).status_code == 404
    assert client.get("/api/coverage", headers={"X-Admin-Token": "test-admin-token"}).status_code == 404


def test_public_health_reveals_almost_nothing(client):
    body = client.get("/api/health").json()
    assert set(body) == {"ok", "version", "focus"}
    detail = client.get("/api/health", headers={"X-Admin-Token": "test-admin-token"}).json()
    assert detail["diseases_registered"] == len(main.DISEASES_BY_ID) and "rice_model_loaded" in detail


def test_api_docs_are_off_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_the_page_never_asks_for_the_debug_routes():
    html = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")
    for route in ("/api/coverage", "/api/dataset"):
        assert route not in html


# ------------------------------------------------------------ headers and cors

def test_security_headers_on_page_and_api(client):
    page = client.get("/")
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "object-src 'none'" in page.headers["content-security-policy"]
    api = client.get("/api/health")
    assert api.headers["cache-control"] == "no-store" and api.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in api.headers


def test_csp_can_be_switched_off(client):
    guard.SETTINGS.csp = "off"
    assert "content-security-policy" not in client.get("/").headers


def test_the_csp_allows_everything_the_page_really_loads():
    html = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")
    csp = guard.DEFAULT_CSP
    for host in ("fonts.googleapis.com", "fonts.gstatic.com", "api.open-meteo.com",
                 "geocoding-api.open-meteo.com", "api.bigdatacloud.net"):
        if host in html:
            assert host in csp, host
    assert "<script src=" not in html.replace(" ", "")


def test_no_cross_site_access_by_default(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    r = client.options("/api/analyze", headers={"Origin": "https://evil.example",
                                                "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_listed_origin_is_allowed_and_others_are_not():
    app = FastAPI()

    @app.get("/x")
    def x():
        return {"ok": 1}
    main.configure_cors(app, ["https://farm.example"])
    c = TestClient(app)
    ok = c.get("/x", headers={"Origin": "https://farm.example"})
    assert ok.headers["access-control-allow-origin"] == "https://farm.example"
    assert "access-control-allow-credentials" not in ok.headers
    bad = c.get("/x", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
    pre = c.options("/x", headers={"Origin": "https://farm.example", "Access-Control-Request-Method": "DELETE",
                                   "Access-Control-Request-Headers": "x-delete-token"})
    assert pre.status_code == 200 and "DELETE" in pre.headers["access-control-allow-methods"]


def test_empty_cors_list_adds_no_middleware():
    app = FastAPI()
    main.configure_cors(app, [])
    assert not app.user_middleware


# ------------------------------------------------------------ forged predictions

def forged_preds():
    return json.dumps([{"class": "Rice___Neck_Blast", "confidence": 0.99}])


def test_browser_sent_predictions_are_ignored_by_default(client):
    r = analyze(client, tflite_top3=forged_preds())
    assert r.status_code == 200
    assert r.json()["evidence_sources"]["tflite"] == []


def test_browser_sent_predictions_work_when_the_operator_allows_them(client, monkeypatch):
    monkeypatch.setattr(main, "ALLOW_CLIENT_ML", True)
    body = analyze(client, tflite_top3=forged_preds()).json()
    assert body["evidence_sources"]["rice_model_used"] is True
    assert body["evidence_sources"]["tflite"][0]["class"] == "Rice___Neck_Blast"


# ------------------------------------------------------------ the pubmed cache

def test_pubmed_answers_are_cached(client, monkeypatch):
    calls = []

    async def fake(term, retmax=4):
        calls.append(term)
        return [{"kind": "pubmed", "pmid": "1", "title": "t", "org": "o", "authors": "", "url": "u", "score": 0.9}]
    monkeypatch.setattr(main, "pubmed_search", fake)
    main._pubmed_cache.clear()
    d = main.DISEASES_BY_ID["rice_blast"]
    import asyncio
    for _ in range(3):
        cits, live, failed = asyncio.run(main.retrieve(d, True))
    assert len(calls) == 1 and live == 1 and not failed
    main._pubmed_cache.clear()


def test_failed_pubmed_lookups_are_not_cached(monkeypatch):
    calls = []

    async def fake(term, retmax=4):
        calls.append(term)
        return []
    monkeypatch.setattr(main, "pubmed_search", fake)
    main._pubmed_cache.clear()
    import asyncio
    for _ in range(2):
        asyncio.run(main.retrieve(main.DISEASES_BY_ID["rice_blast"], True))
    assert len(calls) == 2
    main._pubmed_cache.clear()

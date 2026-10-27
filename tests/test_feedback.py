import io
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

import feedback as fb
import main
import synth
from conftest import analyze


def photo_with_exif():
    """A JPEG that carries GPS and camera data in its EXIF block."""
    im = synth.make_leaf(size=(1600, 1200), spots=6)
    exif = Image.Exif()
    exif[0x010F] = "TestPhoneMaker"                 # Make
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (12.0, 58.0, 18.0), "E", (77.0, 35.0, 41.0)
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def analysed(client, **kw):
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=8), **kw)
    assert r.status_code == 200, r.text
    return r.json()


def send(client, body, verdict="match", **extra):
    files = extra.pop("files", None)
    data = {"analysis_token": body["analysis_token"], "verdict": verdict, "language": "en"}
    data.update(extra)
    return client.post("/api/feedback", data=data, files=files)


# ------------------------------------------------------------ signed token

def test_token_round_trip_and_tamper(store):
    token = store.sign_analysis({"id": "a1", "crop": "rice", "pred": "rice_blast"})
    assert store.verify_analysis(token)["pred"] == "rice_blast"
    payload, mac = token.split(".")
    assert store.verify_analysis(payload + "." + "0" * len(mac)) is None
    forged = fb._b64e(b'{"id":"a1","crop":"rice","pred":"healthy","ts":%d}' % int(time.time()))
    assert store.verify_analysis(forged + "." + mac) is None
    assert store.verify_analysis("garbage") is None and store.verify_analysis("") is None


def test_token_expires(store):
    now = time.time()
    token = store.sign_analysis({"id": "a1", "crop": "rice"}, now=now)
    assert store.verify_analysis(token, now=now + fb.TOKEN_TTL_SECONDS - 5) is not None
    assert store.verify_analysis(token, now=now + fb.TOKEN_TTL_SECONDS + 5) is None
    assert store.verify_analysis(token, now=now - 600) is None          # from the future


def test_token_made_with_another_secret_is_rejected(store, tmp_path):
    other = fb.Store(tmp_path / "other", secret=b"another-secret")
    assert store.verify_analysis(other.sign_analysis({"id": "x", "crop": "rice"})) is None


# ------------------------------------------------------------ the endpoint

def test_answer_without_photo_stores_only_the_answer(client, store):
    body = analysed(client)
    r = send(client, body, "match")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["photo_saved"] is False and out["delete_token"]
    row = store.get(out["id"])
    assert row["verdict"] == "match" and row["consent_photo"] == 0
    assert row["photo_file"] is None and row["lat_r"] is None and row["lon_r"] is None
    assert row["analysis_id"] == body["analysis_id"] and row["crop"] == "rice"
    assert list(store.images.iterdir()) == []


def test_what_the_app_said_comes_from_the_signed_token_not_the_request(client, store):
    body = analysed(client)
    r = send(client, body, "different", reported_id="rice_blast", predicted_id="healthy", confidence="high")
    assert r.status_code == 200
    row = store.get(r.json()["id"])
    signed = store.verify_analysis(body["analysis_token"])
    assert row["predicted_id"] == signed["pred"] and row["confidence"] == signed["conf"]


def test_photo_is_not_kept_while_no_privacy_contact_is_set(client, store):
    body = analysed(client)
    r = send(client, body, "match", consent_photo="true", latitude="12.97", longitude="77.59",
             files={"image": ("p.jpg", photo_with_exif(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["photo_saved"] is False
    row = store.get(r.json()["id"])
    assert row["consent_photo"] == 0 and row["lat_r"] is None
    assert list(store.images.iterdir()) == []


def test_consented_photo_is_saved_without_exif_and_location_is_rounded(client, store, with_contact):
    body = analysed(client)
    assert body["photo_feedback_enabled"] is True
    r = send(client, body, "match", consent_photo="true", latitude="12.97164", longitude="77.59461",
             files={"image": ("p.jpg", photo_with_exif(), "image/jpeg")})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["photo_saved"] is True
    row = store.get(out["id"])
    assert row["consent_photo"] == 1 and (row["lat_r"], row["lon_r"]) == (13.0, 77.6)
    saved = Image.open(store.images / row["photo_file"])
    assert saved.format == "JPEG" and max(saved.size) <= fb.MAX_PHOTO_SIDE
    assert len(saved.getexif()) == 0, "camera and GPS data must be gone"
    assert b"TestPhoneMaker" not in (store.images / row["photo_file"]).read_bytes()


def test_location_without_consent_is_dropped(client, store, with_contact):
    body = analysed(client)
    r = send(client, body, "match", latitude="12.97", longitude="77.59")
    row = store.get(r.json()["id"])
    assert row["lat_r"] is None and row["lon_r"] is None


def test_consent_without_a_photo_is_an_error(client, with_contact):
    body = analysed(client)
    r = send(client, body, "match", consent_photo="true")
    assert r.status_code == 400


def test_bad_photo_is_rejected_and_nothing_is_stored(client, store, with_contact):
    body = analysed(client)
    r = send(client, body, "match", consent_photo="true",
             files={"image": ("p.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400 and store.count() == 0


def test_second_answer_for_the_same_result_is_refused(client, store):
    body = analysed(client)
    assert send(client, body, "match").status_code == 200
    assert send(client, body, "different", reported_id="healthy").status_code == 409
    assert store.count() == 1


def test_duplicate_does_not_leave_a_stray_photo(client, store, with_contact):
    body = analysed(client)
    files = lambda: {"image": ("p.jpg", photo_with_exif(), "image/jpeg")}   # noqa: E731
    assert send(client, body, "match", consent_photo="true", files=files()).status_code == 200
    assert send(client, body, "match", consent_photo="true", files=files()).status_code == 409
    assert len(list(store.images.iterdir())) == 1


def test_invalid_input_is_rejected(client, store):
    body = analysed(client)
    assert client.post("/api/feedback", data={"analysis_token": "x.y", "verdict": "match"}).status_code == 400
    assert send(client, body, "maybe").status_code == 400
    assert send(client, body, "different", reported_id="no_such_disease").status_code == 400
    assert send(client, body, "different").status_code == 400                    # needs a disease
    assert send(client, body, "different", reported_id="tomato_early_blight").status_code == 400   # wrong crop
    assert store.count() == 0


def test_a_different_diagnosis_is_stored(client, store):
    body = analysed(client)
    rice_ids = [o["id"] for o in body["diagnosis_options"]]
    assert rice_ids and all(i in main.DISEASES_BY_ID for i in rice_ids)
    r = send(client, body, "different", reported_id=rice_ids[-1])
    assert r.status_code == 200
    row = store.get(r.json()["id"])
    assert row["verdict"] == "different" and row["reported_id"] == rice_ids[-1]
    r2 = send(client, analysed(client), "different", reported_id="healthy")
    assert r2.status_code == 200 and store.get(r2.json()["id"])["reported_id"] == "healthy"


def test_reported_id_is_ignored_unless_the_answer_is_different(client, store):
    body = analysed(client)
    r = send(client, body, "match", reported_id="rice_blast")
    assert store.get(r.json()["id"])["reported_id"] is None


def test_no_personal_data_columns_exist(store):
    with sqlite3.connect(store.db_path) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(feedback)")}
    assert not ({"ip", "ip_address", "name", "phone", "email", "user_agent"} & cols)


# ------------------------------------------------------------ deleting

def test_delete_needs_the_right_code_and_removes_the_photo(client, store, with_contact):
    body = analysed(client)
    out = send(client, body, "match", consent_photo="true",
               files={"image": ("p.jpg", photo_with_exif(), "image/jpeg")}).json()
    path = store.images / store.get(out["id"])["photo_file"]
    assert path.exists()
    assert client.delete(f"/api/feedback/{out['id']}", headers={"X-Delete-Token": "wrong"}).status_code == 404
    assert client.delete(f"/api/feedback/{out['id']}").status_code == 404
    assert path.exists() and store.get(out["id"]) is not None
    assert client.delete(f"/api/feedback/{out['id']}", headers={"X-Delete-Token": out["delete_token"]}).status_code == 200
    assert not path.exists() and store.get(out["id"]) is None
    assert client.delete(f"/api/feedback/{out['id']}", headers={"X-Delete-Token": out["delete_token"]}).status_code == 404


def test_unknown_id_and_wrong_code_look_the_same(client):
    a = client.delete("/api/feedback/doesnotexist", headers={"X-Delete-Token": "x"})
    body = analysed(client)
    out = send(client, body, "match").json()
    b = client.delete(f"/api/feedback/{out['id']}", headers={"X-Delete-Token": "x"})
    assert (a.status_code, a.json()) == (b.status_code, b.json())


def test_delete_token_is_not_stored_in_clear(store, client):
    out = send(client, analysed(client), "match").json()
    row = store.get(out["id"])
    assert out["delete_token"] not in row.values()


# ------------------------------------------------------------ retention

def test_purge_removes_old_records_and_photos_only(store):
    def add(i):
        return store.add(analysis={"id": f"a{i}", "crop": "rice"}, verdict="match", reported_id=None,
                         consent_photo=True, photo_jpeg=fb.encode_photo(synth.make_leaf(size=(300, 200))),
                         lat=12.9, lon=77.5, language="en", app_version="t")[0]
    old, new = add(1), add(2)
    with sqlite3.connect(store.db_path) as con:
        long_ago = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat(timespec="seconds")
        con.execute("UPDATE feedback SET created_at=? WHERE id=?", (long_ago, old))
    old_file = store.images / store.get(old)["photo_file"]
    assert store.purge_older_than(0) == 0 and old_file.exists()          # 0 = keep forever
    assert store.purge_older_than(730) == 1
    assert store.get(old) is None and not old_file.exists()
    assert store.get(new) is not None and (store.images / store.get(new)["photo_file"]).exists()


def test_secret_is_created_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("FEEDBACK_SECRET", raising=False)
    a = fb.load_secret(tmp_path / "s")
    assert a and fb.load_secret(tmp_path / "s") == a
    monkeypatch.setenv("FEEDBACK_SECRET", "from-env")
    assert fb.load_secret(tmp_path / "s") == b"from-env"


def test_the_exif_test_photo_really_carries_exif():
    raw = photo_with_exif()
    assert b"TestPhoneMaker" in raw
    assert Image.open(io.BytesIO(raw)).getexif().get(0x010F) == "TestPhoneMaker"

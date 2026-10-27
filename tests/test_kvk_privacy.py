import csv
import io
import json
import re

from PIL import Image

import main
import synth

PHONE = re.compile(r"^(0\d{2,4}-?\d{5,8}|[6-9]\d{9}|1800-?\d{3}-?\d{4})$")


def data():
    return json.loads((main.BASE_DIR / "kvk_contacts.json").read_text(encoding="utf-8"))


def test_endpoint_serves_the_file(client):
    r = client.get("/api/kvk")
    assert r.status_code == 200 and r.json() == data()


def test_kisan_call_centre_number():
    kcc = data()["national"]["kisan_call_centre"]
    assert kcc["phone"] == "1800-180-1551"
    for lang in ("en", "hi", "kn"):
        assert kcc["hours"][lang] and kcc["note"][lang]


def test_every_karnataka_district_is_present_with_a_kannada_name():
    d = data()["districts"]
    assert len(d) == 31
    for key in ("bengaluru_urban", "bengaluru_rural", "belagavi", "vijayapura", "kalaburagi", "mysuru",
                "shivamogga", "dakshina_kannada", "uttara_kannada", "vijayanagara", "koppal", "yadgir"):
        assert key in d
    for key, dist in d.items():
        assert dist["name"]["en"] and re.search(r"[ಀ-೿]", dist["name"]["kn"]), key


def test_district_keys_match_the_districts_the_page_offers():
    page = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")
    page_ids = re.findall(r"^\s*\['([a-z_]+)', [\d.]+, [\d.]+, '", page, re.M)
    assert len(page_ids) == 31
    assert sorted(page_ids) == sorted(data()["districts"])


def test_every_district_has_a_kvk_or_points_to_one():
    d = data()["districts"]
    for key, dist in d.items():
        assert dist["kvks"] or dist.get("see"), key
        for other in dist.get("see", []):
            assert other in d and d[other]["kvks"], (key, other)


def test_phone_numbers_look_like_phone_numbers():
    bad = []
    for key, dist in data()["districts"].items():
        for kvk in dist["kvks"]:
            assert kvk["name"].strip()
            for p in kvk["phones"] + kvk.get("more_phones", []):
                if not PHONE.match(p):
                    bad.append((key, p))
    nodal = data()["nodal"]["phone"]
    assert PHONE.match(nodal), nodal
    assert not bad, bad


def test_every_centre_has_a_number_and_the_page_shows_at_most_two():
    for key, dist in data()["districts"].items():
        for kvk in dist["kvks"]:
            assert kvk["phones"], (key, kvk["name"])
            assert len(kvk["phones"]) <= 2, (key, kvk["name"])


def test_owners_directory_numbers_come_first():
    d = data()["districts"]
    assert d["dharwad"]["kvks"][0]["phones"] == ["0836-2484411", "0836-2970246"]
    assert d["uttara_kannada"]["kvks"][0]["phones"] == ["08384-228411"]      # was empty before the directory
    assert d["yadgir"]["kvks"][0]["phones"] == ["08479-202280"]
    assert d["koppal"]["kvks"][0]["phones"] == ["08533-272518"]
    assert d["ramanagara"]["kvks"][0]["phones"] == ["08113-206584"]
    assert d["belagavi"]["kvks"][1]["phones"] == ["9535604747"]               # "+91 " is not part of the number
    # a third number is kept, but behind the two the page shows
    assert d["vijayapura"]["kvks"][0]["more_phones"] == ["9481503971"]


def test_districts_the_directory_added_have_their_own_centre():
    d = data()["districts"]
    assert "GKVK" in d["bengaluru_urban"]["kvks"][0]["name"] and d["bengaluru_urban"]["kvks"][0]["phones"] == ["080-23418833"]
    assert "Vyasanakere" in d["vijayanagara"]["kvks"][0]["name"]
    assert not d["bengaluru_urban"].get("see") and not d["vijayanagara"].get("see")


def test_nodal_office_is_listed():
    nodal = data()["nodal"]
    assert "ATARI" in nodal["name"] and nodal["phone"] == "080-23410614" and "Hebbal" in nodal["address"]


def test_no_phone_appears_under_two_districts():
    seen = {}
    for key, dist in data()["districts"].items():
        for kvk in dist["kvks"]:
            for p in kvk["phones"] + kvk.get("more_phones", []):
                digits = re.sub(r"\D", "", p)
                assert digits not in seen or seen[digits] == key, (p, seen.get(digits), key)
                seen[digits] = key


def test_call_sheet_lists_exactly_the_centres_in_the_file():
    """The sheet people use to phone each centre must not drift from what the app shows."""
    with open(main.BASE_DIR / "kvk_call_sheet.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    in_sheet = sorted((r["district"], r["kvk"]) for r in rows)
    in_file = sorted((key, kvk["name"]) for key, dist in data()["districts"].items() for kvk in dist["kvks"])
    assert in_sheet == in_file
    for r in rows:
        assert r["called_on"] == "" and r["number_works"] == "", "the sheet is for people to fill in"


def test_file_says_it_is_unverified_and_cites_sources():
    f = data()
    assert f["verified"] is False and f["retrieved"]
    assert all(s["url"].startswith("https://") for s in f["sources"])


# ------------------------------------------------------------ privacy page

def test_privacy_page_without_contact_shows_a_notice(client):
    r = client.get("/privacy")
    assert r.status_code == 200 and "{{" not in r.text
    assert "has not yet added a contact" in r.text
    assert "Kindwise" not in r.text


def test_privacy_page_with_contact_and_retention(client, monkeypatch):
    monkeypatch.setattr(main, "PRIVACY_CONTACT", "help@example.org")
    monkeypatch.setattr(main, "PRIVACY_OPERATOR", "Example Trust")
    monkeypatch.setattr(main, "FEEDBACK_RETENTION_DAYS", 365)
    t = client.get("/privacy").text
    assert "help@example.org" in t and "Example Trust" in t and "365 days" in t
    assert "has not yet added a contact" not in t


def test_privacy_page_mentions_kindwise_only_when_it_is_used(client, monkeypatch):
    monkeypatch.setattr(main, "KINDWISE_KEY", "abc")
    assert "Kindwise" in client.get("/privacy").text


def test_privacy_page_escapes_the_contact_text(client, monkeypatch):
    monkeypatch.setattr(main, "PRIVACY_CONTACT", "<script>alert(1)</script>")
    t = client.get("/privacy").text
    assert "<script>alert(1)</script>" not in t and "&lt;script&gt;" in t


def test_privacy_page_has_the_three_language_summary_and_delete_form(client):
    t = client.get("/privacy").text
    assert 'lang="kn"' in t and 'lang="hi"' in t and 'id="delbtn"' in t
    assert "/api/feedback/" in t and "X-Delete-Token" in t


def test_photos_are_sent_to_kindwise_without_gps(client, monkeypatch):
    """What leaves the server for the outside service must not carry the phone's location."""
    seen = {}

    async def spy(raw, crop, lang="en"):
        seen["raw"] = raw
        return None
    monkeypatch.setattr(main, "KINDWISE_KEY", "abc")
    monkeypatch.setattr(main, "kindwise_analyze", spy)
    im = synth.make_leaf(size=(1600, 1200), spots=4)
    exif = Image.Exif()
    exif[0x010F] = "TestPhoneMaker"
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    r = client.post("/api/analyze", data={"crop": "rice"}, files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    assert b"TestPhoneMaker" not in seen["raw"] and len(Image.open(io.BytesIO(seen["raw"])).getexif()) == 0

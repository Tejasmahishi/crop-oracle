"""The real page in headless Chromium against the real app (network stubbed).

Skipped automatically when Playwright or its browser is not installed. To run them:
    pip install playwright && playwright install chromium
"""
import re
import sqlite3

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

import feedback as fb  # noqa: E402
import main  # noqa: E402
import synth  # noqa: E402
from livepage import LiveServer, analyse_through_the_ui, stub_outside_world  # noqa: E402


@pytest.fixture(scope="module")
def live():
    server = LiveServer().start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def browser():
    pw = playwright_sync.sync_playwright().start()
    try:
        b = pw.chromium.launch(args=["--no-sandbox", "--use-fake-device-for-media-stream",
                                     "--use-fake-ui-for-media-stream"])
    except Exception as e:                                  # browser not installed
        pw.stop()
        pytest.skip(f"Chromium is not available: {e}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def page(browser, live):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, permissions=["camera"])
    pg = ctx.new_page()
    stub_outside_world(pg)
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(str(e)))
    pg.on("console", lambda m: pg.problems.append(m.text)
          if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
    pg.goto(live.url)
    yield pg
    ctx.close()
    assert not pg.problems, pg.problems


def save(tmp_path, name, im):
    path = tmp_path / f"{name}.jpg"
    im.convert("RGB").save(path, "JPEG", quality=92)
    return path


def text(page, selector):
    return page.inner_text(selector)


def leaf(tmp_path, name="leaf", **kw):
    return save(tmp_path, name, synth.make_leaf(size=(1200, 900), spots=kw.pop("spots", 8), **kw))


def step_texts(page):
    return [e.inner_text() for e in page.query_selector_all("#plan .step-card .s-text")]


def to_plan(page):
    page.click("#btnGoSolution")
    page.wait_for_selector("#s5.active")


# ------------------------------------------------------------ photo guide and on-phone check

def test_guide_and_guided_camera_button_are_there(page):
    assert page.is_visible("#photoGuide summary")
    page.click("#photoGuide summary")
    assert len(page.query_selector_all("#photoGuide li")) == 5
    assert page.is_visible("#btnGuided")           # localhost counts as a secure page


def test_on_phone_check_warns_about_dark_tiny_and_very_blurry_photos(page, tmp_path):
    cases = {
        "dark": (synth.darkened(synth.make_leaf(size=(1200, 900), spots=8), 0.22), "too dark"),
        "tiny": (synth.make_leaf(size=(300, 225), spots=2), "too small"),
        "smear": (synth.blurred(synth.make_leaf(size=(1200, 900), spots=8), 40), "blurry"),
    }
    for name, (im, words) in cases.items():
        page.set_input_files("#fileIn", str(save(tmp_path, name, im)))
        page.wait_for_selector("#photoCheck:not(.hidden)")
        assert words in text(page, "#photoCheck"), name
    page.set_input_files("#fileIn", str(leaf(tmp_path)))
    page.wait_for_selector("#photoCheck.hidden", state="attached")


def test_on_phone_warning_changes_language(page, tmp_path):
    page.set_input_files("#fileIn", str(save(tmp_path, "d", synth.darkened(synth.make_leaf(size=(1200, 900)), 0.2))))
    page.wait_for_selector("#photoCheck:not(.hidden)")
    page.click('#langpick button[data-l="kn"]')
    page.wait_for_function("() => document.querySelector('#photoCheck').textContent.includes('ಕತ್ತಲಾಗಿದೆ')")


def test_demo_leaves_are_not_checked(page):
    page.click("details.samples-box summary")
    page.click("#samples .sample >> nth=0")
    page.wait_for_selector("#dropHas:not(.hidden)")
    assert page.is_hidden("#photoCheck")


def test_guided_camera_takes_a_picture_and_lets_go_of_the_camera(page):
    page.click("#btnGuided")
    page.wait_for_selector("#cam:not(.hidden)")
    page.wait_for_function("() => document.querySelector('#camVideo').videoWidth > 0")
    assert page.is_visible("#cam .cam-frame")
    page.click("#camShot")
    page.wait_for_selector("#cam.hidden", state="attached")
    page.wait_for_selector("#dropHas:not(.hidden)")
    assert page.evaluate("() => camStream === null")
    page.wait_for_function("() => state.imageBlob !== null")
    assert page.evaluate("() => state.imageSource") == "upload"


def test_cancelling_the_camera_releases_it(page):
    page.click("#btnGuided")
    page.wait_for_selector("#cam:not(.hidden)")
    page.keyboard.press("Escape")
    page.wait_for_selector("#cam.hidden", state="attached")
    assert page.evaluate("() => camStream === null")
    assert page.is_hidden("#dropHas")


def test_camera_failure_falls_back_with_a_message(browser, live):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})      # camera permission not granted
    pg = ctx.new_page()
    stub_outside_world(pg)
    pg.add_init_script("navigator.mediaDevices.getUserMedia = () => Promise.reject(new Error('denied'))")
    pg.goto(live.url)
    pg.click("#btnGuided")
    pg.wait_for_selector("#toast.show")
    assert "camera" in text(pg, "#toast").lower() and pg.is_hidden("#cam")
    ctx.close()


# ------------------------------------------------------------ results: uncertain, marks, help

def test_unsure_result_shows_no_spray_steps_and_offers_help(page, tmp_path, force_confidence, force_diseased):
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))
    assert text(page, "#diagBox .eyebrow") == "Best guess only"
    assert page.query_selector("#diagBox .score") is None
    assert "No spray advice is shown" in text(page, "#diagBox .note-box")
    assert page.get_attribute('#helpBox a[href^="tel:"]', "href") == "tel:18001801551"
    to_plan(page)
    assert "No spray advice yet" in text(page, "#plan .banner")
    assert page.get_attribute("#plan .banner a", "href") == "tel:18001801551"
    steps = " | ".join(step_texts(page))
    assert "photos" in steps.lower() and "Krishi Vigyan Kendra" in steps or "KVK" in steps
    # The KVK step may warn "before you spray anything". What must never appear is advice to spray or fertilise.
    advice = steps.replace("before you spray anything", "")
    assert not re.search(r"spray|fertili[sz]er", advice, re.I), steps
    assert "Best guess only" in text(page, "#summaryHero")
    assert page.is_hidden("#remedyCard")


def test_confident_result_still_shows_the_normal_plan(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path))
    assert page.query_selector("#diagBox .eyebrow") is None and page.query_selector("#diagBox .score")
    assert page.inner_html("#helpBox").strip() == ""
    to_plan(page)
    assert page.query_selector("#plan .banner") is None
    assert re.search(r"spray", " ".join(step_texts(page)), re.I) or page.query_selector("#plan .step-card")


def test_marks_sit_exactly_over_the_photo_and_can_be_hidden(page, tmp_path, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path, spots=14))
    page.wait_for_selector("#ovCard:not(.hidden)")
    page.wait_for_function("() => document.querySelector('#ovMarks').complete && document.querySelector('#ovPhoto').complete")
    box = page.evaluate("""() => { const p = document.querySelector('#ovPhoto').getBoundingClientRect(),
                                 m = document.querySelector('#ovMarks').getBoundingClientRect();
                                 const mi = document.querySelector('#ovMarks'), pi = document.querySelector('#ovPhoto');
                                 return {p: [p.x, p.y, p.width, p.height], m: [m.x, m.y, m.width, m.height],
                                         pa: pi.naturalWidth / pi.naturalHeight, ma: mi.naturalWidth / mi.naturalHeight}; }""")
    for a, b in zip(box["p"], box["m"]):
        assert abs(a - b) <= 1.5
    assert abs(box["pa"] - box["ma"]) / box["pa"] < 0.02, "marks must have the photo's proportions"
    assert page.get_attribute("#btnMarks", "aria-pressed") == "true"
    page.click("#btnMarks")
    assert page.is_hidden("#ovMarks") and page.get_attribute("#btnMarks", "aria-pressed") == "false"
    page.click("#btnMarks")
    assert page.is_visible("#ovMarks")


def test_photo_problems_are_listed_with_a_retake_button(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, save(tmp_path, "g", synth.with_glare(synth.make_leaf(size=(1200, 900), spots=8))))
    assert "glare" in text(page, "#qualityBox").lower()
    page.click("#btnRetake")
    page.wait_for_selector("#s1.active")
    assert page.is_hidden("#dropHas")


def test_unreadable_healthy_looking_photo_is_not_called_healthy(page, tmp_path, monkeypatch):
    original = main.rank_candidates

    def healthy_looking(*a, **k):
        scored = original(*a, **k)
        scored[0] = {**scored[0], "vision_score": 0.05, "score": 0.1}
        return scored
    monkeypatch.setattr(main, "rank_candidates", healthy_looking)
    analyse_through_the_ui(page, save(tmp_path, "t", synth.make_leaf(size=(200, 150), spots=0)))
    assert text(page, "#h-s3") == "We could not check this photo"
    assert "not good enough" in text(page, "#diagBox")
    # The message may say "we cannot say whether it is healthy"; it must never declare the crop healthy.
    assert page.query_selector("#diagBox .diag.healthy") is None and "✅" not in text(page, "#diagBox")
    assert page.is_hidden("#decisionCard")
    to_plan(page)
    assert len(page.query_selector_all("#plan .step-card")) == 2
    assert "No spray advice yet" in text(page, "#plan .banner")
    page.click("#btnBack4")
    page.click("#btnViewEvidence")
    ev = text(page, "#evidence").lower()
    assert "no spots or yellowing" not in ev and "looks healthy" not in ev and "✅" not in ev


def test_dark_photo_error_says_why(page, tmp_path):
    page.set_input_files("#fileIn", str(save(tmp_path, "d", synth.darkened(synth.make_leaf(size=(1200, 900), spots=8), 0.22))))
    page.click("#btnAnalyze1")
    page.wait_for_selector("#s2.active")
    page.click("#btnAnalyze2")
    page.wait_for_selector("#toast.show")
    assert "too dark" in text(page, "#toast") and page.is_visible("#s2.active")


# ------------------------------------------------------------ expert contacts

def serve_kvk(page, change):
    """Answer the page's /api/kvk call with the real file after `change` has edited a copy of it."""
    import json
    data = json.loads((main.BASE_DIR / "kvk_contacts.json").read_text(encoding="utf-8"))
    change(data["districts"])
    page.route("**/api/kvk", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(data)))


def test_help_card_lists_the_district_kvk_with_call_buttons(page, tmp_path, force_confidence, force_diseased):
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))
    page.select_option("#helpDistrict", "dharwad")
    page.wait_for_selector("#kvkList .kvk-item")
    assert "Saidapur" in text(page, "#kvkList")
    hrefs = [a.get_attribute("href") for a in page.query_selector_all('#kvkList a[href^="tel:"]')]
    assert hrefs == ["tel:08362484411", "tel:08362970246"]           # the directory's number first
    assert page.is_visible('#kvkList .kvk-web')                       # website sits on the same row as the call buttons


def test_a_centre_with_no_number_says_so_and_still_offers_the_kisan_call_centre(page, tmp_path, force_confidence, force_diseased):
    serve_kvk(page, lambda d: d["uttara_kannada"]["kvks"][0].update(phones=[]))
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))
    page.select_option("#helpDistrict", "uttara_kannada")
    page.wait_for_function("() => document.querySelector('#kvkList').textContent.includes('No phone number')")
    assert page.query_selector('#kvkList a[href^="tel:"]') is None
    assert page.get_attribute('#helpBox .kcc a[href^="tel:"]', "href") == "tel:18001801551"


def test_bengaluru_urban_has_its_own_centre(page, tmp_path, force_confidence, force_diseased):
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))      # the page starts in Bengaluru Urban
    page.wait_for_selector("#kvkList .kvk-item")
    assert page.input_value("#helpDistrict") == "bengaluru_urban"
    assert "GKVK" in text(page, "#kvkList") and "nearest" not in text(page, "#kvkList").lower()
    assert page.get_attribute('#kvkList a[href^="tel:"]', "href") == "tel:08023418833"


def test_district_without_a_kvk_points_to_its_neighbours(page, tmp_path, force_confidence, force_diseased):
    def no_own_centre(d):
        d["bengaluru_urban"]["kvks"], d["bengaluru_urban"]["see"] = [], ["bengaluru_rural", "ramanagara"]
    serve_kvk(page, no_own_centre)
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))
    page.wait_for_selector("#kvkList .kvk-item")
    assert "nearest" in text(page, "#kvkList").lower() and "Hadonahalli" in text(page, "#kvkList")


# ------------------------------------------------------------ feedback

def feedback_rows(store):
    with sqlite3.connect(store.db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute("SELECT * FROM feedback")]


def test_yes_it_was_right(page, tmp_path, store, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    assert page.is_disabled('#feedbackBox [data-fb="send"]') and page.is_hidden('#feedbackBox [data-fb="send"]')
    assert page.query_selector("#fbConsent") is None, "no photo option without a privacy contact"
    page.click('#feedbackBox [data-v="match"]')
    page.click('#feedbackBox [data-fb="send"]')
    page.wait_for_selector('#feedbackBox [data-fb="delete"]')
    assert "saved" in text(page, "#feedbackBox")
    rows = feedback_rows(store)
    assert len(rows) == 1 and rows[0]["verdict"] == "match" and rows[0]["consent_photo"] == 0
    code = page.inner_text("#feedbackBox .fb-code")
    assert rows[0]["id"] in code


def test_no_it_was_something_else_needs_a_choice(page, tmp_path, store, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    page.click('#feedbackBox [data-v="different"]')
    page.wait_for_selector("#fbWhich")
    options = page.eval_on_selector_all("#fbWhich option", "els => els.map(e => e.value)")
    assert options[0] == "" and options[-2:] == ["healthy", "other"] and len(options) == 1 + len(main.DISEASES_BY_CROP["rice"]) + 2
    page.click('#feedbackBox [data-fb="send"]')
    page.wait_for_selector("#toast.show")
    assert "choose" in text(page, "#toast").lower() and feedback_rows(store) == []
    page.select_option("#fbWhich", options[2])
    page.click('#feedbackBox [data-fb="send"]')
    page.wait_for_selector('#feedbackBox [data-fb="delete"]')
    row = feedback_rows(store)[0]
    assert row["verdict"] == "different" and row["reported_id"] == options[2]


def test_photo_is_kept_only_with_the_tick_and_can_be_deleted(page, tmp_path, store, with_contact, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    assert page.is_hidden("#fbConsent"), "the photo question waits until an answer is chosen"
    page.click('#feedbackBox [data-v="match"]')
    page.wait_for_selector("#fbConsent")
    assert not page.is_checked("#fbConsent"), "the photo box must start unticked"
    page.check("#fbConsent")
    page.click('#feedbackBox [data-fb="send"]')
    page.wait_for_selector('#feedbackBox [data-fb="delete"]')
    assert "photo was saved" in text(page, "#feedbackBox")
    row = feedback_rows(store)[0]
    assert row["consent_photo"] == 1 and row["photo_file"] and (store.images / row["photo_file"]).exists()
    page.click('#feedbackBox [data-fb="delete"]')
    page.wait_for_function("() => document.querySelector('#feedbackBox').textContent.includes('Deleted')")
    assert feedback_rows(store) == [] and list(store.images.iterdir()) == []


def test_unticked_photo_box_sends_no_photo(page, tmp_path, store, with_contact, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    page.click('#feedbackBox [data-v="unsure"]')
    page.wait_for_selector("#fbConsent")
    page.click('#feedbackBox [data-fb="send"]')
    page.wait_for_selector('#feedbackBox [data-fb="delete"]')
    assert feedback_rows(store)[0]["photo_file"] is None and list(store.images.iterdir()) == []


def test_changing_language_keeps_the_answer_in_progress(page, tmp_path, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    page.click('#feedbackBox [data-v="different"]')
    page.wait_for_selector("#fbWhich")
    first = page.eval_on_selector("#fbWhich", "e => e.options[2].value")
    page.select_option("#fbWhich", first)
    page.click('#langpick button[data-l="kn"]')
    page.wait_for_function("() => document.querySelector('#feedbackBox h2').textContent.includes('ಸರಿಯಾಗಿತ್ತೇ')")
    page.wait_for_function("() => !document.querySelector('#results3').classList.contains('busy-veil')")
    assert page.get_attribute('#feedbackBox [data-v="different"]', "aria-pressed") == "true"
    assert page.input_value("#fbWhich") == first


def test_new_analysis_starts_a_fresh_form(page, tmp_path, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    page.click('#feedbackBox [data-v="match"]')
    page.click("#btnGoSolution")
    page.click("#btnRestart")
    page.wait_for_selector("#s1.active")
    assert page.is_hidden("#feedbackBox") and page.is_hidden("#ovCard")
    assert page.inner_html("#helpBox").strip() == "" and page.inner_html("#qualityBox").strip() == ""
    analyse_through_the_ui(page, leaf(tmp_path, "second", spots=5))
    assert page.get_attribute('#feedbackBox [data-v="match"]', "aria-pressed") == "false"


def test_server_busy_and_rate_limit_messages(page, tmp_path, monkeypatch):
    import guard
    guard.SETTINGS.analyze_per_min = 1
    analyse_through_the_ui(page, leaf(tmp_path))
    page.click("#btnRedo")
    page.click("#btnAnalyze1")
    page.click("#btnAnalyze2")
    page.wait_for_selector("#toast.show")
    assert "Too many requests" in text(page, "#toast")


def test_kannada_and_hindi_have_every_new_text():
    page_src = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")

    def keys(lang, nxt):
        block = page_src.split(f"\n  {lang}: {{", 1)[1].split(nxt, 1)[0]
        return set(re.findall(r"^\s+'([a-z0-9_.]+)':", block, re.M))
    en, kn, hi = keys("en", "\n  kn: {"), keys("kn", "\n  hi: {"), keys("hi", "\n};")
    assert {"fb.title", "pq.dark", "help.title", "ov.title", "inc.title", "guide.title"} <= en
    assert not (en - kn), sorted(en - kn)
    assert not (en - hi), sorted(en - hi)


# ------------------------------------------------------------ dark mode

def page_in(browser, live, scheme="light", width=390, height=844):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme)
    pg = ctx.new_page()
    stub_outside_world(pg)
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(str(e)))
    pg.goto(live.url)
    return ctx, pg


def theme_of(pg):
    return pg.evaluate("() => document.documentElement.getAttribute('data-theme')")


def body_colour(pg):
    return pg.evaluate("() => getComputedStyle(document.body).backgroundColor")


def test_page_follows_the_phones_setting(browser, live):
    for scheme in ("dark", "light"):
        ctx, pg = page_in(browser, live, scheme)
        assert theme_of(pg) == scheme
        assert pg.get_attribute("#themeBtn", "aria-pressed") == str(scheme == "dark").lower()
        assert not pg.problems
        ctx.close()


def test_button_switches_theme_and_the_choice_is_remembered(browser, live):
    ctx, pg = page_in(browser, live, "light")
    light_bg = body_colour(pg)
    pg.click("#themeBtn")
    assert theme_of(pg) == "dark" and pg.get_attribute("#themeBtn", "aria-pressed") == "true"
    assert body_colour(pg) != light_bg
    pg.reload()
    assert theme_of(pg) == "dark"                       # kept across a reload
    pg.click("#themeBtn")
    pg.reload()
    assert theme_of(pg) == "light"
    ctx.close()


def test_the_choice_beats_the_phones_setting(browser, live):
    ctx, pg = page_in(browser, live, "dark")
    pg.click("#themeBtn")                               # dark phone, light chosen
    assert theme_of(pg) == "light"
    pg.reload()
    assert theme_of(pg) == "light"
    ctx.close()


def test_phone_setting_is_followed_live_until_the_button_is_used(browser, live):
    ctx, pg = page_in(browser, live, "light")
    pg.emulate_media(color_scheme="dark")
    pg.wait_for_function("() => document.documentElement.getAttribute('data-theme') === 'dark'")
    pg.click("#themeBtn")                               # the person now chooses: light
    pg.emulate_media(color_scheme="light")
    pg.emulate_media(color_scheme="dark")
    pg.wait_for_timeout(300)
    assert theme_of(pg) == "light"
    ctx.close()


def test_switching_theme_does_not_disturb_a_result(browser, live, tmp_path, force_diseased):
    ctx, pg = page_in(browser, live, "light")
    analyse_through_the_ui(pg, leaf(tmp_path))
    before = pg.inner_html("#diagBox")
    pg.click("#themeBtn")
    assert theme_of(pg) == "dark"
    assert pg.is_visible("#results3") and pg.inner_html("#diagBox") == before
    ctx.close()


def test_theme_button_has_a_label_in_every_language(browser, live):
    ctx, pg = page_in(browser, live, "light")
    assert pg.get_attribute("#themeBtn", "aria-label") == "Dark mode"
    pg.click('#langpick [data-l="kn"]')
    assert pg.get_attribute("#themeBtn", "aria-label") == "ಡಾರ್ಕ್ ಮೋಡ್"
    pg.click('#langpick [data-l="hi"]')
    assert pg.get_attribute("#themeBtn", "aria-label") == "डार्क मोड"
    ctx.close()


def test_privacy_page_uses_the_same_choice(browser, live):
    ctx, pg = page_in(browser, live, "light")
    pg.click("#themeBtn")
    pg.goto(live.url + "/privacy")
    assert theme_of(pg) == "dark"
    dark_bg = body_colour(pg)
    pg.evaluate("() => localStorage.setItem('cropOracle.theme', 'light')")
    pg.reload()
    assert theme_of(pg) == "light" and body_colour(pg) != dark_bg
    ctx.close()


BRIGHT_SCAN = """() => {
  const lum = ([r, g, b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const found = [];
  for (const el of document.querySelectorAll('body *')) {
    const box = el.getBoundingClientRect();
    if (!box.width || !box.height) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden') continue;
    const m = cs.backgroundColor.match(/rgba?\\(([^)]+)\\)/);
    if (!m) continue;
    const p = m[1].split(',').map(Number), a = p.length > 3 ? p[3] : 1;
    if (a > 0.5 && lum(p) > 0.5) found.push(el.tagName.toLowerCase() + '.' + String(el.className).split(' ').join('.') + ' in ' + String(el.parentElement && el.parentElement.className));
  }
  return found;
}"""
ON_PURPOSE_BRIGHT = ("sw", "health-meter")          # colour legend squares and the meter in the always-dark health tile


def test_dark_mode_leaves_no_white_panels_on_any_screen(browser, live, tmp_path, force_diseased, force_confidence):
    force_confidence("low")
    ctx, pg = page_in(browser, live, "dark")
    pg.click("#photoGuide summary")
    pg.click(".samples-box summary")
    pg.click(".loc-more summary")
    pg.click("#btnSearchToggle")
    seen = {"photo": pg.evaluate(BRIGHT_SCAN)}
    pg.set_input_files("#fileIn", str(leaf(tmp_path)))
    pg.wait_for_function("() => !document.querySelector('#dropHas').classList.contains('hidden')")
    pg.click("#btnAnalyze1")
    pg.wait_for_selector("#s2.active")
    seen["confirm"] = pg.evaluate(BRIGHT_SCAN)
    pg.click("#btnAnalyze2")
    pg.wait_for_selector("#results3:not(.hidden)", timeout=60000)
    pg.click("#moreDetails summary")
    pg.wait_for_selector("#kvkList .kvk-item")
    seen["result"] = pg.evaluate(BRIGHT_SCAN)
    pg.click("#btnViewEvidence")
    pg.wait_for_selector("#s4.active")
    seen["evidence"] = pg.evaluate(BRIGHT_SCAN)
    pg.click("#btnNext5")
    pg.wait_for_selector("#s5.active")
    seen["plan"] = pg.evaluate(BRIGHT_SCAN)
    bad = {screen: [x for x in found if not any(ok in x for ok in ON_PURPOSE_BRIGHT)] for screen, found in seen.items()}
    bad = {k: v for k, v in bad.items() if v}
    ctx.close()
    assert not bad, bad


# ------------------------------------------------------------ calmer result screen

def test_result_screen_puts_the_main_things_first_and_folds_the_rest(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path))
    # the detail is folded away until asked for ...
    assert page.evaluate("() => document.querySelector('#moreDetails').open") is False
    assert page.is_hidden("#slmReason") and page.is_hidden("#wgrid") and page.is_hidden("#healthScore")
    page.click("#moreDetails summary")
    assert page.is_visible("#slmReason") and page.is_visible("#wgrid") and page.is_visible("#healthScore")
    page.click("#moreDetails summary")
    assert page.is_hidden("#wgrid")
    # ... but the diagnosis, the marks on the leaf, and the question about the result are not
    for visible in ("#diagBox", "#ovCard", "#feedbackBox", "#btnGoSolution", "#btnViewEvidence", "#btnRedo"):
        assert page.is_visible(visible), visible
    inside = page.evaluate("""() => ['#diagBox', '#helpBox', '#confBox', '#ovCard', '#feedbackBox', '#qualityBox']
                              .filter(s => document.querySelector('#moreDetails').contains(document.querySelector(s)))""")
    assert inside == [], "warnings and the main result must never sit inside the fold"


def test_main_button_is_always_on_screen(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path))
    on_screen = """() => { const r = document.querySelector('#btnGoSolution').getBoundingClientRect();
                          return r.top >= 0 && r.bottom <= innerHeight; }"""
    assert page.evaluate(on_screen), "visible straight away, without scrolling"
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight / 2)")
    assert page.evaluate(on_screen)
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    assert page.evaluate(on_screen)


def test_result_screen_stays_short_on_a_phone(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path))
    page.wait_for_function("() => document.querySelector('#ovPhoto') && document.querySelector('#ovPhoto').complete")
    assert page.evaluate("() => document.documentElement.scrollHeight") <= 1500      # it was about 3,100 before the folds


def test_searching_for_another_place_sits_in_a_fold_that_still_works(page):
    assert page.is_visible("#btnGeo")                    # "use my location" stays in front
    assert page.is_hidden("#btnSearchToggle") and page.is_hidden("#btnManual")
    page.click(".loc-more summary")
    assert page.is_visible("#btnSearchToggle") and page.is_visible("#btnManual")
    page.click("#btnSearchToggle")
    assert page.is_visible("#locSearch")
    page.click("#btnManual")
    assert page.is_visible("#lat") and page.is_hidden("#locSearch")


# ------------------------------------------------------------ shorter screens, and the true reason for "low"

def test_unsure_result_names_the_two_diseases_that_look_alike(page, tmp_path, scores):
    scores(0.62, 0.58)
    analyse_through_the_ui(page, leaf(tmp_path))
    note = text(page, "#diagBox .note-box")
    answer = text(page, "#diagBox .name")
    assert "look almost the same" in note and "No spray advice is shown" in note
    assert note.startswith(answer), "the note names the answer first, then the runner-up"
    assert "not clear enough" not in note, "a good photo must not be blamed"
    assert "scored almost the same" not in text(page, "#confBox"), "said once, in the diagnosis card"
    assert page.query_selector("#diagBox .diag-bar") is None, "the three scores live in the fold now"


def test_unsure_result_in_kannada_and_hindi_still_names_both(page, tmp_path, scores):
    scores(0.62, 0.58)
    analyse_through_the_ui(page, leaf(tmp_path))
    for lang in ("kn", "hi"):
        page.click(f'#langpick button[data-l="{lang}"]')
        page.wait_for_function(f"() => document.documentElement.lang === '{lang}'")
        note = text(page, "#diagBox .note-box")
        assert "{" not in note and "}" not in note, note
        assert len(note) > 40


def test_weak_match_says_so(page, tmp_path, scores):
    scores(0.30, 0.10)
    analyse_through_the_ui(page, leaf(tmp_path))
    note = text(page, "#diagBox .note-box")
    assert "does not clearly match" in note and "No spray advice is shown" in note


def test_reason_texts_keep_their_name_slots_in_every_language():
    page_src = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")
    for lang, nxt in (("en", "\n  kn: {"), ("kn", "\n  hi: {"), ("hi", "\n};")):
        block = page_src.split(f"\n  {lang}: {{", 1)[1].split(nxt, 1)[0]
        line = re.search(r"'unsure\.alike':\s*'([^']*)'", block).group(1)
        assert "{a}" in line and "{b}" in line, (lang, line)
        for key in ("unsure.weak", "unsure.model", "unsure.photo", "uncertain.body", "ov.zoom"):
            assert re.search(rf"'{re.escape(key)}':", block), (lang, key)


def test_the_three_scores_sit_in_the_fold(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path))
    assert page.query_selector("#diagBox .diag-bar") is None
    assert page.is_hidden("#diagBars")
    page.click("#moreDetails summary")
    assert page.is_visible("#diagBars")
    assert len(page.query_selector_all("#moreDetails #diagBars .diag-bar")) == 3


def test_marks_start_small_and_a_tap_enlarges_them_without_losing_alignment(page, tmp_path, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path, spots=14))
    page.wait_for_function("() => document.querySelector('#ovMarks').complete && document.querySelector('#ovPhoto').complete")
    width = lambda sel: page.evaluate(f"() => document.querySelector('{sel}').getBoundingClientRect().width")
    assert width("#ovPhoto") <= 160
    assert page.get_attribute("#ovFig", "aria-expanded") == "false"
    page.click("#ovFig")
    assert page.get_attribute("#ovFig", "aria-expanded") == "true" and width("#ovPhoto") > 250
    boxes = page.evaluate("""() => ['#ovPhoto', '#ovMarks'].map(s => { const r = document.querySelector(s).getBoundingClientRect();
                                  return [r.x, r.y, r.width, r.height]; })""")
    assert all(abs(a - b) <= 1.5 for a, b in zip(*boxes))
    page.focus("#ovFig")
    page.keyboard.press("Enter")
    assert page.get_attribute("#ovFig", "aria-expanded") == "false" and width("#ovPhoto") <= 160
    # the Hide marks button still works while it is zoomed or not
    page.click("#btnMarks")
    assert page.is_hidden("#ovMarks")


def test_help_card_is_compact_and_keeps_one_note_line(page, tmp_path, force_confidence, force_diseased):
    force_confidence("low")
    analyse_through_the_ui(page, leaf(tmp_path))
    page.select_option("#helpDistrict", "dharwad")
    page.wait_for_selector("#kvkList .kvk-item")
    assert page.get_attribute('#helpBox .kcc a[href^="tel:"]', "href") == "tel:18001801551"
    label = page.evaluate("""() => { const l = document.querySelector('label[for="helpDistrict"]').getBoundingClientRect();
                                      return [l.width, l.height]; }""")
    assert label[0] <= 2 and label[1] <= 2, "the label is for screen readers; the first option already says it"
    hints = page.query_selector_all("#kvkList .hint")
    assert len(hints) == 1
    hint = hints[0].inner_text()
    assert "Numbers may have changed" in hint and "Full list of KVKs" in hint


def test_feedback_waits_for_an_answer_before_showing_more(page, tmp_path, with_contact, force_diseased):
    analyse_through_the_ui(page, leaf(tmp_path))
    for v in ("match", "different", "unsure"):
        assert page.is_visible(f'#feedbackBox [data-v="{v}"]')
    tops = {page.evaluate(f"() => document.querySelector('#feedbackBox [data-v=\"{v}\"]').getBoundingClientRect().top")
            for v in ("match", "different", "unsure")}
    assert max(tops) - min(tops) < 2, "the three answers sit on one row"
    assert page.is_hidden("#fbConsent") and page.is_hidden('#feedbackBox [data-fb="send"]')
    page.click('#feedbackBox [data-v="different"]')
    assert page.is_visible("#fbWhich") and page.is_visible("#fbConsent") and page.is_visible('#feedbackBox [data-fb="send"]')


def test_unsure_result_is_short_on_a_phone(page, tmp_path, scores):
    scores(0.62, 0.58)
    analyse_through_the_ui(page, leaf(tmp_path))
    page.wait_for_selector("#kvkList .kvk-item")
    page.wait_for_function("() => document.querySelector('#ovPhoto') && document.querySelector('#ovPhoto').complete")
    assert page.evaluate("() => document.documentElement.scrollHeight") <= 1900      # it was about 2,500 before


def test_first_screen_is_shorter_and_everything_still_works(page):
    assert page.evaluate("() => document.documentElement.scrollHeight") <= 1650      # it was about 1,800
    assert page.query_selector(".screen-head ~ .card .hint[data-i18n='s1.tips']") is None
    top = lambda sel: page.evaluate(f"() => document.querySelector('{sel}').getBoundingClientRect().top")
    assert abs(top("#btnCamera") - top("#btnGallery")) < 2, "take and choose sit side by side"
    for sel in ("#district", "#crop", "#btnGeo", "#symptoms", "#btnVoiceInput", "#photoGuide summary", ".samples-box summary"):
        assert page.is_visible(sel), sel
    assert page.is_hidden("#btnSearchToggle")
    # the two photo folds sit on one row until one is opened
    assert abs(top("#photoGuide summary") - top(".samples-box summary")) < 4
    page.click("#photoGuide summary")
    assert len(page.query_selector_all("#photoGuide li")) == 5 and page.is_visible(".samples-box summary")


# ------------------------------------------------------------ leaf damage estimate

def test_damage_estimate_is_a_percentage_and_says_it_is_rough(page, tmp_path, force_confidence, force_diseased):
    force_confidence("high")
    analyse_through_the_ui(page, leaf(tmp_path, spots=14))
    page.click("#moreDetails summary")
    card = text(page, "#healthScore")
    assert "Leaf damage" in card and "%" in card and "/100" not in card and "rough estimate" in card.lower()
    assert page.evaluate("() => document.querySelector('#healthScore .health-meter i').style.width").endswith("%")
    assert page.get_attribute("#healthScore .health-meter i", "class") in ("good", "warn", "bad")


def test_old_health_signal_wording_is_gone_in_every_language():
    page_src = (main.BASE_DIR / "index.html").read_text(encoding="utf-8")
    for old in ("Crop health signal", "ಬೆಳೆ ಆರೋಗ್ಯ ಸೂಚಕ", "फसल स्वास्थ्य संकेत", "/100</small>"):
        assert old not in page_src, old

# Crop Oracle Karnataka

A photo-based crop-disease advisor for Karnataka farmers, in English, Kannada and Hindi. A farmer takes a
photo of a leaf, picks the crop and district, and gets a best guess, the reasons for it, and an action plan.
When the app is not sure, it says so, gives no spray advice, and points to the district Krishi Vigyan Kendra (KVK)
and the Kisan Call Centre.

This is decision support. It can be wrong. Read **"Before you put it on the internet"** and
**"Measuring accuracy"** below before inviting real farmers.

## Run it on your computer

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000
```

Nothing is required to start. Copy `.env.example` to `.env` to change settings. The app works without the
optional pieces (Kindwise, Sarvam, the local language model, the rice model file); the start-up log says
which ones it found.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

About 230 tests (the two scripts in `scripts/` are tested too). They never use the internet: weather, place search,
PubMed and Kindwise are replaced with fixed answers. `tests/test_browser.py` drives the real page in headless Chromium and skips itself if Chromium is not
installed (`playwright install chromium`).

The tests use synthetic leaf photos. They show that the code does what it says. They say **nothing** about how
well the app recognises diseases on real photos (see "Measuring accuracy").

## What is in the folder

| File | What it does |
|---|---|
| `main.py` | The web app and the diagnosis. |
| `index.html` | The page farmers see (one file, no build step). |
| `quality.py` | Photo checks (blur, dark, glare, small leaf, several leaves) and the "what the app saw" overlay. |
| `feedback.py` | "Was this right?" answers and consented photos (SQLite + JPEG files). |
| `guard.py` | Rate limits, size limits, security headers, admin token. |
| `kvk_contacts.json` | District KVK phone numbers shown when the app is unsure. **Not yet verified.** |
| `kvk_call_sheet.csv` | One row per centre, with every number from both lists, for the person who phones them. |
| `privacy.html` | Privacy page, served at `/privacy`. |
| `scripts/calibrate_quality.py` | Check the photo-check thresholds against real photos. |
| `scripts/feedback_report.py` | Farmer agreement, export for experts, accuracy against expert labels. |
| `requirements.txt`, `requirements-dev.txt` | What to install to run the app, and to run the tests. |
| `Dockerfile`, `.dockerignore`, `.env.example` | Deployment. |

## How the app decides when to hold back spray advice

1. The app gives each result a confidence: **high**, **moderate** or **low** (`decide_confidence()` in `main.py`).
2. A bad photo can lower the confidence but never raise it (`quality.cap_confidence`). A badly blurred or dark
   photo goes straight to low.
3. If the result is a disease and the confidence is **low**, the app shows only non-chemical steps (retake the photo,
   remove infected leaves, call the KVK). Every step that tells the farmer to spray, or to buy a fungicide,
   bactericide or fertiliser, is removed (`apply_input_gate()`). The page says "No spray advice yet" and shows the
   district KVK and the Kisan Call Centre.
3. If the leaf looks healthy but the measurements show severe damage, the app says it cannot decide
   (`inconclusive`) instead of choosing one of the two.

The pesticide approvals and waiting periods in the plan come from a table in the code. **Have someone who knows
Indian pesticide registration (CIB&RC, Karnataka Department of Agriculture, or the KVK) read that table before
launch.** Rules change; the app cannot know if a product has been withdrawn.

## Why is it often "low"?

When the result says **Confidence: Low**, the page now says which of four reasons applies (`confidence_reason` in the
answer; `low_confidence_reason()` in `main.py`):

| Reason | What it means | What to do |
|---|---|---|
| `alike` | Two or more diseases fit about equally well (the best two scores are less than 0.08 apart). This is the usual one. | Ask an expert. A second photo (underside of the leaf, whole plant) can help. |
| `weak` | Nothing fits well (best score under 0.40). | Could be something not in the app's list, or not a disease. |
| `model` | The trained rice model is at least 85% sure of a different disease than the colour check. | Ask an expert. |
| `photo` | The photo is too poor to read (badly blurred, tiny leaf, and so on). | Retake it. |

Why `alike` is so common: without a trained model, the app compares colour, spot count and shape with a hand-written
profile for each disease, and for rice, ragi and arecanut those profiles overlap a lot. In a test with made-up
leaves the two best scores were usually only 0.03 to 0.04 apart, and the app needs 0.08 to call a result moderate.
It is the app being honest: it cannot tell those diseases apart from colour alone.

**What will make "low" less common:**

1. **Run the trained rice model.** Look at the start-up log. `[tflite] rice model loaded` means it is running.
   `rice model not found` means `rice_model.tflite` (and its class-name files) are not in the folder.
   `none of tflite-runtime, ai-edge-litert or tensorflow is installed` means the file is there but nothing can run it:
   `pip install ai-edge-litert` (or `tflite-runtime` on Python 3.12 or older). When the model agrees with the colour
   check, the result is "high". Only rice has such a model; ragi, arecanut and the other crops never will until one
   is trained.
2. **Better photos.** One leaf, daylight, in focus, with the spots clearly visible.
3. **Measure, then tune.** The cut-offs (`HIGH_MIN_SCORE`, `HIGH_MIN_GAP`, `MODERATE_MIN_SCORE`, `MODERATE_MIN_GAP`
   at the top of `decide_confidence()`) are hand-set and unmeasured. Do not lower them just to see fewer "low"
   results: that only makes the label claim more than the evidence supports. Collect expert-labelled answers (see
   "Measuring accuracy") and move them only when the numbers say a lower cut-off is still right most of the time.

## Before you put it on the internet

Do these in order. The first four are about not harming people.

1. **Have the pesticide table reviewed** by a person in the field (above).
2. **Phone every number in `kvk_contacts.json`** once and fix the file. Two lists were merged (the ICAR-ATARI
   Bengaluru page, and the directory the site owner supplied) and they disagree for 17 of the 27 centres that
   both lists give a number for, so nobody
   knows yet which number works. For each centre the owner's number is shown first and the other list's number
   second; a third, if any, is kept in `more_phones`. `kvk_call_sheet.csv` lists both numbers per centre with empty
   columns for the date called and the number that works. Keep only the working ones, delete `more_phones`, and set
   `"verified": true`. A wrong number shown to a farmer who is already unsure is worse than none. Two centres
   (Belagavi BIRDS and Kolar) are placed in different villages by the two lists; check those on the phone call.
3. **Have a native Kannada and a native Hindi reader go through every screen.** The Kannada and Hindi text for
   the new features (photo checks, "was this right?", the unsure state, the privacy page) was written by a machine.
4. **Run the accuracy check** (next section) at least on ragi and arecanut, and make the "high" and "moderate"
   labels true.
5. **Set the privacy variables.** Without `PRIVACY_CONTACT` the app will not offer to keep photos (the box is
   hidden) and only stores the yes/no answer. Set `PRIVACY_CONTACT` (an email or phone farmers can write to) and
   `PRIVACY_OPERATOR` (who runs the site). Read `privacy.html` and make sure every sentence is true for how you run it.
6. **Set `ADMIN_TOKEN`** to a long random string. Without it the `/api/coverage` and `/api/dataset/*` routes are off.
7. **Set `TRUSTED_PROXY_COUNT`** to the number of reverse proxies in front of the app (see below).
8. **Serve it over HTTPS.** The camera in "Camera with guide" only works on HTTPS (or localhost).
9. **Mount a volume at `DATA_DIR`** so answers and photos survive a restart, and back it up.
10. **Calibrate the photo checks** on real photos (`scripts/calibrate_quality.py`, below).

### Docker

```bash
cp .env.example .env        # fill it in
docker build -t crop-oracle .
docker run -d --name crop-oracle -p 8000:8000 -v crop-oracle-data:/data --env-file .env crop-oracle
```

The image runs one worker as an ordinary user and keeps its data in `/data`. (The Dockerfile has not been built
in the place this was written, because no Docker was available there. The dependency list was checked by
installing it into a clean virtual environment and importing the app.)

### Behind a proxy

Put nginx, Caddy or your platform's router in front and let it handle HTTPS. Then:

- `TRUSTED_PROXY_COUNT=1` for one proxy. The app then uses the address the proxy reports. **If you leave it at 0
  behind a proxy, every farmer shares one address and one rate limit.** If you set it higher than the real number,
  a caller can invent their own address and dodge the limits.
- Rate limits are counted in memory **per process**. With `N` workers the real limit is about `N` times the
  setting; divide the `RATE_*` values by `N`. Replace this with a shared store (for example Redis) if you scale out.
- Let the proxy also set `client_max_body_size` (nginx) to about 11 MB.

## Settings

All settings are environment variables (or lines in `.env`). Defaults are shown.

| Variable | Default | What it does |
|---|---|---|
| `PRIVACY_CONTACT` | empty | Where farmers write to ask for deletion. **Photos are only kept when this is set.** |
| `PRIVACY_OPERATOR` | empty | Who runs the site; shown on `/privacy`. |
| `DATA_DIR` | `./data` | Where `feedback/feedback.db` and the consented photos are kept. |
| `FEEDBACK_RETENTION_DAYS` | `730` | Delete answers and photos older than this. `0` keeps everything. |
| `FEEDBACK_SECRET` | random, kept in `DATA_DIR` | Signs results so feedback is stored against what the app really said. Set it yourself if you run more than one copy. |
| `ADMIN_TOKEN` | empty | Needed for `/api/coverage` and `/api/dataset/*` (send it in `X-Admin-Token`). Empty switches them off. |
| `TRUSTED_PROXY_COUNT` | `0` | How many reverse proxies to believe `X-Forwarded-For` from. |
| `CORS_ALLOWED_ORIGINS` | empty | Other sites allowed to call the API from a browser (comma separated). Empty = same site only. |
| `RATE_ANALYZE_PER_MIN` | `30` | Photo analyses per address per minute. |
| `RATE_FEEDBACK_PER_MIN` | `20` | Feedback and delete calls per address per minute. |
| `RATE_GENERAL_PER_MIN` | `240` | Everything else. |
| `MAX_BODY_BYTES` | `11534336` | Largest request accepted (about 11 MB; the photo limit is 10 MB). |
| `MAX_CONCURRENT_ANALYSES` | `4` | Photos analysed at the same time. Others wait up to `ANALYSIS_QUEUE_SECONDS` (default 20), then get a "busy, try again" answer. |
| `ENABLE_DOCS` | `false` | Turns on `/docs` and `/openapi.json`. |
| `ALLOW_CLIENT_ML` | `false` | Let the browser send its own model predictions. Leave off: anyone could forge them to get a "high confidence" result. |
| `CSP` | `default` | Content-Security-Policy header: `default`, `off`, or your own policy. The default allows the page's inline script; tighten it if you move the script to its own file. |
| `KINDWISE_API_KEY`, `SARVAM_API_KEY`, `SLM_ENABLED`, `SLM_MODEL`, `OLLAMA_URL` | empty / off | Optional services. |

Mobile networks put many farmers behind one address, so the rate limits are deliberately generous. A farmer
who is limited gets a message telling them how long to wait.

## Privacy, in short

- A photo that is analysed is measured and thrown away. It is not saved.
- After a result the farmer can answer "Was this right?". That stores the answer, what the app said, the crop,
  language, app version, photo-check codes and the date. No name, phone number or IP address.
- Only if the farmer ticks the box, and `PRIVACY_CONTACT` is set, the photo (shrunk, camera data and GPS removed)
  and the location rounded to about 11 km are stored too.
- The farmer gets a record number and a delete code. Deleting from `/privacy` removes the answer and the photo.
  The code is stored only as a hash.
- Records are deleted after `FEEDBACK_RETENTION_DAYS`.
- The page itself loads Google Fonts, and the browser calls Open-Meteo and BigDataCloud for weather and place
  names. `privacy.html` says so. If that is not acceptable, self-host the fonts and add a server-side weather call.

If a farmer asks you to delete their data and has lost the code, you cannot find it (there is no name). Say so
plainly, as the privacy page does.

## Calibrating the photo checks

The thresholds in `quality.py` (`THRESHOLDS`) were chosen by hand and tried only on synthetic photos. On real
photos the app may warn about good photos (annoying) or miss bad ones (harmful). The same goes for the quick
check the page runs on the phone before upload (`QUICK` in `index.html`).

1. Collect real photos from the phones your farmers use, and sort them:

   ```
   photos/good/                  photos an expert says are fine
   photos/bad/blur/  dark/  glare/  small_leaf/  low_res/  multiple_leaves/
   ```

   At least 50 good and 20 of each kind of bad photo.
2. `python scripts/calibrate_quality.py photos/ --csv metrics.csv`
3. It prints, for each check, how the good and bad photos are spread, how many good photos get a false warning
   and how many bad ones are caught, and a suggested range. Edit `THRESHOLDS`, run again.
4. Keep false **severe** warnings near zero. A severe warning drops the confidence to low, which removes spray advice.

Known gap: the "several leaves" check works on separate patches of leaf. Two leaves that touch look like one.

## Measuring accuracy

**Why this matters.** Today nobody knows how often the app is right. "High", "moderate" and "low" come from
numbers chosen by hand in `decide_confidence()` (for example "score at least 0.60 and a gap of at least 0.15 is
high"). They have never been compared with what is really on the leaf. A farmer reads "high" as "very likely
right". Until it has been measured, it should not be called that.

Also note: only rice has a trained image model, and only if you supply `rice_model.tflite`. For ragi, arecanut
and the other crops the answer comes from colour and shape measurements, weather and a symptom match. That can be
useful, but its accuracy on field photos is unknown and may be much lower than the screen suggests.

### What to measure

For each crop, on photos taken by farmers on ordinary phones in the field (not lab or internet photos):

- **Accuracy**: of the photos where the app gave a diagnosis, how often did it match the expert? Give it with a
  range (the script does), not just a percentage.
- **How often it said "not sure"**, and whether the experts found those photos easy. Too often means the app
  is too cautious to be useful.
- **Spray advice for the wrong disease**: of the photos where the app showed spray advice, how often was the
  disease wrong? This is the number that matters most for safety.
- **Per confidence level**: is "high" right more often than "moderate", and "moderate" more than "low"?

### How to do it

1. **Get the experts.** Ask the University of Agricultural Sciences (Bengaluru, Dharwad, Raichur) and the KVKs of
   the districts you serve. For ragi, the small-millets research group at UAS Bengaluru; for arecanut, ICAR-CPCRI
   and the horticultural university at Shivamogga are natural people to ask. You need people who can tell the
   diseases apart on a leaf, and ideally confirm doubtful ones with a lab test.
2. **Collect labelled photos.** Two sources:
   - *From the app:* farmers who tick the photo box add their photos. Run
     `python scripts/feedback_report.py --export for_experts/` to copy them into folders and a `labels.csv`.
     Experts fill in `expert_label` for each.
   - *On purpose:* hold collection days with KVK or university staff. They photograph leaves in plots with a phone,
     the way a farmer would, and write down the true diagnosis on the spot. This gives you cases the app never
     sees from farmers, and a fair sample instead of only the farmers who bothered to answer.
3. **How many.** At least 200 photos per crop to say anything per crop, with healthy leaves and diseases the app
   does not cover included, because farmers will send those. With 100 photos and 90% right, the true accuracy could
   be anywhere from about 83% to 94%; with 300, about 86% to 93%. Spread the photos over seasons, districts,
   phones and varieties. Do not pick the "nice" photos.
4. **Check the experts.** Have two experts label about one in five photos independently, and look at how often they
   disagree. If the experts disagree with each other 20% of the time, the app cannot be expected to do better.
   Photos that even experts cannot judge are marked `unclear` and left out.
5. **Keep a test set apart.** Do not tune thresholds on the photos you use to report accuracy. Set aside a test
   set (for example one in three photos, chosen at random) and look at it only to report.
6. **Run the score.** `python scripts/feedback_report.py --expert for_experts/labels.csv`. It prints accuracy per
   crop, per confidence level, the wrong-spray rate and the most common mix-ups.
7. **Make the words true.** Decide the targets before looking (a starting suggestion: "high" right at least 90% of
   the time, "moderate" at least 70%). If a level misses, **make the rule stricter so fewer results get that label**;
   do not lower the target. Change `decide_confidence()`, re-run, and check again on fresh photos.
8. **Start with ragi and arecanut.** They matter most in Karnataka and are the crops where the app has no trained
   model. Arecanut also needs a question for the experts: several of its diseases (mahali / fruit rot, bud rot,
   stem bleeding) show on the nut, the crown or the trunk, not on a leaf. Ask which can be recognised from a leaf
   photo at all. For the rest the honest answer may be "photograph the nut or the crown", or "call the KVK".
9. **Repeat.** Re-measure every season and after every change to the diagnosis code. Keep the numbers in a file
   with the date and app version.
10. **Tell farmers the truth.** When you have numbers, show them in the app ("tested on N photos of ragi: right
    X in 10 times"). If a crop is below what you would accept, switch that crop to "photo plus KVK" until it improves.

`scripts/feedback_report.py` without options shows what farmers say ("yes, looks right" / "no, it is something
else") per crop and confidence. That is a **quick warning light, not accuracy**: farmers can be wrong, and the ones
who answer are not a fair sample.

## Appearance

The page follows the phone's light or dark setting, and the moon/sun button in the header overrides it. The choice is
remembered on that phone (the privacy page uses it too). Every colour is a token at the top of the `<style>` block
in `index.html`; `tests/test_ui_colours.py` checks that text and borders stay readable in both themes, so run the tests
after changing a colour. On the result screen the diagnosis, expert help, marks on the leaf, the "was this right?"
question and the action-plan button are always shown; the reasoning, the health signal and the weather sit behind
one "How we decided" fold (the three scores behind the answer are in that fold too). The marks-on-the-leaf photo
starts small; tapping it makes it larger.

## Known limits

- Photo-check thresholds are uncalibrated (see above). The "several leaves" check misses leaves that touch.
- KVK numbers are unverified (see the checklist); Kannada and Hindi for the new features need native review.
- Rate limits are per process. The security-header policy allows the page's inline script.
- "Camera with guide" is an extra button; the normal camera and gallery buttons still open the phone's own camera.
- The page needs the internet for weather and place names, and falls back to typical values with a warning.
- Not built: offline use, a WhatsApp bot, Kannada voice input/output beyond the phone's own voices, and more crops.

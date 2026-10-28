#!/usr/bin/env python3
"""Measure how often the app is right, on REAL photos that an expert has labelled.

Until this is run on real photos nobody knows. The confidence labels ("high", "moderate", "low") and the
rice model's vote are hand-set or untested; this script is how to test them.

Folder layout (the folder names are the answers; use the disease ids the app uses, or "healthy"):

    photos/
      rice_blast/        photos an expert says show rice blast
      rice_blb/          bacterial leaf blight
      rice_brown_spot/   ...
      healthy/           photos with no disease

Photos taken in the field with a phone are what matters. Pictures from the internet are fine for a first look but
are often cleaner than farmers' photos, so the numbers will be better than real life. Aim for at least 30 per
disease; with 5 or 10 per disease the percentages below are only a hint.

Usage:
    python scripts/evaluate_photos.py photos/
    python scripts/evaluate_photos.py photos/ --crop rice --csv results.csv

Every photo is sent through the real /api/analyze code (weather and the internet are replaced by fixed answers), twice:

  advisory   the app as shipped: the rice model is run but does not change the result
  score      the rice model's answer counts (RICE_MODEL_ROLE=score), i.e. what the app did before

and the rice model's own answer is listed on its own. For each mode it prints how many photos got

  right, sure      the right disease, confidence "high" or "moderate"
  right, unsure    the right disease on top, but confidence "low" (the farmer is told to ask an expert)
  wrong, sure      a WRONG disease with confidence "high" or "moderate"   <- the dangerous column
  wrong, unsure    wrong disease on top, but "low", so no spray advice is shown
  no result        no leaf found, or "inconclusive"

"Wrong, sure" is what to push towards zero. It never edits the app: change the profiles or cut-offs in main.py
yourself, then run this again on the SAME photos (and keep some photos aside that you never tune on).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="croporacle-eval-"))   # never touch the real feedback data
for _name in ("KINDWISE_API_KEY", "KINDWISE_KEY", "SARVAM_API_KEY", "SARVAM_KEY", "PRIVACY_CONTACT", "ALLOW_CLIENT_ML"):
    os.environ.pop(_name, None)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODES = ("advisory", "score")
# What the app measures on each photo. They go into the --csv file so the disease profiles in main.py can later be
# fitted to real photos (mean and spread per disease) instead of being guessed.
FEATURE_KEYS = ("leaf_fraction", "spot_count", "lesion_ratio", "damage_ratio", "chlorosis_ratio", "brown_ratio",
                "white_ratio", "mean_spot_pct", "mean_aspect", "lab_a_red_frac")


def list_photos(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def load_app():
    """Import the app with everything that needs the internet replaced by fixed answers."""
    from fastapi.testclient import TestClient
    import guard
    import main

    async def weather(lat, lon):
        return dict(main.WEATHER_FALLBACK)

    async def place(lat, lon):
        return {"label": "Evaluation", "country_code": "in", "region": "ka"}

    async def kindwise(raw, crop, lang="en"):
        return None

    main.fetch_weather_cached, main.reverse_geocode_cached, main.kindwise_analyze = weather, place, kindwise
    guard.SETTINGS = guard.Settings(analyze_per_min=10 ** 9, general_per_min=10 ** 9, max_concurrent_analyses=2)
    guard.LIMITER = guard.RateLimiter()
    return main, TestClient(main.app)


def outcome(body: dict, status: int, truth: str):
    """(top disease id or None, confidence, bucket)"""
    if status != 200:
        return None, "-", "no result"
    if body.get("inconclusive"):
        return None, "low", "no result"
    top = "healthy" if body.get("is_healthy") else (body["candidates"][0]["diagnosis"]["id"] if body.get("candidates") else None)
    conf = body.get("confidence", "low")
    right = top == truth
    sure = conf in ("high", "moderate")
    if top is None:
        return None, conf, "no result"
    return top, conf, ("right, sure" if right and sure else "right, unsure" if right
                       else "wrong, sure" if sure else "wrong, unsure")


def evaluate(folder: Path, crop: str, region: str, say=print):
    main, client = load_app()
    valid = set(main.DISEASES_BY_ID) | {"healthy"}
    rows = []
    skipped = []
    with client:
        for label_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            truth = label_dir.name
            if truth not in valid:
                skipped.append(truth)
                continue
            for photo in list_photos(label_dir):
                raw = photo.read_bytes()
                row = {"photo": f"{truth}/{photo.name}", "truth": truth}
                model = []
                try:
                    from PIL import Image
                    model = main._run_rice_model(main._decode_image(raw)) if crop == "rice" else []
                except Exception:
                    model = []
                try:
                    feats = main.analyze_leaf(main._decode_image(raw))
                    for k in FEATURE_KEYS:
                        row[f"m_{k}"] = round(float(feats.get(k, 0.0)), 4)
                except Exception:
                    for k in FEATURE_KEYS:
                        row[f"m_{k}"] = ""
                row["model"] = (f"{model[0]['class']} {model[0]['confidence'] * 100:.0f}%" if model else "")
                tid, tconf, _ = main.tflite_hint_to_internal(model)
                row["model_id"] = tid if (tid and tconf >= 0.55) else ""
                for mode in MODES:
                    main.RICE_MODEL_ROLE = mode
                    r = client.post("/api/analyze", data={"crop": crop, "language": "en", "region": region},
                                    files={"image": (photo.name, raw, "image/jpeg")})
                    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    top, conf, bucket = outcome(body, r.status_code, truth)
                    row[f"{mode}_top"], row[f"{mode}_conf"], row[f"{mode}_bucket"] = top or "-", conf, bucket
                    cands = [c["diagnosis"]["id"] for c in body.get("candidates", [])] if r.status_code == 200 else []
                    row[f"{mode}_rank"] = (cands.index(truth) + 1) if truth in cands else ""
                    row[f"{mode}_reason"] = body.get("confidence_reason") or ""
                rows.append(row)
    return rows, skipped


def report(rows, skipped, say=print):
    if skipped:
        say(f"Skipped folders that are not a disease id the app knows: {', '.join(skipped)}")
    if not rows:
        say("No photos found.")
        return
    say(f"\n{len(rows)} photos\n")
    w = max(len(r["photo"]) for r in rows)
    say(f"{'photo':{w}}  {'rice model alone':26}  {'advisory (as shipped)':44}  score (model counts)")
    for r in rows:
        a = f"{r['advisory_top']} / {r['advisory_conf']}"
        s = f"{r['score_top']} / {r['score_conf']}"
        mark = lambda m: "ok " if m.startswith("right") else ".. " if m == "no result" else "NO "   # noqa: E731
        say(f"{r['photo']:{w}}  {r['model']:26}  {mark(r['advisory_bucket'])}{a:41}  {mark(r['score_bucket'])}{s}")

    say("\nPer mode (counts of photos):")
    buckets = ["right, sure", "right, unsure", "wrong, sure", "wrong, unsure", "no result"]
    say(f"  {'':10}" + "".join(f"{b:>16}" for b in buckets) + f"{'top-1 right':>14}{'truth in top 2':>16}")
    for mode in MODES:
        c = Counter(r[f"{mode}_bucket"] for r in rows)
        right = sum(1 for r in rows if r[f"{mode}_bucket"].startswith("right"))
        top2 = sum(1 for r in rows if r[f"{mode}_rank"] in (1, 2) or (r["truth"] == "healthy" and r[f"{mode}_top"] == "healthy"))
        say(f"  {mode:10}" + "".join(f"{c.get(b, 0):>16}" for b in buckets) + f"{right:>10}/{len(rows):<3}{top2:>12}/{len(rows):<3}")

    model_rows = [r for r in rows if r["model"]]
    if model_rows:
        right = sum(1 for r in model_rows if r["model_id"] == r["truth"])
        sure_wrong = [r for r in model_rows if r["model_id"] and r["model_id"] != r["truth"]]
        say(f"\nThe rice model on its own (counting only answers of 55% or more): right on {right} of {len(model_rows)}; "
            f"a confident WRONG disease on {len(sure_wrong)}.")

    say("\nWhat the app said, by true disease (advisory mode):")
    by_truth = defaultdict(Counter)
    for r in rows:
        by_truth[r["truth"]][r["advisory_top"]] += 1
    for truth, counts in sorted(by_truth.items()):
        say(f"  {truth:18} -> " + ", ".join(f"{k} x{v}" for k, v in counts.most_common()))
    if len(rows) < 100:
        say(f"\nOnly {len(rows)} photos: treat these as a first look, not as an accuracy figure.")


def main_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the app on labelled photos.")
    ap.add_argument("folder", type=Path, help="folder with one sub-folder per disease id (and 'healthy')")
    ap.add_argument("--crop", default="rice")
    ap.add_argument("--region", default="ka")
    ap.add_argument("--csv", type=Path, help="also write one row per photo (answers and measurements) to this file")
    args = ap.parse_args(argv)
    if not args.folder.is_dir():
        print(f"{args.folder} is not a folder", file=sys.stderr)
        return 2
    rows, skipped = evaluate(args.folder, args.crop, args.region)
    report(rows, skipped)
    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main_cli())

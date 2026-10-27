#!/usr/bin/env python3
"""Read the farmers' "was this right?" answers and measure how well the app does.

It only reads the feedback database. It never changes it.

Three jobs:

1. Farmer agreement report (default).  Per crop and per confidence level: how often farmers said
   "yes, it looks right" versus "no, it is something else". This is AGREEMENT, not accuracy. A
   farmer can be wrong, and farmers who are unhappy answer more often. Use it to spot problems early.

2. Export for experts (--export FOLDER).  Copies every photo a farmer agreed to share into
   FOLDER/<crop>/<what the app said>/ and writes FOLDER/labels.csv with an empty `expert_label`
   column. Give this to UAS / KVK staff. They fill in the true diagnosis for each photo.

3. Real accuracy (--expert FOLDER/labels.csv).  Once experts have filled in `expert_label`, this
   prints accuracy per crop and per confidence level, and checks that "high" and "moderate" mean
   what they say. THIS is the number to publish, not the farmer agreement.

Usage:
    python scripts/feedback_report.py                       # uses DATA_DIR or ./data
    python scripts/feedback_report.py --data /srv/crop/data
    python scripts/feedback_report.py --export for_experts/
    python scripts/feedback_report.py --expert for_experts/labels.csv

Valid values for expert_label: a disease id exactly as in the `predicted_id` column of the app's
disease list, or `healthy`, or `other` (something the app does not cover), or `unclear`
(the photo is too poor even for an expert; these are left out of the accuracy numbers).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONF_ORDER = ["high", "moderate", "low"]

# Honest reading of a small sample: below this many answers a percentage means very little.
MIN_N = 30


def wilson(k: int, n: int, z: float = 1.96):
    """95% confidence interval for k successes in n tries. Wide when n is small, as it should be."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "    -"
    lo, hi = wilson(k, n)
    flag = "  (too few to trust)" if n < MIN_N else ""
    return f"{100 * k / n:5.1f}%  ({100 * lo:.0f}-{100 * hi:.0f}%)  n={n}{flag}"


def data_root(arg: str | None) -> Path:
    base = Path(arg or os.getenv("DATA_DIR") or (ROOT / "data"))
    return base if (base / "feedback.db").exists() else base / "feedback"


def open_db(root: Path) -> sqlite3.Connection:
    db = root / "feedback.db"
    if not db.exists():
        sys.exit(f"No feedback database at {db}. Use --data to point at the folder that holds it.")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)     # read-only on purpose
    con.row_factory = sqlite3.Row
    return con


# ------------------------------------------------------------------ 1. farmer agreement

def agreement_report(con: sqlite3.Connection):
    rows = con.execute("SELECT * FROM feedback").fetchall()
    print(f"{len(rows)} answers in total.\n")
    if not rows:
        return

    by_verdict = Counter(r["verdict"] for r in rows)
    print("All answers:", ", ".join(f"{k} {v}" for k, v in sorted(by_verdict.items())))
    print("'unsure' answers are left out of the agreement rates below.\n")

    def block(title, groups):
        print(f"== {title}")
        for key in sorted(groups, key=lambda k: (CONF_ORDER.index(k[-1]) if k and k[-1] in CONF_ORDER else 0, k)):
            g = groups[key]
            yes = sum(1 for r in g if r["verdict"] == "match")
            no = sum(1 for r in g if r["verdict"] == "different")
            label = " / ".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            print(f"   {label:<30} {fmt_rate(yes, yes + no)}")
        print()

    for kind in ("diagnosis", "healthy"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        print(f"######## results of kind '{kind}' ({len(sub)} answers)\n")
        g = defaultdict(list)
        for r in sub:
            g[r["crop"]].append(r)
        block("Farmers who agree, per crop", g)
        g = defaultdict(list)
        for r in sub:
            g[r["confidence"] or "?"].append(r)
        block("Per confidence level (high should agree more often than moderate, and moderate more than low)", g)
        g = defaultdict(list)
        for r in sub:
            g[(r["crop"], r["confidence"] or "?")].append(r)
        block("Per crop and confidence level", g)

    unsure_results = [r for r in rows if r["kind"] == "inconclusive"]
    if unsure_results:
        print(f"== Results where the app said 'not sure' ({len(unsure_results)} answers)")
        print("   What farmers say it really was:", dict(Counter(r["reported_id"] or r["verdict"] for r in unsure_results)))
        print()

    wrong = Counter((r["crop"], r["predicted_id"], r["reported_id"]) for r in rows
                    if r["verdict"] == "different" and r["kind"] != "inconclusive")
    if wrong:
        print("== Most common 'it is something else' answers: (crop, app said, farmer said)")
        for (crop, said, was), n in wrong.most_common(10):
            print(f"   {n:>3} x  {crop}: {said}  ->  {was}")
        print()

    q = Counter()
    for r in rows:
        try:
            import json
            for code in json.loads(r["quality"] or "[]"):
                q[code] += 1
        except Exception:
            pass
    if q:
        print("== Photo problems the app noticed:", dict(q.most_common()))
        print()

    print("Remember: farmers are not the final judge. For real accuracy, use --export and --expert.")


# ------------------------------------------------------------------ 2. export for experts

def export_for_experts(con: sqlite3.Connection, root: Path, out: Path):
    rows = con.execute("SELECT * FROM feedback WHERE consent_photo=1 AND photo_file IS NOT NULL ORDER BY created_at").fetchall()
    if not rows:
        sys.exit("No photos with consent to export yet.")
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "labels.csv"
    already = {}
    if existing.exists():            # never lose labels experts already typed in
        with open(existing, newline="", encoding="utf-8") as f:
            already = {r["id"]: r for r in csv.DictReader(f)}

    fields = ["id", "file", "crop", "predicted_id", "confidence", "farmer_verdict", "farmer_says",
              "photo_issues", "latitude", "longitude", "expert_label", "expert_name", "expert_note"]
    n = 0
    with open(existing, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            src = root / "images" / r["photo_file"]
            if not src.exists():
                continue
            said = r["predicted_id"] or "unsure"
            dest_dir = out / r["crop"] / said
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / r["photo_file"]
            shutil.copy2(src, dest)
            prev = already.get(r["id"], {})
            w.writerow({
                "id": r["id"], "file": str(dest.relative_to(out)), "crop": r["crop"],
                "predicted_id": r["predicted_id"] or "", "confidence": r["confidence"] or "",
                "farmer_verdict": r["verdict"], "farmer_says": r["reported_id"] or "",
                "photo_issues": r["quality"] or "", "latitude": r["lat_r"] if r["lat_r"] is not None else "",
                "longitude": r["lon_r"] if r["lon_r"] is not None else "",
                "expert_label": prev.get("expert_label", ""), "expert_name": prev.get("expert_name", ""),
                "expert_note": prev.get("expert_note", ""),
            })
            n += 1
    print(f"Exported {n} photos to {out} and wrote {existing}.")
    print("Give this folder to the experts. They fill in expert_label (and expert_name) for each row.")
    print("These photos are shared by farmers for improving the app. Share them only with the people "
          "your privacy page says they will go to, and delete the export when you are done.")


# ------------------------------------------------------------------ 3. real accuracy

def expert_report(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if (r.get("expert_label") or "").strip()]
    unclear = [r for r in labelled if r["expert_label"].strip().lower() == "unclear"]
    usable = [r for r in labelled if r["expert_label"].strip().lower() != "unclear"]
    # An inconclusive result has no prediction. It is neither right nor wrong; count it apart.
    decided = [r for r in usable if r["predicted_id"]]
    abstained = [r for r in usable if not r["predicted_id"]]
    print(f"{len(rows)} photos in the file, {len(labelled)} labelled by experts, "
          f"{len(unclear)} marked unclear (left out).")
    print(f"{len(decided)} photos where the app gave an answer, {len(abstained)} where it said 'not sure'.\n")
    if not decided:
        sys.exit("Nothing to score yet.")
    if len(decided) < 100:
        print("WARNING: fewer than 100 expert-labelled answers. Treat every number below as a first look only.\n")

    def right(r):
        return r["predicted_id"].strip() == r["expert_label"].strip()

    def block(title, key):
        print(f"== {title}")
        groups = defaultdict(list)
        for r in decided:
            groups[key(r)].append(r)
        for k in sorted(groups, key=lambda x: (CONF_ORDER.index(x) if x in CONF_ORDER else 0, str(x))):
            g = groups[k]
            print(f"   {str(k):<30} {fmt_rate(sum(1 for r in g if right(r)), len(g))}")
        print()

    k_all = sum(1 for r in decided if right(r))
    print(f"== Overall   {fmt_rate(k_all, len(decided))}\n")
    block("Accuracy per crop (report these separately; an average hides the weak crops)", lambda r: r["crop"])
    block("Accuracy per confidence level", lambda r: r["confidence"] or "?")
    block("Per crop and confidence level", lambda r: f"{r['crop']} / {r['confidence'] or '?'}")

    # Do the words mean what they say?  Target: "high" at least ~90%, "moderate" at least ~70%.
    # These targets are a starting suggestion. Choose your own and write them down before you look.
    print("== Do 'high' and 'moderate' mean what they say?  (suggested targets: high >= 90%, moderate >= 70%)")
    for level, target in (("high", 0.90), ("moderate", 0.70)):
        g = [r for r in decided if r["confidence"] == level]
        if not g:
            print(f"   {level:<9} no photos")
            continue
        k = sum(1 for r in g if right(r))
        lo, hi = wilson(k, len(g))
        if len(g) < MIN_N:
            verdict = "too few photos to say"
        elif lo >= target:
            verdict = "OK (even the low end of the range meets the target)"
        elif hi < target:
            verdict = "TOO OPTIMISTIC: this level is not as reliable as its name. Tighten decide_confidence()"
        else:
            verdict = "unclear: the range straddles the target; collect more photos"
        print(f"   {level:<9} {100 * k / len(g):5.1f}%  ({100 * lo:.0f}-{100 * hi:.0f}%)  {verdict}")
    print()

    # Spray advice is only shown when confidence is not 'low'. How often would a farmer have been
    # told to spray for the wrong disease?
    shown = [r for r in decided if r["confidence"] in ("high", "moderate") and r["predicted_id"] != "healthy"]
    if shown:
        wrong = sum(1 for r in shown if not right(r))
        lo, hi = wilson(wrong, len(shown))
        print(f"== Spray advice shown for the WRONG disease: {wrong} of {len(shown)} "
              f"({100 * wrong / len(shown):.1f}%, range {100 * lo:.0f}-{100 * hi:.0f}%)")
        print("   This is the number that matters most for safety. Look at each one below.\n")
        for r in shown:
            if not right(r):
                print(f"   {r['id']}  {r['crop']}: app said {r['predicted_id']} ({r['confidence']}), "
                      f"expert said {r['expert_label']}  {r.get('expert_note', '')}")
        print()

    # What the app got wrong most often
    conf = Counter((r["crop"], r["predicted_id"], r["expert_label"]) for r in decided if not right(r))
    if conf:
        print("== Most common mix-ups: (crop, app said, expert said)")
        for (crop, said, truth), n in conf.most_common(10):
            print(f"   {n:>3} x  {crop}: {said}  ->  {truth}")
        print()

    if abstained:
        print(f"== When the app said 'not sure' ({len(abstained)} photos) the experts said:")
        print("  ", dict(Counter(r["expert_label"] for r in abstained)))
        print("   If most of these were easy for the experts, the app is too cautious.\n")

    # Disagreement between farmer and expert tells you how much to trust the farmers' answers
    both = [r for r in decided if r.get("farmer_verdict") in ("match", "different")]
    if both:
        agree = sum(1 for r in both if (r["farmer_verdict"] == "match") == right(r))
        print(f"== Farmer and expert agree about whether the app was right: {agree} of {len(both)} "
              f"({100 * agree / len(both):.0f}%)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder that holds feedback.db (default: DATA_DIR or ./data)")
    ap.add_argument("--export", type=Path, metavar="FOLDER", help="copy consented photos here for expert labelling")
    ap.add_argument("--expert", type=Path, metavar="labels.csv", help="score the app against expert labels")
    args = ap.parse_args()

    if args.expert:
        expert_report(args.expert)
        return
    root = data_root(args.data)
    con = open_db(root)
    try:
        if args.export:
            export_for_experts(con, root, args.export)
        else:
            agreement_report(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()

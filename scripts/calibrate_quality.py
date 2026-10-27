#!/usr/bin/env python3
"""Check the photo-quality thresholds in quality.py against REAL photos.

The shipped thresholds were only tried on synthetic images. Run this on photos taken with the
phones your farmers use before you rely on the "blurry / dark / glare" warnings.

Folder layout (the folder names are the labels):

    photos/
      good/            photos an expert says are fine to diagnose from
      bad/
        blur/          out-of-focus photos
        dark/          too dark
        glare/         flash or sun reflections on the leaf
        small_leaf/    leaf far away / tiny in the frame
        low_res/       tiny files
        multiple_leaves/   several leaves in one photo
      (photos straight in bad/ count as "bad for some reason" and are only used for the overall catch rate)

Aim for at least 50 good and 20 of each kind of bad photo. Fewer than that and the numbers below
are guesses.

Usage:
    python scripts/calibrate_quality.py photos/
    python scripts/calibrate_quality.py photos/ --csv metrics.csv

What it prints:
  * for every metric, how the good and bad photos are spread (min / 10th / median / 90th / max)
  * with the CURRENT thresholds: how many good photos get a false warning, and how many bad
    photos are caught
  * a suggested "minor" threshold range: warns on at most 5% of your good photos and catches at
    least 90% of that kind of bad photo (needs 10+ of each; otherwise only a limit is given)
It never edits quality.py. Change THRESHOLDS there yourself, then run this again.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# metric -> (issue code it feeds, True if a HIGHER value is worse)
METRICS = {
    "sharpness": ("blur", False),
    "image_mean_v": ("dark", False),
    "glare_frac": ("glare", True),
    "leaf_fraction": ("small_leaf", False),
    "long_side_px": ("low_res", False),
}


def list_photos(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def measure(path: Path, analyze_leaf):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)      # same orientation fix the server applies
        res = analyze_leaf(im.convert("RGB"), with_extras=True)
    q = res["_extras"]["quality"]
    return q["metrics"], {i["code"]: i["severity"] for i in q["issues"]}


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def spread(values):
    if not values:
        return "no photos"
    return " ".join(f"{label} {pct(values, p):.3g}" for label, p in
                    (("min", 0), ("p10", .1), ("med", .5), ("p90", .9), ("max", 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--csv", type=Path, help="also write every photo's numbers to this file")
    args = ap.parse_args()

    good_dir, bad_dir = args.folder / "good", args.folder / "bad"
    if not good_dir.is_dir() or not bad_dir.is_dir():
        sys.exit(f"Expected {good_dir} and {bad_dir}. See the top of this file for the layout.")

    import main  # imported late: it loads the models and data files, which takes a few seconds
    import quality

    rows = []   # (label, expected_code or "", path, metrics, issues)
    for p in list_photos(good_dir):
        rows.append(("good", "", p))
    for p in list_photos(bad_dir):
        rel = p.relative_to(bad_dir)
        rows.append(("bad", rel.parts[0] if len(rel.parts) > 1 else "", p))
    if not rows:
        sys.exit("No photos found.")

    measured = []
    for label, code, p in rows:
        try:
            m, issues = measure(p, main.analyze_leaf)
        except Exception as e:          # a photo the app cannot open is worth knowing about
            print(f"SKIPPED {p}: {e!r}")
            continue
        measured.append((label, code, p, m, issues))

    good = [r for r in measured if r[0] == "good"]
    bad = [r for r in measured if r[0] == "bad"]
    print(f"\n{len(good)} good and {len(bad)} bad photos measured.\n")
    if len(good) < 50:
        print("WARNING: fewer than 50 good photos. The suggestions below are not reliable yet.\n")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "expected_issue", "file", *METRICS, "warnings_raised"])
            for label, code, p, m, issues in measured:
                w.writerow([label, code, str(p), *[m[k] for k in METRICS],
                            ";".join(f"{c}:{s}" for c, s in issues.items())])
        print(f"Wrote {args.csv}\n")

    thresholds = quality.THRESHOLDS
    for metric, (issue, higher_worse) in METRICS.items():
        gvals = [r[3][metric] for r in good]
        mine = [r for r in bad if r[1] == issue]
        bvals = [r[3][metric] for r in mine]
        lim = thresholds[metric]
        print(f"== {metric}   (feeds the '{issue}' warning; {'higher' if higher_worse else 'lower'} is worse)")
        print(f"   current thresholds   minor {lim['minor']}   severe {lim['severe']}")
        print(f"   good photos          {spread(gvals)}")
        print(f"   bad '{issue}' photos   {spread(bvals)}")

        false_alarm = sum(1 for r in good if issue in r[4])
        if good:
            print(f"   false warnings now   {false_alarm} of {len(good)} good photos "
                  f"({100 * false_alarm / len(good):.0f}%)")
        if mine:
            caught = sum(1 for r in mine if issue in r[4])
            print(f"   caught now           {caught} of {len(mine)} bad '{issue}' photos "
                  f"({100 * caught / len(mine):.0f}%)")
        else:
            print(f"   (no photos in bad/{issue}/, so the catch rate is unknown)")

        if len(gvals) >= 10 and len(bvals) >= 10:
            # A 'minor' threshold that false-warns on <=5% of good photos and catches >=90% of bad ones.
            if higher_worse:
                lo, hi = pct(gvals, 0.95), pct(bvals, 0.10)      # warn when value > threshold
            else:
                lo, hi = pct(bvals, 0.90), pct(gvals, 0.05)      # warn when value < threshold
            if lo <= hi:
                print(f"   suggested 'minor'    anything from {lo:.3g} to {hi:.3g} (middle: {(lo + hi) / 2:.3g}) "
                      f"false-warns on <=5% of good and catches >=90% of bad photos")
            else:
                print(f"   suggested 'minor'    none: good and bad photos overlap on this measure "
                      f"(good p5/p95 {hi:.3g}, bad p90/p10 {lo:.3g}). Look at the photos in the lists below.")
        elif len(gvals) >= 10:
            edge = pct(gvals, 0.95) if higher_worse else pct(gvals, 0.05)
            print(f"   limit for 'minor'    no further than {edge:.3g} from the good photos "
                  f"(past that, more than 5% of good photos get a needless warning). "
                  f"Add 10+ photos in bad/{issue}/ to get a real suggestion.")
        print()

    # overall catch rate, including bad photos that are not sorted by kind
    if bad:
        caught_any = sum(1 for r in bad if r[4])
        print(f"Any warning on a bad photo: {caught_any} of {len(bad)} ({100 * caught_any / len(bad):.0f}%)")
    if good:
        warned = sum(1 for r in good if r[4])
        severe = sum(1 for r in good if any(s == 'severe' for s in r[4].values()))
        print(f"Any warning on a good photo: {warned} of {len(good)} ({100 * warned / len(good):.0f}%); "
              f"'severe' on {severe}")
        print("A severe warning lowers confidence to 'low' and withholds spray advice, so false severe "
              "warnings cost the farmer real advice. Keep them near zero.")

    # worst offenders, so you can look at the photos themselves
    print("\nGood photos that got a warning (look at these first):")
    shown = 0
    for label, code, p, m, issues in good:
        if issues and shown < 15:
            print(f"   {p}  {issues}")
            shown += 1
    if shown == 0:
        print("   none")
    print("\nBad photos that got NO warning (the check missed these):")
    shown = 0
    for label, code, p, m, issues in bad:
        if not issues and shown < 15:
            print(f"   {p}  ({code or 'unsorted'})")
            shown += 1
    if shown == 0:
        print("   none")


if __name__ == "__main__":
    main()

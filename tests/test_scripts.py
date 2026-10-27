"""The two helper scripts in scripts/ are run the way a person would run them, on data made by
the real app, so a change to the database or the checks cannot quietly break them."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

import synth
from conftest import ROOT, analyze

SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run([sys.executable, "-I", str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def answer(client, store, verdict, photo=True, **extra):
    r = analyze(client, synth.make_leaf(size=(1200, 900), spots=8))
    assert r.status_code == 200, r.text
    body = r.json()
    data = {"analysis_token": body["analysis_token"], "verdict": verdict, "language": "en"}
    files = None
    if photo:
        data["consent_photo"] = "true"
        files = {"image": ("p.jpg", synth.to_jpeg(synth.make_leaf(size=(900, 700), spots=8)), "image/jpeg")}
    data.update(extra)
    out = client.post("/api/feedback", data=data, files=files)
    assert out.status_code == 200, out.text
    return body, out.json()


# ------------------------------------------------------------------ feedback_report.py

def test_report_without_a_database_says_so(tmp_path):
    empty = tmp_path / "nothing-here"
    empty.mkdir()
    r = run("feedback_report.py", "--data", empty)
    assert r.returncode != 0 and "No feedback database" in (r.stdout + r.stderr)


def test_report_on_an_empty_database_says_zero(store):
    r = run("feedback_report.py", "--data", store.root)
    assert r.returncode == 0, r.stderr
    assert "0 answers in total" in r.stdout


def test_agreement_report_reads_the_real_database(client, store, with_contact):
    answer(client, store, "match", photo=False)
    answer(client, store, "match", photo=False)
    answer(client, store, "different", photo=False, reported_id="healthy")
    r = run("feedback_report.py", "--data", store.root)
    assert r.returncode == 0, r.stderr
    assert "rice" in r.stdout
    assert "n=3" in r.stdout or "3" in r.stdout


def test_report_never_changes_the_database(client, store, with_contact):
    answer(client, store, "match", photo=False)
    before = store.db_path.read_bytes()
    count = store.count()
    assert run("feedback_report.py", "--data", store.root).returncode == 0
    assert store.count() == count
    # The script opens the file read-only; the main file's bytes must be untouched.
    assert store.db_path.read_bytes() == before


def test_export_then_expert_scoring(client, store, with_contact, tmp_path):
    answer(client, store, "match")
    answer(client, store, "different", reported_id="healthy")
    answer(client, store, "match", photo=False)          # no consent: must not be exported
    out = tmp_path / "for_experts"

    r = run("feedback_report.py", "--data", store.root, "--export", out)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out / "labels.csv", newline="", encoding="utf-8")))
    assert len(rows) == 2, "only photos with consent are exported"
    assert all((out / row["file"]).exists() for row in rows)
    assert all(row["expert_label"] == "" for row in rows)
    assert all(row["latitude"] == "" for row in rows)    # no location was sent

    # An expert fills in the labels; re-exporting must keep them.
    rows[0]["expert_label"], rows[0]["expert_name"] = "healthy", "Dr Test"
    rows[1]["expert_label"] = "unclear"
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert run("feedback_report.py", "--data", store.root, "--export", out).returncode == 0
    again = {row["id"]: row for row in csv.DictReader(open(out / "labels.csv", newline="", encoding="utf-8"))}
    assert again[rows[0]["id"]]["expert_label"] == "healthy"
    assert again[rows[0]["id"]]["expert_name"] == "Dr Test"
    assert again[rows[1]["id"]]["expert_label"] == "unclear"

    r = run("feedback_report.py", "--expert", out / "labels.csv")
    assert r.returncode == 0, r.stderr
    assert "labelled by experts" in r.stdout
    assert "1 marked unclear" in r.stdout


def test_expert_scoring_with_nothing_labelled_stops_politely(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("id,file,crop,predicted_id,confidence,farmer_verdict,farmer_says,photo_issues,"
                    "latitude,longitude,expert_label,expert_name,expert_note\n"
                    "a1,x.jpg,rice,rice_blast,high,match,,,,,,,\n", encoding="utf-8")
    r = run("feedback_report.py", "--expert", path)
    assert r.returncode != 0 and "Nothing to score" in (r.stdout + r.stderr)


def test_expert_scoring_flags_an_overconfident_level(tmp_path):
    """40 'high' answers, 20 of them wrong: the report must say 'high' is too optimistic."""
    header = ("id,file,crop,predicted_id,confidence,farmer_verdict,farmer_says,photo_issues,"
              "latitude,longitude,expert_label,expert_name,expert_note\n")
    lines = []
    for i in range(40):
        truth = "rice_blast" if i % 2 == 0 else "rice_brown_spot"
        lines.append(f"id{i},x{i}.jpg,rice,rice_blast,high,match,,,,,{truth},Dr T,\n")
    path = tmp_path / "labels.csv"
    path.write_text(header + "".join(lines), encoding="utf-8")
    r = run("feedback_report.py", "--expert", path)
    assert r.returncode == 0, r.stderr
    assert "TOO OPTIMISTIC" in r.stdout
    assert "Spray advice shown for the WRONG disease: 20 of 40" in r.stdout


# ------------------------------------------------------------------ calibrate_quality.py

@pytest.fixture
def photo_folder(tmp_path):
    for sub in ("good", "bad/blur", "bad/dark"):
        (tmp_path / "p" / sub).mkdir(parents=True)
    for i in range(3):
        synth.make_leaf(size=(1000, 750), spots=5, seed=100 + i).convert("RGB").save(
            tmp_path / "p" / "good" / f"g{i}.jpg", "JPEG", quality=92)
        synth.blurred(synth.make_leaf(size=(1000, 750), spots=5, seed=110 + i), 7).convert("RGB").save(
            tmp_path / "p" / "bad" / "blur" / f"b{i}.jpg", "JPEG", quality=92)
        synth.darkened(synth.make_leaf(size=(1000, 750), spots=5, seed=120 + i), 0.15).convert("RGB").save(
            tmp_path / "p" / "bad" / "dark" / f"d{i}.jpg", "JPEG", quality=92)
    return tmp_path / "p"


def test_calibration_script_measures_good_and_bad_photos(photo_folder, tmp_path):
    csv_path = tmp_path / "metrics.csv"
    r = run("calibrate_quality.py", photo_folder, "--csv", csv_path)
    assert r.returncode == 0, r.stderr
    assert "3 good and 6 bad photos measured" in r.stdout
    assert "fewer than 50 good photos" in r.stdout            # small-sample warning must appear
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    assert len(rows) == 9
    bad = {row["expected_issue"] for row in rows if row["label"] == "bad"}
    assert bad == {"blur", "dark"}


def test_calibration_script_with_an_empty_folder_stops_politely(tmp_path):
    r = run("calibrate_quality.py", tmp_path)
    assert r.returncode != 0

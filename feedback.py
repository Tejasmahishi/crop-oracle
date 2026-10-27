"""Consent-based "was this result right?" feedback.

What is stored (SQLite file + one JPEG per consenting farmer):
  - the farmer's answer (matches / different / not sure) and what they think it is
  - what the app said (disease id, confidence, score, photo-quality codes)
  - crop, language, app version, time
  - ONLY if the farmer ticked the photo box: the photo (re-encoded, EXIF removed)
    and the location rounded to 0.1 degrees (about 11 km)

No name, phone number or IP address is stored.

What the app said is carried in a signed token that the server issued with the
analysis, so a feedback record cannot claim the app said something it did not.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

VERDICTS = ("match", "different", "unsure")
TOKEN_TTL_SECONDS = 7 * 24 * 3600
MAX_PHOTO_SIDE = 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    analysis_id TEXT NOT NULL UNIQUE,
    crop TEXT NOT NULL,
    kind TEXT NOT NULL,
    predicted_id TEXT,
    confidence TEXT,
    score REAL,
    quality TEXT,
    verdict TEXT NOT NULL,
    reported_id TEXT,
    consent_photo INTEGER NOT NULL DEFAULT 0,
    lat_r REAL,
    lon_r REAL,
    language TEXT,
    app_version TEXT,
    photo_file TEXT,
    delete_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
"""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def load_secret(root: Path) -> bytes:
    """FEEDBACK_SECRET from the environment, else a random key kept next to the data."""
    env = (os.getenv("FEEDBACK_SECRET") or "").strip()
    if env:
        return env.encode()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "secret.key"
    if path.exists():
        return path.read_bytes().strip()
    key = secrets.token_hex(32).encode()
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def encode_photo(img: Image.Image, max_side: int = MAX_PHOTO_SIDE) -> bytes:
    """Re-encode as a plain JPEG: shrinks it and drops EXIF (GPS, camera model, date)."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88, optimize=True)
    return buf.getvalue()


def round_coord(value: float) -> float:
    return round(float(value), 1)


class Store:
    def __init__(self, root: Path, secret: Optional[bytes] = None):
        self.root = Path(root)
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "feedback.db"
        self.secret = secret or load_secret(self.root)
        self._lock = threading.Lock()
        with self._db() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        """A connection that commits on success, rolls back on error and is always closed."""
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            try:
                con.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    # ---------------------------------------------------- signed analysis token

    def sign_analysis(self, summary: dict, now: Optional[float] = None) -> str:
        body = dict(summary)
        body["ts"] = int(now if now is not None else time.time())
        payload = _b64e(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        mac = hmac.new(self.secret, payload.encode("ascii"), hashlib.sha256).hexdigest()[:32]
        return f"{payload}.{mac}"

    def verify_analysis(self, token: str, now: Optional[float] = None) -> Optional[dict]:
        try:
            payload, mac = (token or "").split(".", 1)
            good = hmac.new(self.secret, payload.encode("ascii"), hashlib.sha256).hexdigest()[:32]
            if not hmac.compare_digest(mac, good):
                return None
            body = json.loads(_b64d(payload).decode("utf-8"))
            age = (now if now is not None else time.time()) - int(body.get("ts", 0))
            if age < 0 or age > TOKEN_TTL_SECONDS:
                return None
            return body
        except Exception:
            return None

    # ---------------------------------------------------- write

    def add(self, *, analysis: dict, verdict: str, reported_id: Optional[str],
            consent_photo: bool, photo_jpeg: Optional[bytes], lat: Optional[float],
            lon: Optional[float], language: str, app_version: str) -> tuple[str, str]:
        """Store one answer. Returns (feedback_id, delete_token).

        Raises ValueError for a bad verdict and FileExistsError if this analysis
        already has an answer.
        """
        if verdict not in VERDICTS:
            raise ValueError("bad verdict")
        fid = uuid.uuid4().hex[:12]
        delete_token = secrets.token_urlsafe(18)
        delete_hash = hashlib.sha256(delete_token.encode()).hexdigest()
        keep_photo = bool(consent_photo and photo_jpeg)
        photo_file = None
        if keep_photo:
            photo_file = f"{fid}.jpg"
            tmp = self.images / f".{fid}.tmp"
            tmp.write_bytes(photo_jpeg)
            os.replace(tmp, self.images / photo_file)
        lat_r = round_coord(lat) if keep_photo and lat is not None else None
        lon_r = round_coord(lon) if keep_photo and lon is not None else None
        try:
            with self._lock, self._db() as con:
                con.execute(
                    "INSERT INTO feedback (id, created_at, analysis_id, crop, kind, predicted_id,"
                    " confidence, score, quality, verdict, reported_id, consent_photo, lat_r, lon_r,"
                    " language, app_version, photo_file, delete_hash)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fid, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     analysis["id"], analysis["crop"], analysis.get("kind", "diagnosis"),
                     analysis.get("pred"), analysis.get("conf"), analysis.get("score"),
                     json.dumps(analysis.get("q", [])), verdict, reported_id,
                     1 if keep_photo else 0, lat_r, lon_r, language, app_version,
                     photo_file, delete_hash))
        except sqlite3.IntegrityError:
            if photo_file:
                (self.images / photo_file).unlink(missing_ok=True)
            raise FileExistsError("already submitted")
        except Exception:
            if photo_file:
                (self.images / photo_file).unlink(missing_ok=True)
            raise
        return fid, delete_token

    def delete(self, fid: str, token: str) -> str:
        """Remove a record and its photo. Returns 'deleted', 'forbidden' or 'missing'."""
        with self._lock, self._db() as con:
            row = con.execute("SELECT delete_hash, photo_file FROM feedback WHERE id=?", (fid,)).fetchone()
            if not row:
                return "missing"
            given = hashlib.sha256((token or "").encode()).hexdigest()
            if not hmac.compare_digest(row["delete_hash"], given):
                return "forbidden"
            con.execute("DELETE FROM feedback WHERE id=?", (fid,))
        if row["photo_file"]:
            (self.images / row["photo_file"]).unlink(missing_ok=True)
        return "deleted"

    def purge_older_than(self, days: int, now: Optional[datetime] = None) -> int:
        """Delete records (and photos) older than `days`. days <= 0 keeps everything."""
        if days <= 0:
            return 0
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock, self._db() as con:
            rows = con.execute("SELECT id, photo_file FROM feedback WHERE created_at < ?", (cutoff,)).fetchall()
            con.execute("DELETE FROM feedback WHERE created_at < ?", (cutoff,))
        for r in rows:
            if r["photo_file"]:
                (self.images / r["photo_file"]).unlink(missing_ok=True)
        return len(rows)

    # ---------------------------------------------------- read

    def count(self) -> int:
        with self._db() as con:
            return int(con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])

    def get(self, fid: str) -> Optional[dict]:
        with self._db() as con:
            row = con.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
        return dict(row) if row else None

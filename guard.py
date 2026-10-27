"""Public-hosting guard rails: rate limits, size caps, security headers, admin token.

Everything is configured with environment variables (see .env.example) and
read once at start-up into SETTINGS. Tests replace guard.SETTINGS.

Rate limits are kept in memory, per process. If you run several worker
processes, each one counts separately, so the real limit is roughly the
setting times the number of workers.
"""
from __future__ import annotations

import hmac
import os
import time
from collections import deque
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _csv(name: str) -> list:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass
class Settings:
    # Requests allowed per client IP per minute. Mobile networks put many farmers
    # behind one IP address, so keep these generous.
    analyze_per_min: int = 30
    feedback_per_min: int = 20
    general_per_min: int = 240
    # Largest request body accepted (photo plus form fields).
    max_body_bytes: int = 11 * 1024 * 1024
    # How many reverse proxies sit in front of the app (0 = none, use the socket address).
    trusted_proxies: int = 0
    # Browser origins allowed to call the API from another site. Empty = same site only.
    cors_origins: list = field(default_factory=list)
    # Token for the /api/coverage and /api/dataset/* routes. Empty = those routes are off.
    admin_token: str = ""
    # "default", "off" or a custom policy string.
    csp: str = "default"
    max_concurrent_analyses: int = 4
    analysis_queue_seconds: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            analyze_per_min=_int("RATE_ANALYZE_PER_MIN", 30),
            feedback_per_min=_int("RATE_FEEDBACK_PER_MIN", 20),
            general_per_min=_int("RATE_GENERAL_PER_MIN", 240),
            max_body_bytes=_int("MAX_BODY_BYTES", 11 * 1024 * 1024),
            trusted_proxies=_int("TRUSTED_PROXY_COUNT", 0),
            cors_origins=_csv("CORS_ALLOWED_ORIGINS"),
            admin_token=(os.getenv("ADMIN_TOKEN") or "").strip(),
            csp=(os.getenv("CSP") or "default").strip(),
            max_concurrent_analyses=max(1, _int("MAX_CONCURRENT_ANALYSES", 4)),
            analysis_queue_seconds=max(1, _int("ANALYSIS_QUEUE_SECONDS", 20)),
        )


SETTINGS = Settings.from_env()


# ------------------------------------------------------------ rate limiting

class RateLimiter:
    """Sliding-window counter: at most `limit` hits per `window` seconds per key."""

    def __init__(self):
        self._hits: dict = {}
        self._last_sweep: float | None = None      # set from the first timestamp seen

    def check(self, bucket: str, key: str, limit: int, window: float = 60.0,
              now: float | None = None) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, seconds_to_wait)."""
        now = time.monotonic() if now is None else now
        if limit <= 0:
            return True, 0
        self._sweep(now, window)
        q = self._hits.setdefault((bucket, key), deque())
        while q and now - q[0] >= window:
            q.popleft()
        if len(q) >= limit:
            return False, max(1, int(window - (now - q[0])) + 1)
        q.append(now)
        return True, 0

    def _sweep(self, now: float, window: float):
        # Drop idle keys now and then so memory cannot grow without bound.
        if self._last_sweep is None:
            self._last_sweep = now
            return
        if now - self._last_sweep < 60 and len(self._hits) < 50_000:
            return
        self._last_sweep = now
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= window]:
            self._hits.pop(k, None)


LIMITER = RateLimiter()


def client_ip(headers, peer_host: str | None, trusted_proxies: int) -> str:
    """The caller's address. X-Forwarded-For is believed only for as many proxies as you declare."""
    if trusted_proxies > 0:
        parts = [p.strip() for p in (headers.get("x-forwarded-for") or "").split(",") if p.strip()]
        if len(parts) >= trusted_proxies:
            return parts[-trusted_proxies]
    return peer_host or "unknown"


def bucket_for(method: str, path: str) -> tuple[str, int]:
    """Which limit applies to this request."""
    if path == "/api/analyze" and method == "POST":
        return "analyze", SETTINGS.analyze_per_min
    if path.startswith("/api/feedback") and method in ("POST", "DELETE"):
        return "feedback", SETTINGS.feedback_per_min
    return "general", SETTINGS.general_per_min


# ------------------------------------------------------------ headers

DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' https://api.open-meteo.com https://geocoding-api.open-meteo.com "
    "https://api.bigdatacloud.net; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)

BASE_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(self), microphone=(self), camera=(self), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def security_headers(content_type: str) -> dict:
    headers = dict(BASE_SECURITY_HEADERS)
    if content_type.startswith("text/html"):
        csp = SETTINGS.csp
        if csp == "default":
            headers["Content-Security-Policy"] = DEFAULT_CSP
        elif csp and csp.lower() != "off":
            headers["Content-Security-Policy"] = csp
    return headers


# ------------------------------------------------------------ admin token

def admin_allowed(provided: str | None) -> bool:
    """True when ADMIN_TOKEN is set and the caller sent exactly that token."""
    expected = SETTINGS.admin_token
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected.encode(), provided.encode())

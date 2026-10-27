"""Helpers for driving the real page in headless Chromium against the real app.

The app runs in a background thread of this process, so the usual test patches (no network, a
temporary feedback folder) apply to it too. The page's own outside calls (weather, place search,
fonts) are answered locally by `stub_outside_world`.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import uvicorn

import main


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    def __init__(self):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=self.port,
                                                    log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


WEATHER = {
    "current": {"time": "2026-10-15T12:00", "temperature_2m": 27.5, "relative_humidity_2m": 78, "precipitation": 0.0},
    "daily": {"time": ["2026-10-13", "2026-10-14", "2026-10-15"], "precipitation_sum": [3.0, 5.5, 0.0]},
}


def stub_outside_world(page):
    """Answer the browser's own calls to other websites without using the network."""
    def weather(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(WEATHER))

    def geocode(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"results": []}))

    def reverse(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"locality": "Test"}))

    def fonts(route):
        route.fulfill(status=200, content_type="text/css", body="")

    page.route("https://api.open-meteo.com/**", weather)
    page.route("https://geocoding-api.open-meteo.com/**", geocode)
    page.route("https://api.bigdatacloud.net/**", reverse)
    page.route("https://fonts.googleapis.com/**", fonts)
    page.route("https://fonts.gstatic.com/**", lambda r: r.abort())


def analyse_through_the_ui(page, image_path):
    """Photo in, 'Analyse' clicked twice (screen 1 then 2), wait until the results are showing."""
    page.set_input_files("#fileIn", str(image_path))
    page.wait_for_function("() => document.querySelector('#dropHas') && !document.querySelector('#dropHas').classList.contains('hidden')")
    page.wait_for_function("() => window.__state && true || true")
    page.click("#btnAnalyze1")
    page.wait_for_selector("#s2.active")
    page.click("#btnAnalyze2")
    page.wait_for_selector("#results3:not(.hidden)", timeout=60000)

"""The demo app serves the page and the replays with a strict content-security policy, and the page
never inserts HTML (SPEC §6.13). Offline: FastAPI's test client, no network."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from voltproof.api.app import CONTENT_SECURITY_POLICY, WEB_DIR, create_app

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
HTML_SINKS = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function")


def client() -> TestClient:
    return TestClient(create_app())


def test_page_and_replays_are_served_with_security_headers() -> None:
    c = client()
    page = c.get("/")
    assert page.status_code == 200 and "VoltProof AI" in page.text
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert page.headers["x-content-type-options"] == "nosniff" and page.headers["referrer-policy"] == "no-referrer"
    index = c.get("/replays/index.json")
    assert index.status_code == 200 and index.json()["chips"]
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/replays/does-not-exist.json").status_code == 404


def test_policy_allows_no_inline_code_and_only_the_pinned_chart_library() -> None:
    assert "unsafe-inline" not in CONTENT_SECURITY_POLICY and "unsafe-eval" not in CONTENT_SECURITY_POLICY
    assert "script-src 'self' https://cdnjs.cloudflare.com" in CONTENT_SECURITY_POLICY and "connect-src 'self'" in CONTENT_SECURITY_POLICY
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    sources = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert sources == [CHART_JS, "charts.js", "app.js"]
    assert re.search(rf'src="{re.escape(CHART_JS)}" integrity="sha512-[A-Za-z0-9+/=]+" crossorigin="anonymous"', html)
    assert not re.search(r"<script(?![^>]*src=)[^>]*>", html)  # no inline script
    assert " style=" not in html and "<style" not in html and not re.search(r"<[^>]+\son[a-z]+=", html)  # no inline style or handlers
    assert 'http-equiv="Content-Security-Policy"' in html  # the static copy carries the policy too


def test_the_page_never_inserts_html() -> None:
    for name in ("app.js", "charts.js"):
        source = (WEB_DIR / name).read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)  # comments may mention the rule itself
        code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
        for sink in HTML_SINKS:
            assert sink not in code, f"{name} uses {sink}"


def test_goal_box_limits_and_notice() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="goal"' in html and 'maxlength="500"' in html and '<label for="goal">' in html and '<label for="day">' in html
    assert "Typed goals are sent to the model provider" in html and "do not enter personal information" in html
    assert "Weather data by" in html and "Open-Meteo.com" in html and "Built with Muse Spark Open (Meta), served by Baseten" in html

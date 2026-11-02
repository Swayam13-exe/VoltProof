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
    assert sources == ["theme.js", CHART_JS, "charts.js", "dayplayer.js", "app.js", "landing.js"]
    assert '<script src="theme.js"></script>' in html.split("</head>")[0]  # synchronous in <head>: no theme flash
    assert re.search(rf'src="{re.escape(CHART_JS)}" integrity="sha512-[A-Za-z0-9+/=]+" crossorigin="anonymous"', html)
    assert not re.search(r"<script(?![^>]*src=)[^>]*>", html)  # no inline script
    assert " style=" not in html and "<style" not in html and not re.search(r"<[^>]+\son[a-z]+=", html)  # no inline style or handlers
    assert 'http-equiv="Content-Security-Policy"' in html  # the static copy carries the policy too


def test_the_page_never_inserts_html() -> None:
    for name in ("theme.js", "app.js", "charts.js", "dayplayer.js", "landing.js"):
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
    conversation = re.search(r'<section id="conversation"[^>]*>', html).group(0)
    assert "aria-live" not in conversation and 'id="status"' in html and 'role="status"' in html  # the status line announces
    assert "Weather: <a" in html and "/ DWD ICON" in html and "MIT license" in html


def test_landing_sections_menu_and_no_hand_typed_numbers() -> None:
    """The shell around Try it: the five menu targets exist, Try it keeps every id the tool uses, and the
    only digits in visible text are step labels; every result number is filled in from a file."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for target in ("problem", "how", "try", "evidence", "limits"):
        assert f'<a href="#{target}">' in html and f'id="{target}"' in html
    tool = html[html.index('<section id="try"'):html.index('<section id="evidence"')]
    for tool_id in ("setup", "site-list", "day", "chips", "goal-form", "goal", "goal-count", "live-note", "status", "conversation"):
        assert f'id="{tool_id}"' in tool, tool_id
    visible = re.sub(r"<(script|svg)\b.*?</\1>", "", html, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", visible)  # text only, attributes dropped
    visible = visible[visible.index("Skip to"):]
    numbers = set(re.findall(r"(?<![\w-])\d+(?:[.,:]\d+)*%?(?![\w-])", visible))  # standalone numbers, not names like Swayam13
    assert numbers <= {"1", "2", "3", "4", "0", "500", "4.0"}, numbers  # step labels, the goal box counter, CC BY 4.0


def test_content_is_visible_without_javascript() -> None:
    """Reveal-on-scroll is progressive: CSS hides .reveal content only under the class landing.js adds
    (and removes after a safety timer), so with no script, or a failed one, every section shows."""
    css = re.sub(r"/\*.*?\*/", "", (WEB_DIR / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    hiding = [rule for rule in re.findall(r"([^{}]+)\{([^}]*)\}", css) if "opacity: 0" in rule[1] and ".reveal" in rule[0]]
    assert hiding and all(selector.strip().startswith(".reveal-armed") for selector, _ in hiding)
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    assert 'classList.add("reveal-armed")' in landing and 'classList.remove("reveal-armed"), 1500' in landing

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
    reveal = landing[landing.index("function setupReveal("):landing.index("/* --- hero:")]
    assert 'if (reduceMotion.matches || !("IntersectionObserver" in window)) return;' in reveal  # armed only with IO, no reduced motion
    assert 'classList.add("reveal-armed")' in reveal and "if (inView(node)) show(node);" in reveal  # in view at load: shown at once
    assert "window.setTimeout(() => show(node), 1000)" in reveal  # the safety net, while scrolling, the whole way down
    assert "1500" not in reveal  # no blanket timer that ends the effect after the first seconds
    assert re.search(r"\.reveal \{ transition: opacity 0\.4s ease, transform 0\.4s ease; \}", css)
    assert "translateY(16px)" in css and "height" not in css[css.index(".reveal-armed"):css.index(".reveal-armed") + 120]  # no layout shift


def test_every_section_shares_one_frame() -> None:
    """Try it is part of the page, not a separate tool: every section below the hero is a plain
    .section with the same .section-inner frame (only the Problem two-column split and the reveal
    hook on top), and no CSS rule gives one section its own background band or a narrower column."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", (WEB_DIR / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    for section_id in ("problem", "how", "try", "evidence", "limits"):
        tag = re.search(rf'<section id="{section_id}" class="([^"]+)"', html).group(1)
        assert tag == "section", (section_id, tag)
        start = html.index(f'<section id="{section_id}"')
        inner = re.search(r'<div class="([^"]+)"', html[start:]).group(1).split()
        assert inner[0] == "section-inner" and set(inner[1:]) <= {"split", "reveal"}, (section_id, inner)
    assert not re.search(r"\.section-inner\.[a-z]", css)  # no narrowed variant of the frame
    assert "#try" not in css and not re.search(r"\.try[\s{.,]", css)  # no special band or width for Try it
    assert re.search(r"\.hero \{[^}]*max-width: 1160px", css) and re.search(r"\.section-inner \{[^}]*max-width: 1160px", css)  # hero and sections share the width


def test_top_bar_and_hero_link_to_github_and_the_video() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'<a [^>]*href="https://github.com/Swayam13-exe/VoltProof"[^>]*>', html)
    top_and_hero = [a for a in links if ('id="github-link"' in a or 'class="hero-link"' in a) and "video" not in a]
    assert len(top_and_hero) == 2 and all('target="_blank"' in a and 'rel="noopener noreferrer"' in a and "aria-label=" in a for a in top_and_hero)
    for video_id in ("video-link", "hero-video"):  # hidden until /api/status gives an https URL
        tag = re.search(rf'<a id="{video_id}"[^>]*>', html).group(0)
        assert " hidden" in tag and 'target="_blank"' in tag and 'rel="noopener noreferrer"' in tag and "aria-label=" in tag
    assert "copy-link" not in html and "copy-status" not in html
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    assert "setupCopyLink" not in landing and 'url.startsWith("https://")' in landing
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    assert ".actions .tool-text { display: none; }" in css  # phones: icons only


# --- Part C1: favicon and link previews -------------------------------------------------------------------


def preview_tags(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r'<meta (?:property|name)="((?:og|twitter):[a-z:]+)" content="([^"]*)">', text)}


def test_the_page_has_its_favicon_and_link_preview_tags() -> None:
    from voltproof.config import load_settings

    page = TestClient(create_app(settings=load_settings({}))).get("/")
    assert page.status_code == 200 and page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' in page.text
    description = re.search(r'<meta name="description" content="([^"]+)">', page.text).group(1)
    tags = preview_tags(page.text)
    assert tags == {"og:title": "VoltProof AI", "og:description": description, "og:type": "website", "twitter:card": "summary_large_image"}  # no image tags without a base
    assert "og:image" not in page.text and "twitter:image" not in page.text


def test_image_tags_are_absolute_when_the_public_address_is_set() -> None:
    from voltproof.config import load_settings

    app = create_app(settings=load_settings({"PUBLIC_BASE_URL": "https://example.hf.space/"}))
    for path in ("/", "/index.html"):
        tags = preview_tags(TestClient(app).get(path).text)
        assert tags["og:image"] == tags["twitter:image"] == "https://example.hf.space/og.png", path
        assert (tags["og:image:width"], tags["og:image:height"]) == ("1200", "630")
    assert all(not v.startswith("/") for v in tags.values())


def test_favicon_and_preview_image_are_served_with_their_types_and_sizes() -> None:
    from PIL import Image

    c = client()
    icon = c.get("/favicon.svg")
    assert icon.status_code == 200 and icon.headers["content-type"].startswith("image/svg+xml")
    assert icon.headers["content-security-policy"] == CONTENT_SECURITY_POLICY and "img-src 'self'" in CONTENT_SECURITY_POLICY
    assert icon.text.startswith("<svg") and "#f2a516" in icon.text and "<script" not in icon.text
    og = c.get("/og.png")
    assert og.status_code == 200 and og.headers["content-type"] == "image/png"
    assert len(og.content) < 300 * 1024
    with Image.open(WEB_DIR / "og.png") as image:
        assert image.size == (1200, 630) and image.format == "PNG"


# --- Part D7: the page route -----------------------------------------------------------------------------


def test_head_and_get_on_the_page_routes_and_the_static_types() -> None:
    c = client()
    for path in ("/", "/index.html"):
        get, head = c.get(path), c.head(path)
        assert get.status_code == head.status_code == 200 and head.content == b""
        assert get.headers["content-type"] == head.headers["content-type"] == "text/html; charset=utf-8"
        assert head.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    for path, kind in (("/favicon.svg", "image/svg+xml"), ("/og.png", "image/png")):
        for response in (c.get(path), c.head(path)):
            assert response.status_code == 200 and response.headers["content-type"].startswith(kind), path


def test_the_injected_image_url_is_html_escaped() -> None:
    from voltproof.api.app import with_image_tags

    page = with_image_tags("<html><head>\n</head><body></body></html>", 'https://x.example/a?b=1&c="><script>alert(1)</script>')
    assert "<script>" not in page and '"><' not in page
    assert 'content="https://x.example/a?b=1&amp;c=&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;/og.png"' in page
    assert page.count("</head>") == 1 and with_image_tags("<head></head>", "") == "<head></head>"


def test_a_non_https_video_url_is_ignored_and_the_app_still_starts() -> None:
    from voltproof.config import load_settings

    for bad in ("http://example.org/video", "javascript:alert(1)"):
        status = TestClient(create_app(settings=load_settings({"DEMO_VIDEO_URL": bad}))).get("/api/status").json()
        assert status["demo_video_url"] is None  # the page keeps both video links hidden
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    assert 'if (typeof url !== "string" || !url.startsWith("https://")) return;' in landing  # and the page checks again

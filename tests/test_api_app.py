"""The demo app serves the page and the replays with a strict content-security policy, and the page
never inserts HTML (SPEC §6.13). Offline: FastAPI's test client, no network."""

from __future__ import annotations

import hashlib
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
    """Reveal-on-scroll is progressive: CSS hides .reveal blocks only under the class landing.js adds,
    and only with IntersectionObserver and without reduced motion, so with no script, a failed one, an
    old browser or reduced motion, every block shows with no transition."""
    css = re.sub(r"/\*.*?\*/", "", (WEB_DIR / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    hiding = [rule for rule in re.findall(r"([^{}]+)\{([^}]*)\}", css) if "opacity: 0" in rule[1] and ".reveal" in rule[0]]
    assert hiding and all(selector.strip().startswith(".reveal-armed") for selector, _ in hiding)
    assert re.search(r"\.reveal \{ transition: opacity 0\.35s ease-out, transform 0\.35s ease-out; \}", css)  # one motion for every block
    assert ".reveal-armed .reveal:not(.in) { opacity: 0; transform: translateY(12px); }" in css  # transform only: no layout shift
    assert "transition: none !important" in css[css.index("@media (prefers-reduced-motion: reduce)"):]
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    reveal = landing[landing.index("function setupReveal("):landing.index("/* --- hero:")]
    assert 'if (reduceMotion.matches || !("IntersectionObserver" in window)) return;' in reveal
    assert 'document.documentElement.classList.add("reveal-armed");' in reveal
    # on screen at load, or within the margin: shown before arming (no fade on first paint)
    assert reveal.index('if (box(node).top < window.innerHeight * (1 + REVEAL_MARGIN)) node.classList.add("in")') < reveal.index('classList.add("reveal-armed")')
    # the trigger: threshold 0 with a positive bottom margin on the document's own viewport (so it also holds in a frame)
    assert "root: document," in reveal and "threshold: 0," in reveal and "rootMargin: `0px 0px ${Math.round(REVEAL_MARGIN * 100)}% 0px`" in reveal
    assert "unobserve" not in reveal  # blocks stay observed: they replay
    # every scrolled frame judges every block: a jump can cross the viewport with no intersection change
    assert "blocks.forEach(apply);" in reveal and 'addEventListener("scroll", queueSweep' in reveal
    # the safety net: only blocks in or above the viewport, independent of scrolling and of the observer
    assert 'if (!node.classList.contains("in") && node.getBoundingClientRect().top < window.innerHeight) node.classList.add("in");' in reveal and "}, 1000);" in reveal
    # the Try it frame is pinned once used or once a run has started
    assert 'frame.dataset.used === "1" || !$("try-pipeline").hidden' in reveal
    assert all(f'"{kind}"' in reveal for kind in ("focusin", "pointerdown", "input"))


def test_every_fixed_block_outside_the_hero_fades_and_run_output_never_does() -> None:
    """Headings, leads, content blocks, both charts with their notes, the number cards, the Limits list and
    the footer fade; the hero, the result area, the bill panel, the pipeline strip and anything a run
    renders never do, and there is no per-child stagger (no card is a block of its own)."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    blocks = re.findall(r'<(\w+)([^>]*class="[^"]*\breveal\b[^"]*"[^>]*)>', html)
    ids = [re.search(r'id="([^"]+)"', attrs).group(1) if 'id="' in attrs else re.search(r'class="([^"]+)"', attrs).group(1) for _, attrs in blocks]
    assert ids == ["reveal", "problem-dots", "how-title", "lead reveal", "pipeline reveal", "pipe-loop reveal", "pipe-key reveal", "try-title", "lead reveal", "setup",
                   "evidence-title", "evidence-size", "evidence-numbers", "evidence-footnote", "evidence-chart", "links reveal", "limits-title", "limits reveal", "section-inner reveal"]
    for never in ("conversation", "bill-panel", "try-pipeline", "day-panel"):
        tag = re.search(rf'<\w+ id="{never}"[^>]*>', html).group(0)
        assert "reveal" not in tag, never
    hero = html[html.index('<section id="top"'):html.index('<section id="problem"')]
    assert "reveal" not in hero
    assert 'class="section-inner reveal"' in html[html.index('<footer class="site-footer">'):]
    # the two Problem columns are one group, so they start at the same moment when side by side
    problem = html[html.index('<section id="problem"'):html.index('<section id="how"')]
    assert problem.count('data-reveal-group="problem"') == 2 and html.count("data-reveal-group") == 2
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    assert "if (o.top < r.bottom && o.bottom > r.top) {" in landing and "top = Math.min(top, o.top);" in landing  # one row, one moment; stacked: its own top
    for path in WEB_DIR.glob("*.js"):
        js = path.read_text(encoding="utf-8")
        assert '"reveal"' not in js and "watchReveal" not in js, path.name  # nothing created at run time is a block
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    assert ".pipe-node.reveal" not in css and "transition-delay" not in css  # no stagger
    # neither landing-page chart has a drawing animation of its own to double up with the fade
    assert "new Chart" not in landing
    for selector in (r"\.bar-fill \{", r"\.dot\.pass \{", r"\.dot-svg \{"):
        assert "transition" not in re.search(selector + r"([^}]*)\}", css).group(1) and "animation" not in re.search(selector + r"([^}]*)\}", css).group(1), selector


def test_no_lines_between_sections_and_one_spacing_token() -> None:
    css = re.sub(r"/\*.*?\*/", "", (WEB_DIR / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    assert ".section + .section" not in css
    footer = re.search(r"\.site-footer \{([^}]*)\}", css).group(1)
    assert "border" not in footer and "padding: calc(var(--section-gap) / 2) 0 40px" in footer
    assert re.search(r"\.topbar \{[^}]*\}", css) and "border-bottom: 1px solid var(--line)" in css  # the sticky header keeps its line
    assert ":root { --section-gap: 88px; }" in css and "@media (min-width: 900px) { :root { --section-gap: 160px; } }" in css
    assert ".section { padding: calc(var(--section-gap) / 2) 0; }" in css
    assert "padding: 40px 16px calc(var(--section-gap) / 2)" in re.search(r"\.hero \{([^}]*)\}", css).group(1)
    assert re.findall(r"\.section \{[^}]*\}", css) == [".section { padding: calc(var(--section-gap) / 2) 0; }"]  # no per-section override
    for section_id in ("problem", "how", "try", "evidence", "limits"):
        assert f"#{section_id} {{" not in css and f"#{section_id}{{" not in css
    assert "section[id] { scroll-margin-top: calc(var(--topbar-h) + 8px); }" in css


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


# --- Part F4: links that leave the site -------------------------------------------------------------------


def external(href: str) -> bool:
    return bool(re.match(r"https?://", href)) and not href.startswith(("http://127.0.0.1", "http://localhost"))


def test_every_external_link_opens_a_new_tab_without_opener_or_referrer() -> None:
    """The page runs in a frame on huggingface.co; a link to another site without target="_blank" tries to
    open inside the frame and is refused. In-page anchors stay in the same tab."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    anchors = re.findall(r"<a\s[^>]*>", html)
    outside = [a for a in anchors if external(re.search(r'href="([^"]*)"', a).group(1))]
    assert len(outside) >= 9  # header (2), hero (2), evidence (2), footer (3)
    for a in outside:
        assert 'target="_blank"' in a and 'rel="noopener noreferrer"' in a, a
    for a in anchors:
        if re.search(r'href="#', a):
            assert "target=" not in a, a  # in-page jumps never open a tab
    for name in ("github.com/Swayam13-exe/VoltProof/blob/main/LICENSE", "open-meteo.com"):
        assert any(name in a for a in outside), name
    # anchors created by the scripts: only same-origin ones (the CSV download is a blob: URL) or hrefs set on
    # the existing, already-safe video links
    for path in WEB_DIR.glob("*.js"):
        js = path.read_text(encoding="utf-8")
        for created in re.findall(r'make\("a",\s*\{([^}]*)\}', js) + re.findall(r'createElement\("a"\)', js):
            assert "http" not in created and ("target" in created or "download" in created), (path.name, created)
        for target in re.findall(r'\$\("([\w-]+)"\)\.href\s*=', js) + re.findall(r'\$\(id\)\.href = url', js):
            pass
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    for video_id in ("video-link", "hero-video"):
        tag = re.search(rf'<a id="{video_id}"[^>]*>', html).group(0)
        assert 'target="_blank" rel="noopener noreferrer"' in tag


def test_the_footer_has_no_video_link_and_the_header_and_hero_follow_the_rule() -> None:
    """ "Watch video" lives in the header and the hero line only, shown only for an https DEMO_VIDEO_URL;
    the footer never has one, with the URL set or unset (the script never touches the footer)."""
    from voltproof.config import load_settings

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    footer = html[html.index('<footer class="site-footer">'):html.index("</footer>")]
    assert "video" not in footer.lower()
    assert [re.sub(r"<[^>]+>", "", p).strip() for p in re.findall(r"<p[^>]*>.*?</p>", footer, re.S)] == [
        "Built with Muse Spark Open (Meta), served by Baseten. Code and results: github.com/Swayam13-exe/VoltProof",
        "Weather: Open-Meteo / DWD ICON. Weather data by Open-Meteo.com, CC BY 4.0.",
        "MIT license",
    ]
    landing = (WEB_DIR / "landing.js").read_text(encoding="utf-8")
    video = landing[landing.index("async function setupVideoLinks("):landing.index("/* --- menu:")]
    assert 'for (const id of ["video-link", "hero-video"]) {' in video and "footer" not in video
    assert video.index('!url.startsWith("https://")) return;') < video.index('$(id).hidden = false;')
    for video_id in ("video-link", "hero-video"):
        tag = re.search(rf'<a id="{video_id}"[^>]*>', html).group(0)
        assert " hidden" in tag and 'target="_blank" rel="noopener noreferrer"' in tag
    for env in ({}, {"DEMO_VIDEO_URL": "https://example.org/video"}):  # the served page's footer, URL unset and set
        page = TestClient(create_app(settings=load_settings(env))).get("/").text
        assert "video" not in page[page.index('<footer class="site-footer">'):].lower(), env


# --- Part H3: a deploy never leaves a visitor on old files --------------------------------------------------


def test_the_served_page_has_no_cache_and_versions_its_own_files_only() -> None:
    from voltproof.config import load_settings

    c = TestClient(create_app(settings=load_settings({})))
    for path in ("/", "/index.html"):
        page = c.get(path)
        assert page.headers["cache-control"] == "no-cache", path
    links = re.findall(r'<(?:script|link)\b[^>]*?\b(?:src|href)="([^"]+)"', page.text)
    own = [link for link in links if link.endswith(tuple("0123456789abcdef")) and "?v=" in link]
    assert sorted(link.split("?")[0] for link in own) == ["app.js", "charts.js", "dayplayer.js", "landing.js", "styles.css", "theme.js"]
    for link in own:
        name, version = link.split("?v=")
        assert version == hashlib.sha256((WEB_DIR / name).read_bytes()).hexdigest()[:10], name
        served = c.get(f"/{link}")  # the static route ignores the query
        assert served.status_code == 200 and served.content == (WEB_DIR / name).read_bytes(), name
        assert served.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert not any("?v=" in link for link in links if link.startswith("http"))  # the pinned Chart.js keeps its integrity-checked URL
    assert "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js\"" in page.text
    assert CONTENT_SECURITY_POLICY == page.headers["content-security-policy"]  # 'self' covers a query string: the policy is unchanged


def test_the_version_changes_when_the_file_changes(tmp_path) -> None:
    from voltproof.api.app import with_asset_versions

    (tmp_path / "styles.css").write_text("a { color: red; }", encoding="utf-8")
    (tmp_path / "app.js").write_text("let a = 1;", encoding="utf-8")
    page = '<link rel="stylesheet" href="styles.css"><script defer src="app.js"></script><script src="gone.js"></script><script src="https://x.example/lib.js"></script><a href="other.js">not an asset</a>'
    first = with_asset_versions(page, tmp_path)
    assert first == with_asset_versions(page, tmp_path)  # stable while nothing changes
    (tmp_path / "styles.css").write_text("a { color: blue; }", encoding="utf-8")
    second = with_asset_versions(page, tmp_path)
    css = lambda text: re.search(r'href="(styles\.css\?v=[0-9a-f]{10})"', text).group(1)
    js = lambda text: re.search(r'src="(app\.js\?v=[0-9a-f]{10})"', text).group(1)
    assert css(first) != css(second) and js(first) == js(second)  # only the changed file gets a new URL
    assert 'src="gone.js"' in second and 'src="https://x.example/lib.js"' in second and 'href="other.js"' in second  # missing, external and non-asset links are left alone

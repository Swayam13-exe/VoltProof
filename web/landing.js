/* VoltProof landing page: the shell around Try it (app.js) and the hero's day player (dayplayer.js).
   - the hero mounts the shared day player on three recorded days (web/replays)
   - the Problem, How-it-works and Evidence numbers come from /api/benchmark (the committed summary
     files in results, plus constants from the frozen code); every number shows its source
   - top menu (nothing highlighted while the hero is in view), theme toggle, GitHub and video links,
     reveal on scroll
   Everything is built with createElement and textContent: no HTML is ever inserted. */
"use strict";

(() => {
  /* --- pure: begin (no DOM; tested in tests/test_web_logic.py) --------------------------------- */

  /* Reveal on scroll: what to do with one block, from where it is (px, viewport coordinates). Its top
     passing 88% of the viewport shows it; a block that has gone completely below the viewport (the
     reader scrolled back up past it) is hidden again so it fades in the next time; a block above the
     viewport (left through the top, or jumped past by a menu link) is shown; a pinned block (the Try it
     frame once used) always shows. Anything else, partly on screen below the line, stays as it is. */
  const REVEAL_LINE = 0.88;

  function revealDecision(top, bottom, viewportHeight, pinned) {
    if (pinned || bottom <= 0) return "show";
    if (top < viewportHeight * REVEAL_LINE) return "show";
    if (top >= viewportHeight) return "hide";
    return "keep";
  }

  const HERO_PAIRS = [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"], ["munich_home", "es"]];
  const HERO_KIND_RANK = { english: 0, multilingual: 1, hard: 2 };
  const BENCHMARK_FILES = { main: "results/report/summary.json", e2: "results/e2/summary.json" };

  /* The recorded run for each (site, language) pair, in the hero's order; null where none exists.
     Several matches: an everyday goal before a translated one before a deliberately hard one. */
  function heroPicks(chips, pairs = HERO_PAIRS) {
    return pairs.map(([site, language]) => {
      const matches = chips.filter((c) => c.site_id === site && c.language === language);
      matches.sort((a, b) => (HERO_KIND_RANK[a.kind] ?? 9) - (HERO_KIND_RANK[b.kind] ?? 9));
      return matches[0] || null;
    });
  }

  /* Dots for a printed rate over n episodes: the whole count that reproduces the printed rate exactly
     (e.g. "17.8%" of 45 -> 8, since 8/45 prints as 17.8%), or null when no single whole count does. */
  function dotCount(rateText, n) {
    const match = /^\s*(\d+(?:\.(\d+))?)%/.exec(String(rateText));
    if (!match || !(n > 0)) return null;
    const decimals = match[2] ? match[2].length : 0;
    const passed = Math.round((Number(match[1]) * n) / 100);
    const candidates = [passed - 1, passed, passed + 1].filter((k) => k >= 0 && k <= n && ((100 * k) / n).toFixed(decimals) === match[1]);
    return candidates.length === 1 ? { passed: candidates[0], failed: n - candidates[0] } : null;
  }

  function sourceLine(file, detail) {
    return `Source: ${file}${detail ? ` · ${detail}` : ""}`;
  }

  /* "98.8% (257/260)" -> "98.8%": the number exactly as the file prints it. */
  function leadingNumber(text) {
    const match = /^\s*([+-]?\d+(?:\.\d+)?%?)/.exec(String(text));
    return match ? match[1] : null;
  }

  function parseShare(text) {
    const match = /^\s*(\d+(?:\.\d+)?)%(?:\s*\((\d+)\/(\d+)\))?/.exec(String(text));
    if (!match) return null;
    return { pct: Number(match[1]), num: match[2] ? Number(match[2]) : null, den: match[3] ? Number(match[3]) : null };
  }

  function parseDelta(text) {
    const match = /^\s*([+-]?\d+(?:\.\d+)?)\s*pts\s*\[\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\s*\]/.exec(String(text));
    return match ? { value: match[1], low: match[2], high: match[3] } : null;
  }

  function rowFor(rows, key, value) {
    return (rows || []).find((row) => row[key] === value) || null;
  }

  function armNames(runSummary) {
    return Object.fromEntries((runSummary || []).map((row) => [row.arm, row.name]));
  }

  /* --- pure: end ------------------------------------------------------------------------------- */

  const make = VPCharts.make;
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const THEME_KEY = "voltproof-theme";
  const REFERENCE_ARMS = new Set(["A5", "A6"]); // given the hand-written rules or the actual weather
  const OURS = "A4";
  const LLM_SCHEDULE = "A2";

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) if (value !== null && value !== undefined) node.setAttribute(key, String(value));
    for (const child of children.flat()) if (child) node.append(child instanceof Node ? child : String(child));
    return node;
  }

  /* --- theme: the button names the mode it switches to ---------------------------------------- */

  function storedTheme() {
    try {
      const value = window.localStorage.getItem(THEME_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (_err) {
      return null;
    }
  }

  function effectiveDark() {
    const set = document.documentElement.dataset.theme;
    return set ? set === "dark" : darkScheme.matches;
  }

  function labelThemeButton() {
    const button = $("theme-toggle");
    const toDark = !effectiveDark();
    button.setAttribute("aria-label", toDark ? "Switch to dark theme" : "Switch to light theme");
    button.querySelector(".tool-icon").textContent = toDark ? "☾" : "☀";
    button.querySelector(".tool-text").textContent = toDark ? "Dark" : "Light";
  }

  function setupTheme() {
    labelThemeButton();
    $("theme-toggle").addEventListener("click", () => {
      const next = effectiveDark() ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        window.localStorage.setItem(THEME_KEY, next);
      } catch (_err) {
        // storage blocked: the choice lasts for this page view only
      }
      labelThemeButton();
      VPCharts.rerenderAll();
    });
    darkScheme.addEventListener("change", () => {
      if (!storedTheme()) labelThemeButton();
    });
  }

  /* --- the demo video link: shown only when the server has DEMO_VIDEO_URL (https) ----------------- */

  async function setupVideoLinks() {
    try {
      const response = await fetch("api/status", { cache: "no-store" });
      const url = response.ok ? (await response.json()).demo_video_url : null;
      if (typeof url !== "string" || !url.startsWith("https://")) return;
      for (const id of ["video-link", "hero-video"]) {
        $(id).href = url;
        $(id).hidden = false;
      }
    } catch (_err) {
      // no server here: the link stays hidden
    }
  }

  /* --- menu: the section under the top bar is highlighted; nothing while the hero is in view ---- */

  function setupMenu() {
    const links = [...document.querySelectorAll(".menu a")];
    const menu = document.querySelector(".menu");
    const sections = links.map((link) => [link, $(link.hash.slice(1))]).filter(([, section]) => section);
    let current = null;
    const update = () => {
      const bar = document.querySelector(".topbar").offsetHeight;
      const hero = $("top");
      let active = null;
      if (window.scrollY + bar >= hero.offsetTop + hero.offsetHeight - 80) {
        const probe = window.scrollY + bar + Math.min(window.innerHeight, 900) * 0.35;
        for (const [link, section] of sections) if (section.offsetTop <= probe) active = link;
      }
      if (active === current) return;
      current = active;
      for (const [link] of sections) {
        if (link === active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      }
      if (active && menu.scrollWidth > menu.clientWidth) menu.scrollTo({ left: active.offsetLeft - 16, behavior: reduceMotion.matches ? "auto" : "smooth" });
    };
    let queued = false;
    window.addEventListener("scroll", () => {
      if (queued) return;
      queued = true;
      window.requestAnimationFrame(() => { queued = false; update(); });
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
    // phones: while more items sit past the right edge, the row fades out there (CSS .menu.more-right),
    // so a cut-off item reads as "scroll for more"
    const edge = () => menu.classList.toggle("more-right", menu.scrollLeft + menu.clientWidth < menu.scrollWidth - 2);
    menu.addEventListener("scroll", edge, { passive: true });
    window.addEventListener("resize", edge);
    edge();
  }

  /* --- reveal on scroll ------------------------------------------------------------------------- */

  /* The top-level blocks of each section (the .reveal elements in index.html: headings, leads and main
     content blocks, never charts, tables, results or bill-panel contents) fade in and rise as their top
     passes 88% of the viewport, and again after the reader has scrolled back up past them (see
     revealDecision). Content is visible without JavaScript: the hidden state (CSS under .reveal-armed)
     is applied only here, only with IntersectionObserver and without reduced motion.
     - two observers (threshold 0): one whose root ends 12% above the viewport bottom (the 88% line), one
       at the viewport edge (a block leaving below); both just ask revealDecision about the block
     - a scroll sweep shows blocks above the viewport at once (a smooth menu jump can carry a block from
       below to above without it ever intersecting), and is the safety net: a block in the viewport still
       hidden 1 s later is shown; a block below the viewport is never shown by it
     - the Try it frame (#setup) is pinned once used (any focus, click or typing in it, or a run in the
       pipeline strip): it never hides again, so nothing a run renders starts at opacity 0 */
  function setupReveal() {
    if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
    const blocks = [...document.querySelectorAll(".reveal")];
    const frame = $("setup");
    const pinned = (node) => node === frame && (frame.dataset.used === "1" || !$("try-pipeline").hidden);
    const apply = (node) => {
      const r = node.getBoundingClientRect();
      const action = revealDecision(r.top, r.bottom, window.innerHeight, pinned(node));
      if (action === "show") node.classList.add("in");
      else if (action === "hide") node.classList.remove("in");
    };
    const onEntries = (entries) => entries.forEach((entry) => apply(entry.target));
    const atLine = new IntersectionObserver(onEntries, { threshold: 0, rootMargin: `0px 0px -${Math.round((1 - REVEAL_LINE) * 100)}% 0px` });
    const atEdge = new IntersectionObserver(onEntries, { threshold: 0 });
    // at load: whatever is already on screen shows at once (no fade on first paint)
    for (const node of blocks) if (node.getBoundingClientRect().top < window.innerHeight) node.classList.add("in");
    document.documentElement.classList.add("reveal-armed");
    for (const node of blocks) {
      atLine.observe(node);
      atEdge.observe(node);
    }
    if (frame) {
      const use = () => {
        frame.dataset.used = "1";
        frame.classList.add("in");
      };
      for (const kind of ["focusin", "pointerdown", "input"]) frame.addEventListener(kind, use, { passive: true });
    }
    let queued = false;
    const sweep = () => {
      queued = false;
      for (const node of blocks) {
        if (node.classList.contains("in")) continue;
        const r = node.getBoundingClientRect();
        if (r.bottom <= 0) node.classList.add("in"); // above: show now
        else if (r.top < window.innerHeight && !node.dataset.revealDue) {
          node.dataset.revealDue = "1"; // the safety net, for blocks in the viewport only
          window.setTimeout(() => {
            delete node.dataset.revealDue;
            if (node.getBoundingClientRect().top < window.innerHeight) node.classList.add("in");
          }, 1000);
        }
      }
    };
    window.addEventListener("scroll", () => {
      if (queued) return;
      queued = true;
      window.requestAnimationFrame(sweep);
    }, { passive: true });
  }

  /* --- hero: three recorded days in the shared day player ------------------------------------------ */

  async function setupHero() {
    try {
      const index = await (await fetch("replays/index.json", { cache: "no-cache" })).json();
      const picks = heroPicks(index.chips).filter(Boolean);
      const views = await Promise.all(picks.map((chip) => fetch(`replays/${encodeURIComponent(chip.file)}`, { cache: "no-cache" }).then((r) => r.json())));
      const examples = views.map((view, i) => ({ view, file: `web/replays/${picks[i].file}`, language: view.goal.language || picks[i].language }));
      if (examples.length) VPDay.mount($("day-mount"), examples, { id: "hero", switcher: true, loop: true, live: $("day-live"), cost: "muted" });
    } catch (_err) {
      $("day-mount").replaceChildren(make("p", { class: "hint" }, "The recorded days could not load. Try it below still works."));
    }
  }

  /* --- problem: dot grid ------------------------------------------------------------------------- */

  function dotGroup(name, count, total, ours) {
    const columns = 15;
    const rows = Math.ceil(total / columns);
    const dots = [];
    for (let i = 0; i < total; i += 1) {
      const pass = i < count.passed;
      dots.push(svg("circle", { class: `dot ${pass ? "pass" : "fail"}`, cx: 10 + (i % columns) * 22, cy: 10 + Math.floor(i / columns) * 22, r: pass ? 8 : 7 }));
    }
    return make(
      "div",
      { class: "dot-group" },
      make("p", { class: "dot-head" }, make("span", { class: "dot-name" }, ours ? name : `"${name}"`), make("span", { class: "dot-count" }, String(count.passed), make("small", {}, ` of ${total} passed`))),
      svg("svg", { class: "dot-svg", viewBox: `0 0 ${columns * 22} ${rows * 22}`, role: "img", "aria-label": `${name}: ${count.passed} of ${total} outage-day plans passed the physics check, ${count.failed} failed.` }, dots)
    );
  }

  function renderProblem(bench) {
    const main = bench.main || {};
    const names = armNames(main.run_summary);
    const llm = rowFor(main.outage_safety, "arm", LLM_SCHEDULE);
    const ours = rowFor(main.outage_safety, "arm", OURS);
    if (!llm || !ours || !names[LLM_SCHEDULE] || !names[OURS]) return;
    const llmCount = dotCount(llm["plan physics valid"], Number(llm.episodes));
    const ourCount = dotCount(ours["plan physics valid"], Number(ours.episodes));
    if (!llmCount || !ourCount) return;
    $("problem-dots").replaceChildren(
      make("figcaption", { class: "chart-heading" }, `Outage-day plans that passed the physics check, ${llm.episodes} tests each`),
      dotGroup(names[LLM_SCHEDULE], llmCount, Number(llm.episodes), false),
      dotGroup(names[OURS], ourCount, Number(ours.episodes), true),
      make("ul", { class: "dot-legend" }, make("li", {}, make("span", { class: "dot-key pass", "aria-hidden": "true" }), "passed"), make("li", {}, make("span", { class: "dot-key fail", "aria-hidden": "true" }), "failed")),
      make("p", { class: "dots-note" }, `"${names[LLM_SCHEDULE]}" means an AI wrote the hourly schedule itself. When the plans ran, a safety override still protected the critical load; the point is that the AI's own plan could not be trusted.`),
      make("p", { class: "source" }, sourceLine(BENCHMARK_FILES.main, `table outage_safety, column "plan physics valid" (${llm["plan physics valid"]} and ${ours["plan physics valid"]} of ${llm.episodes})`))
    );
  }

  function renderPipeline(bench) {
    const code = bench.code || {};
    if (!(code.max_repairs > 0)) return;
    $("repair-count").replaceChildren(`, at most ${code.max_repairs} repairs`);
    $("repair-source").replaceChildren(sourceLine(code.max_repairs_source));
  }

  /* --- evidence ------------------------------------------------------------------------------------ */

  function bigNumber(value, label, sources, ours = false) {
    return make("div", { class: "big-number" }, make("p", { class: `big-value${ours ? " ours" : ""}` }, value), make("p", { class: "big-label" }, label), ...sources.map((s) => make("p", { class: "source" }, s)));
  }

  function renderEvidence(bench) {
    const main = bench.main || {};
    const code = bench.code || {};
    const names = armNames(main.run_summary);
    const satisfied = main.goal_satisfaction || [];
    const oursRow = rowFor(satisfied, "arm", OURS);
    const share = oursRow && parseShare(oursRow.overall);
    if (!share || !names[OURS]) return;

    $("evidence-size").replaceChildren(
      `${share.den} test episodes, never used while building. Every approach gets the same goals, sites and days.`,
      make("span", { class: "source source-inline" }, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction, episodes"))
    );

    const numbers = [bigNumber(leadingNumber(oursRow.overall), `of goals met by ${names[OURS]} (${share.num} of ${share.den})`, [sourceLine(BENCHMARK_FILES.main, "goal_satisfaction")], true)];
    const delta = parseDelta((rowFor(main.goal_satisfaction_bootstrap, "comparison", `${OURS} - ${LLM_SCHEDULE}`) || {}).overall);
    if (delta && names[LLM_SCHEDULE]) {
      numbers.push(bigNumber(`${delta.value} pts`, `more goals met than "${names[LLM_SCHEDULE]}" (bootstrap interval ${delta.low} to ${delta.high} pts)`, [sourceLine(BENCHMARK_FILES.main, "goal_satisfaction_bootstrap")]));
    }
    const languages = (main.a4_spec_by_language || []).filter((row) => String(row.group).startsWith("multilingual:"));
    if (languages.length) {
      // "exact match": the share of goals whose rules matched the hand-written gold rules item for item
      // (spec precision and recall both 1): same rule kinds and hours, levels within the tolerances
      const lowest = languages.reduce((a, b) => (parseShare(b["exact match"]).pct < parseShare(a["exact match"]).pct ? b : a));
      const allSame = languages.every((row) => row["exact match"] === lowest["exact match"]);
      const sizes = [...new Set(languages.map((row) => row.episodes))];
      const languageList = languages.map((row) => VPDay.languageName(String(row.group).split(":")[1].trim()).split(" · ")[0]).join(", ");
      const size = sizes.length === 1 ? `${sizes[0]} goals each` : `${languages.reduce((sum, row) => sum + Number(row.episodes), 0)} goals`;
      const label = allSame
        ? `of goals turned into exactly the hand-written rules, in all ${languages.length} languages tested (${size})`
        : `of goals turned into exactly the hand-written rules, in the weakest of ${languages.length} languages tested (${size})`;
      numbers.push(bigNumber(leadingNumber(lowest["exact match"]), label, [sourceLine(BENCHMARK_FILES.main, 'a4_spec_by_language, "exact match"')]));
      // the fine print, once, under the card row: what "exactly" means and where the tolerances live
      const tolerance = code.spec_match_pct_points !== undefined ? ` Charge levels count as equal within ±${code.spec_match_pct_points} percentage point and power within ±${code.spec_match_kw} kW.` : "";
      $("evidence-footnote").replaceChildren(
        `"Exactly the hand-written rules": every rule item matches the hand-written rules for that goal (same rule kinds and hours), nothing missing and nothing extra. Languages: ${languageList}.${tolerance} `,
        make("span", { class: "source source-inline" }, [sourceLine(BENCHMARK_FILES.main, 'a4_spec_by_language, "exact match"'), tolerance ? sourceLine(code.spec_match_source) : null].filter(Boolean).join(" · "))
      );
    }
    $("evidence-numbers").replaceChildren(...numbers.slice(0, 3));

    const rows = (main.run_summary || []).map((summary) => {
      const row = rowFor(satisfied, "arm", summary.arm);
      const parsed = row && parseShare(row.overall);
      if (!parsed) return null;
      const reference = REFERENCE_ARMS.has(summary.arm);
      const fill = make("span", { class: "bar-fill" });
      fill.style.width = `${parsed.pct}%`;
      return make(
        "li",
        { class: `bar-row${summary.arm === OURS ? " ours" : ""}${reference ? " ref" : ""}` },
        make("span", { class: "bar-label" }, summary.name, reference ? make("span", { class: "bar-tag" }, "reference") : null),
        make("span", { class: "bar-track", "aria-hidden": "true" }, fill),
        make("span", { class: "bar-value" }, leadingNumber(row.overall), make("small", {}, `(${parsed.num}/${parsed.den})`))
      );
    }).filter(Boolean);
    const hardOurs = rowFor(bench.e2, "arm", OURS);
    const hardLlm = rowFor(bench.e2, "arm", LLM_SCHEDULE);
    const hardShare = hardOurs && parseShare(hardOurs.overall);
    $("evidence-chart").append(
      make("ol", { class: "bars" }, rows),
      make("p", { class: "bar-note" }, 'The reference rows are not real alternatives: they are given the hand-written rules ("gold spec") or the weather that actually happened ("oracle"), to show the ceiling.'),
      make("p", { class: "source" }, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction, overall; names from run_summary")),
      hardShare && hardLlm ? make("p", { class: "bar-note" }, `On ${hardShare.den} harder goals (relative times, negations, distractors): ${names[OURS]} ${leadingNumber(hardOurs.overall)}, "${names[LLM_SCHEDULE]}" ${leadingNumber(hardLlm.overall)}.`, make("span", { class: "source source-inline" }, sourceLine(BENCHMARK_FILES.e2, "goal_satisfaction"))) : null
    );
  }

  async function setupBenchmark() {
    try {
      const response = await fetch("api/benchmark", { cache: "no-cache" });
      if (!response.ok) throw new Error(String(response.status));
      const bench = await response.json();
      renderProblem(bench);
      renderPipeline(bench);
      renderEvidence(bench);
    } catch (_err) {
      const note = "The benchmark numbers load from the app server, which is not running here. They are in results/report/summary.json in the repository.";
      $("problem-dots").replaceChildren(make("p", { class: "hint" }, note));
      $("evidence-numbers").replaceChildren(make("p", { class: "hint" }, note));
    }
  }

  setupTheme();
  setupVideoLinks();
  setupMenu();
  setupReveal();
  setupHero();
  setupBenchmark();
})();

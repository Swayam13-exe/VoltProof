/* VoltProof landing page: the shell around Try it (app.js runs the tool itself, unchanged).
   - the hero plays three recorded runs (web/replays/*.json) in steps: goal, rules, battery chart, result
   - the Problem and Evidence numbers come from /api/benchmark (the committed summary.json files in results);
     every number shows its source file, and the strings are shown as the files print them
   - top menu with the section in view highlighted, light/dark toggle, copy link, reveal on scroll
   Everything is built with createElement and textContent: no HTML is ever inserted. */
"use strict";

(() => {
  /* --- pure: begin (no DOM; tested in tests/test_web_logic.py) --------------------------------- */

  const HERO_PAIRS = [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"], ["munich_home", "es"]];
  const HERO_KIND_RANK = { english: 0, multilingual: 1, hard: 2 };
  const HERO_EXAMPLE_MS = 9000;
  const HERO_STEPS = [["goal", 0], ["rules", 1800], ["chart", 3200], ["result", 5600]];
  const SPEC_RULES = [
    ["grid_outages", "OUTAGE_DEFICIT"],
    ["reserve_windows", "RESERVE"],
    ["soc_targets", "SOC_TARGET"],
    ["min_soc_pct_all_day", "ALL_DAY_MIN"],
    ["end_of_day_min_soc_pct", "END_SOC"],
    ["import_caps", "IMPORT_CAP"],
    ["no_export_windows", "NO_EXPORT"],
    ["discharge_only_windows", "DISCHARGE_WINDOW"],
    ["max_throughput_kwh", "THROUGHPUT"],
  ];
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

  /* Where the player is after `elapsedMs` of playing `count` examples in a loop. */
  function heroFrame(elapsedMs, count) {
    if (!count) return null;
    const total = HERO_EXAMPLE_MS * count;
    const t = ((elapsedMs % total) + total) % total;
    const index = Math.floor(t / HERO_EXAMPLE_MS);
    const local = t - index * HERO_EXAMPLE_MS;
    let stepIndex = 0;
    HERO_STEPS.forEach(([, at], i) => { if (local >= at) stepIndex = i; });
    return { index, stepIndex, step: HERO_STEPS[stepIndex][0], progress: local / HERO_EXAMPLE_MS };
  }

  function heroStart(index) {
    return index * HERO_EXAMPLE_MS;
  }

  /* The rule codes a spec contains, in a fixed order (goalWords in app.js turns each into words). */
  function specCodes(spec) {
    return SPEC_RULES.filter(([key]) => {
      const value = spec ? spec[key] : undefined;
      return Array.isArray(value) ? value.length > 0 : value !== null && value !== undefined;
    }).map(([, code]) => code);
  }

  /* Contiguous hour windows: [9, 10, 11, 14] -> [[9, 12], [14, 15]]. */
  function hourWindows(hours) {
    const out = [];
    for (const hour of [...new Set(hours)].sort((a, b) => a - b)) {
      const last = out[out.length - 1];
      if (last && last[1] === hour) last[1] = hour + 1;
      else out.push([hour, hour + 1]);
    }
    return out;
  }

  /* The small line under every number: which file (and table) it comes from. */
  function sourceLine(file, detail) {
    return `Source: ${file}${detail ? ` · ${detail}` : ""}`;
  }

  /* "98.8% (257/260)" -> "98.8%": the number exactly as the file prints it. */
  function leadingNumber(text) {
    const match = /^\s*([+-]?\d+(?:\.\d+)?%?)/.exec(String(text));
    return match ? match[1] : null;
  }

  /* "98.8% (257/260)" -> {pct: 98.8, num: 257, den: 260}; "100.0%" -> {pct: 100, num: null, den: null}. */
  function parseShare(text) {
    const match = /^\s*(\d+(?:\.\d+)?)%(?:\s*\((\d+)\/(\d+)\))?/.exec(String(text));
    if (!match) return null;
    return { pct: Number(match[1]), num: match[2] ? Number(match[2]) : null, den: match[3] ? Number(match[3]) : null };
  }

  /* "+8.5 pts [+5.0, +11.9]" -> {value: "+8.5", low: "+5.0", high: "+11.9"} (strings as printed). */
  function parseDelta(text) {
    const match = /^\s*([+-]?\d+(?:\.\d+)?)\s*pts\s*\[\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\s*\]/.exec(String(text));
    return match ? { value: match[1], low: match[2], high: match[3] } : null;
  }

  function rowFor(rows, key, value) {
    return (rows || []).find((row) => row[key] === value) || null;
  }

  /* Arm code -> the name the report gives it (run_summary), e.g. A2 -> "LLM writes schedule". */
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

  /* --- theme ----------------------------------------------------------------------------------- */

  function storedTheme() {
    try {
      const value = window.localStorage.getItem(THEME_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (_err) {
      return null;
    }
  }

  function applyTheme(theme) {
    const root = document.documentElement;
    if (theme) root.dataset.theme = theme;
    else delete root.dataset.theme;
    const dark = theme ? theme === "dark" : darkScheme.matches;
    $("theme-toggle").setAttribute("aria-pressed", String(dark));
    VPCharts.rerenderAll();
  }

  function setupTheme() {
    applyTheme(storedTheme());
    $("theme-toggle").addEventListener("click", () => {
      const next = $("theme-toggle").getAttribute("aria-pressed") === "true" ? "light" : "dark";
      try {
        window.localStorage.setItem(THEME_KEY, next);
      } catch (_err) {
        // private mode or blocked storage: the choice lasts for this page view only
      }
      applyTheme(next);
    });
    darkScheme.addEventListener("change", () => { if (!storedTheme()) applyTheme(null); });
  }

  /* --- copy link ------------------------------------------------------------------------------- */

  function setupCopyLink() {
    const note = $("copy-status");
    let timer = null;
    $("copy-link").addEventListener("click", async () => {
      const url = window.location.href;
      let copied = false;
      try {
        await navigator.clipboard.writeText(url);
        copied = true;
      } catch (_err) {
        const area = make("textarea", { class: "offscreen", readonly: "readonly", "aria-hidden": "true", tabindex: "-1" });
        area.value = url;
        document.body.append(area);
        area.select();
        try {
          copied = document.execCommand("copy");
        } catch (_e) {
          copied = false;
        }
        area.remove();
      }
      note.textContent = copied ? "Link copied" : "Copy the link from the address bar";
      clearTimeout(timer);
      timer = setTimeout(() => { note.textContent = ""; }, 2500);
    });
  }

  /* --- menu: highlight the section in view ------------------------------------------------------ */

  function setupMenu() {
    const links = [...document.querySelectorAll(".menu a")];
    const menu = document.querySelector(".menu");
    const byId = new Map(links.map((a) => [a.hash.slice(1), a]));
    const setCurrent = (id) => {
      for (const [target, link] of byId) {
        if (target === id) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      }
      const active = byId.get(id);
      if (active && menu.scrollWidth > menu.clientWidth) menu.scrollTo({ left: active.offsetLeft - 16, behavior: reduceMotion.matches ? "auto" : "smooth" });
    };
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting) setCurrent(entry.target.id);
    }, { rootMargin: "-45% 0px -50% 0px" });
    for (const id of [...byId.keys(), "top"]) {
      const section = $(id);
      if (section) observer.observe(section);
    }
  }

  /* --- reveal on scroll ------------------------------------------------------------------------- */

  function setupReveal() {
    if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
    document.documentElement.classList.add("motion");
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      }
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.05 });
    document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
  }

  /* --- hero ------------------------------------------------------------------------------------- */

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    for (const child of children) if (child) node.append(child instanceof Node ? child : String(child));
    return node;
  }

  function languageName(code) {
    try {
      const english = new Intl.DisplayNames(["en"], { type: "language" }).of(code);
      const own = new Intl.DisplayNames([code], { type: "language" }).of(code);
      return own && own.toLowerCase() !== english.toLowerCase() ? `${english} · ${own}` : english;
    } catch (_err) {
      return code;
    }
  }

  function siteLabel(site) {
    return `${site.name}, ${site.country}`;
  }

  /* State of charge over the day (the plan), with any outage shaded; drawn in when the step shows. */
  function heroChart(view) {
    const plan = view.plan;
    const W = 320, H = 128, L = 30, R = 6, T = 8, B = 22;
    const x = (hour) => L + (hour / 24) * (W - L - R);
    const y = (pct) => T + (1 - pct / 100) * (H - T - B);
    const values = [plan.initial_soc_pct, ...plan.soc_pct];
    const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    const outages = hourWindows(plan.goals.outage_hours);
    const nodes = [
      svg("line", { class: "grid-line", x1: L, x2: W - R, y1: y(100), y2: y(100) }),
      svg("line", { class: "grid-line", x1: L, x2: W - R, y1: y(50), y2: y(50) }),
      ...outages.map(([from, to]) => svg("rect", { class: "outage-band", x: x(from), y: T, width: x(to) - x(from), height: H - T - B })),
      ...outages.map(([from, to]) => svg("text", { class: "band-text", x: (x(from) + x(to)) / 2, y: y(0) - 4, "text-anchor": "middle" }, "grid off")), // at the bottom, away from a full battery
      svg("path", { class: "soc-area", d: `M${x(0)},${y(0)} L${points.join(" L")} L${x(24)},${y(0)} Z` }),
      svg("path", { class: "soc-line", d: `M${points.join(" L")}`, pathLength: 1 }),
      svg("line", { class: "axis-line", x1: L, x2: W - R, y1: y(0), y2: y(0) }),
      svg("text", { class: "axis-text", x: L - 4, y: y(100) + 3, "text-anchor": "end" }, "100%"),
      svg("text", { class: "axis-text", x: L - 4, y: y(0) + 3, "text-anchor": "end" }, "0%"),
      ...[0, 6, 12, 18, 24].map((h) => svg("text", { class: "axis-text", x: x(h), y: H - 6, "text-anchor": h === 0 ? "start" : h === 24 ? "end" : "middle" }, `${h}:00`)),
    ];
    const low = Math.min(...values), high = Math.max(...values);
    const outageWords = outages.map(([a, b]) => `${a}:00 to ${b}:00`).join(" and ");
    const label = `Planned battery charge over the day: ${values[0]}% at midnight, lowest ${low}%, highest ${high}%${outageWords ? `; grid off ${outageWords}` : ""}.`;
    return make(
      "figure",
      { class: "slide-chart" },
      svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label }, ...nodes),
      make("figcaption", {}, "Battery charge over the day, as planned")
    );
  }

  function heroSlide(view, chip) {
    const page = window.VPPage;
    const language = view.goal.language || chip.language;
    const rules = specCodes(view.spec).slice(0, 3).map((code) => page.goalWords(code, view.spec));
    rules.push("then: the lowest cost");
    const sd = view.same_day;
    const cost = page.costLine(sd, view.site.currency, view.spec, "Realized cost on the same-day weather, counting the energy left in the battery.");
    const met = sd.agent.goals_met;
    const slide = make(
      "article",
      { class: "hero-slide", "aria-hidden": "true" },
      make("p", { class: "slide-meta" }, make("span", { class: "slide-site" }, siteLabel(view.site)), make("span", { class: "slide-lang" }, languageName(language))),
      make("p", { class: "slide-goal", lang: language }, view.goal.text),
      make("ul", { class: "slide-rules", "aria-label": "Rules the AI wrote" }, rules.map((rule) => make("li", { class: "rule-chip" }, rule))),
      heroChart(view),
      make(
        "div",
        { class: "slide-result" },
        make("p", { class: "result-line" }, make("span", { class: met ? "ok" : "fail" }, met ? "✓ Goal met" : "✗ Goal missed"), " · ", make("span", { class: cost.tone }, cost.value), " than the rule-based battery"),
        make("p", { class: "result-explain" }, cost.explain),
        make("p", { class: "source" }, sourceLine(`web/replays/${chip.file}`))
      )
    );
    return { slide, language, label: `${siteLabel(view.site)}, ${languageName(language)}`, goal: view.goal.text };
  }

  function startPlayer(examples) {
    const card = $("hero-card");
    const slides = examples.map((e) => e.slide);
    $("hero-slides").replaceChildren(...slides);
    const fills = [];
    const buttons = examples.map((example, i) => {
      const fill = make("span", { class: "progress-fill" });
      fills.push(fill);
      const button = make("button", { type: "button", class: "progress-line", "aria-label": `Example ${i + 1} of ${examples.length}: ${example.label}` }, make("span", { class: "progress-track" }, fill));
      button.addEventListener("click", () => jump(i));
      return button;
    });
    $("hero-progress").replaceChildren(...buttons);

    const still = reduceMotion.matches;
    let elapsed = 0, last = null, current = null, hovered = false, focused = false;

    function announce(i) {
      const example = examples[i];
      $("hero-live").replaceChildren(`Example ${i + 1} of ${examples.length}: ${example.label}. Goal: `, make("span", { lang: example.language }, example.goal));
    }

    function render(frame, announceChange) {
      const changed = !current || current.index !== frame.index;
      slides.forEach((slide, i) => {
        const on = i === frame.index;
        slide.classList.toggle("active", on);
        slide.setAttribute("aria-hidden", String(!on));
        HERO_STEPS.forEach(([name], s) => slide.classList.toggle(`show-${name}`, on && s <= frame.stepIndex));
      });
      buttons.forEach((button, i) => {
        if (i === frame.index) button.setAttribute("aria-current", "true");
        else button.removeAttribute("aria-current");
        fills[i].style.transform = `scaleX(${still ? (i === frame.index ? 1 : 0) : i < frame.index ? 1 : i === frame.index ? frame.progress : 0})`;
      });
      if (changed && announceChange) announce(frame.index);
      current = frame;
    }

    function jump(i) {
      elapsed = heroStart(i);
      const frame = heroFrame(elapsed, slides.length);
      render(still ? { ...frame, stepIndex: HERO_STEPS.length - 1, progress: 1 } : frame, true);
    }

    function paused() {
      return hovered || focused || document.hidden;
    }

    card.addEventListener("pointerenter", () => { hovered = true; card.classList.add("paused"); });
    card.addEventListener("pointerleave", () => { hovered = false; card.classList.toggle("paused", paused()); });
    card.addEventListener("focusin", () => { focused = true; card.classList.add("paused"); });
    card.addEventListener("focusout", (event) => {
      if (card.contains(event.relatedTarget)) return;
      focused = false;
      card.classList.toggle("paused", paused());
    });

    if (still) {
      // reduced motion: no autoplay; the first example shows complete, the progress lines switch
      render({ ...heroFrame(0, slides.length), stepIndex: HERO_STEPS.length - 1, progress: 1 }, false);
      return;
    }
    render(heroFrame(0, slides.length), false);
    const tick = (now) => {
      if (last !== null && !paused()) elapsed += Math.min(now - last, 100);
      last = now;
      const frame = heroFrame(elapsed, slides.length);
      if (frame.index !== current.index || frame.stepIndex !== current.stepIndex || !paused()) render(frame, true);
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  }

  async function setupHero() {
    try {
      const index = await (await fetch("replays/index.json", { cache: "no-cache" })).json();
      const picks = heroPicks(index.chips).filter(Boolean);
      const views = await Promise.all(picks.map((chip) => fetch(`replays/${encodeURIComponent(chip.file)}`, { cache: "no-cache" }).then((r) => r.json())));
      const examples = views.map((view, i) => heroSlide(view, picks[i])).filter((e) => e.slide);
      if (examples.length) startPlayer(examples);
    } catch (_err) {
      $("hero-slides").replaceChildren(make("p", { class: "hint" }, "The recorded examples could not load. Try it below still works."));
    }
  }

  /* --- problem and evidence (from /api/benchmark) ------------------------------------------------ */

  function renderProblem(bench) {
    const main = bench.main || {};
    const names = armNames(main.run_summary);
    const llm = rowFor(main.outage_safety, "arm", "A2");
    const ours = rowFor(main.outage_safety, "arm", OURS);
    if (!llm || !ours || !names.A2 || !names[OURS]) return;
    $("problem-number").replaceChildren(
      make("p", { class: "figure-number" }, leadingNumber(llm["plan physics valid"])),
      make("p", { class: "figure-label" }, `of outage-day plans passed the physics check when an AI wrote the hourly schedule directly ("${names.A2}"), on ${llm.episodes} outage-day tests.`),
      make("p", { class: "figure-compare" }, make("span", { class: "compare-value" }, leadingNumber(ours["plan physics valid"])), ` for ${names[OURS]} on the same ${ours.episodes} tests.`),
      make("p", { class: "figure-note" }, "When the plans ran, a safety override still protected the critical load; the point is that the AI's own plan could not be trusted."),
      make("p", { class: "source" }, sourceLine(BENCHMARK_FILES.main, 'table outage_safety, column "plan physics valid"'))
    );
  }

  function bigNumber(value, label, source, ours = false) {
    return make("div", { class: "big-number" }, make("p", { class: `big-value${ours ? " ours" : ""}` }, value), make("p", { class: "big-label" }, label), make("p", { class: "source" }, source));
  }

  function renderEvidence(bench) {
    const main = bench.main || {};
    const names = armNames(main.run_summary);
    const satisfied = main.goal_satisfaction || [];
    const oursRow = rowFor(satisfied, "arm", OURS);
    const share = oursRow && parseShare(oursRow.overall);
    if (!share || !names[OURS]) return;

    $("evidence-size").replaceChildren(
      `${share.den} test episodes, never used while building. Every approach gets the same goals, sites and days.`,
      make("span", { class: "source source-inline" }, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction, episodes"))
    );

    const numbers = [bigNumber(leadingNumber(oursRow.overall), `of goals met by ${names[OURS]} (${share.num} of ${share.den})`, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction"), true)];
    const delta = parseDelta((rowFor(main.goal_satisfaction_bootstrap, "comparison", `${OURS} - A2`) || {}).overall);
    if (delta && names.A2) {
      numbers.push(bigNumber(`${delta.value} pts`, `more goals met than "${names.A2}" (95% interval ${delta.low} to ${delta.high} pts)`, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction_bootstrap")));
    }
    const hardOurs = rowFor(bench.e2, "arm", OURS);
    const hardLlm = rowFor(bench.e2, "arm", "A2");
    const hardShare = hardOurs && parseShare(hardOurs.overall);
    if (hardShare && hardLlm) {
      numbers.push(bigNumber(leadingNumber(hardOurs.overall), `on ${hardShare.den} harder goals (relative times, negations, distractors); "${names.A2}": ${leadingNumber(hardLlm.overall)}`, sourceLine(BENCHMARK_FILES.e2, "goal_satisfaction")));
    }
    const languages = (main.a4_spec_by_language || []).filter((row) => String(row.group).startsWith("multilingual:"));
    if (languages.length) {
      const lowest = languages.reduce((a, b) => (parseShare(b["exact match"]).pct < parseShare(a["exact match"]).pct ? b : a));
      const allSame = languages.every((row) => row["exact match"] === lowest["exact match"]);
      const sizes = [...new Set(languages.map((row) => row.episodes))];
      const languageList = languages.map((row) => languageName(String(row.group).split(":")[1].trim()).split(" · ")[0]).join(", ");
      const size = sizes.length === 1 ? `${sizes[0]} goals each` : `${languages.reduce((sum, row) => sum + Number(row.episodes), 0)} goals`;
      const label = allSame
        ? `of goals became exactly the right rules in every language tested: ${languageList} (${size})`
        : `of goals became exactly the right rules in the weakest language tested (${languageList}; ${size})`;
      numbers.push(bigNumber(leadingNumber(lowest["exact match"]), label, sourceLine(BENCHMARK_FILES.main, "a4_spec_by_language, multilingual rows")));
    }
    $("evidence-numbers").replaceChildren(...numbers);

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
    $("evidence-chart").append(
      make("ol", { class: "bars" }, rows),
      make("p", { class: "bar-note" }, 'The two reference rows are not real alternatives: they are given the hand-written rules ("gold spec") or the weather that actually happened ("oracle"), to show the ceiling.'),
      make("p", { class: "source" }, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction, overall; names from run_summary"))
    );
  }

  async function setupBenchmark() {
    try {
      const response = await fetch("api/benchmark", { cache: "no-cache" });
      if (!response.ok) throw new Error(String(response.status));
      const bench = await response.json();
      renderProblem(bench);
      renderEvidence(bench);
    } catch (_err) {
      const note = "The benchmark numbers load from the app server, which is not running here. They are in results/report/summary.json in the repository.";
      $("problem-number").replaceChildren(make("p", { class: "hint" }, note));
      $("evidence-numbers").replaceChildren(make("p", { class: "hint" }, note));
    }
  }

  setupTheme();
  setupCopyLink();
  setupMenu();
  setupReveal();
  setupHero();
  setupBenchmark();
})();

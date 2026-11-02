/* VoltProof landing page: the shell around Try it (app.js runs the tool itself, unchanged).
   - the hero plays three recorded days (web/replays): the goal, energy flowing between sun, grid,
     battery and site hour by hour, the rules drawn on a 24-hour timeline, then the rules checked
   - the Problem, How-it-works and Evidence numbers come from /api/benchmark (the committed summary
     files in results, plus one constant from the frozen agent code); every number shows its source
   - top menu (nothing highlighted while the hero is in view), theme toggle, copy link, reveal on scroll
   Everything is built with createElement and textContent: no HTML is ever inserted. */
"use strict";

(() => {
  /* --- pure: begin (no DOM; tested in tests/test_web_logic.py) --------------------------------- */

  const HERO_PAIRS = [["nairobi_clinic", "en"], ["ahmedabad_cold_store", "hi"], ["munich_home", "es"]];
  const HERO_KIND_RANK = { english: 0, multilingual: 1, hard: 2 };
  const HOUR_MS = 350; // one hour of the day on screen
  const DAY_MS = 24 * HOUR_MS; // the day plays, then the checklist holds until the next site
  const SITE_MS = 10000;
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
  const DRAWN_RULES = new Set(["OUTAGE_DEFICIT", "RESERVE", "SOC_TARGET", "ALL_DAY_MIN", "END_SOC", "IMPORT_CAP"]);
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

  /* Where autoplay is after `elapsedMs`: which site, which hour (0-23, or 24 = end of day). */
  function heroClock(elapsedMs, count) {
    if (!count) return null;
    const total = SITE_MS * count;
    const t = ((elapsedMs % total) + total) % total;
    const index = Math.floor(t / SITE_MS);
    const local = t - index * SITE_MS;
    return { index, hour: local < DAY_MS ? Math.floor(local / HOUR_MS) : 24, progress: local / SITE_MS };
  }

  /* The autoplay time that shows `hour` of site `index` (used when the user picks a site or scrubs). */
  function heroTime(index, hour) {
    return index * SITE_MS + (hour >= 24 ? DAY_MS : hour * HOUR_MS);
  }

  /* One hour of the executed VoltProof run on the same-day weather, exactly as the replay stores it.
     Curtailed solar follows from the executor's own balance (surplus beyond export is curtailed). */
  function hourFlows(view, hour) {
    const sd = view.same_day;
    const run = sd.agent;
    const plan = view.plan;
    if (hour >= 24) {
      return { hour: 24, end: true, socStart: run.soc_pct[23], socEnd: run.soc_pct[23], outage: false };
    }
    const pv = sd.pv_actual_kw[hour];
    const load = sd.load_actual_kw[hour];
    const battery = run.battery_kw[hour];
    const gridImport = run.grid_import_kw[hour];
    const gridExport = run.grid_export_kw[hour];
    const residual = load + battery - pv;
    const curtail = residual < 0 ? Math.max(0, -residual - gridExport) : 0;
    const pvUsed = pv - curtail;
    const charge = Math.max(battery, 0);
    const discharge = Math.max(-battery, 0);
    const critical = plan.critical_load_kw[hour];
    return {
      hour,
      end: false,
      pv,
      pvUsed,
      curtail,
      load,
      critical,
      other: Math.max(0, load - critical),
      charge,
      discharge,
      gridImport,
      gridExport,
      socStart: hour === 0 ? plan.initial_soc_pct : run.soc_pct[hour - 1],
      socEnd: run.soc_pct[hour],
      outage: plan.goals.outage_hours.includes(hour),
      balance: pvUsed + gridImport + discharge - (load + charge + gridExport),
      fullLoadServed: run.unserved_critical_kwh === 0 && run.non_critical_shed_kwh === 0,
    };
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

  /* What the timeline draws: outage bands, the required minimum charge per hour (reserves, targets,
     all-day and end-of-day minimums, as the plan resolved them), the import cap per hour, and the
     rule kinds it does not draw (listed in words instead). */
  function timelineRules(view) {
    const goals = view.plan.goals;
    const floor = goals.required_soc_pct.map((value, hour) => (value === null || value === undefined ? null : { hour, pct: value }));
    const cap = goals.import_cap_kw.map((value, hour) => (value === null || value === undefined ? null : { hour, kw: value }));
    return {
      outages: hourWindows(goals.outage_hours),
      floor: floor.filter(Boolean),
      importCap: cap.filter(Boolean),
      listed: specCodes(view.spec).filter((code) => !DRAWN_RULES.has(code)),
    };
  }

  /* Each rule of the spec with its result from the replay (met unless its code was missed). */
  function ruleResults(spec, missedCodes) {
    const missed = new Set(missedCodes || []);
    return specCodes(spec).map((code) => ({ code, met: !missed.has(code) }));
  }

  /* The hour worth showing first: mid-outage, else the middle of the first drawn rule, else noon. */
  function focusHour(view) {
    const rules = timelineRules(view);
    const middle = ([a, b]) => Math.floor((a + b - 1) / 2);
    if (rules.outages.length) return middle(rules.outages[0]);
    const ruled = hourWindows([...rules.floor.map((f) => f.hour), ...rules.importCap.map((c) => c.hour)]);
    return ruled.length ? middle(ruled[0]) : 12;
  }

  /* Dots for a printed rate over n episodes: the whole count that reproduces the printed rate exactly
     (e.g. "17.8%" of 45 -> 8, since 8/45 prints as 17.8%), or null when no whole count does. */
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

  function hourText(hour) {
    return `${String(hour).padStart(2, "0")}:00`;
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

  function kw(value) {
    return `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 })} kW`;
  }

  function pct(value) {
    return `${Math.round(Number(value))}%`;
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
      if (window.scrollY + bar < hero.offsetTop + hero.offsetHeight - 80) active = null;
      else {
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
  }

  /* --- reveal on scroll: progressive, with a safety timer ---------------------------------------- */

  function setupReveal() {
    if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
    const root = document.documentElement;
    root.classList.add("reveal-armed");
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      }
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.05 });
    document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
    window.setTimeout(() => root.classList.remove("reveal-armed"), 1500); // nothing stays hidden
  }

  /* --- hero: the day ------------------------------------------------------------------------------ */

  const SITE_ICONS = {
    health_clinic: ["M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"],
    commercial_cold_store: ["M12 2v20", "M3.5 7l17 10", "M3.5 17l17-10", "M9 3.5l3 2.5 3-2.5", "M9 20.5l3-2.5 3 2.5"],
    household: ["M3 11.5 12 4l9 7.5", "M5.5 10v10h13V10", "M10 20v-5h4v5"],
  };

  function siteIcon(type, className) {
    return svg("svg", { class: className, viewBox: "0 0 24 24", "aria-hidden": "true", focusable: "false" },
      (SITE_ICONS[type] || SITE_ICONS.household).map((d) => svg("path", { d, fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" })));
  }

  /* The energy-flow diagram: four nodes around a junction; each line's width follows its kW. Two
     layouts from one drawing: "wide" (nodes around the junction) and "compact" for phones (nodes in
     the four corners, readable at 390 px). Node contents are drawn in local coordinates. */
  const FLOW_LAYOUTS = {
    wide: {
      W: 640, H: 276, hub: { x: 320, y: 146 }, labelShift: { solar: [34, 4], battery: [34, 4], grid: [0, -10], site: [0, -10] },
      boxes: { solar: [240, 14, 160, 58], grid: [8, 116, 160, 60], battery: [240, 212, 160, 58], site: [452, 104, 180, 78] },
      ends: { solar: [320, 72], grid: [168, 146], battery: [320, 212], site: [452, 146] },
      hour: { x: 634, y: 30, anchor: "end", sub: 48 },
    },
    compact: {
      W: 360, H: 330, hub: { x: 180, y: 154 }, labelShift: { solar: [-30, 0], battery: [-30, 4], grid: [30, 0], site: [30, 4] },
      boxes: { solar: [4, 8, 164, 60], grid: [192, 8, 164, 60], battery: [4, 240, 164, 60], site: [192, 232, 164, 78] },
      ends: { solar: [110, 68], grid: [250, 68], battery: [110, 240], site: [250, 232] },
      hour: { x: 180, y: 326, anchor: "middle", sub: null },
    },
  };

  function flowDiagram(layoutName) {
    const layout = FLOW_LAYOUTS[layoutName];
    const { W, H, hub } = layout;
    const marker = (id, cls) => svg("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 11, markerHeight: 11, markerUnits: "userSpaceOnUse", orient: "auto" }, svg("path", { d: "M0 0 10 5 0 10z", class: cls }));
    const defs = svg("defs", {}, marker("arrow-solar", "arrow solar"), marker("arrow-grid", "arrow grid"), marker("arrow-battery", "arrow battery"), marker("arrow-site", "arrow site"));
    const lines = {};
    const labels = {};
    for (const name of ["solar", "grid", "battery", "site"]) {
      lines[name] = svg("path", { class: `flow-line ${name}` });
      labels[name] = svg("text", { class: "flow-kw", "text-anchor": "middle" });
    }
    const node = (name, ...content) => {
      const [x, y, w, h] = layout.boxes[name];
      return svg("g", { class: "flow-node", transform: `translate(${x},${y})` }, svg("rect", { class: "flow-node-bg", x: 0, y: 0, width: w, height: h, rx: 12 }), ...content);
    };
    const text = (cls, x, y, value) => svg("text", { class: cls, x, y }, value || "");

    const sun = svg("g", { "aria-hidden": "true" }, svg("circle", { class: "sun-core", cx: 25, cy: 30, r: 9 }),
      [0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
        const r = (a * Math.PI) / 180;
        return svg("line", { class: "sun-ray", x1: 25 + 13 * Math.cos(r), y1: 30 + 13 * Math.sin(r), x2: 25 + 17 * Math.cos(r), y2: 30 + 17 * Math.sin(r) });
      }));
    const solarValue = text("flow-value", 52, 44);
    const solarNode = node("solar", sun, text("flow-label", 52, 24, "Solar"), solarValue);

    const gridValue = text("flow-value", 52, 44);
    const gridOff = text("flow-off-label", 52, 44);
    const gridNode = node("grid", svg("path", { class: "grid-icon", d: "M24 16 l-8 32 M24 16 l8 32 M16 26 h16 M13 38 h22 M24 10 v6" }), text("flow-label", 52, 24, "Grid"), gridValue, gridOff);

    const level = svg("rect", { class: "battery-level", x: 17, y: 48, width: 16, height: 0, rx: 2 });
    const batteryValue = text("flow-value", 52, 44);
    const batteryNode = node("battery",
      svg("rect", { class: "battery-shell", x: 14, y: 14, width: 22, height: 38, rx: 4 }),
      svg("rect", { class: "battery-shell", x: 20, y: 10, width: 10, height: 4, rx: 1 }),
      level, text("flow-label", 52, 24, "Battery"), batteryValue);

    const siteValue = text("flow-value", 52, 44);
    const siteCritical = text("flow-sub", 24, 66);
    const siteNode = node("site",
      svg("path", { class: "site-icon-path", d: "M12 32 l12-10 12 10 v12 h-24z" }),
      svg("circle", { class: "critical-dot", cx: 15, cy: 62, r: 3.5 }),
      text("flow-label", 52, 24, "Site"), siteValue, siteCritical);

    const hourBig = svg("text", { class: "flow-hour", x: layout.hour.x, y: layout.hour.y, "text-anchor": layout.hour.anchor });
    const hourSub = layout.hour.sub === null ? null : svg("text", { class: "flow-hour-sub", x: layout.hour.x, y: layout.hour.sub, "text-anchor": layout.hour.anchor });
    const root = svg("svg", { class: `flow-svg ${layoutName}`, viewBox: `0 0 ${W} ${H}`, role: "img" },
      defs, ...Object.values(lines), svg("circle", { class: "flow-hub", cx: hub.x, cy: hub.y, r: 7 }),
      solarNode, gridNode, batteryNode, siteNode, ...Object.values(labels), hourBig, hourSub);

    function setLine(name, kwValue, towardHub, scale) {
      const [ex, ey] = layout.ends[name];
      const end = { x: ex, y: ey };
      const line = lines[name];
      const idle = !(kwValue > 0.005);
      const [a, b] = towardHub ? [end, hub] : [hub, end];
      const length = Math.hypot(b.x - a.x, b.y - a.y);
      const stop = idle ? 0 : 12; // the arrowhead ends just short of the node or the junction
      const bx = b.x - ((b.x - a.x) / length) * stop;
      const by = b.y - ((b.y - a.y) / length) * stop;
      line.setAttribute("d", `M${a.x},${a.y} L${bx.toFixed(1)},${by.toFixed(1)}`);
      line.setAttribute("stroke-width", idle ? 2 : (2.5 + 11 * Math.min(1, kwValue / scale)).toFixed(2));
      line.classList.toggle("idle", idle);
      line.classList.toggle("moving", !idle);
      if (idle) line.removeAttribute("marker-end");
      else line.setAttribute("marker-end", `url(#arrow-${name})`);
      const [dx, dy] = layout.labelShift[name];
      const label = labels[name];
      label.setAttribute("x", ((end.x + hub.x) / 2 + dx).toFixed(1));
      label.setAttribute("y", ((end.y + hub.y) / 2 + dy).toFixed(1));
      label.textContent = idle ? "" : kw(kwValue);
    }

    function update(view, flows, scale) {
      const criticalName = String(view.site.critical_load_name).split(",")[0];
      if (flows.end) {
        for (const name of Object.keys(lines)) setLine(name, 0, true, scale);
        solarValue.textContent = "";
        gridValue.textContent = "";
        gridOff.textContent = "";
        gridNode.classList.remove("off");
        batteryValue.textContent = `${pct(flows.socEnd)} at midnight`;
        siteValue.textContent = "";
        siteCritical.textContent = criticalName;
        hourBig.textContent = hourSub ? "24:00" : "24:00 · end of day";
        if (hourSub) hourSub.textContent = "end of day";
      } else {
        setLine("solar", flows.pvUsed, true, scale);
        setLine("grid", flows.gridImport > 0 ? flows.gridImport : flows.gridExport, flows.gridImport > 0, scale);
        setLine("battery", flows.discharge > 0 ? flows.discharge : flows.charge, flows.discharge > 0, scale);
        setLine("site", flows.load, false, scale);
        solarValue.textContent = kw(flows.pvUsed);
        gridNode.classList.toggle("off", flows.outage);
        gridOff.textContent = flows.outage ? "grid off" : "";
        gridValue.textContent = flows.outage ? "" : flows.gridImport > 0 ? `buying ${kw(flows.gridImport)}` : flows.gridExport > 0 ? `selling ${kw(flows.gridExport)}` : "idle";
        batteryValue.textContent = `${pct(flows.socStart)}${flows.charge > 0 ? " · charging" : flows.discharge > 0 ? " · supplying" : ""}`;
        siteValue.textContent = `uses ${kw(flows.load)}`;
        siteCritical.textContent = `${criticalName} ${kw(flows.critical)}`;
        hourBig.textContent = hourSub ? hourText(flows.hour) : `${hourText(flows.hour)}${flows.outage ? " · grid off" : ""}`;
        if (hourSub) hourSub.textContent = flows.outage ? "grid off this hour" : `${hourText(flows.hour)} to ${hourText(flows.hour + 1)}`;
      }
      const height = 32 * Math.max(0, Math.min(100, flows.socStart)) / 100;
      level.setAttribute("height", height.toFixed(1));
      level.setAttribute("y", (49 - height).toFixed(1));
      root.setAttribute("aria-label", flows.end
        ? `End of day: the battery ends at ${pct(flows.socEnd)}.`
        : `${hourText(flows.hour)}: solar ${kw(flows.pvUsed)}, ${flows.outage ? "grid off" : `grid ${flows.gridImport > 0 ? `supplies ${kw(flows.gridImport)}` : flows.gridExport > 0 ? `takes ${kw(flows.gridExport)}` : "idle"}`}, battery at ${pct(flows.socStart)}${flows.charge > 0 ? ` charging ${kw(flows.charge)}` : flows.discharge > 0 ? ` supplying ${kw(flows.discharge)}` : ""}, site uses ${kw(flows.load)}.`);
    }
    return { root, update };
  }

  /* The 24-hour timeline: charge curve, rules drawn on the day, grid strip, and the playhead. */
  function timeline(view, id, W = 640) {
    const L = 40, R = 10, T = 8, SOC_B = 118, STRIP_T = 134, STRIP_ZERO = 168, STRIP_B = 196, H = 230; // axis labels on their own row, below the outage label
    const x = (hour) => L + (hour / 24) * (W - L - R);
    const ySoc = (value) => T + (1 - value / 100) * (SOC_B - T);
    const sd = view.same_day;
    const run = sd.agent;
    const rules = timelineRules(view);
    const capMax = rules.importCap.reduce((m, c) => Math.max(m, c.kw), 0);
    const stripMax = Math.max(0.1, ...run.grid_import_kw, ...run.grid_export_kw, capMax) * 1.1;
    const yUp = (v) => STRIP_ZERO - (v / stripMax) * (STRIP_ZERO - STRIP_T);
    const yDown = (v) => STRIP_ZERO + (v / stripMax) * (STRIP_B - STRIP_ZERO);
    const hatchId = `hatch-${id}`;
    const nodes = [
      svg("defs", {}, svg("pattern", { id: hatchId, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, svg("line", { class: "hatch-line", x1: 0, y1: 0, x2: 0, y2: 6 }))),
      svg("line", { class: "tl-grid-line", x1: L, x2: W - R, y1: ySoc(100), y2: ySoc(100) }),
      svg("line", { class: "tl-grid-line", x1: L, x2: W - R, y1: ySoc(50), y2: ySoc(50) }),
    ];
    for (const [from, to] of rules.outages) {
      nodes.push(svg("rect", { class: "tl-outage", x: x(from), y: T, width: x(to) - x(from), height: STRIP_B - T, fill: `url(#${hatchId})` }));
      nodes.push(svg("text", { class: "tl-outage-label", x: (x(from) + x(to)) / 2, y: STRIP_B + 13, "text-anchor": "middle" }, "grid off"));
    }
    let lastFloor = null;
    for (const { hour, pct: floor } of rules.floor) {
      nodes.push(svg("rect", { class: "tl-floor-zone", x: x(hour), y: ySoc(floor), width: x(hour + 1) - x(hour), height: ySoc(0) - ySoc(floor) }));
      nodes.push(svg("line", { class: "tl-floor-line", x1: x(hour), x2: x(hour + 1), y1: ySoc(floor), y2: ySoc(floor) }));
      if (lastFloor === null || lastFloor.pct !== floor || lastFloor.hour !== hour - 1) nodes.push(svg("text", { class: "tl-rule-label", x: x(hour) + 4, y: ySoc(floor) + 13 }, `at least ${floor}%`));
      lastFloor = { hour, pct: floor };
    }
    const points = [view.plan.initial_soc_pct, ...run.soc_pct].map((v, i) => `${x(i).toFixed(1)},${ySoc(v).toFixed(1)}`);
    nodes.push(svg("path", { class: "tl-soc-area", d: `M${x(0)},${ySoc(0)} L${points.join(" L")} L${x(24)},${ySoc(0)} Z` }));
    nodes.push(svg("path", { class: "tl-soc-line", d: `M${points.join(" L")}` }));
    run.grid_import_kw.forEach((v, hour) => {
      if (v > 0.005) nodes.push(svg("rect", { class: "tl-import", x: x(hour) + 1, y: yUp(v), width: x(1) - x(0) - 2, height: STRIP_ZERO - yUp(v) }));
    });
    run.grid_export_kw.forEach((v, hour) => {
      if (v > 0.005) nodes.push(svg("rect", { class: "tl-export", x: x(hour) + 1, y: STRIP_ZERO, width: x(1) - x(0) - 2, height: yDown(v) - STRIP_ZERO }));
    });
    let lastCap = null;
    for (const { hour, kw: cap } of rules.importCap) {
      nodes.push(svg("line", { class: "tl-cap-line", x1: x(hour), x2: x(hour + 1), y1: yUp(cap), y2: yUp(cap) }));
      if (lastCap === null || lastCap.kw !== cap || lastCap.hour !== hour - 1) nodes.push(svg("text", { class: "tl-rule-label", x: x(hour) + 3, y: yUp(cap) - 4 }, `import cap ${cap} kW`));
      lastCap = { hour, kw: cap };
    }
    nodes.push(
      svg("line", { class: "tl-axis", x1: L, x2: W - R, y1: ySoc(0), y2: ySoc(0) }),
      svg("line", { class: "tl-axis", x1: L, x2: W - R, y1: STRIP_ZERO, y2: STRIP_ZERO }),
      svg("text", { class: "tl-text", x: L - 6, y: ySoc(100) + 3, "text-anchor": "end" }, "100%"),
      svg("text", { class: "tl-text", x: L - 6, y: ySoc(0) + 3, "text-anchor": "end" }, "0%"),
      svg("text", { class: "tl-strip-label", x: L - 6, y: STRIP_T + 8, "text-anchor": "end" }, "grid"),
      svg("text", { class: "tl-strip-label", x: L + 2, y: T + 10 }, "battery charge"),
      ...[0, 6, 12, 18, 24].map((h) => svg("text", { class: "tl-text", x: x(h), y: H - 2, "text-anchor": h === 0 ? "start" : h === 24 ? "end" : "middle" }, `${h}:00`)),
    );
    const band = svg("rect", { class: "tl-hour-band", y: T, height: STRIP_B - T, width: x(1) - x(0) });
    const head = svg("line", { class: "tl-playhead", y1: T - 4, y2: STRIP_B + 2 });
    nodes.splice(3, 0, band); // behind the data
    nodes.push(head);
    const outageWords = rules.outages.map(([a, b]) => `${hourText(a)} to ${hourText(b)}`).join(" and ");
    const label = `Timeline of the day: battery charge from ${pct(view.plan.initial_soc_pct)} to ${pct(run.soc_pct[23])}${outageWords ? `; grid off ${outageWords}` : ""}${rules.floor.length ? "; shaded: the minimum charge the rules require" : ""}${rules.importCap.length ? "; dashed line on the grid strip: the import cap" : ""}.`;
    const root = svg("svg", { class: "timeline-svg", viewBox: `0 0 ${W} ${H + 14}`, role: "img", "aria-label": label }, nodes);
    function setHour(hour) {
      const at = Math.min(hour, 24);
      head.setAttribute("x1", x(at));
      head.setAttribute("x2", x(at));
      band.setAttribute("x", x(Math.min(at, 23)));
      band.setAttribute("visibility", hour >= 24 ? "hidden" : "visible");
    }
    return { root, setHour, listed: rules.listed };
  }

  function goalBlock(example) {
    const view = example.view;
    const english = view.goal.english;
    return make(
      "div",
      { class: "goal-variant", "aria-hidden": "true" },
      make("blockquote", { class: "goal-quote" }, make("p", { class: "goal-text", lang: example.language }, view.goal.text)),
      english ? make("p", { class: "goal-english" }, make("span", { class: "goal-english-label" }, "In English: "), english) : null,
      make("p", { class: "goal-site" }, `${view.site.name}, ${view.site.country} · ${view.day.date} · ${languageName(example.language)}`)
    );
  }

  function resultBlock(example) {
    const page = window.VPPage;
    const view = example.view;
    const sd = view.same_day;
    const items = ruleResults(view.spec, sd.agent.goals_missed).map(({ code, met }) => {
      const mark = make("span", { class: "check-mark", "aria-hidden": "true" }, "○");
      const state = make("span", { class: "check-state" }, "checked at the end of the day");
      const item = make("li", { class: "check-item pending", "data-met": String(met) }, mark, make("span", {}, page.goalWords(code, view.spec), state));
      return { item, mark, state, met };
    });
    const listed = timelineRules(view).listed;
    const cost = page.costLine(sd, view.site.currency, view.spec, "Realized cost on the same-day weather, counting the energy left in the battery.");
    const block = make(
      "div",
      { class: "result-variant", "aria-hidden": "true" },
      make("p", { class: "result-title" }, "The rules, checked on the day"),
      make("ul", { class: "checklist" }, items.map((i) => i.item)),
      listed.length ? make("p", { class: "hint" }, "Not drawn on the timeline: ", listed.map((code) => page.goalWords(code, view.spec)).join("; "), ".") : null,
      make("p", { class: "cost-line" }, make("strong", { class: cost.tone }, cost.value), cost.same ? ". " : " than the rule-based battery. ", cost.explain),
      make("p", { class: "source" }, sourceLine(`web/replays/${example.chip.file}`))
    );
    function reveal(show) {
      for (const i of items) {
        i.item.classList.toggle("pending", !show);
        i.item.classList.toggle("met", show && i.met);
        i.item.classList.toggle("missed", show && !i.met);
        i.mark.textContent = show ? (i.met ? "✓" : "✗") : "○";
        i.state.textContent = show ? (i.met ? "met" : "missed") : "checked at the end of the day";
      }
    }
    return { block, reveal };
  }

  function dayScale(view) {
    const sd = view.same_day;
    const run = sd.agent;
    return Math.max(0.1, ...sd.pv_actual_kw, ...sd.load_actual_kw, ...run.grid_import_kw, ...run.grid_export_kw, ...run.battery_kw.map(Math.abs));
  }

  function startDay(examples) {
    const panel = $("day-panel");
    const still = reduceMotion.matches;
    const phone = window.matchMedia("(max-width: 560px)"); // compact drawings, readable at 390 px
    let flow = flowDiagram(phone.matches ? "compact" : "wide");
    const flowSlot = make("div", { class: "flow-slot" }, flow.root);
    const goalStack = make("div", { class: "stack" }, examples.map((e) => e.goal));
    const resultStack = make("div", { class: "stack" }, examples.map((e) => e.result.block));
    const timelineSlot = make("div", { class: "timeline-slot" });
    const range = make("input", { type: "range", id: "day-hour", min: "0", max: "24", step: "1", value: "0", "aria-describedby": "day-readout" });
    const readout = make("p", { id: "day-readout", class: "scrub-readout" });
    // two balanced columns: the goal over the live flows, the timeline over the checked rules
    $("day-body").replaceChildren(
      make("div", { class: "day-col" }, make("div", { class: "day-goal" }, goalStack), make("figure", { class: "day-flow" }, flowSlot, make("figcaption", { class: "flow-caption" }, "Energy this hour: line width follows kW, arrows show the direction. The red dot marks the critical load."))),
      make("div", { class: "day-col" }, make("div", { class: "day-timeline" }, timelineSlot, make("div", { class: "scrub" }, make("label", { for: "day-hour" }, "Hour of the day"), range, readout)), make("div", { class: "day-result" }, resultStack))
    );

    const fills = [];
    const buttons = examples.map((example, i) => {
      const fill = make("span", { class: "site-fill", "aria-hidden": "true" });
      fills.push(fill);
      const button = make("button", { type: "button", class: "site-btn", "aria-pressed": "false", "aria-label": `${example.view.site.name}, goal in ${languageName(example.language)}` },
        fill, siteIcon(example.view.site.site_type, "site-icon"), make("span", { class: "site-name" }, example.view.site.name), make("span", { class: "site-lang", lang: "en" }, example.language.toUpperCase()));
      button.addEventListener("click", () => choose(i));
      return button;
    });
    $("site-switch").replaceChildren(...buttons);

    let index = -1, hour = -1, elapsed = 0, last = null, hovered = false, focused = false, line = null;
    const scales = examples.map((e) => dayScale(e.view));

    function show(nextIndex, nextHour, announce) {
      const example = examples[nextIndex];
      if (nextIndex !== index) {
        examples.forEach((e, i) => {
          const on = i === nextIndex;
          e.goal.classList.toggle("active", on);
          e.goal.setAttribute("aria-hidden", String(!on));
          e.result.block.classList.toggle("active", on);
          e.result.block.setAttribute("aria-hidden", String(!on));
          buttons[i].setAttribute("aria-pressed", String(on));
          if (!on) fills[i].style.transform = "scaleX(0)";
        });
        line = timeline(example.view, nextIndex, phone.matches ? 360 : 640);
        timelineSlot.replaceChildren(line.root);
        if (announce) $("day-live").replaceChildren(`${example.view.site.name}. Goal: `, make("span", { lang: example.language }, example.view.goal.text));
      }
      if (nextIndex !== index || nextHour !== hour) {
        const flows = hourFlows(example.view, nextHour);
        flow.update(example.view, flows, scales[nextIndex]);
        line.setHour(nextHour);
        example.result.reveal(still || nextHour >= 24);
        range.value = String(nextHour);
        const words = nextHour >= 24 ? "24:00, end of day" : `${hourText(nextHour)}${flows.outage ? ", grid off" : ""}, battery ${pct(flows.socStart)}`;
        range.setAttribute("aria-valuetext", words);
        readout.textContent = nextHour >= 24 ? "24:00 · end of day: the rules are checked" : `${hourText(nextHour)} · battery ${pct(flows.socStart)}${flows.outage ? " · grid off" : ""}`;
      }
      index = nextIndex;
      hour = nextHour;
    }

    function choose(i) {
      if (still) {
        show(i, focusHour(examples[i].view), true);
        return;
      }
      elapsed = heroTime(i, 0);
      show(i, 0, true);
    }

    phone.addEventListener("change", () => { // crossing the phone width: rebuild both drawings in place
      flow = flowDiagram(phone.matches ? "compact" : "wide");
      flowSlot.replaceChildren(flow.root);
      const [i, h] = [index, hour];
      index = -1;
      show(i, h, false);
    });

    range.addEventListener("input", () => {
      const h = Number(range.value);
      elapsed = heroTime(index, h);
      show(index, h, false);
    });

    const paused = () => hovered || focused || document.hidden;
    const markPaused = () => panel.classList.toggle("paused", !still && paused());
    panel.addEventListener("pointerenter", () => { hovered = true; markPaused(); });
    panel.addEventListener("pointerleave", () => { hovered = false; markPaused(); });
    panel.addEventListener("focusin", () => { focused = true; markPaused(); });
    panel.addEventListener("focusout", (event) => {
      if (panel.contains(event.relatedTarget)) return;
      focused = false;
      markPaused();
    });

    if (still) {
      show(0, focusHour(examples[0].view), false); // reduced motion: no autoplay
      return;
    }
    show(0, 0, false);
    const tick = (now) => {
      if (last !== null && !paused()) elapsed += Math.min(now - last, 100);
      last = now;
      const frame = heroClock(elapsed, examples.length);
      show(frame.index, frame.hour, frame.index !== index);
      fills[frame.index].style.transform = `scaleX(${frame.progress.toFixed(4)})`;
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  }

  async function setupHero() {
    try {
      const index = await (await fetch("replays/index.json", { cache: "no-cache" })).json();
      const picks = heroPicks(index.chips).filter(Boolean);
      const views = await Promise.all(picks.map((chip) => fetch(`replays/${encodeURIComponent(chip.file)}`, { cache: "no-cache" }).then((r) => r.json())));
      const examples = views.map((view, i) => {
        const example = { view, chip: picks[i], language: view.goal.language || picks[i].language };
        example.goal = goalBlock(example);
        example.result = resultBlock(example);
        return example;
      });
      if (examples.length) startDay(examples);
    } catch (_err) {
      $("day-body").replaceChildren(make("p", { class: "hint" }, "The recorded days could not load. Try it below still works."));
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
    $("repair-note").replaceChildren(
      `On a fail the AI repairs its rules and resubmits, at most ${code.max_repairs} times. `,
      make("span", { class: "source source-inline" }, sourceLine(code.max_repairs_source))
    );
  }

  /* --- evidence ------------------------------------------------------------------------------------ */

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
    const delta = parseDelta((rowFor(main.goal_satisfaction_bootstrap, "comparison", `${OURS} - ${LLM_SCHEDULE}`) || {}).overall);
    if (delta && names[LLM_SCHEDULE]) {
      numbers.push(bigNumber(`${delta.value} pts`, `more goals met than "${names[LLM_SCHEDULE]}" (bootstrap interval ${delta.low} to ${delta.high} pts)`, sourceLine(BENCHMARK_FILES.main, "goal_satisfaction_bootstrap")));
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
  setupCopyLink();
  setupMenu();
  setupReveal();
  setupHero();
  setupBenchmark();
})();

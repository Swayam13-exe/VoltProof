/* VoltProof day player: one recorded or live run played hour by hour. Used by the hero (three days,
   site switcher, looping) and by Try it (one result, played once). One component, two mounts.
   - energy-flow diagram: sun, grid, battery and site around a junction; line width follows kW
   - 24-hour timeline: battery charge, the rules drawn exactly where the checker tests them, the grid
     strip (import up, export down), and a playhead that is also a range input
   - the rules checked at the end of the day, then one muted or emphasised cost line
   Every value comes from the run's view (the replay file or the live result). No HTML is inserted. */
"use strict";

const VPDay = (() => {
  /* --- pure: begin (no DOM; tested in tests/test_web_logic.py) --------------------------------- */

  const DAY_MS = 14000; // the 24 hours play in 14 s
  const HOLD_MS = 3000; // then the checked rules hold for 3 s
  const SITE_MS = DAY_MS + HOLD_MS;
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

  /* Where playback is after `elapsedMs`: which run, which hour (0-23, or 24 = end of day), and how far
     through that run's slot. `loop` false: one pass that stops on the end of day. */
  function playClock(elapsedMs, count, loop = true) {
    if (!count) return null;
    if (!loop) {
      const t = Math.max(0, elapsedMs);
      return { index: 0, hour: t >= DAY_MS ? 24 : Math.floor((t * 24) / DAY_MS), progress: Math.min(1, t / DAY_MS), done: t >= DAY_MS };
    }
    const total = SITE_MS * count;
    const t = ((elapsedMs % total) + total) % total;
    const index = Math.floor(t / SITE_MS);
    const local = t - index * SITE_MS;
    return { index, hour: local < DAY_MS ? Math.floor((local * 24) / DAY_MS) : 24, progress: local / SITE_MS, done: false };
  }

  /* The playback time that shows `hour` of run `index`. */
  function playTime(index, hour) {
    return index * SITE_MS + (hour >= 24 ? DAY_MS : Math.ceil((hour * DAY_MS) / 24));
  }

  /* One hour of the executed run on the same-day weather, exactly as the view stores it. Curtailed
     solar follows from the executor's own balance (surplus beyond export is curtailed). */
  function hourFlows(view, hour) {
    const sd = view.same_day;
    const run = sd.agent;
    const plan = view.plan;
    if (hour >= 24) return { hour: 24, end: true, socStart: run.soc_pct[23], socEnd: run.soc_pct[23], outage: false };
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
      hour, end: false, pv, pvUsed, curtail, load, critical, other: Math.max(0, load - critical), charge, discharge, gridImport, gridExport,
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

  /* What the timeline draws, in hours of the day, exactly where the checker tests each rule:
     - outage: the hours [from, to) with the grid off
     - floor: a minimum charge checked at the instants from..to (inclusive). The checker tests the
       charge at the end of each hour, so "at least r% during [12, 20)" is checked at 12:00 (the end of
       hour 11), 13:00, ..., 20:00 (the end of hour 19); "by hour h" is checked at h:00
     - cap: an import cap over the hours [from, to)
     `listed`: rule kinds not drawn (they are named in words instead). */
  function ruleBands(view) {
    const goals = view.plan.goals;
    const floors = [];
    const instants = [];
    if (goals.initial_required_soc_pct !== null && goals.initial_required_soc_pct !== undefined) instants.push({ at: 0, pct: goals.initial_required_soc_pct });
    goals.required_soc_pct.forEach((pct, index) => {
      if (pct !== null && pct !== undefined) instants.push({ at: index + 1, pct }); // end of hour `index`
    });
    for (const point of instants) {
      const last = floors[floors.length - 1];
      if (last && last.to === point.at - 1 && last.pct === point.pct) last.to = point.at;
      else floors.push({ kind: "floor", from: point.at, to: point.at, pct: point.pct });
    }
    const caps = [];
    goals.import_cap_kw.forEach((kw, hour) => {
      if (kw === null || kw === undefined) return;
      const last = caps[caps.length - 1];
      if (last && last.to === hour && last.kw === kw) last.to = hour + 1;
      else caps.push({ kind: "cap", from: hour, to: hour + 1, kw });
    });
    return {
      outages: hourWindows(goals.outage_hours).map(([from, to]) => ({ kind: "outage", from, to })),
      floors,
      caps,
      listed: specCodes(view.spec).filter((code) => !DRAWN_RULES.has(code)),
    };
  }

  /* Each rule of the spec with its result (met unless its code was missed). */
  function ruleResults(spec, missedCodes) {
    const missed = new Set(missedCodes || []);
    return specCodes(spec).map((code) => ({ code, met: !missed.has(code) }));
  }

  /* The hour worth showing first: mid-outage, else inside the first drawn rule, else noon. */
  function focusHour(view) {
    const bands = ruleBands(view);
    if (bands.outages.length) return Math.floor((bands.outages[0].from + bands.outages[0].to - 1) / 2);
    const first = [...bands.floors.map((f) => [f.from, Math.max(f.from + 1, f.to)]), ...bands.caps.map((c) => [c.from, c.to])].sort((a, b) => a[0] - b[0])[0];
    return first ? Math.min(23, Math.floor((first[0] + first[1] - 1) / 2)) : 12;
  }

  /* A short place name for small buttons: the last capitalised word of the site's own name
     ("Rural health clinic near Nairobi" -> "Nairobi"). */
  function shortSiteName(name) {
    const words = String(name).match(/\b\p{Lu}[\p{L}-]*/gu);
    return words ? words[words.length - 1] : String(name);
  }

  function hourText(hour) {
    return `${String(hour).padStart(2, "0")}:00`;
  }

  /* --- pure: end ------------------------------------------------------------------------------- */

  const SVG_NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const phone = window.matchMedia("(max-width: 560px)"); // compact drawings, readable at 390 px

  function make(tag, props, ...children) {
    return VPCharts.make(tag, props, ...children);
  }

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

  const SITE_ICONS = {
    health_clinic: ["M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"],
    commercial_cold_store: ["M12 2v20", "M3.5 7l17 10", "M3.5 17l17-10", "M9 3.5l3 2.5 3-2.5", "M9 20.5l3-2.5 3 2.5"],
    household: ["M3 11.5 12 4l9 7.5", "M5.5 10v10h13V10", "M10 20v-5h4v5"],
  };

  function siteIcon(type, className) {
    return svg("svg", { class: className, viewBox: "0 0 24 24", "aria-hidden": "true", focusable: "false" },
      (SITE_ICONS[type] || SITE_ICONS.household).map((d) => svg("path", { d, fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" })));
  }

  /* --- the energy-flow diagram ----------------------------------------------------------------- */

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

  function flowDiagram(layoutName, idPrefix) {
    const layout = FLOW_LAYOUTS[layoutName];
    const { W, H, hub } = layout;
    const arrowId = (name) => `${idPrefix}-arrow-${name}`;
    const marker = (name) => svg("marker", { id: arrowId(name), viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 11, markerHeight: 11, markerUnits: "userSpaceOnUse", orient: "auto" }, svg("path", { d: "M0 0 10 5 0 10z", class: `arrow ${name}` }));
    const names = ["solar", "grid", "battery", "site"];
    const defs = svg("defs", {}, names.map(marker));
    const lines = Object.fromEntries(names.map((n) => [n, svg("path", { class: `flow-line ${n}` })]));
    const labels = Object.fromEntries(names.map((n) => [n, svg("text", { class: "flow-kw", "text-anchor": "middle" })]));
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
    const gridValue = text("flow-value", 52, 44);
    const gridOff = text("flow-off-label", 52, 44);
    const gridNode = node("grid", svg("path", { class: "grid-icon", d: "M24 16 l-8 32 M24 16 l8 32 M16 26 h16 M13 38 h22 M24 10 v6" }), text("flow-label", 52, 24, "Grid"), gridValue, gridOff);
    const level = svg("rect", { class: "battery-level", x: 17, y: 48, width: 16, height: 0, rx: 2 });
    const batteryValue = text("flow-value", 52, 44);
    const siteValue = text("flow-value", 52, 44);
    const siteCritical = text("flow-sub", 24, 66);
    const hourBig = svg("text", { class: "flow-hour", x: layout.hour.x, y: layout.hour.y, "text-anchor": layout.hour.anchor });
    const hourSub = layout.hour.sub === null ? null : svg("text", { class: "flow-hour-sub", x: layout.hour.x, y: layout.hour.sub, "text-anchor": layout.hour.anchor });
    const root = svg("svg", { class: `flow-svg ${layoutName}`, viewBox: `0 0 ${W} ${H}`, role: "img" },
      defs, ...Object.values(lines), svg("circle", { class: "flow-hub", cx: hub.x, cy: hub.y, r: 7 }),
      node("solar", sun, text("flow-label", 52, 24, "Solar"), solarValue),
      gridNode,
      node("battery", svg("rect", { class: "battery-shell", x: 14, y: 14, width: 22, height: 38, rx: 4 }), svg("rect", { class: "battery-shell", x: 20, y: 10, width: 10, height: 4, rx: 1 }), level, text("flow-label", 52, 24, "Battery"), batteryValue),
      node("site", svg("path", { class: "site-icon-path", d: "M12 32 l12-10 12 10 v12 h-24z" }), svg("circle", { class: "critical-dot", cx: 15, cy: 62, r: 3.5 }), text("flow-label", 52, 24, "Site"), siteValue, siteCritical),
      ...Object.values(labels), hourBig, hourSub);

    function setLine(name, kwValue, towardHub, scale) {
      const [ex, ey] = layout.ends[name];
      const end = { x: ex, y: ey };
      const line = lines[name];
      const idle = !(kwValue > 0.005);
      const [a, b] = towardHub ? [end, hub] : [hub, end];
      const length = Math.hypot(b.x - a.x, b.y - a.y);
      const stop = idle ? 0 : 12; // the arrowhead ends just short of the node or the junction
      line.setAttribute("d", `M${a.x},${a.y} L${(b.x - ((b.x - a.x) / length) * stop).toFixed(1)},${(b.y - ((b.y - a.y) / length) * stop).toFixed(1)}`);
      line.setAttribute("stroke-width", idle ? 2 : (2.5 + 11 * Math.min(1, kwValue / scale)).toFixed(2));
      line.classList.toggle("idle", idle);
      line.classList.toggle("moving", !idle);
      if (idle) line.removeAttribute("marker-end");
      else line.setAttribute("marker-end", `url(#${arrowId(name)})`);
      const [dx, dy] = layout.labelShift[name];
      labels[name].setAttribute("x", ((end.x + hub.x) / 2 + dx).toFixed(1));
      labels[name].setAttribute("y", ((end.y + hub.y) / 2 + dy).toFixed(1));
      labels[name].textContent = idle ? "" : kw(kwValue);
    }

    function update(view, flows, scale) {
      const criticalName = String(view.site.critical_load_name).split(",")[0];
      if (flows.end) {
        for (const name of names) setLine(name, 0, true, scale);
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
      const height = (32 * Math.max(0, Math.min(100, flows.socStart))) / 100;
      level.setAttribute("height", height.toFixed(1));
      level.setAttribute("y", (49 - height).toFixed(1));
      root.setAttribute("aria-label", flows.end
        ? `End of day: the battery ends at ${pct(flows.socEnd)}.`
        : `${hourText(flows.hour)}: solar ${kw(flows.pvUsed)}, ${flows.outage ? "grid off" : `grid ${flows.gridImport > 0 ? `supplies ${kw(flows.gridImport)}` : flows.gridExport > 0 ? `takes ${kw(flows.gridExport)}` : "idle"}`}, battery at ${pct(flows.socStart)}${flows.charge > 0 ? ` charging ${kw(flows.charge)}` : flows.discharge > 0 ? ` supplying ${kw(flows.discharge)}` : ""}, site uses ${kw(flows.load)}.`);
    }
    return { root, update };
  }

  /* --- the 24-hour timeline -------------------------------------------------------------------- */

  function timeline(view, hatchId, W) {
    const L = 40, R = 10, T = 8, SOC_B = 118, STRIP_T = 134, STRIP_ZERO = 168, STRIP_B = 196, H = 230; // axis labels on their own row
    const x = (hour) => L + (hour / 24) * (W - L - R);
    const ySoc = (value) => T + (1 - value / 100) * (SOC_B - T);
    const run = view.same_day.agent;
    const bands = ruleBands(view);
    const capMax = bands.caps.reduce((m, c) => Math.max(m, c.kw), 0);
    const stripMax = Math.max(0.1, ...run.grid_import_kw, ...run.grid_export_kw, capMax) * 1.1;
    const yUp = (v) => STRIP_ZERO - (v / stripMax) * (STRIP_ZERO - STRIP_T);
    const yDown = (v) => STRIP_ZERO + (v / stripMax) * (STRIP_B - STRIP_ZERO);
    const band = svg("rect", { class: "tl-hour-band", y: T, height: STRIP_B - T, width: x(1) - x(0) });
    const nodes = [
      svg("defs", {}, svg("pattern", { id: hatchId, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, svg("line", { class: "hatch-line", x1: 0, y1: 0, x2: 0, y2: 6 }))),
      svg("line", { class: "tl-grid-line", x1: L, x2: W - R, y1: ySoc(100), y2: ySoc(100) }),
      svg("line", { class: "tl-grid-line", x1: L, x2: W - R, y1: ySoc(50), y2: ySoc(50) }),
      band,
    ];
    for (const o of bands.outages) {
      nodes.push(svg("rect", { class: "tl-outage", x: x(o.from), y: T, width: x(o.to) - x(o.from), height: STRIP_B - T, fill: `url(#${hatchId})`, "data-band": `outage ${o.from} ${o.to}` }));
      nodes.push(svg("text", { class: "tl-outage-label", x: (x(o.from) + x(o.to)) / 2, y: STRIP_B + 13, "text-anchor": "middle" }, "grid off"));
    }
    for (const f of bands.floors) {
      const left = f.from === f.to ? x(f.from) - 3 : x(f.from); // a single instant gets a narrow marker
      const right = f.from === f.to ? x(f.to) + 3 : x(f.to);
      nodes.push(svg("rect", { class: "tl-floor-zone", x: left, y: ySoc(f.pct), width: right - left, height: ySoc(0) - ySoc(f.pct), "data-band": `floor ${f.from} ${f.to}` }));
      nodes.push(svg("line", { class: "tl-floor-line", x1: left, x2: right, y1: ySoc(f.pct), y2: ySoc(f.pct) }));
      nodes.push(svg("text", { class: "tl-rule-label", x: left + 4, y: ySoc(f.pct) + 13 }, `at least ${f.pct}%`));
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
    for (const c of bands.caps) {
      nodes.push(svg("line", { class: "tl-cap-line", x1: x(c.from), x2: x(c.to), y1: yUp(c.kw), y2: yUp(c.kw), "data-band": `cap ${c.from} ${c.to}` }));
      nodes.push(svg("text", { class: "tl-rule-label", x: x(c.from) + 3, y: yUp(c.kw) - 4 }, `import cap ${c.kw} kW`));
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
    const head = svg("line", { class: "tl-playhead", y1: T - 4, y2: STRIP_B + 2 });
    nodes.push(head);
    const outageWords = bands.outages.map((o) => `${hourText(o.from)} to ${hourText(o.to)}`).join(" and ");
    const label = `Timeline of the day: battery charge from ${pct(view.plan.initial_soc_pct)} to ${pct(run.soc_pct[23])}${outageWords ? `; grid off ${outageWords}` : ""}${bands.floors.length ? "; shaded: the minimum charge the rules require, at the times the checker tests it" : ""}${bands.caps.length ? "; dashed line on the grid strip: the import cap" : ""}.`;
    const root = svg("svg", { class: "timeline-svg", viewBox: `0 0 ${W} ${H + 14}`, role: "img", "aria-label": label }, nodes);
    function setHour(hour) {
      const at = Math.min(hour, 24);
      head.setAttribute("x1", x(at));
      head.setAttribute("x2", x(at));
      band.setAttribute("x", x(Math.min(at, 23)));
      band.setAttribute("visibility", hour >= 24 ? "hidden" : "visible");
    }
    return { root, setHour, bands };
  }

  /* What the timeline's colours mean (only the parts this run uses). */
  function timelineLegend(bands) {
    const item = (cls, words) => make("li", {}, make("span", { class: `tl-key ${cls}`, "aria-hidden": "true" }), words);
    return make("ul", { class: "tl-legend" },
      item("charge", "battery charge"),
      bands.floors.length ? item("floor", "minimum the rules require") : null,
      bands.outages.length ? item("outage", "grid off") : null,
      item("import", "grid import ↑"),
      item("export", "export ↓"),
      bands.caps.length ? item("cap", "import cap") : null);
  }

  /* --- goal and result blocks ------------------------------------------------------------------ */

  function goalBlock(example) {
    const view = example.view;
    const english = view.goal.english;
    return make(
      "div",
      { class: "goal-variant", "aria-hidden": "true" },
      make("blockquote", { class: "goal-quote" }, make("p", { class: "goal-text", lang: example.language || null }, view.goal.text)),
      english ? make("p", { class: "goal-english" }, make("span", { class: "goal-english-label" }, "In English: "), english) : null,
      make("p", { class: "goal-site" }, `${view.site.name}, ${view.site.country} · ${view.day.date}${example.language ? ` · ${languageName(example.language)}` : ""}`)
    );
  }

  /* cost: "full" (value and explanation), "muted" (the hero's quiet line) or "none" (the page shows it
     elsewhere, as Try it does in its comparison panel). */
  function resultBlock(example, cost) {
    const page = window.VPPage;
    const view = example.view;
    const sd = view.same_day;
    const items = ruleResults(view.spec, sd.agent.goals_missed).map(({ code, met }) => {
      const mark = make("span", { class: "check-mark", "aria-hidden": "true" }, "○");
      const state = make("span", { class: "check-state" }, "checked at the end of the day");
      return { item: make("li", { class: "check-item pending" }, mark, make("span", {}, page.goalWords(code, view.spec), state)), mark, state, met };
    });
    const listed = ruleBands(view).listed;
    const line = page.costLine(sd, view.site.currency, view.spec, "Realized cost on the same-day weather, counting the energy left in the battery.");
    const muted = cost === "muted";
    const value = muted ? make("span", { class: "cost-value" }, line.value) : make("strong", { class: line.tone }, line.value);
    const block = make(
      "div",
      { class: "result-variant", "aria-hidden": "true" },
      make("p", { class: "result-title" }, "The rules, checked on the day"),
      items.length ? make("ul", { class: "checklist" }, items.map((i) => i.item)) : make("p", { class: "hint" }, "No rules besides the lowest cost."),
      listed.length ? make("p", { class: "hint" }, "Not drawn on the timeline: ", listed.map((code) => page.goalWords(code, view.spec)).join("; "), ".") : null,
      cost === "none" ? null : make("p", { class: `cost-line${muted ? " muted" : ""}` }, value, line.same ? ". " : " than the rule-based battery. ", line.explain),
      example.file ? make("p", { class: "source" }, `Source: ${example.file}`) : null
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

  /* --- mount: the player ------------------------------------------------------------------------- */

  /* examples: [{view, file (source path or null), language}]; options:
     id (unique per page), switcher (site buttons), loop (cycle forever, else one pass), live (an
     aria-live element announcing each new goal), showGoal (the goal quote; off when the page shows the
     goal already), cost (see resultBlock). */
  function mount(host, examples, options) {
    const { id, switcher = false, loop = true, live = null, showGoal = true, cost = "full" } = options || {};
    const still = reduceMotion.matches;
    examples.forEach((e) => {
      e.goal = goalBlock(e);
      e.result = resultBlock(e, cost);
    });
    let flow = flowDiagram(phone.matches ? "compact" : "wide", id);
    const flowSlot = make("div", { class: "flow-slot" }, flow.root);
    const timelineSlot = make("div", { class: "timeline-slot" });
    const legendSlot = make("div", { class: "legend-slot" });
    const range = make("input", { type: "range", id: `${id}-hour`, min: "0", max: "24", step: "1", value: "0", "aria-describedby": `${id}-readout` });
    const readout = make("p", { id: `${id}-readout`, class: "scrub-readout" });
    const body = make("div", { class: "day-body" },
      make("div", { class: "day-col" },
        showGoal ? make("div", { class: "day-goal" }, make("div", { class: "stack" }, examples.map((e) => e.goal))) : null,
        make("figure", { class: "day-flow" }, flowSlot, make("figcaption", { class: "flow-caption" }, "Energy this hour: line width follows kW, arrows show the direction. The red dot marks the critical load."))),
      make("div", { class: "day-col" },
        make("div", { class: "day-timeline" }, timelineSlot, legendSlot, make("div", { class: "scrub" }, make("label", { for: `${id}-hour` }, "Hour of the day"), range, readout)),
        make("div", { class: "day-result" }, make("div", { class: "stack" }, examples.map((e) => e.result.block)))));

    const fills = [];
    const buttons = switcher ? examples.map((example, i) => {
      const fill = make("span", { class: "site-progress", "aria-hidden": "true" });
      fills.push(fill);
      const site = example.view.site;
      const button = make("button", { type: "button", class: "site-btn", "aria-pressed": "false", "aria-label": `${site.name}, goal in ${languageName(example.language)}` },
        siteIcon(site.site_type, "site-icon"),
        make("span", { class: "site-name" }, make("span", { class: "name-full" }, site.name), make("span", { class: "name-short" }, shortSiteName(site.name))),
        make("span", { class: "site-lang", lang: "en" }, String(example.language).toUpperCase()), fill);
      button.addEventListener("click", () => choose(i));
      return button;
    }) : [];
    host.replaceChildren(...(switcher ? [make("div", { class: "site-switch", role: "group", "aria-label": "Recorded days" }, buttons)] : []), body);

    let index = -1, hour = -1, elapsed = 0, last = null, hovered = false, focused = false, line = null, frameId = null;
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
          if (buttons[i]) buttons[i].setAttribute("aria-pressed", String(on));
          if (fills[i] && !on) fills[i].style.transform = "scaleX(0)";
        });
        line = timeline(example.view, `${id}-hatch-${nextIndex}`, phone.matches ? 360 : 640);
        timelineSlot.replaceChildren(line.root);
        legendSlot.replaceChildren(timelineLegend(line.bands));
        if (announce && live) live.replaceChildren(`${example.view.site.name}. Goal: `, make("span", { lang: example.language || null }, example.view.goal.text));
      }
      if (nextIndex !== index || nextHour !== hour) {
        const flows = hourFlows(example.view, nextHour);
        flow.update(example.view, flows, scales[nextIndex]);
        line.setHour(nextHour);
        example.result.reveal(still || nextHour >= 24);
        range.value = String(nextHour);
        range.setAttribute("aria-valuetext", nextHour >= 24 ? "24:00, end of day" : `${hourText(nextHour)}${flows.outage ? ", grid off" : ""}, battery ${pct(flows.socStart)}`);
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
      elapsed = playTime(i, 0);
      show(i, 0, true);
      if (!loop) start();
    }

    range.addEventListener("input", () => {
      const h = Number(range.value);
      elapsed = playTime(index, h);
      show(index, h, false);
    });
    phone.addEventListener("change", () => { // crossing the phone width: rebuild both drawings in place
      flow = flowDiagram(phone.matches ? "compact" : "wide", id);
      flowSlot.replaceChildren(flow.root);
      const [i, h] = [index, hour];
      index = -1;
      show(i, h, false);
    });

    const paused = () => hovered || focused || document.hidden;
    const markPaused = () => host.classList.toggle("paused", !still && paused());
    host.addEventListener("pointerenter", () => { hovered = true; markPaused(); });
    host.addEventListener("pointerleave", () => { hovered = false; markPaused(); });
    host.addEventListener("focusin", () => { focused = true; markPaused(); });
    host.addEventListener("focusout", (event) => {
      if (host.contains(event.relatedTarget)) return;
      focused = false;
      markPaused();
    });

    function tick(now) {
      frameId = null;
      if (!host.isConnected) return;
      if (last !== null && !paused()) elapsed += Math.min(now - last, 100);
      last = now;
      const frame = playClock(elapsed, examples.length, loop);
      show(frame.index, frame.hour, frame.index !== index);
      if (fills[frame.index]) fills[frame.index].style.transform = `scaleX(${frame.progress.toFixed(4)})`;
      if (frame.done) return; // a single pass ends on the checked rules
      frameId = window.requestAnimationFrame(tick);
    }

    function start() {
      if (frameId === null) {
        last = null;
        frameId = window.requestAnimationFrame(tick);
      }
    }

    if (still) {
      show(0, focusHour(examples[0].view), false); // reduced motion: no autoplay
    } else {
      show(0, 0, false);
      start();
    }
    return { show };
  }

  return { mount, specCodes, ruleBands, ruleResults, hourFlows, focusHour, hourText, shortSiteName, siteIcon, languageName };
})();

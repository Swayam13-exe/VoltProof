/* Charts for the VoltProof page (Chart.js 4.4.1 from cdnjs, nothing else).
   Colours come from the page's CSS tokens, so light and dark mode both work; the series colours
   are a colour-blind-safe set (validated) and every chart also has a text legend and a table view.
   Every axis carries its unit; money axes carry the site's currency. No innerHTML anywhere. */
"use strict";

const VPCharts = (() => {
  const charts = new Map(); // canvas -> {chart, build}
  const HOURS = 24;

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function make(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function windows(hours) {
    const sorted = [...new Set(hours)].sort((a, b) => a - b);
    const out = [];
    for (const hour of sorted) {
      const last = out[out.length - 1];
      if (last && last[1] === hour) last[1] = hour + 1;
      else out.push([hour, hour + 1]);
    }
    return out;
  }

  /* Shaded hour bands (outages, tariff periods), drawn behind the data, each with a text label. */
  const bandPlugin = {
    id: "vpBands",
    beforeDatasetsDraw(chart, _args, options) {
      const bands = (options && options.bands) || [];
      if (!bands.length) return;
      const { ctx, chartArea, scales } = chart;
      ctx.save();
      for (const band of bands) {
        const left = scales.x.getPixelForValue(band.from);
        const right = scales.x.getPixelForValue(band.to);
        ctx.fillStyle = band.strong ? token("--band-strong") : token("--band");
        ctx.fillRect(left, chartArea.top, right - left, chartArea.bottom - chartArea.top);
        if (band.label) {
          ctx.fillStyle = token("--ink-2");
          ctx.font = "600 11px system-ui, -apple-system, 'Segoe UI', sans-serif";
          ctx.textBaseline = "bottom"; // labels sit at the bottom of the band, away from most lines
          const text = band.label;
          const width = ctx.measureText(text).width;
          if (width + 6 <= right - left) ctx.fillText(text, left + 3, chartArea.bottom - 3);
        }
      }
      ctx.restore();
    },
  };

  function axes({ yTitle, yMin, yMax, stacked, currency }) {
    const tick = { color: token("--muted"), font: { size: 11 }, maxRotation: 0 };
    const narrow = window.matchMedia("(max-width: 520px)").matches;
    return {
      x: {
        type: "linear", min: 0, max: HOURS, stacked: !!stacked, offset: false, // every chart shares the same 0-24 h axis
        title: { display: true, text: "Hour of day (site's local time)", color: token("--ink-2") },
        ticks: { ...tick, stepSize: narrow ? 6 : 3, callback: (value) => `${value}:00` },
        grid: { color: token("--line") }, border: { color: token("--axis") },
      },
      y: {
        min: yMin, max: yMax, stacked: !!stacked,
        title: { display: true, text: yTitle, color: token("--ink-2") },
        ticks: { ...tick, callback: currency ? (value) => `${value}` : undefined },
        grid: { color: token("--line") }, border: { color: token("--axis") },
      },
    };
  }

  function baseOptions(scaleOptions, bands, tooltipUnit) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false, // the data is static; no motion needed
      interaction: { mode: "nearest", axis: "x", intersect: false },
      scales: axes(scaleOptions),
      plugins: {
        legend: { display: false },
        vpBands: { bands },
        tooltip: {
          callbacks: {
            title: (items) => (items.length ? `Hour ${Math.floor(items[0].parsed.x)}:00` : ""),
            label: (item) => `${item.dataset.label}: ${formatNumber(item.parsed.y)} ${tooltipUnit}`,
          },
        },
      },
    };
  }

  function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "-";
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  /* SOC lines: points at hour boundaries 0..24 (the initial SOC, then the end of each hour). */
  function boundaryPoints(initial, endOfHour) {
    return [{ x: 0, y: initial }, ...endOfHour.map((value, hour) => ({ x: hour + 1, y: value }))];
  }

  function requiredPoints(initialRequired, required) {
    const values = [initialRequired, ...required];
    return values.map((value, index) => ({ x: index, y: value === null || value === undefined ? null : value }));
  }

  function hourPoints(values, sign = 1) {
    return values.map((value, hour) => ({ x: hour + 0.5, y: sign * value }));
  }

  function register(canvas, build) {
    if (!ready()) return; // no Chart.js (offline copy): the legend and the table still show everything
    const existing = charts.get(canvas);
    if (existing) existing.chart.destroy();
    charts.set(canvas, { chart: build(), build });
  }

  function rerenderAll() {
    for (const [canvas, entry] of charts) {
      if (!canvas.isConnected) {
        entry.chart.destroy();
        charts.delete(canvas);
        continue;
      }
      entry.chart.destroy();
      entry.chart = entry.build();
    }
  }

  function destroyAll() {
    for (const entry of charts.values()) entry.chart.destroy();
    charts.clear();
  }

  /* --- legend and table (the text channels every chart has) --------------------------------- */

  function legend(items) {
    return make(
      "ul",
      { class: "legend" },
      items.map((item) => {
        const key = make("span", { class: `key ${item.kind || ""}`.trim(), "aria-hidden": "true" });
        if (item.color) key.style.background = item.color;
        return make("li", {}, key, item.label);
      })
    );
  }

  function table(caption, headers, rows) {
    const head = make("tr", {}, headers.map((h) => make("th", { scope: "col" }, h)));
    const body = rows.map((row) => make("tr", {}, row.map((cell, i) => (i === 0 ? make("th", { scope: "row" }, cell) : make("td", {}, cell)))));
    return make(
      "details",
      {},
      make("summary", {}, "Show the numbers as a table"),
      make("div", { class: "table-wrap" }, make("table", {}, make("caption", { class: "visually-hidden" }, caption), make("thead", {}, head), make("tbody", {}, body)))
    );
  }

  function block(title, caption, canvasLabel, legendItems, tableNode) {
    const canvas = make("canvas", { role: "img", "aria-label": canvasLabel });
    const node = make(
      "figure",
      { class: "chart-block" },
      make("figcaption", {}, make("p", { class: "chart-title" }, title), make("p", { class: "chart-caption" }, caption)),
      ready() ? make("div", { class: "chart-box" }, canvas) : make("p", { class: "hint" }, "The chart could not load (it needs Chart.js from cdnjs); the table below has the same numbers."),
      legendItems.length > 1 ? legend(legendItems) : null,
      tableNode
    );
    return { node, canvas };
  }

  /* --- the four charts ----------------------------------------------------------------------- */

  function outageBands(hours, label) {
    return windows(hours).map(([from, to]) => ({ from, to, label, strong: true }));
  }

  function socChart(plan, site) {
    const goals = plan.goals;
    const hasRequired = goals.initial_required_soc_pct !== null || goals.required_soc_pct.some((v) => v !== null);
    const items = [{ label: "Planned state of charge", color: token("--series-1") }];
    if (hasRequired) items.push({ label: "Minimum the goal requires", color: token("--ink-2") });
    if (goals.outage_hours.length) items.push({ label: "Grid off (announced outage)", kind: "band" });
    const rows = plan.soc_pct.map((v, h) => [`${h}:00-${h + 1}:00`, formatNumber(v), formatNumber(goals.required_soc_pct[h]), goals.outage_hours.includes(h) ? "off" : ""]);
    const out = block(
      "Battery state of charge (plan, on the forecast)",
      "Energy stored at the end of each hour, as a share of the battery's capacity.",
      `Line chart of the planned state of charge over the day for ${site.name}.`,
      items,
      table("Planned state of charge by hour", ["Hour", "State of charge (%)", "Goal minimum (%)", "Grid"], rows)
    );
    register(out.canvas, () => {
      const datasets = [{
        label: "Planned state of charge", data: boundaryPoints(plan.initial_soc_pct, plan.soc_pct),
        borderColor: token("--series-1"), backgroundColor: token("--series-1"), borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0,
      }];
      if (hasRequired) {
        datasets.push({
          label: "Minimum the goal requires", data: requiredPoints(goals.initial_required_soc_pct, goals.required_soc_pct),
          borderColor: token("--ink-2"), backgroundColor: token("--ink-2"), borderWidth: 2, stepped: "before", pointRadius: 0, spanGaps: false,
        });
      }
      return new Chart(out.canvas, { type: "line", data: { datasets }, options: baseOptions({ yTitle: "State of charge (%)", yMin: 0, yMax: 100 }, outageBands(goals.outage_hours, "grid off"), "%") });
    });
    return out.node;
  }

  function flowsChart(plan, site) {
    const solarUsed = plan.pv_forecast_kw.map((pv, h) => Math.max(0, pv - plan.curtail_kw[h]));
    const series = [
      { label: "Solar", data: hourPoints(solarUsed), color: "--series-4", stack: "flow" },
      { label: "Battery discharging", data: hourPoints(plan.discharge_kw), color: "--series-3", stack: "flow" },
      { label: "Grid import", data: hourPoints(plan.grid_import_kw), color: "--series-1", stack: "flow" },
      { label: "Battery charging", data: hourPoints(plan.charge_kw, -1), color: "--series-3", stack: "flow" },
      { label: "Grid export", data: hourPoints(plan.grid_export_kw, -1), color: "--series-1", stack: "flow" },
    ];
    const items = [
      { label: "Solar", color: token("--series-4"), kind: "box" },
      { label: "Battery (discharging above zero, charging below)", color: token("--series-3"), kind: "box" },
      { label: "Grid (import above zero, export below)", color: token("--series-1"), kind: "box" },
      { label: "Demand (forecast)", color: token("--ink") },
    ];
    const rows = plan.load_forecast_kw.map((load, h) => [`${h}:00`, formatNumber(load), formatNumber(solarUsed[h]), formatNumber(plan.discharge_kw[h]), formatNumber(plan.charge_kw[h]), formatNumber(plan.grid_import_kw[h]), formatNumber(plan.grid_export_kw[h])]);
    const out = block(
      "Where the energy comes from (plan, on the forecast)",
      "Above zero: what supplies the site each hour. Below zero: energy sent into the battery or exported. The line is the forecast demand.",
      `Stacked bar chart of planned energy flows by hour for ${site.name}.`,
      items,
      table("Planned flows by hour (kW)", ["Hour", "Demand", "Solar", "Battery out", "Battery in", "Grid import", "Grid export"], rows)
    );
    register(out.canvas, () => {
      const surface = token("--surface");
      const datasets = series.map((s) => ({
        type: "bar", label: s.label, data: s.data, stack: s.stack, backgroundColor: token(s.color),
        borderColor: surface, borderWidth: 1, borderRadius: 2, maxBarThickness: 24, barPercentage: 0.85, categoryPercentage: 1,
      }));
      datasets.push({ type: "line", label: "Demand (forecast)", data: hourPoints(plan.load_forecast_kw), borderColor: token("--ink"), backgroundColor: token("--ink"), borderWidth: 2, pointRadius: 0, stepped: "middle" });
      return new Chart(out.canvas, { type: "bar", data: { datasets }, options: baseOptions({ yTitle: "Power (kW)", stacked: true }, outageBands(plan.goals.outage_hours, "grid off"), "kW") });
    });
    return out.node;
  }

  function priceChart(plan, site) {
    const currency = site.currency;
    const hasExport = plan.export_price_per_kwh.some((v) => v > 0);
    const periodBands = [];
    for (const [name, wins] of Object.entries(plan.tariff_periods || {})) {
      if (name === "normal") continue;
      for (const [from, to] of wins) periodBands.push({ from, to, label: name, strong: name === "peak" });
    }
    const items = [{ label: `Import price (${currency}/kWh)`, color: token("--series-1") }];
    if (hasExport) items.push({ label: `Export price (${currency}/kWh)`, color: token("--series-2") });
    if (periodBands.length) items.push({ label: "Named tariff periods (shaded, labelled)", kind: "band" });
    const rows = plan.import_price_per_kwh.map((p, h) => [`${h}:00`, formatNumber(p), formatNumber(plan.export_price_per_kwh[h])]);
    const out = block(
      `Electricity price (${currency} per kWh)`,
      `What the grid charges for each kWh bought, and pays for each kWh sold, in ${currency}.`,
      `Line chart of hourly electricity prices in ${currency} per kWh for ${site.name}.`,
      items,
      table(`Prices by hour (${currency}/kWh)`, ["Hour", "Import", "Export"], rows)
    );
    register(out.canvas, () => {
      const datasets = [{ label: "Import price", data: hourPoints(plan.import_price_per_kwh), borderColor: token("--series-1"), backgroundColor: token("--series-1"), borderWidth: 2, pointRadius: 0, stepped: "middle" }];
      if (hasExport) datasets.push({ label: "Export price", data: hourPoints(plan.export_price_per_kwh), borderColor: token("--series-2"), backgroundColor: token("--series-2"), borderWidth: 2, pointRadius: 0, stepped: "middle" });
      return new Chart(out.canvas, { type: "line", data: { datasets }, options: baseOptions({ yTitle: `Price (${currency}/kWh)`, yMin: 0, currency }, periodBands, `${currency}/kWh`) });
    });
    return out.node;
  }

  function sameDayChart(view) {
    const plan = view.plan;
    const sd = view.same_day;
    const goals = plan.goals;
    const hasRequired = goals.initial_required_soc_pct !== null || goals.required_soc_pct.some((v) => v !== null);
    const items = [
      { label: "VoltProof plan, as executed", color: token("--series-1") },
      { label: "Rule-based battery", color: token("--series-2") },
    ];
    if (hasRequired) items.push({ label: "Minimum the goal requires", color: token("--ink-2") });
    if (goals.outage_hours.length) items.push({ label: "Grid off (announced outage)", kind: "band" });
    const rows = sd.agent.soc_pct.map((v, h) => [`${h}:00-${h + 1}:00`, formatNumber(v), formatNumber(sd.rule_based.soc_pct[h]), formatNumber(goals.required_soc_pct[h])]);
    const out = block(
      "State of charge on the same-day weather",
      "Both batteries run on what the weather model says happened that day, starting from the same charge.",
      `Line chart comparing the executed state of charge of the VoltProof plan and the rule-based battery for ${view.site.name}.`,
      items,
      table("Executed state of charge by hour (%)", ["Hour", "VoltProof", "Rule-based", "Goal minimum"], rows)
    );
    register(out.canvas, () => {
      const datasets = [
        { label: "VoltProof plan", data: boundaryPoints(plan.initial_soc_pct, sd.agent.soc_pct), borderColor: token("--series-1"), backgroundColor: token("--series-1"), borderWidth: 2, pointRadius: 0 },
        { label: "Rule-based battery", data: boundaryPoints(plan.initial_soc_pct, sd.rule_based.soc_pct), borderColor: token("--series-2"), backgroundColor: token("--series-2"), borderWidth: 2, pointRadius: 0 },
      ];
      if (hasRequired) {
        datasets.push({ label: "Minimum the goal requires", data: requiredPoints(goals.initial_required_soc_pct, goals.required_soc_pct), borderColor: token("--ink-2"), backgroundColor: token("--ink-2"), borderWidth: 2, stepped: "before", pointRadius: 0 });
      }
      return new Chart(out.canvas, { type: "line", data: { datasets }, options: baseOptions({ yTitle: "State of charge (%)", yMin: 0, yMax: 100 }, outageBands(goals.outage_hours, "grid off"), "%") });
    });
    return out.node;
  }

  function ready() {
    return typeof window.Chart !== "undefined";
  }

  if (ready()) Chart.register(bandPlugin);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerenderAll);

  return { socChart, flowsChart, priceChart, sameDayChart, destroyAll, ready, make, formatNumber };
})();

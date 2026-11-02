/* VoltProof page, "Try it": pick a site and a goal, press one button, watch the pipeline, see the day.
   - a recorded goal replays a recorded run of the agent (replays/*.json): free, labelled
     "replay (recorded run)"; its pipeline stages light up quickly from the recorded trace
   - a typed goal goes to the live API (api/plan) and the stages light up from the events it streams;
     if a live run is refused or fails, a recorded run for the same site plays, with a notice
   - the result is the shared day player (dayplayer.js), then the comparison with the rule-based
     battery, the rules (plain words, raw JSON folded), the agent's steps (folded), and a CSV download
   Every string from the model or the data is inserted as text (textContent / text nodes), never as
   HTML: there is no innerHTML in this file. */
"use strict";

(() => {
  const make = VPCharts.make;
  const $ = (id) => document.getElementById(id);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const MAX_GOAL = 500;
  const RECORDED_STEP_MS = 220; // recorded runs: stages light up quickly (the label says it is a replay)

  const CHECK_NAMES = {
    verified_plan: "Plan verified: physics checked, and every shortfall stated",
    coverage: "Every checklist item is covered by the spec",
    grounded_explanation: "Every number in the explanation comes from the tools or the goal",
    review: "The independent AI reviewer approves the spec",
  };
  const INFO_TOOLS = { get_site_info: "the site", get_forecast: "the day-ahead forecast", get_tariff: "the tariff" };
  const KIND_TAGS = { english: "EN", hard: "hard", multilingual: null };

  const state = { chips: [], sites: [], siteId: null, chipId: null, playToken: 0, live: null, busy: false };

  /* --- formatting -------------------------------------------------------------------------- */

  function money(value, currency) {
    if (value === null || value === undefined) return "-";
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency, maximumFractionDigits: 2 }).format(value);
    } catch (_err) {
      return `${value.toFixed(2)} ${currency}`;
    }
  }

  function pct(value) {
    return value === null || value === undefined ? "-" : `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }

  function amount(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function kwh(value) {
    return `${amount(value)} kWh`;
  }

  function status(text, warn = false) {
    const node = $("status");
    node.textContent = text;
    node.classList.toggle("warn", warn);
  }

  /* --- setup: sites, days, chips ------------------------------------------------------------- */

  function siteFacts(site) {
    const tariff = { time_of_use: "time-of-use tariff", hourly_dynamic: "hourly dynamic tariff", flat: "flat tariff" }[site.tariff_kind] || site.tariff_kind;
    return `${site.country} · ${tariff} · ${site.battery_kwh} kWh battery · ${site.pv_kwp} kWp solar`;
  }

  function renderSites() {
    const list = $("site-list");
    list.replaceChildren(
      ...state.sites.map((site) => {
        const input = make("input", { type: "radio", name: "site", value: site.site_id, id: `site-${site.site_id}` });
        input.checked = site.site_id === state.siteId;
        input.addEventListener("change", () => selectSite(site.site_id));
        return make(
          "label",
          { class: "site-card", for: `site-${site.site_id}` },
          input,
          make("span", { class: "site-card-head" }, VPDay.siteIcon(site.site_type, "site-card-icon"), make("span", { class: "site-name" }, site.name)),
          make("span", { class: "site-facts" }, siteFacts(site)),
          make("span", { class: "site-facts" }, `Critical load: ${site.critical_load_name} (${site.critical_load_kw} kW)`)
        );
      })
    );
  }

  function renderDays() {
    const recorded = new Set(state.chips.filter((c) => c.site_id === state.siteId).map((c) => c.date));
    const select = $("day");
    select.replaceChildren(...[...recorded].sort().map((d) => make("option", { value: d }, `${d} (recorded runs)`)));
    updateButton();
    if (!state.live || !state.live.live) return;
    const siteId = state.siteId;
    fetch(`api/days?site=${encodeURIComponent(siteId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((days) => {
        if (!days || siteId !== state.siteId) return;
        const keep = select.value;
        select.replaceChildren(...days.map((d) => make("option", { value: d.date }, `${d.date} · ${d.split} day${recorded.has(d.date) ? ", recorded runs" : ""}`)));
        if (keep) select.value = keep;
        updateButton();
      })
      .catch(() => {});
  }

  function renderChips() {
    const list = $("chips");
    const mine = state.chips.filter((c) => c.site_id === state.siteId);
    list.replaceChildren(
      ...mine.map((chip) => {
        const tag = KIND_TAGS[chip.kind] === undefined ? chip.kind : KIND_TAGS[chip.kind] || chip.language.toUpperCase();
        const button = make(
          "button",
          { type: "button", class: "chip", "aria-pressed": String(chip.id === state.chipId), lang: chip.language },
          make("span", { class: "chip-tag", lang: "en" }, tag),
          chip.goal
        );
        button.addEventListener("click", () => fillFromChip(chip));
        return make("li", {}, button);
      })
    );
  }

  function selectSite(siteId) {
    state.siteId = siteId;
    renderSites();
    renderDays();
    renderChips();
  }

  /* A recorded goal fills the box (and sets its day); the button then offers the free replay. */
  function fillFromChip(chip) {
    state.chipId = chip.id;
    if (state.siteId !== chip.site_id) selectSite(chip.site_id);
    const select = $("day");
    if (![...select.options].some((o) => o.value === chip.date)) select.append(make("option", { value: chip.date }, `${chip.date} (recorded runs)`));
    select.value = chip.date;
    $("goal").value = chip.goal;
    updateCount();
    renderChips();
    updateButton();
  }

  /* --- missed goals in plain words (rule code -> what it means, with the spec's window and level) --- */

  function hourText(h) {
    return h === 24 ? "midnight" : `${h}:00`;
  }

  function windowText(w) {
    return `from ${hourText(w.start_hour)} to ${hourText(w.end_hour)}`;
  }

  function goalWords(code, spec) {
    const s = spec || {};
    const list = (items, fmt) => (items && items.length ? `: ${items.map(fmt).join("; ")}` : "");
    switch (code) {
      case "RESERVE": return `battery reserve${list(s.reserve_windows, (w) => `at least ${w.min_soc_pct}% ${windowText(w)}`)}`;
      case "SOC_TARGET": return `charge target${list(s.soc_targets, (t) => `at least ${t.min_soc_pct}% by ${hourText(t.by_hour)}`)}`;
      case "ALL_DAY_MIN": return `all-day minimum${s.min_soc_pct_all_day !== undefined && s.min_soc_pct_all_day !== null ? `: at least ${s.min_soc_pct_all_day}% at every hour` : ""}`;
      case "END_SOC": return `end-of-day level${s.end_of_day_min_soc_pct !== undefined && s.end_of_day_min_soc_pct !== null ? `: at least ${s.end_of_day_min_soc_pct}% at midnight` : ""}`;
      case "IMPORT_CAP": return `grid import cap${list(s.import_caps, (c) => `at most ${c.max_kw} kW ${windowText(c)}`)}`;
      case "NO_EXPORT": return `no export to the grid${list(s.no_export_windows, windowText)}`;
      case "DISCHARGE_WINDOW": return `battery discharging only${list(s.discharge_only_windows, windowText)}`;
      case "THROUGHPUT": return `battery use limit${s.max_throughput_kwh !== undefined && s.max_throughput_kwh !== null ? `: at most ${s.max_throughput_kwh} kWh charged and discharged` : ""}`;
      case "OUTAGE_DEFICIT": return `critical load during the outage${list(s.grid_outages, windowText)}`;
      default: return String(code).toLowerCase().replaceAll("_", " ");
    }
  }

  function missedText(codes, spec) {
    return codes.map((code) => goalWords(code, spec)).join("; ");
  }

  /* Non-critical ("other") load left unserved. Realized cost does not count it, so a plan that keeps
     the whole site running through an outage can cost more than a rule-based battery that switched
     other load off. */
  const SHED_SHOWN_KWH = 0.01;
  const SHED_EXPLAINS_KWH = 0.1;

  function shed(side) {
    return Number(side.non_critical_shed_kwh || 0);
  }

  function shedWords(side) {
    const total = shed(side);
    const outage = side.non_critical_shed_outage_kwh;
    if (outage === null || outage === undefined || outage < SHED_SHOWN_KWH) return `${kwh(total)} of other load`;
    if (Math.abs(total - outage) < SHED_SHOWN_KWH) return `${kwh(total)} of other load during the outage`;
    return `${kwh(total)} of other load (${kwh(outage)} of it during the outage)`;
  }

  /* The cost comparison with the rule-based battery. When VoltProof costs more only because the
     rule-based battery missed goals that VoltProof met, or switched off other load that VoltProof
     kept on, the value is neutral and the reason is said. A difference that rounds to zero in the
     currency's two decimals is the same cost. */
  function costLine(sd, currency, spec, explain) {
    const diff = sd.cost_difference;
    if (Math.round(Math.abs(diff) * 100) === 0) return { value: "Same cost as the rule-based battery", tone: "neutral", explain, same: true };
    if (diff >= 0) return { value: `${money(diff, currency)} cheaper`, tone: "ok", explain };
    const value = `${money(-diff, currency)} more expensive`;
    const metOnly = sd.agent.goals_met && !sd.rule_based.goals_met;
    if (shed(sd.rule_based) - shed(sd.agent) >= SHED_EXPLAINS_KWH - 1e-9) {
      const kept = shed(sd.agent) < SHED_SHOWN_KWH ? "the whole site" : "more of the site";
      const off = `switched off ${shedWords(sd.rule_based)}`;
      if (metOnly) {
        return { value, tone: "neutral", explain: `Costs more because it met the goal and kept ${kept} running; the rule-based battery missed: ${missedText(sd.rule_based.goals_missed, spec)}, and ${off}.` };
      }
      return { value, tone: "neutral", explain: `Costs more because it kept ${kept} running; the rule-based battery ${off}.` };
    }
    if (metOnly) {
      return { value, tone: "neutral", explain: `Costs more because it met the goal; the rule-based battery was cheaper only by missing: ${missedText(sd.rule_based.goals_missed, spec)}.` };
    }
    return { value, tone: "fail", explain };
  }

  /* The "Other load kept on" line, or null when neither side switched any load off. */
  function otherLoadLine(sd) {
    const mine = shed(sd.agent);
    const theirs = shed(sd.rule_based);
    if (mine < SHED_SHOWN_KWH && theirs < SHED_SHOWN_KWH) return null;
    const explain = theirs < SHED_SHOWN_KWH ? "The rule-based battery kept all of it on." : `Rule-based battery switched off ${shedWords(sd.rule_based)}.`;
    if (mine < SHED_SHOWN_KWH) return { value: "✓ all of it", tone: "ok", explain };
    return { value: `all but ${kwh(mine)}`, tone: mine <= theirs ? "neutral" : "fail", explain };
  }

  /* --- try pure: begin (no DOM; tested in tests/test_web_logic.py) ----------------------------- */

  const PIPELINE_STAGES = [
    ["rules", "AI writes the rules"],
    ["review", "AI review"],
    ["optimizer", "Optimizer plans the day"],
    ["physics", "Physics check"],
    ["replay", "Replay on the same-day weather"],
  ];
  const PIPELINE_TOOLS = new Set(["set_checklist", "solve_plan", "finalize"]);

  /* Recorded goals for the chosen site and day. */
  function recordedFor(chips, siteId, day) {
    return chips.filter((c) => c.site_id === siteId && c.date === day);
  }

  /* What the one main button offers for this text: a free replay when it is a recorded goal of this
     site and day, a live run when it is typed and live runs are available, nothing when empty. */
  function runButtonState(text, recorded, live) {
    const goal = String(text || "").trim();
    if (!goal) return { mode: "empty", label: "Plan my day", disabled: true, hint: "Type a goal or pick one below." };
    const chip = recorded.find((c) => c.goal.trim() === goal) || null;
    if (chip) return { mode: "recorded", label: "Play recorded run (free)", disabled: false, hint: "A recorded run of the agent for this goal, replayed at no cost.", chip };
    const left = live && live.live ? Number(live.runs_left_this_hour) : 0;
    if (live && live.live && left > 0) {
      return { mode: "live", label: `Run live (${left} left this hour)`, disabled: false, hint: live.per_visitor_limit === false ? "Runs the AI on your goal now; live runs are shared by all visitors." : "Runs the AI on your goal now; it usually takes 10 to 30 seconds." };
    }
    const why = !live ? "the live AI is not available here" : !live.live ? reasonText(live.reason) : "no live runs are left this hour";
    return { mode: "unavailable", label: "Live runs unavailable: play a recorded run", disabled: true, hint: `Live runs unavailable (${why}). Pick a recorded goal below.` };
  }

  function pipelineStart() {
    return { rules: "idle", review: "idle", optimizer: "idle", physics: "idle", replay: "idle", repairs: 0, failed: [] };
  }

  /* One agent event -> the stages it moves. Events are {kind, name, data}: live events from the
     stream, or the recorded trace (traceEvents). A failed check sends the run back to the AI (a repair). */
  function pipelineStep(previous, event) {
    const s = { ...previous, failed: [...previous.failed] };
    const result = (event.data && event.data.result) || {};
    if (event.kind === "tool_call") {
      if (event.name === "set_checklist" && s.rules === "idle") s.rules = "active";
      else if (event.name === "solve_plan") {
        s.rules = "done";
        s.optimizer = "active";
      } else if (event.name === "finalize") s.physics = "active";
    } else if (event.kind === "tool_result") {
      if (event.name === "solve_plan") {
        s.optimizer = result.error ? "fail" : "done";
        if (result.error) s.rules = "active";
      } else if (event.name === "finalize") {
        const failed = (result.failed_checks || []).map((f) => (typeof f === "string" ? f : f && f.check)).filter(Boolean);
        if (result.accepted) s.physics = "done";
        else if (failed.length) {
          s.repairs += 1;
          s.failed = failed;
          s.physics = "fail";
          if (failed.includes("review")) s.review = "fail";
          if (!result.escalated) s.rules = "active";
        }
      }
    } else if (event.kind === "review") {
      s.review = event.data && event.data.approved === false ? "fail" : "done";
    } else if (event.kind === "end") {
      s.replay = event.data && event.data.status === "done" ? "active" : "fail";
    } else if (event.kind === "replayed") {
      s.replay = event.data && event.data.ok ? "done" : "fail";
    }
    return s;
  }

  /* The recorded trace as pipeline events (only the steps the stages show). */
  function traceEvents(view) {
    return (view.trace || [])
      .filter((e) => e.kind === "review" || ((e.kind === "tool_call" || e.kind === "tool_result") && PIPELINE_TOOLS.has(e.name)))
      .map((e) => ({ kind: e.kind, name: e.name, data: e.data || {} }));
  }

  /* The comparison panel: each rule met or missed on both sides, then cost, other load, critical load. */
  function comparisonRows(view, codes) {
    const sd = view.same_day;
    const currency = view.site.currency;
    const missedAgent = new Set(sd.agent.goals_missed || []);
    const missedRule = new Set(sd.rule_based.goals_missed || []);
    return {
      rules: codes.map((code) => ({ label: goalWords(code, view.spec), agent: !missedAgent.has(code), rule: !missedRule.has(code) })),
      facts: [
        { label: "Cost on the same-day weather", agent: money(sd.agent.realized_cost, currency), rule: money(sd.rule_based.realized_cost, currency) },
        { label: "Other load switched off", agent: kwh(shed(sd.agent)), rule: kwh(shed(sd.rule_based)) },
        { label: `Critical load served (${view.site.critical_load_name})`, agent: pct(sd.agent.critical_served_pct), rule: pct(sd.rule_based.critical_served_pct) },
      ],
      cost: costLine(sd, currency, view.spec, "Realized cost on the same-day weather, counting the energy left in the battery."),
    };
  }

  /* The executed plan hour by hour, for the CSV download. */
  function csvRows(view) {
    const sd = view.same_day;
    const run = sd.agent;
    const rows = [["hour", "solar kW", "load kW", "battery kW (+ charging, - discharging)", "battery % (end of hour)", "grid import kW", "grid export kW"]];
    for (let h = 0; h < 24; h += 1) rows.push([hourText(h), sd.pv_actual_kw[h], sd.load_actual_kw[h], run.battery_kw[h], run.soc_pct[h], run.grid_import_kw[h], run.grid_export_kw[h]]);
    return rows;
  }

  function csvText(rows) {
    const cell = (v) => {
      const text = String(v);
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    return `${rows.map((row) => row.map(cell).join(",")).join("\r\n")}\r\n`;
  }

  /* --- try pure: end --------------------------------------------------------------------------- */

  function stat(label, value, explain) {
    return make("div", { class: "stat" }, make("p", { class: "stat-label" }, label), make("p", { class: "stat-value" }, value), explain ? make("p", { class: "stat-explain" }, explain) : null);
  }

  /* --- the main button and the pipeline strip ------------------------------------------------------ */

  function currentButtonState() {
    return runButtonState($("goal").value, recordedFor(state.chips, state.siteId, $("day").value), state.live);
  }

  function updateButton() {
    const s = currentButtonState();
    const button = $("run-button");
    button.textContent = s.label;
    button.disabled = s.disabled || state.busy;
    button.dataset.mode = s.mode;
    $("run-hint").textContent = s.hint;
    const chipId = s.chip ? s.chip.id : null;
    if (chipId !== state.chipId) {
      state.chipId = chipId;
      renderChips();
    }
  }

  function setPipeline(p, kind) {
    const strip = $("try-pipeline");
    strip.hidden = false;
    strip.dataset.kind = kind || strip.dataset.kind || "";
    for (const [stage, label] of PIPELINE_STAGES) {
      const item = strip.querySelector(`[data-stage="${stage}"]`);
      const value = p[stage];
      item.className = `stage ${value}`;
      item.querySelector(".stage-state").textContent = { idle: "waiting", active: "working", done: "done", fail: stage === "physics" ? "failed: back to the AI" : "failed" }[value];
      item.setAttribute("aria-label", `${label}: ${item.querySelector(".stage-state").textContent}`);
    }
    const repairs = $("try-repairs");
    repairs.textContent = p.repairs ? `Repairs: ${p.repairs}. The checks failed (${p.failed.map((c) => CHECK_NAMES[c] || c).join("; ")}), so the AI fixed its rules and resubmitted.` : "";
  }

  function buildPipeline() {
    $("try-pipeline").replaceChildren(
      ...PIPELINE_STAGES.map(([stage, label]) => make("li", { class: "stage idle", "data-stage": stage }, make("span", { class: "stage-dot", "aria-hidden": "true" }), make("span", { class: "stage-name" }, label), make("span", { class: "stage-state" }, "waiting")))
    );
  }

  /* --- the result ------------------------------------------------------------------------------------ */

  function metaLine(view) {
    const m = view.meta || {};
    const parts = [];
    if (view.kind === "replay" && m.source_run) parts.push(`Recorded run ${m.source_run}`);
    if (m.model) parts.push(`${m.model}, reasoning ${m.reasoning_effort || "medium"}`);
    if (m.tokens_in) parts.push(`${Number(m.tokens_in).toLocaleString()} tokens in, ${Number(m.tokens_out).toLocaleString()} out`);
    if (m.llm_cost_usd !== undefined && m.llm_cost_usd !== null) parts.push(`AI calls cost about $${Number(m.llm_cost_usd).toFixed(3)}`);
    if (m.latency_s) parts.push(`${Number(m.latency_s).toFixed(1)} s`);
    return make("p", { class: "meta-line" }, parts.join(" · "));
  }

  function solveResults(view) {
    return view.trace.filter((e) => e.kind === "tool_result" && e.name === "solve_plan").map((e) => e.data.result || {});
  }

  function mark(ok, label) {
    if (ok === null) return make("span", { class: "not-run" }, `– ${label}`);
    return make("span", { class: ok ? "ok" : "fail" }, `${ok ? "✓ passed" : "✗ failed"}`);
  }

  function gateSection(view) {
    const gate = view.gate;
    const attempts = gate.attempts.map((attempt) => {
      const failed = new Map(attempt.failed_checks.map((f) => [f.check, f.message]));
      const items = Object.entries(CHECK_NAMES).map(([check, name]) => {
        if (failed.has(check)) return make("li", {}, mark(false), ` ${name}: `, failed.get(check));
        if (check === "review" && failed.size && !attempt.final) return make("li", {}, mark(null, "not run yet"), ` ${name}`);
        return make("li", {}, mark(true), ` ${name}`);
      });
      const after = failed.size && !attempt.final ? make("p", { class: "hint" }, "The agent fixed this and submitted again.") : null;
      return make("li", {}, make("span", { class: "step-title" }, `Attempt ${attempt.attempt}`), make("ul", { class: "checks" }, items), after);
    });
    const reviews = gate.reviews.map((r) =>
      make(
        "p",
        {},
        "Independent reviewer (a separate AI call that sees only the goal, the site and the spec): ",
        r.approved === true ? make("span", { class: "ok" }, "✓ approved") : r.approved === false ? make("span", { class: "fail" }, "✗ raised issues") : make("span", { class: "not-run" }, "no verdict"),
        r.issues && r.issues.length ? make("ul", { class: "list" }, r.issues.map((i) => make("li", {}, `[${i.kind}] ${i.detail}`))) : null
      )
    );
    const lastAttempt = gate.attempts[gate.attempts.length - 1];
    let outcome;
    if (!gate.outcome) outcome = make("p", {}, `Plan source: ${(view.meta && view.meta.plan_source) || "unknown"}`);
    else if (gate.outcome.decision === "accepted") outcome = make("p", {}, make("strong", {}, `Accepted: plan ${lastAttempt ? lastAttempt.plan_id : ""}`));
    else outcome = make("p", {}, make("strong", {}, `Outcome: ${gate.outcome.decision.replaceAll("_", " ")}.`), " ", gate.outcome.message || "");
    return make("div", {}, make("ol", { class: "list" }, attempts), reviews, outcome);
  }

  function stepsFold(view) {
    const steps = [];
    if (view.checklist.length) steps.push(make("li", {}, make("span", { class: "step-title" }, "Listed what the goal requires"), make("ul", { class: "list", lang: view.goal.language || null }, view.checklist.map((item) => make("li", {}, item)))));
    const read = [...new Set(view.trace.filter((e) => e.kind === "tool_call" && INFO_TOOLS[e.name]).map((e) => INFO_TOOLS[e.name]))];
    if (read.length) steps.push(make("li", {}, make("span", { class: "step-title" }, `Read ${read.join(", ")}`)));
    const solved = solveResults(view).map((r) => {
      if (r.error) return make("li", {}, `Rejected spec: ${(r.spec_errors || [r.error]).join("; ")}`);
      const verdict = r.valid ? "verified" : `not verified (${r.total_violations} problems)`;
      const shortfalls = r.shortfalls && r.shortfalls.length ? `, ${r.shortfalls.length} goal(s) cannot be fully met` : "";
      return make("li", {}, `Plan ${r.plan_id}: ${verdict}, expected cost ${money(r.energy_cost, view.site.currency)}${shortfalls}`);
    });
    steps.push(make("li", {}, make("span", { class: "step-title" }, "Solved the plan for the rules"), solved.length ? make("ul", { class: "list" }, solved) : null));
    steps.push(make("li", {}, make("span", { class: "step-title" }, "Submitted the plan for checking"), gateSection(view)));
    if (view.explanation) steps.push(make("li", {}, make("span", { class: "step-title" }, "Explained the plan (in the goal's language)"), make("p", { class: "explanation", lang: view.goal.language || null }, view.explanation)));
    return make(
      "details",
      { class: "fold" },
      make("summary", {}, "Agent steps"),
      make("ol", { class: "steps" }, steps),
      make("details", {}, make("summary", {}, `All ${view.trace.length} recorded steps`), make("ol", { class: "list" }, view.trace.map((e) => make("li", {}, `${e.kind}${e.name ? ` · ${e.name}` : ""}${e.summary ? `: ${e.summary}` : ""}`))))
    );
  }

  function rulesSection(view) {
    const codes = VPDay.specCodes(view.spec);
    const specText = view.spec && Object.keys(view.spec).length ? JSON.stringify(view.spec, null, 2) : "{}  (no constraints: minimise cost only)";
    return make(
      "section",
      { class: "result-block", "aria-label": "The rules" },
      make("h4", {}, "The rules the AI wrote"),
      make("ul", { class: "rule-chips" }, codes.length ? codes.map((code) => make("li", { class: "rule-chip" }, goalWords(code, view.spec))) : make("li", { class: "rule-chip" }, "no rules: the lowest cost only")),
      make(
        "details",
        { class: "fold" },
        make("summary", {}, "The exact rules the AI wrote"),
        make("pre", { class: "spec", "aria-label": "Spec as JSON" }, specText),
        view.assumptions.length ? make("div", {}, make("h4", {}, "Assumptions the AI made"), make("ul", { class: "list", lang: view.goal.language || null }, view.assumptions.map((a) => make("li", {}, a)))) : null
      )
    );
  }

  function chartsFold(view) {
    const fold = make("details", { class: "fold" }, make("summary", {}, "The plan on the day-ahead forecast (charts)"));
    let drawn = false;
    fold.addEventListener("toggle", () => {
      if (!fold.open || drawn) return;
      drawn = true;
      fold.append(VPCharts.socChart(view.plan, view.site), VPCharts.flowsChart(view.plan, view.site), VPCharts.priceChart(view.plan, view.site));
    });
    return fold;
  }

  function comparisonPanel(view) {
    const c = comparisonRows(view, VPDay.specCodes(view.spec));
    const met = (ok) => make("span", { class: ok ? "ok" : "fail" }, ok ? "✓ met" : "✗ missed");
    const row = (label, agent, rule) => make("tr", {}, make("th", { scope: "row" }, label), make("td", {}, agent), make("td", {}, rule));
    return make(
      "section",
      { class: "result-block compare", "aria-label": "Compared with a rule-based battery" },
      make("h4", {}, "Compared with a rule-based battery"),
      make("p", { class: "hint" }, "Both batteries run on the same-day weather, from the same charge. The rule-based battery charges from solar surplus and discharges to cover demand, ignoring prices and goals."),
      make("div", { class: "table-wrap" }, make(
        "table",
        { class: "compare-table" },
        make("thead", {}, make("tr", {}, make("th", { scope: "col" }, ""), make("th", { scope: "col" }, "VoltProof"), make("th", { scope: "col" }, "Rule-based battery"))),
        make("tbody", {}, ...c.rules.map((r) => row(r.label, met(r.agent), met(r.rule))), ...c.facts.map((f) => row(f.label, f.agent, f.rule)))
      )),
      make("p", { class: "compare-cost" }, make("strong", { class: c.cost.tone }, c.cost.value), c.cost.same ? ". " : " than the rule-based battery. ", c.cost.explain)
    );
  }

  function csvButton(view) {
    const button = make("button", { type: "button", class: "secondary" }, "Download plan (CSV)");
    button.addEventListener("click", () => {
      const blob = new Blob([csvText(csvRows(view))], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = make("a", { href: url, download: `voltproof-${view.site.site_id}-${view.day.date}.csv`, class: "offscreen" });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    });
    return make("div", { class: "result-actions" }, button, make("p", { class: "hint" }, "Hour by hour as executed on the same-day weather: solar, load, battery power and charge, grid import and export."));
  }

  function renderResult(view, { notice, file } = {}) {
    VPCharts.destroyAll();
    const head = make(
      "div",
      { class: "result-head" },
      make("div", { class: "badges" }, make("span", { class: "badge" }, view.label)),
      make("p", { class: "checked-against" }, view.checked_against),
      metaLine(view),
      view.plan_notice ? make("p", { class: "fallback", role: "note" }, view.plan_notice) : null
    );
    const parts = [notice ? make("p", { class: "fallback", role: "status" }, notice) : null, head];
    if (!view.plan || !view.same_day) {
      parts.push(make("p", {}, make("span", { class: "fail" }, "✗ No accepted plan."), " Nothing would be sent to the battery; the agent's steps show what happened."), stepsFold(view));
      $("conversation").replaceChildren(...parts.filter(Boolean));
      return;
    }
    const player = make("div", { class: "try-player day-panel", "aria-label": "The result, hour by hour", role: "region" });
    parts.push(player, comparisonPanel(view), rulesSection(view), stepsFold(view), chartsFold(view), csvButton(view));
    $("conversation").replaceChildren(...parts.filter(Boolean));
    VPDay.mount(player, [{ view, file: file || null, language: view.goal.language || null }], { id: "try", loop: false });
  }

  function emptyResult() {
    $("conversation").replaceChildren(make("p", { class: "result-empty" }, "Your result plays here, hour by hour: the energy flows, the rules drawn on the day, and the rules checked."));
  }

  /* --- recorded runs --------------------------------------------------------------------------------- */

  async function loadReplay(chip) {
    const response = await fetch(`replays/${encodeURIComponent(chip.file)}`, { cache: "no-cache" });
    if (!response.ok) throw new Error(`replay ${chip.id} not found`);
    return response.json();
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function playRecorded(chip, { notice, keepUrl } = {}) {
    const token = ++state.playToken;
    if (!keepUrl) history.replaceState(null, "", `#replay=${encodeURIComponent(chip.id)}`);
    status(notice ? "" : `Playing a recorded run (${chip.site_name}, ${chip.date}).`);
    let p = pipelineStart();
    setPipeline(p, "recorded");
    let view;
    try {
      view = await loadReplay(chip);
    } catch (err) {
      status(`Could not load the recorded run: ${err.message}`, true);
      return;
    }
    for (const event of [...traceEvents(view), { kind: "end", data: { status: view.plan ? "done" : "unavailable" } }]) {
      if (token !== state.playToken) return;
      p = pipelineStep(p, event);
      setPipeline(p);
      if (!reducedMotion) await sleep(RECORDED_STEP_MS);
    }
    if (token !== state.playToken) return;
    renderResult(view, { notice, file: `web/replays/${chip.file}` });
    setPipeline(pipelineStep(p, { kind: "replayed", data: { ok: Boolean(view.plan) } }));
  }

  /* --- typed goals: live API, with a recorded run as the fallback --------------------------------- */

  function reasonText(reason, body) {
    const minutes = body && body.retry_after_s ? Math.max(1, Math.ceil(body.retry_after_s / 60)) : null;
    switch (reason) {
      case "hourly_cap": return `this hour's live runs for all visitors are used up${minutes ? `; try again in about ${minutes} min` : ""}`;
      case "rate_limit": return `you have used this hour's 10 live runs${minutes ? `; try again in about ${minutes} min` : ""}`;
      case "daily_cap": return "today's live runs are used up";
      case "busy": return "two live runs are already in progress; try again in a minute";
      case "budget": return "the demo's AI budget is used up";
      case "live_mode_off":
      case "live_off": return "the live AI is switched off here";
      case "paused":
      case "busy_upstream": return "the model provider is busy, so live runs are paused for a few minutes";
      case "network": return "the live AI could not be reached";
      default: return "the live run failed";
    }
  }

  /* The one-line live status under the goal box (status: /api/status, or null when unreachable). */
  function liveNote(live) {
    if (!live) return "Live AI not available here: pick a recorded goal below to play a recorded run.";
    if (!live.live) return `Live AI unavailable (${reasonText(live.reason)}${live.paused_until ? `, until ${new Date(live.paused_until).toLocaleTimeString()}` : ""}): pick a recorded goal below to play a recorded run.`;
    if (live.per_visitor_limit === false) return `Live AI on: ${live.runs_left_this_hour} live runs left this hour (shared by all visitors).`;
    return `Live AI on: ${live.runs_left_this_hour} live runs left for you this hour.`;
  }

  async function fallBack(reason, body, started) {
    const day = $("day").value;
    const mine = state.chips.filter((c) => c.site_id === state.siteId);
    const fallback = mine.find((c) => c.date === day) || mine[0];
    const notice = `Live run unavailable: ${reasonText(reason, body)}. Showing a recorded run for this site instead, labelled "replay (recorded run)". Your goal was not ${started ? "completed" : "run"}.`;
    status(notice, true);
    if (fallback) await playRecorded(fallback, { notice });
    refreshLive();
  }

  /* While the agent works: one plain line per step, text only. */
  function liveStepText(kind, data) {
    if (kind === "tool_call") {
      if (data.name === "set_checklist") return "Listing what the goal requires";
      if (INFO_TOOLS[data.name]) return `Reading ${INFO_TOOLS[data.name]}`;
      if (data.name === "solve_plan") return "Writing the rules and solving a plan";
      if (data.name === "finalize") return "Submitting the plan for checking";
      return `Calling ${data.name}`;
    }
    if (kind === "tool_result") {
      const r = data.result || {};
      if (r.error) return `Tool refused: ${r.error}`;
      if (data.name === "solve_plan" && r.plan_id) return `Plan ${r.plan_id}: ${r.valid ? "verified" : "not verified"}, expected cost ${money(r.energy_cost, r.currency)}`;
      if (data.name === "finalize" && r.failed_checks && r.failed_checks.length) return `Checks not passed: ${r.failed_checks.map((c) => CHECK_NAMES[c] || c).join("; ")}. The AI is fixing it.`;
      return null;
    }
    if (kind === "review") return data.approved === true ? "Independent reviewer: ✓ approved" : data.approved === false ? "Independent reviewer: ✗ raised issues" : "Independent reviewer: no verdict";
    return null;
  }

  function streamRun(runId) {
    return new Promise((resolve) => {
      const log = make("ol", { class: "live-log", "aria-label": "What the agent is doing" });
      $("conversation").replaceChildren(make("div", { class: "result-head" }, make("div", { class: "badges" }, make("span", { class: "badge" }, "live run")), make("p", { class: "checked-against" }, "The agent is working: it writes rules, the optimizer plans, and the checks run.")), log);
      let p = pipelineStart();
      setPipeline(p, "live");
      const source = new EventSource(`api/plan/${encodeURIComponent(runId)}/events`);
      let ended = false;
      const add = (kind) => (event) => {
        const data = JSON.parse(event.data);
        p = pipelineStep(p, { kind, name: data.name, data });
        setPipeline(p);
        const text = liveStepText(kind, data);
        if (text) log.append(make("li", {}, text));
      };
      for (const kind of ["tool_call", "tool_result", "review"]) source.addEventListener(kind, add(kind));
      source.addEventListener("end", async (event) => {
        ended = true;
        source.close();
        const end = JSON.parse(event.data);
        p = pipelineStep(p, { kind: "end", data: end });
        setPipeline(p);
        if (end.status !== "done") {
          resolve(fallBack(end.reason, null, true));
          return;
        }
        try {
          const result = await (await fetch(`api/plan/${encodeURIComponent(runId)}`)).json();
          status("Live run finished.");
          renderResult(result.view);
          setPipeline(pipelineStep(p, { kind: "replayed", data: { ok: Boolean(result.view && result.view.plan) } }));
        } catch (_err) {
          await fallBack("network", null, true);
        }
        refreshLive();
        resolve();
      });
      source.onerror = () => {
        if (ended) return;
        ended = true;
        source.close();
        resolve(fallBack("network", null, true));
      };
    });
  }

  async function runLive(goal) {
    let response;
    try {
      response = await fetch("api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ site: state.siteId, day: $("day").value, goal }) });
    } catch (_err) {
      await fallBack("network");
      return;
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.run_id) {
      await fallBack(body.reason || (response.status === 422 ? "invalid" : "error"), body);
      return;
    }
    history.replaceState(null, "", location.pathname);
    status("Running live: the agent is working.");
    await streamRun(body.run_id);
  }

  async function submitGoal(event) {
    event.preventDefault();
    const s = currentButtonState();
    if (s.disabled || state.busy) return; // the button says why
    state.busy = true;
    updateButton();
    try {
      if (s.mode === "recorded") await playRecorded(s.chip);
      else if (s.mode === "live") await runLive($("goal").value.trim());
    } finally {
      state.busy = false;
      updateButton();
    }
  }

  /* --- live status ---------------------------------------------------------------------------------- */

  async function refreshLive() {
    try {
      const response = await fetch("api/status", { cache: "no-store" });
      state.live = response.ok ? await response.json() : null;
    } catch (_err) {
      state.live = null;
    }
    $("live-note").textContent = liveNote(state.live);
    updateButton();
  }

  function updateCount() {
    $("goal-count").textContent = `${$("goal").value.length} / ${MAX_GOAL}`;
  }

  /* --- start ------------------------------------------------------------------------------------ */

  async function start() {
    buildPipeline();
    $("goal-form").addEventListener("submit", submitGoal);
    $("goal").addEventListener("input", () => {
      updateCount();
      updateButton();
    });
    $("day").addEventListener("change", updateButton);
    try {
      const response = await fetch("replays/index.json", { cache: "no-cache" });
      state.chips = (await response.json()).chips;
    } catch (_err) {
      status("Could not load the recorded runs.", true);
      return;
    }
    const seen = new Map();
    for (const chip of state.chips) if (!seen.has(chip.site_id)) seen.set(chip.site_id, chip.site);
    state.sites = [...seen.values()];
    const wanted = new URLSearchParams(location.hash.slice(1)).get("replay");
    const chip = state.chips.find((c) => c.id === wanted);
    selectSite(chip ? chip.site_id : state.sites[0].site_id);
    emptyResult();
    await refreshLive();
    if (chip) {
      // a shared link to a recorded run: fill the goal and play it
      fillFromChip(chip);
      await playRecorded(chip, { keepUrl: true });
    }
  }

  // The landing page and the day player reuse these helpers, so they speak exactly like the result here
  window.VPPage = Object.freeze({ money, goalWords, costLine });

  start();
})();

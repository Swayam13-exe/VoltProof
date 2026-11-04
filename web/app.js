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

  const CHECK_NAMES = {
    verified_plan: "Plan verified: physics checked, and every shortfall stated",
    coverage: "Every checklist item is covered by the spec",
    grounded_explanation: "Every number in the explanation comes from the tools or the goal",
    review: "The independent AI reviewer approves the spec",
  };
  const INFO_TOOLS = { get_site_info: "the site", get_forecast: "the day-ahead forecast", get_tariff: "the tariff" };
  const KIND_TAGS = { english: "EN", hard: "hard", multilingual: null };

  const state = { chips: [], sites: [], siteId: null, chipId: null, playToken: 0, live: null, busy: false, bill: null }; // bill: a confirmed tariff from a bill, in use

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

  const TARIFF_TAGS = { time_of_use: "TOU", hourly_dynamic: "hourly", flat: "flat" };

  /* The sites as the hero's switcher buttons (same CSS), and one line of facts about the chosen one. */
  function renderSites() {
    $("site-list").replaceChildren(
      ...state.sites.map((site) => {
        const button = make(
          "button",
          { type: "button", class: "site-btn", "aria-pressed": String(site.site_id === state.siteId), "aria-label": `${site.name} (${siteFacts(site)})` },
          VPDay.siteIcon(site.site_type, "site-icon"),
          make("span", { class: "site-name" }, make("span", { class: "name-full" }, site.name), make("span", { class: "name-short" }, VPDay.shortSiteName(site.name))),
          make("span", { class: "site-lang", lang: "en" }, TARIFF_TAGS[site.tariff_kind] || site.tariff_kind)
        );
        button.addEventListener("click", () => {
          if (site.site_id !== state.siteId) selectSite(site.site_id);
        });
        return button;
      })
    );
    const site = state.sites.find((s) => s.site_id === state.siteId);
    $("site-facts").textContent = site ? `${siteFacts(site)} · critical load: ${site.critical_load_name} (${site.critical_load_kw} kW)` : "";
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
        button.dataset.chip = chip.id;
        button.addEventListener("click", () => fillFromChip(chip));
        return make("li", {}, button);
      })
    );
  }

  function markChips() {
    for (const button of $("chips").querySelectorAll(".chip")) button.setAttribute("aria-pressed", String(button.dataset.chip === state.chipId));
  }

  function selectSite(siteId) {
    state.siteId = siteId;
    renderSites();
    renderDays();
    renderChips();
    billSiteChanged();
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
    markChips();
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

  /* The stages in the order the agent really runs them (SPEC §6.10): the AI writes the rules, the
     optimizer plans, the physics and coverage checks run, and only then the independent AI reviewer;
     a failed check or a rejection sends the run back to the AI. [key, label, short label, who] */
  const PIPELINE_STAGES = [
    ["rules", "AI writes the rules", "Rules", "ai"],
    ["optimizer", "Optimizer plans the day", "Optimizer", "tool"],
    ["checks", "Physics and coverage checks", "Checks", "tool"],
    ["review", "Independent AI review", "Review", "ai"],
    ["accepted", "Accepted", "Accepted", "tool"],
    ["replay", "Replay on the same-day weather", "Replay", "tool"],
  ];
  const PIPELINE_TOOLS = new Set(["set_checklist", "solve_plan", "finalize"]);
  const STAGE_DONE_MS = 700; // recorded runs: a stage finishing holds this long
  const STAGE_ACTIVE_MS = 200; // and a stage starting this long

  /* Recorded goals for the chosen site and day. */
  function recordedFor(chips, siteId, day) {
    return chips.filter((c) => c.site_id === siteId && c.date === day);
  }

  /* How long the recorded runs took (index.json latency_s), whole seconds: {min, max} or null. */
  function recordedTiming(chips) {
    const seconds = chips.map((c) => Number(c.latency_s)).filter((s) => s > 0);
    return seconds.length ? { min: Math.floor(Math.min(...seconds)), max: Math.ceil(Math.max(...seconds)), runs: seconds.length } : null;
  }

  /* What the one main button offers for this text: a free replay when it is a recorded goal of this
     site and day, a live run when it is typed and live runs are available, nothing when empty. With a
     tariff from a bill in use (`bill`), only live runs: the recorded runs used the site's tariff, so a
     recorded goal runs live too (its chip stays marked), and the label says it is the bill's tariff. */
  function runButtonState(text, recorded, live, timing, bill = false) {
    const goal = String(text || "").trim();
    if (!goal) return { mode: "empty", label: "Plan my day", disabled: true, hint: bill ? "Type a goal or pick one above: it is planned live with the tariff from your bill." : "Type a goal above or pick a recorded one." };
    const chip = recorded.find((c) => c.goal.trim() === goal) || null;
    if (chip && !bill) return { mode: "recorded", label: "Play recorded run (free)", disabled: false, hint: "A recorded run of the agent for this goal, replayed at no cost.", chip };
    const left = live && live.live ? Number(live.runs_left_this_hour) : 0;
    if (live && live.live && left > 0) {
      const took = timing ? ` The ${timing.runs} recorded runs took ${timing.min}-${timing.max} s (web/replays/index.json).` : "";
      const shared = live.per_visitor_limit === false ? " Live runs are shared by all visitors." : "";
      const tariff = bill ? " Planned with the tariff from your bill; the rule-based battery is compared on the same tariff." : "";
      const label = bill ? `Run live with your bill's tariff (${left} left)` : `Run live (${left} left this hour)`;
      return { mode: "live", label, disabled: false, hint: `Runs the AI on your goal now.${tariff}${took}${shared}`, chip };
    }
    const why = !live ? "the live AI is not available here" : !live.live ? reasonText(live.reason, null, live) : "no live runs are left this hour";
    if (bill) return { mode: "unavailable", label: "Live runs unavailable", disabled: true, hint: "Needs a live run.", chip }; // the reason is in the note above and the status line below
    return { mode: "unavailable", label: "Live runs unavailable: play a recorded run", disabled: true, hint: `Live runs unavailable (${why}). Pick a recorded goal above.` };
  }

  /* The line under the recorded goals while a bill's tariff is in use (nothing with the site's tariff).
     Live runs unavailable: it also offers "Use the site's tariff", which plays the recorded run instead. */
  const BILL_CHIP_LINE = "Recorded runs used the site's own tariff. With your bill's tariff, a recorded goal runs live and uses one of your live runs.";

  function billChipNote(bill, live) {
    if (!bill) return { show: false, text: "", offerSiteTariff: false };
    if (live && live.live && Number(live.runs_left_this_hour) > 0) return { show: true, text: BILL_CHIP_LINE, offerSiteTariff: false };
    const why = !live ? "the live AI is not available here" : !live.live ? reasonText(live.reason, null, live) : "no live runs are left this hour";
    return { show: true, text: `${BILL_CHIP_LINE} Live runs are unavailable right now (${why}).`, offerSiteTariff: true };
  }

  function pipelineStart() {
    return { rules: "idle", optimizer: "idle", checks: "idle", review: "idle", accepted: "idle", replay: "idle", repairs: 0, failed: [] };
  }

  /* One agent event -> the stages it moves. Events are {kind, name, data}: live events from the
     stream, or the recorded trace (traceEvents). The reviewer only runs once the checks pass, so its
     verdict also closes the checks; a failed check or a rejection is a repair, back to the AI. */
  function pipelineStep(previous, event) {
    const s = { ...previous, failed: [...previous.failed] };
    const result = (event.data && event.data.result) || {};
    if (event.kind === "tool_call") {
      if (event.name === "set_checklist" && s.rules === "idle") s.rules = "active";
      else if (event.name === "solve_plan") {
        s.rules = "done";
        s.optimizer = "active";
      } else if (event.name === "finalize") s.checks = "active";
    } else if (event.kind === "tool_result") {
      if (event.name === "solve_plan") {
        s.optimizer = result.error ? "fail" : "done";
        if (result.error) s.rules = "active";
      } else if (event.name === "finalize") {
        const failed = (result.failed_checks || []).map((f) => (typeof f === "string" ? f : f && f.check)).filter(Boolean);
        if (result.accepted) {
          s.checks = "done";
          if (s.review !== "done") s.review = "done";
          s.accepted = "done";
        } else if (failed.length) {
          s.repairs += 1;
          s.failed = failed;
          if (failed.includes("review")) {
            s.checks = "done";
            s.review = "fail";
          } else s.checks = "fail";
          if (!result.escalated) s.rules = "active";
        }
      }
    } else if (event.kind === "review") {
      s.checks = "done"; // the reviewer runs only after the checks passed
      s.review = event.data && event.data.approved === false ? "fail" : "done";
    } else if (event.kind === "gate") {
      // the gate's decision (api/live.py streams it after the run; recorded traces log it): accepted, or
      // escalated to the operator (escalated_with_plan / escalated_fallback)
      if (event.data && event.data.decision === "accepted") accept(s);
      else s.accepted = "fail";
    } else if (event.kind === "end") {
      s.replay = event.data && event.data.status === "done" ? "active" : "fail";
    } else if (event.kind === "replayed") {
      // the final result: a successful one with an accepted plan closes every stage, whatever the stream missed
      const ok = Boolean(event.data && event.data.ok);
      if (ok && event.data.accepted) accept(s);
      s.replay = ok ? "done" : "fail";
    }
    return s;
  }

  /* An accepted plan: the rules, the plan and the checks passed, and so did the review (unless the
     gate accepted it on its final attempt with the reviewer's dispute flagged). */
  function accept(s) {
    s.rules = "done";
    s.optimizer = "done";
    s.checks = "done";
    if (s.review !== "fail") s.review = "done";
    s.accepted = "done";
  }

  /* Did the run end with the gate accepting a plan (view.gate.outcome)? */
  function gateAccepted(view) {
    return Boolean(view && view.gate && view.gate.outcome && view.gate.outcome.decision === "accepted");
  }

  /* The states a run passes through, each with how long to hold it when replaying a recorded run. */
  function pipelineFrames(events, doneMs = STAGE_DONE_MS, activeMs = STAGE_ACTIVE_MS) {
    let state = pipelineStart();
    return events.map((event) => {
      const next = pipelineStep(state, event);
      const done = PIPELINE_STAGES.some(([k]) => next[k] === "done" && state[k] !== "done");
      const active = PIPELINE_STAGES.some(([k]) => next[k] === "active" && state[k] !== "active");
      state = next;
      return { state: next, delayMs: done ? doneMs : active ? activeMs : 0 };
    });
  }

  /* The recorded trace as pipeline events (only the steps the stages show). */
  function traceEvents(view) {
    return (view.trace || [])
      .filter((e) => e.kind === "review" || (e.kind === "gate" && e.name !== "finalize") || ((e.kind === "tool_call" || e.kind === "tool_result") && PIPELINE_TOOLS.has(e.name)))
      .map((e) => (e.kind === "gate" ? { kind: "gate", name: e.name, data: { decision: e.name } } : { kind: e.kind, name: e.name, data: e.data || {} }));
  }

  /* The operator's approval of a result (UI only: nothing is sent anywhere). */
  function approvalState(view, approved) {
    if (!view || !view.plan) return { show: false };
    if (!view.plan.verified) return { show: true, state: "unverified", text: "Not verified: this plan would not be sent to the battery.", button: null };
    if (approved) return { show: true, state: "approved", text: "Approved by you. Marked as sent to the battery (demo: no device is controlled).", button: null };
    return { show: true, state: "waiting", text: "Verified - waiting for your approval", button: "Approve plan" };
  }

  /* Why the explanation's cost and the table's cost differ (voltproof/core): the explanation quotes the
     plan's energy_cost from solve_plan, on the day-ahead forecast (optimizer.py: import price x import -
     export price x export + battery wear); the table's realized_cost replays the same battery schedule on
     the same-day weather with the same grid and wear costs plus the change in stored energy, each kWh
     valued at the day's average import price x discharge efficiency (metrics.trajectory_total_cost,
     rules.terminal_value_per_kwh). Load switched off is a separate row, not priced. */
  const COST_NOTE = "About the costs: a cost in the AI's explanation is the plan's expected cost on the day-ahead forecast. The table replays the same battery schedule on the same-day weather and adds the change in battery energy over the day, valued at the day's average import price × discharge efficiency.";

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

  /* --- the bill reader (SPEC §6.12): read it, then confirm it. The server checks every edit with the
     same functions it uses before a run (api/tariff/check); these only shape the table. ------------ */

  const BILL_KINDS = { time_of_use: "time-of-use", flat: "flat" };

  /* The confirm table's rows from what the model read (raw values, as text, so a wrong value shows). */
  function billRows(answer) {
    const periods = answer && Array.isArray(answer.periods) ? answer.periods : [];
    const text = (v) => (v === null || v === undefined ? "" : String(v));
    return periods.map((p) => ({ start: text(p && p.start_hour), end: text(p && p.end_hour), rate: text(p && p.import_rate) }));
  }

  /* The table as the tariff the API checks. A number stays a number; anything else goes as typed and
     the server says what is wrong with it. */
  function billTariff(kind, rows, exportText, currency) {
    const value = (v) => {
      const t = String(v === null || v === undefined ? "" : v).trim();
      return t === "" ? null : Number.isFinite(Number(t)) ? Number(t) : t;
    };
    return { kind, currency: currency || null, periods: rows.map((r) => ({ start_hour: value(r.start), end_hour: value(r.end), import_rate: value(r.rate) })), export_rate: value(exportText) };
  }

  /* Which table rows an error or warning names ("Row 2: ..."), as 0-based indexes. */
  function billRowFlags(messages) {
    const rows = new Set();
    for (const m of messages || []) {
      const match = /^Row (\d+)/.exec(m);
      if (match) rows.add(Number(match[1]) - 1);
    }
    return rows;
  }

  /* The 24-hour price bar: each hour's height as a share of the dearest rate, and its band for colour
     (cheapest, dearest, in between; one band for a flat tariff; "missing" for an hour with no rate). */
  function billBars(hourly) {
    const rates = (hourly || []).filter((r) => typeof r === "number");
    if (!rates.length) return [];
    const max = Math.max(...rates);
    const min = Math.min(...rates);
    return hourly.map((rate, hour) => {
      if (typeof rate !== "number") return { hour, rate: null, height: 0, band: "missing" };
      return { hour, rate, height: rate / max, band: max === min ? "flat" : rate === min ? "cheap" : rate === max ? "peak" : "normal" };
    });
  }

  /* One plain example of what the tariff means: "1 kWh at 18:00 costs ₹9.90" (the dearest hour). */
  function billExample(hourly, currency) {
    if (!hourly || hourly.length !== 24 || hourly.some((r) => typeof r !== "number")) return "";
    const top = Math.max(...hourly);
    if (hourly.every((r) => r === top)) return `1 kWh costs ${money(top, currency)} at any hour`;
    return `1 kWh at ${hourText(hourly.indexOf(top))} costs ${money(top, currency)}`;
  }

  /* The result's tariff line: a run planned with a bill's tariff says so, and that the rule-based
     battery was priced with the same tariff (api/live.py sets meta.tariff_source). */
  function tariffNote(view) {
    if (!view || !view.meta || view.meta.tariff_source !== "bill") return null;
    return { tag: "Tariff: from your bill", text: "Planned with the tariff from your bill. The comparison with the rule-based battery used the same tariff." };
  }

  /* Why a sample bill shows its recorded read instead of a live one (the API's `why`). */
  const RECORDED_WHY = {
    live_mode_off: "the live AI is off here",
    live_off: "the live AI is off here",
    paused: "the model provider is busy right now",
    budget: "the demo's AI budget is used up",
    rate_limit: "you have used this hour's bill reads",
    hourly_cap: "this hour's bill reads for all visitors are used up",
    daily_cap: "today's bill reads are used up",
    busy: "other bills are being read right now",
    error: "the live read failed",
  };

  /* The line above the confirm table: who read the bill. A recorded read says so and why (never "read
     just now"); a live read says "Read by Muse Spark Open in X s" (the API's label). */
  function billSourceText(body) {
    if (body.source === "recorded") {
      const why = RECORDED_WHY[body.why];
      return `Recorded read of the sample bill${why ? `, because ${why}` : ""}. It was read earlier by Muse Spark Open (${body.recorded_in || "results/vision_probe/results.json"}).`;
    }
    return `${body.label || "Read by Muse Spark Open"}.`;
  }

  /* Why a bill read was refused, from the API's answer. */
  function billRefusalText(body) {
    const b = body || {};
    const minutes = b.retry_after_s ? Math.max(1, Math.ceil(b.retry_after_s / 60)) : null;
    const later = minutes ? `; try again in about ${minutes} min` : "";
    switch (b.reason) {
      case "rate_limit": return `You have used this hour's ${b.per_visitor_per_hour ? `${b.per_visitor_per_hour} ` : ""}bill reads${later}. The sample bills still work.`;
      case "hourly_cap": return `This hour's bill reads for all visitors are used up${later}. The sample bills still work.`;
      case "daily_cap": return "Today's bill reads are used up. The sample bills still work.";
      case "busy": return "Other bills are being read right now; try again in a minute.";
      default: return b.message || "The bill could not be read right now. Try again or use a sample bill.";
    }
  }

  /* --- try pure: end --------------------------------------------------------------------------- */

  function stat(label, value, explain) {
    return make("div", { class: "stat" }, make("p", { class: "stat-label" }, label), make("p", { class: "stat-value" }, value), explain ? make("p", { class: "stat-explain" }, explain) : null);
  }

  /* --- the main button and the pipeline strip ------------------------------------------------------ */

  function currentButtonState() {
    return runButtonState($("goal").value, recordedFor(state.chips, state.siteId, $("day").value), state.live, recordedTiming(state.chips), Boolean(state.bill));
  }

  function updateButton() {
    const s = currentButtonState();
    const button = $("run-button");
    button.textContent = s.label;
    button.disabled = s.disabled || state.busy;
    button.dataset.mode = s.mode;
    $("run-hint").textContent = s.hint;
    $("live-note").textContent = liveNote(state.live, Boolean(state.bill));
    const note = billChipNote(Boolean(state.bill), state.live);
    $("bill-chip-note").hidden = !note.show;
    $("bill-chip-text").textContent = note.text;
    $("bill-chip-site").hidden = !note.offerSiteTariff;
    const chipId = s.chip ? s.chip.id : null;
    if (chipId !== state.chipId) {
      state.chipId = chipId;
      markChips();
    }
  }

  function setPipeline(p, kind) {
    const strip = $("try-pipeline");
    strip.hidden = false;
    strip.dataset.kind = kind || strip.dataset.kind || "";
    for (const [stage, label, , who] of PIPELINE_STAGES) {
      const item = strip.querySelector(`[data-stage="${stage}"]`);
      const value = p[stage];
      item.className = `stage ${who} ${value}`;
      const words = { idle: "waiting", active: "working", done: "done", fail: stage === "review" ? "rejected: back to the AI" : stage === "checks" ? "failed: back to the AI" : "failed" }[value];
      item.querySelector(".stage-state").textContent = words;
      item.setAttribute("aria-label", `${label}: ${words}`);
    }
    $("try-repairs").textContent = p.repairs ? `Repairs: ${p.repairs}. ${p.failed.map((c) => CHECK_NAMES[c] || c).join("; ")}: not passed, so the AI fixed its rules and resubmitted.` : "";
  }

  function buildPipeline() {
    $("try-pipeline").replaceChildren(
      ...PIPELINE_STAGES.map(([stage, label, short, who]) =>
        make("li", { class: `stage ${who} idle`, "data-stage": stage },
          make("span", { class: "stage-dot", "aria-hidden": "true" }),
          make("span", { class: "stage-name" }, make("span", { class: "label-full" }, label), make("span", { class: "label-short", "aria-hidden": "true" }, short)),
          make("span", { class: "stage-state" }, "waiting")))
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
        make("pre", { class: "spec", "aria-label": "Spec as JSON" }, specText)
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
      make("p", { class: "compare-cost" }, make("strong", { class: c.cost.tone }, c.cost.value), c.cost.same ? ". " : " than the rule-based battery. ", c.cost.explain),
      make("p", { class: "cost-note" }, COST_NOTE)
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
    const tariff = tariffNote(view);
    const head = make(
      "div",
      { class: "result-head" },
      make("div", { class: "badges" }, make("span", { class: "badge" }, view.label), tariff ? make("span", { class: "tariff-tag" }, tariff.tag) : null),
      make("p", { class: "checked-against" }, view.checked_against),
      tariff ? make("p", { class: "tariff-note" }, tariff.text) : null,
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
    parts.push(player, approvalBlock(view), comparisonPanel(view), rulesSection(view), stepsFold(view), chartsFold(view), csvButton(view));
    $("conversation").replaceChildren(...parts.filter(Boolean));
    VPDay.mount(player, [{ view, file: file || null, language: view.goal.language || null }], { id: "try", loop: false, showGoal: false, cost: "none", checklistUnderFlow: true, assumptions: true });
  }

  /* "Verified - waiting for your approval", then the operator approves (UI only, SPEC §6.13). */
  function approvalBlock(view) {
    const first = approvalState(view, false);
    if (!first.show) return null;
    const text = make("p", { class: "approval-text", role: "status", tabindex: "-1" }, first.text);
    const block = make("div", { class: `approval ${first.state}` }, text);
    if (first.button) {
      const button = make("button", { type: "button", class: "button primary" }, first.button);
      button.addEventListener("click", () => {
        const done = approvalState(view, true);
        block.className = `approval ${done.state}`;
        text.textContent = done.text;
        button.remove();
        text.focus(); // keyboard users land on the confirmation
      });
      block.append(button);
    }
    return block;
  }

  function emptyResult(working = false) {
    // a new run clears the last result, including its approval
    $("conversation").replaceChildren(make("p", { class: "result-empty" }, working ? "Working: the result appears when the run finishes." : "Your result plays here, hour by hour: the energy flows, the rules drawn on the day, and the rules checked."));
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
    setPipeline(pipelineStart(), "recorded");
    emptyResult(true);
    let view;
    try {
      view = await loadReplay(chip);
    } catch (err) {
      status(`Could not load the recorded run: ${err.message}`, true);
      return;
    }
    const frames = pipelineFrames([...traceEvents(view), { kind: "end", data: { status: view.plan ? "done" : "unavailable" } }, { kind: "replayed", data: { ok: Boolean(view.plan), accepted: gateAccepted(view) } }]);
    for (const frame of frames.slice(0, -1)) {
      if (token !== state.playToken) return;
      setPipeline(frame.state);
      if (!reducedMotion && frame.delayMs) await sleep(frame.delayMs);
    }
    if (token !== state.playToken) return;
    renderResult(view, { notice, file: `web/replays/${chip.file}` });
    setPipeline(frames[frames.length - 1].state);
  }

  /* --- typed goals: live API, with a recorded run as the fallback --------------------------------- */

  function reasonText(reason, body, live) {
    const minutes = body && body.retry_after_s ? Math.max(1, Math.ceil(body.retry_after_s / 60)) : null;
    switch (reason) {
      case "hourly_cap": return `this hour's live runs for all visitors are used up${minutes ? `; try again in about ${minutes} min` : ""}`;
      case "rate_limit": return `you have used this hour's ${live && live.per_visitor_per_hour ? `${live.per_visitor_per_hour} ` : ""}live runs${minutes ? `; try again in about ${minutes} min` : ""}`;
      case "daily_cap": return "today's live runs are used up";
      case "busy": return "two live runs are already in progress; try again in a minute";
      case "budget": return "the demo's AI budget is used up";
      case "live_mode_off":
      case "live_off": return "the live AI is switched off here";
      case "paused":
      case "busy_upstream": return "the model provider is busy, so live runs are paused for a few minutes";
      case "network": return "the live AI could not be reached";
      case "tariff_invalid": return "the tariff from your bill did not pass the checks";
      default: return "the live run failed";
    }
  }

  /* The one-line live status under the goal box (status: /api/status, or null when unreachable). With a
     bill's tariff in use (`bill`) a recorded run is no way out, so the line gives only the reason. */
  function liveNote(live, bill = false) {
    if (!live) return bill ? "Live AI not available here." : "Live AI not available here: pick a recorded goal above to play a recorded run.";
    if (!live.live) {
      const why = `${reasonText(live.reason, null, live)}${live.paused_until ? `, until ${new Date(live.paused_until).toLocaleTimeString()}` : ""}`;
      return bill ? `Live AI unavailable (${why}).` : `Live AI unavailable (${why}): pick a recorded goal above to play a recorded run.`;
    }
    if (live.per_visitor_limit === false) return `Live AI on: ${live.runs_left_this_hour} live runs left this hour (shared by all visitors).`;
    return `Live AI on: ${live.runs_left_this_hour} live runs left for you this hour.`;
  }

  async function fallBack(reason, body, started) {
    const day = $("day").value;
    const mine = state.chips.filter((c) => c.site_id === state.siteId);
    const fallback = mine.find((c) => c.date === day) || mine[0];
    const tariff = state.bill ? ", which used the site's tariff, not your bill's" : "";
    const notice = `Live run unavailable: ${reasonText(reason, body, state.live)}. Showing a recorded run for this site instead${tariff}, labelled "replay (recorded run)". Your goal was not ${started ? "completed" : "run"}.`;
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
    if (kind === "gate") return data.decision === "accepted" ? `Accepted: plan ${data.plan_id} passed the checks` : "Not accepted: escalated to the operator";
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
      for (const kind of ["tool_call", "tool_result", "review", "gate"]) source.addEventListener(kind, add(kind));
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
          setPipeline(pipelineStep(p, { kind: "replayed", data: { ok: Boolean(result.view && result.view.plan), accepted: gateAccepted(result.view) } }));
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
      response = await fetch("api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ site: state.siteId, day: $("day").value, goal, tariff: state.bill ? state.bill.tariff : undefined }) });
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
    billUploadState();
    updateButton(); // also writes the live status line
  }

  function updateCount() {
    $("goal-count").textContent = `${$("goal").value.length} / ${MAX_GOAL}`;
  }

  /* --- the bill reader: samples, upload, the confirm table ---------------------------------------- */

  const BILL_MAX_BYTES = 2 * 1024 * 1024;
  const bill = { samples: [], sampleId: null, draft: null, checkSeq: 0, readSeq: 0, timer: null };

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    node.append(...children);
    return node;
  }

  function billStatus(text, warn = false) {
    $("bill-status").textContent = text;
    $("bill-status").classList.toggle("warn", warn);
  }

  function showTariffTags() {
    $("tariff-tag").hidden = !state.bill; // one tag, next to the panel's button
  }

  /* The two actions appear only once a bill has been read (a reading on the table). */
  function showBillActions() {
    $("bill-actions").hidden = !bill.draft;
  }

  /* The 24-hour price bar (SVG, text labels only). */
  function priceBar(hourly, currency) {
    const bars = billBars(hourly);
    if (!bars.length) return null;
    const top = 4;
    const base = 44;
    const rects = bars.map((b) => {
      const h = b.band === "missing" ? base - top : Math.max(2, b.height * (base - top));
      return svg("rect", { x: b.hour * 10 + 1, y: base - h, width: 8, height: h, rx: 1.5, class: `bar-${b.band}` });
    });
    const ticks = [0, 6, 12, 18, 24].map((h) => svg("text", { x: h * 10, y: 56, "text-anchor": h === 0 ? "start" : h === 24 ? "end" : "middle" }, `${h}:00`));
    const runs = [];
    bars.forEach((b) => {
      const last = runs[runs.length - 1];
      if (last && last.rate === b.rate) last.end = b.hour + 1;
      else runs.push({ start: b.hour, end: b.hour + 1, rate: b.rate });
    });
    const label = `Price by hour: ${runs.map((r) => `${hourText(r.start)} to ${hourText(r.end)} ${r.rate === null ? "no rate" : money(r.rate, currency)}`).join(", ")}`;
    return svg("svg", { viewBox: "0 0 240 60", role: "img", "aria-label": label }, ...rects, ...ticks);
  }

  function billInput(row, field, label, index) {
    const input = make("input", { class: "bill-input", type: "text", inputmode: "decimal", autocomplete: "off", "aria-label": `Row ${index + 1}: ${label}` });
    input.value = row[field];
    input.addEventListener("input", () => {
      row[field] = input.value;
      billEdited();
    });
    return input;
  }

  function renderBillRows() {
    const d = bill.draft;
    $("bill-rows").replaceChildren(
      ...d.rows.map((row, i) => {
        const remove = make("button", { type: "button", class: "bill-remove", "aria-label": `Remove row ${i + 1}`, title: "Remove this row" }, "✕");
        remove.addEventListener("click", () => {
          d.rows.splice(i, 1);
          renderBillRows();
          billEdited();
        });
        return make("tr", {}, make("td", {}, billInput(row, "start", "from hour", i)), make("td", {}, billInput(row, "end", "to hour", i)), make("td", {}, billInput(row, "rate", `rate per kWh (${d.currency})`, i)), make("td", {}, remove));
      })
    );
  }

  /* A reading (or a re-check): the table, then what the check says. */
  function renderBill(fresh) {
    const d = bill.draft;
    $("bill-confirm").hidden = false;
    if (fresh) {
      const kinds = { ...BILL_KINDS };
      if (d.kind && !kinds[d.kind]) kinds[d.kind] = `${d.kind} (not supported)`;
      $("bill-kind").replaceChildren(...Object.entries(kinds).map(([value, text]) => make("option", { value }, text)));
      $("bill-kind").value = d.kind && kinds[d.kind] ? d.kind : "time_of_use";
      $("bill-currency").textContent = d.currency;
      $("bill-export").value = d.exportText;
      $("bill-rate-head").textContent = `Rate per kWh (${d.currency})`;
      renderBillRows();
    }
    showBillCheck();
  }

  function showBillCheck() {
    const check = bill.draft.check;
    const flagged = billRowFlags(check.errors);
    [...$("bill-rows").children].forEach((tr, i) => {
      for (const input of tr.querySelectorAll("input")) input.setAttribute("aria-invalid", String(flagged.has(i)));
    });
    $("bill-export").setAttribute("aria-invalid", String(check.errors.some((e) => e.startsWith("The export rate"))));
    $("bill-errors").replaceChildren(...check.errors.map((e) => make("li", {}, e)));
    $("bill-warnings").replaceChildren(...check.warnings.map((w) => make("li", {}, w)));
    const fixed = check.fixed_charge;
    $("bill-fixed").textContent = fixed === null || fixed === undefined ? "" : `Fixed charge on the bill: ${amount(fixed)} ${bill.draft.currency} (shown only; it does not change the plan).`;
    const bar = priceBar(check.hourly, check.currency);
    $("bill-bar").replaceChildren(...(bar ? [bar] : []));
    $("bill-example").textContent = check.ok ? billExample(check.hourly, check.currency) : "";
    $("bill-use").disabled = !check.ok;
  }

  function billTariffNow() {
    const d = bill.draft;
    return billTariff($("bill-kind").value, d.rows, $("bill-export").value, d.currency);
  }

  /* Any edit: the tariff in use (if any) no longer matches, and the server checks the table again. */
  function billEdited() {
    if (state.bill) {
      state.bill = null;
      showTariffTags();
      updateButton();
    }
    $("bill-use").disabled = true;
    clearTimeout(bill.timer);
    bill.timer = setTimeout(checkBill, 250);
  }

  async function checkBill() {
    if (!bill.draft) return;
    const seq = ++bill.checkSeq;
    let check;
    try {
      const response = await fetch("api/tariff/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ site_id: state.siteId, day: $("day").value || null, tariff: billTariffNow() }) });
      check = response.ok ? await response.json() : null;
    } catch (_err) {
      check = null;
    }
    if (seq !== bill.checkSeq || !bill.draft) return;
    bill.draft.check = check || { ...bill.draft.check, ok: false, errors: ["The tariff could not be checked right now. Try again."], warnings: [] };
    showBillCheck();
  }

  function showReading(body) {
    const check = body.check;
    bill.checkSeq += 1;
    const answer = body.answer && typeof body.answer === "object" ? body.answer : {};
    if (check.unreadable || !Array.isArray(answer.periods)) {
      bill.draft = null;
      showBillActions();
      $("bill-confirm").hidden = true;
      $("bill-use").disabled = true;
      billStatus(check.errors[0] || "The bill could not be read.", true);
      return;
    }
    const site = state.sites.find((s) => s.site_id === state.siteId);
    const exportText = answer.export_rate === null || answer.export_rate === undefined ? "" : String(answer.export_rate);
    bill.draft = { kind: answer.tariff_kind || null, rows: billRows(answer), exportText, currency: String(answer.currency || (site && site.currency) || "").toUpperCase(), check };
    billStatus(check.ok ? "Read. Check the table below, then plan with it." : "Read, but something needs fixing before you can plan with it.", !check.ok);
    $("bill-source").textContent = billSourceText(body);
    renderBill(true);
    showBillActions();
  }

  async function readBill(payload) {
    const seq = ++bill.readSeq;
    bill.checkSeq += 1; // a check still on its way belongs to the last reading
    clearTimeout(bill.timer);
    bill.draft = null; // the actions wait for this reading
    showBillActions();
    $("bill-confirm").hidden = true;
    $("bill-use").disabled = true;
    billStatus(payload.sample_id && !(state.live && state.live.live) ? "Loading the recorded read." : "Reading the bill. This takes a few seconds.");
    let response;
    let body;
    try {
      response = await fetch("api/tariff-from-image", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ site_id: state.siteId, day: $("day").value || null, ...payload }) });
      body = await response.json().catch(() => ({}));
    } catch (_err) {
      if (seq === bill.readSeq) billStatus("The bill could not be sent. Check your connection and try again.", true);
      return;
    }
    if (seq !== bill.readSeq) return;
    if (!response.ok || !body.check) {
      billStatus(billRefusalText(body), true);
      return;
    }
    showReading(body);
  }

  function markSamples() {
    for (const button of $("bill-samples").querySelectorAll(".bill-sample")) button.setAttribute("aria-pressed", String(button.dataset.sample === bill.sampleId));
  }

  function pickSample(sample) {
    if (sample.site_id !== state.siteId) selectSite(sample.site_id);
    bill.sampleId = sample.id;
    markSamples();
    $("bill-file").value = "";
    $("bill-file-name").textContent = "";
    $("bill-preview").src = `api/bill-samples/${encodeURIComponent(sample.id)}.png`;
    $("bill-preview").hidden = false;
    readBill({ sample_id: sample.id });
  }

  function uploadBill(file) {
    if (!file) return;
    $("bill-file-name").textContent = file.name; // shown as text only
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      billStatus("Only PNG and JPEG images are accepted.", true);
      return;
    }
    if (file.size > BILL_MAX_BYTES) {
      billStatus("The image is larger than 2 MB. Take a smaller photo or crop it.", true);
      return;
    }
    bill.sampleId = null;
    markSamples();
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const url = String(reader.result);
      $("bill-preview").src = url; // a data: URL (img-src allows data:), kept in this page only
      $("bill-preview").hidden = false;
      readBill({ image_b64: url.slice(url.indexOf(",") + 1), mime: file.type });
    });
    reader.addEventListener("error", () => billStatus("The image could not be opened.", true));
    reader.readAsDataURL(file);
  }

  function useBill() {
    if (!bill.draft || !bill.draft.check.ok) return;
    state.bill = { tariff: billTariffNow() };
    showTariffTags();
    billStatus("This tariff is in use: type a goal and run it live.");
    updateButton();
    $("goal").focus();
  }

  function useSiteTariff() {
    state.bill = null;
    showTariffTags();
    $("bill-panel").hidden = true;
    $("bill-open").setAttribute("aria-expanded", "false");
    updateButton();
    $("bill-open").focus();
  }

  /* Another site: a bill's tariff in use belongs to the old site, so it is dropped; a reading on the
     table is checked again against the new site (its currency may not match). */
  function billSiteChanged() {
    if (state.bill) {
      state.bill = null;
      showTariffTags();
    }
    if (bill.draft) checkBill();
  }

  function billUploadState() {
    const on = Boolean(state.live && state.live.live);
    $("bill-file").disabled = !on;
    $("bill-upload-note").textContent = on ? "" : "Reading your own bill needs the live AI, which is not available here right now. The sample bills show recorded reads.";
  }

  async function setupBill() {
    let samples = [];
    try {
      const response = await fetch("api/bill-samples", { cache: "no-cache" });
      samples = response.ok ? await response.json() : [];
    } catch (_err) {
      samples = [];
    }
    if (!samples.length) return; // the page on its own (no API): no bill reader
    bill.samples = samples;
    $("bill-samples").replaceChildren(
      ...samples.map((sample) => {
        const button = make("button", { type: "button", class: "bill-sample", "aria-pressed": "false" }, make("img", { src: `api/bill-samples/${encodeURIComponent(sample.id)}.png`, alt: "", loading: "lazy" }), make("span", {}, sample.label));
        button.dataset.sample = sample.id;
        button.addEventListener("click", () => pickSample(sample));
        return make("li", {}, button);
      })
    );
    $("bill-open").hidden = false;
    $("bill-open").addEventListener("click", () => {
      const open = $("bill-panel").hidden;
      $("bill-panel").hidden = !open;
      $("bill-open").setAttribute("aria-expanded", String(open));
    });
    $("bill-file").addEventListener("change", (event) => uploadBill(event.target.files[0]));
    $("bill-kind").addEventListener("change", billEdited);
    $("bill-export").addEventListener("input", billEdited);
    $("bill-add").addEventListener("click", () => {
      bill.draft.rows.push({ start: "", end: "", rate: "" });
      renderBillRows();
      billEdited();
    });
    $("bill-use").addEventListener("click", useBill);
    $("bill-site").addEventListener("click", useSiteTariff);
    $("bill-chip-site").addEventListener("click", () => {
      useSiteTariff(); // the recorded goal in the box now plays its recorded run
      $("run-button").focus();
    });
    $("day").addEventListener("change", () => {
      if (bill.draft) checkBill();
    });
    billUploadState();
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
    await setupBill();
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

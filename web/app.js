/* VoltProof page: site cards, recorded-goal chips, the conversation with the agent.
   Replays come from replays/*.json (precomputed from recorded runs), so the page works with no API.
   A typed goal goes to the live API (api/plan); if that is unavailable, the page plays a recorded run
   for the same site instead, with a notice and the "replay (recorded run)" label.
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

  const state = { chips: [], sites: [], siteId: null, chipId: null, playToken: 0, live: null };

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

  function kwh(value) {
    return `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })} kWh`;
  }

  function status(text, warn = false) {
    const node = $("status");
    node.textContent = text;
    node.classList.toggle("warn", warn);
  }

  /* --- setup: sites, days, chips ------------------------------------------------------------- */

  function siteFacts(site) {
    const tariff = { time_of_use: "time-of-use tariff", hourly_dynamic: "hourly dynamic tariff", flat: "flat tariff" }[site.tariff_kind] || site.tariff_kind;
    return `${site.country} · ${site.currency} · ${tariff} · ${site.pv_kwp} kWp solar · ${site.battery_kwh} kWh battery`;
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
          make("span", { class: "site-name" }, site.name),
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
    if (!state.live || !state.live.live) return;
    const siteId = state.siteId;
    fetch(`api/days?site=${encodeURIComponent(siteId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((days) => {
        if (!days || siteId !== state.siteId) return;
        const keep = select.value;
        select.replaceChildren(...days.map((d) => make("option", { value: d.date }, `${d.date} · ${d.split} day${recorded.has(d.date) ? ", recorded runs" : ""}`)));
        if (keep) select.value = keep;
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
        button.addEventListener("click", () => playChip(chip, { instant: false }));
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

  /* --- the conversation ------------------------------------------------------------------------ */

  function userMessage(view) {
    return make(
      "article",
      { class: "message user", "aria-label": "Your goal" },
      make("p", { class: "who" }, "Goal"),
      make("p", { class: "goal-text", lang: view.goal.language || null }, view.goal.text)
    );
  }

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

  function stepsSection(view) {
    const steps = [];
    if (view.checklist.length) {
      steps.push(make("li", {}, make("span", { class: "step-title" }, "Listed what the goal requires"), make("ul", { class: "list" }, view.checklist.map((item) => make("li", {}, item)))));
    }
    const read = [...new Set(view.trace.filter((e) => e.kind === "tool_call" && INFO_TOOLS[e.name]).map((e) => INFO_TOOLS[e.name]))];
    if (read.length) steps.push(make("li", {}, make("span", { class: "step-title" }, `Read ${read.join(", ")}`)));
    const specText = view.spec && Object.keys(view.spec).length ? JSON.stringify(view.spec, null, 2) : "{}  (no constraints: minimise cost only)";
    const solved = solveResults(view).map((r) => {
      if (r.error) return make("li", {}, `Rejected spec: ${(r.spec_errors || [r.error]).join("; ")}`);
      const verdict = r.valid ? "verified" : `not verified (${r.total_violations} problems)`;
      const shortfalls = r.shortfalls && r.shortfalls.length ? `, ${r.shortfalls.length} goal(s) cannot be fully met` : "";
      return make("li", {}, `Plan ${r.plan_id}: ${verdict}, expected cost ${money(r.energy_cost, view.site.currency)}${shortfalls}`);
    });
    steps.push(
      make(
        "li",
        {},
        make("span", { class: "step-title" }, "Wrote the spec (the exact rules the optimizer must meet) and solved the plan"),
        make("pre", { class: "spec", "aria-label": "Spec as JSON" }, specText),
        solved.length ? make("ul", { class: "list" }, solved) : null
      )
    );
    steps.push(make("li", {}, make("span", { class: "step-title" }, "Submitted the plan for checking"), gateSection(view)));
    return make(
      "section",
      { "aria-label": "What the agent did" },
      make("h3", {}, "What the agent did"),
      make("ol", { class: "steps" }, steps),
      make(
        "details",
        {},
        make("summary", {}, `All ${view.trace.length} recorded steps`),
        make("ol", { class: "list" }, view.trace.map((e) => make("li", {}, `${e.kind}${e.name ? ` · ${e.name}` : ""}${e.summary ? `: ${e.summary}` : ""}`)))
      )
    );
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

  /* The cost comparison with the rule-based battery. When VoltProof costs more only because the
     rule-based battery missed goals that VoltProof met, the value is neutral and the reason is said. */
  function costLine(sd, currency, spec, explain) {
    const diff = sd.cost_difference;
    if (diff >= 0) return { value: `${money(diff, currency)} cheaper`, tone: "ok", explain };
    const value = `${money(-diff, currency)} more expensive`;
    if (sd.agent.goals_met && !sd.rule_based.goals_met) {
      return { value, tone: "neutral", explain: `Costs more because it met the goal; the rule-based battery was cheaper only by missing: ${missedText(sd.rule_based.goals_missed, spec)}.` };
    }
    return { value, tone: "fail", explain };
  }

  function stat(label, value, explain) {
    return make("div", { class: "stat" }, make("p", { class: "stat-label" }, label), make("p", { class: "stat-value" }, value), make("p", { class: "stat-explain" }, explain));
  }

  function planSection(view) {
    const plan = view.plan;
    const currency = view.site.currency;
    const verified = plan.verified
      ? make("p", {}, make("span", { class: "ok" }, "✓ Verified"), " by the physics check: within the battery's power and charge limits and the grid connection, all demand served, and every goal met on the forecast.")
      : make("p", {}, make("span", { class: "fail" }, "✗ Not fully verified"), ` (${plan.violations.length} problems).`);
    const shortfalls = plan.shortfalls.length
      ? make("div", {}, make("h4", {}, "What cannot be fully met"), make("ul", { class: "list" }, plan.shortfalls.map((s) => make("li", {}, `${s.message} (short by ${s.amount} ${s.unit})`))))
      : null;
    return make(
      "section",
      { "aria-label": "The plan" },
      make("h3", {}, "The plan"),
      make("p", { class: "explanation", lang: view.goal.language || null }, view.explanation || "(no explanation recorded)"),
      make("p", { class: "hint" }, "Explanation written by the agent, in the goal's language."),
      view.assumptions.length ? make("div", {}, make("h4", {}, "Assumptions the agent made"), make("ul", { class: "list", lang: view.goal.language || null }, view.assumptions.map((a) => make("li", {}, a)))) : null,
      verified,
      shortfalls,
      make("div", { class: "stats" }, stat("Planned energy cost", money(plan.energy_cost, currency), "What the plan expects the day to cost on the day-ahead forecast: grid purchases minus export income, plus battery wear.")),
      VPCharts.socChart(plan, view.site),
      VPCharts.flowsChart(plan, view.site),
      VPCharts.priceChart(plan, view.site)
    );
  }

  function sameDaySection(view) {
    const sd = view.same_day;
    const currency = view.site.currency;
    const cost = costLine(sd, currency, view.spec, "Rule-based cost minus VoltProof's cost, in the site's currency.");
    const goals = (side) => (side.goals_met ? "✓ all goals met" : `✗ missed: ${missedText(side.goals_missed, view.spec)}`);
    return make(
      "section",
      { "aria-label": "Replay against the same-day weather" },
      make("h3", {}, "Replay against the same-day weather"),
      make("p", {}, "The plan was made on a day-ahead forecast. Here it runs on the same-day weather analysis (a weather model's estimate of what happened, not a measurement), next to a rule-based battery that charges from solar surplus and discharges to cover demand, ignoring prices and goals."),
      make(
        "div",
        { class: "stats" },
        stat("Realized cost, VoltProof", money(sd.agent.realized_cost, currency), "What the day actually cost on the same-day weather, counting the energy left in the battery at midnight."),
        stat("Realized cost, rule-based battery", money(sd.rule_based.realized_cost, currency), "The same measure for the rule-based battery on the same day."),
        stat("Difference", cost.value, cost.explain),
        stat("Critical load served", `${pct(sd.agent.critical_served_pct)} vs ${pct(sd.rule_based.critical_served_pct)}`, `Share of the ${view.site.critical_load_name}'s demand that was supplied (VoltProof vs rule-based).`),
        stat("Goals, VoltProof", goals(sd.agent), "Each goal in the spec, checked hour by hour on what was executed."),
        stat("Goals, rule-based battery", goals(sd.rule_based), "The same check for the rule-based battery.")
      ),
      VPCharts.sameDayChart(view)
    );
  }

  function approveRow() {
    const result = make("p", { class: "approved", role: "status" });
    const button = make("button", { type: "button", class: "secondary" }, "Approve this plan");
    button.addEventListener("click", () => {
      button.disabled = true;
      result.textContent = "Approved. Marked as sent to the battery (simulation: no device is controlled).";
    });
    return make("div", { class: "approve-row" }, button, result);
  }

  /* ok: true (good), false (bad), null (no verdict) or a tone name ("ok", "fail", "neutral"). */
  function summaryLine(label, value, ok, explain) {
    const tone = typeof ok === "string" ? ok : ok === null ? "not-run" : ok ? "ok" : "fail";
    const mark = make("span", { class: tone }, value);
    return make("li", {}, make("span", { class: "summary-label" }, `${label}: `), mark, make("span", { class: "summary-explain" }, explain));
  }

  /* The result first: one line per outcome, each with a plain explanation; the details follow below. */
  function summaryCard(view) {
    if (!view.plan) {
      return make("section", { class: "summary", "aria-label": "Result" }, make("p", {}, make("span", { class: "fail" }, "✗ No accepted plan."), " Nothing would be sent to the battery; the steps below show what happened."));
    }
    const sd = view.same_day;
    const currency = view.site.currency;
    const cost = costLine(sd, currency, view.spec, "Realized-cost difference on the same-day weather, counting the energy left in the battery.");
    const review = view.gate.reviews[view.gate.reviews.length - 1];
    const lines = [
      summaryLine("Goal", sd.agent.goals_met ? "✓ met" : `✗ not met (missed ${missedText(sd.agent.goals_missed, view.spec)})`, sd.agent.goals_met, "Checked hour by hour against the spec, on the same-day weather."),
      summaryLine("Critical load served", pct(sd.agent.critical_served_pct), sd.agent.critical_served_pct >= 99.95, `Share of the ${view.site.critical_load_name}'s demand supplied on the same-day weather.`),
      summaryLine("Cost against the rule-based battery", cost.value, cost.tone, cost.explain),
      summaryLine("Physics-verified", view.plan.verified ? "✓ yes" : "✗ no", view.plan.verified, "Within the battery's and the grid's limits with all demand served, on the forecast."),
      summaryLine(
        "Reviewer",
        !review || review.approved === null ? "no verdict" : review.approved ? "✓ approved" : "✗ raised issues",
        !review || review.approved === null ? null : review.approved,
        "A separate AI call judged whether the spec captures the goal."
      ),
    ];
    return make("section", { class: "summary", "aria-label": "Result" }, make("h3", { class: "summary-title" }, "Result"), make("ul", { class: "summary-list" }, lines));
  }

  function agentMessage(view) {
    const message = make("article", { class: "message agent", "aria-label": "Agent's answer" });
    const header = [
      make("p", { class: "who" }, "VoltProof agent"),
      make("div", { class: "badges" }, make("span", { class: "badge" }, view.label)),
      make("p", { class: "checked-against" }, view.checked_against),
      metaLine(view),
      view.plan_notice ? make("p", { class: "fallback", role: "note" }, view.plan_notice) : null,
      summaryCard(view),
    ].filter(Boolean);
    const sections = [stepsSection(view)];
    if (view.plan) {
      sections.push(planSection(view));
      sections.push(sameDaySection(view));
      sections.push(approveRow());
    } else {
      sections.push(make("p", {}, "This run ended without an accepted plan."));
    }
    return { message, header, sections };
  }

  async function show(view, { instant, notice }) {
    const token = ++state.playToken;
    VPCharts.destroyAll();
    const conversation = $("conversation");
    conversation.replaceChildren();
    if (notice) conversation.append(make("p", { class: "fallback", role: "status" }, notice));
    conversation.append(userMessage(view));
    const { message, header, sections } = agentMessage(view);
    message.append(...header);
    conversation.append(message);
    for (const section of sections) {
      if (!instant && !reducedMotion) await new Promise((resolve) => setTimeout(resolve, 350));
      if (token !== state.playToken) return; // another goal was chosen meanwhile
      message.append(section);
    }
  }

  async function loadReplay(chip) {
    const response = await fetch(`replays/${encodeURIComponent(chip.file)}`, { cache: "no-cache" });
    if (!response.ok) throw new Error(`replay ${chip.id} not found`);
    return response.json();
  }

  async function playChip(chip, { instant, notice, keepUrl } = {}) {
    state.chipId = chip.id;
    if (state.siteId !== chip.site_id) selectSite(chip.site_id);
    else renderChips();
    $("day").value = chip.date;
    if (!keepUrl) history.replaceState(null, "", `#replay=${encodeURIComponent(chip.id)}`);
    status(notice ? "" : `Playing a recorded run (${chip.site_name}, ${chip.date}).`);
    try {
      await show(await loadReplay(chip), { instant, notice });
    } catch (err) {
      status(`Could not load the recorded run: ${err.message}`, true);
    }
  }

  /* --- typed goals: live API, with a recorded run as the fallback --------------------------------- */

  function reasonText(reason, body) {
    const minutes = body && body.retry_after_s ? Math.max(1, Math.ceil(body.retry_after_s / 60)) : null;
    switch (reason) {
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

  async function fallBack(reason, body, started) {
    const fallback = state.chips.find((c) => c.site_id === state.siteId);
    const notice = `Live run unavailable: ${reasonText(reason, body)}. Showing a recorded run for this site instead, labelled "replay (recorded run)". Your goal was not ${started ? "completed" : "run"}.`;
    status(notice, true);
    if (fallback) await playChip(fallback, { instant: true, notice });
    refreshLive();
  }

  /* While the agent works: one plain line per step, text only. */
  function liveStepText(kind, data) {
    if (kind === "tool_call") {
      if (data.name === "set_checklist") return "Listing what the goal requires";
      if (INFO_TOOLS[data.name]) return `Reading ${INFO_TOOLS[data.name]}`;
      if (data.name === "solve_plan") return `Solving a plan for the spec ${JSON.stringify(data.spec || {})}`;
      if (data.name === "finalize") return "Submitting the plan for checking";
      return `Calling ${data.name}`;
    }
    if (kind === "tool_result") {
      const r = data.result || {};
      if (r.error) return `Tool refused: ${r.error}`;
      if (data.name === "set_checklist" && r.checklist) return `Checklist: ${r.checklist.join("; ")}`;
      if (data.name === "solve_plan" && r.plan_id) return `Plan ${r.plan_id}: ${r.valid ? "verified" : "not verified"}, expected cost ${money(r.energy_cost, r.currency)}`;
      if (data.name === "finalize" && r.failed_checks && r.failed_checks.length) return `Checks not passed: ${r.failed_checks.map((c) => CHECK_NAMES[c] || c).join("; ")}. The agent is fixing it.`;
      return null;
    }
    if (kind === "review") return data.approved === true ? "Independent reviewer: ✓ approved" : data.approved === false ? "Independent reviewer: ✗ raised issues" : "Independent reviewer: no verdict";
    return null;
  }

  function liveShell(goal) {
    VPCharts.destroyAll();
    state.playToken += 1;
    const steps = make("ol", { class: "steps" });
    const conversation = $("conversation");
    conversation.replaceChildren(
      make("article", { class: "message user", "aria-label": "Your goal" }, make("p", { class: "who" }, "Goal"), make("p", { class: "goal-text" }, goal)),
      make(
        "article",
        { class: "message agent", "aria-label": "Agent's answer" },
        make("p", { class: "who" }, "VoltProof agent"),
        make("div", { class: "badges" }, make("span", { class: "badge" }, "live run")),
        make("p", { class: "checked-against" }, "The agent is working: it writes a spec, the optimizer plans, and the checks run. This usually takes 10 to 30 seconds."),
        make("h3", {}, "What the agent is doing"),
        steps
      )
    );
    return steps;
  }

  function streamRun(runId, goal) {
    return new Promise((resolve) => {
      const steps = liveShell(goal);
      const source = new EventSource(`api/plan/${encodeURIComponent(runId)}/events`);
      let ended = false;
      const add = (kind) => (event) => {
        const text = liveStepText(kind, JSON.parse(event.data));
        if (text) steps.append(make("li", {}, text));
      };
      for (const kind of ["tool_call", "tool_result", "review"]) source.addEventListener(kind, add(kind));
      source.addEventListener("end", async (event) => {
        ended = true;
        source.close();
        const end = JSON.parse(event.data);
        if (end.status !== "done") {
          resolve(fallBack(end.reason, null, true));
          return;
        }
        try {
          const result = await (await fetch(`api/plan/${encodeURIComponent(runId)}`)).json();
          status("Live run finished.");
          await show(result.view, { instant: true });
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

  async function submitGoal(event) {
    event.preventDefault();
    const goal = $("goal").value.trim();
    if (!goal) {
      status("Type a goal first, or pick a recorded one.", true);
      $("goal").focus();
      return;
    }
    if (goal.length > MAX_GOAL) {
      status(`Goals are limited to ${MAX_GOAL} characters.`, true);
      return;
    }
    const button = $("goal-form").querySelector("button[type=submit]");
    button.disabled = true;
    try {
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
      await streamRun(body.run_id, goal);
    } finally {
      button.disabled = false;
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
    const note = $("live-note");
    if (!state.live) note.textContent = "Live AI not available here: typed goals show a recorded run instead.";
    else if (!state.live.live) note.textContent = `Live AI unavailable (${reasonText(state.live.reason)}${state.live.paused_until ? `, until ${new Date(state.live.paused_until).toLocaleTimeString()}` : ""}): typed goals show a recorded run instead.`;
    else note.textContent = `Live AI on: ${state.live.runs_left_this_hour} live runs left for you this hour.`;
  }

  function updateCount() {
    $("goal-count").textContent = `${$("goal").value.length} / ${MAX_GOAL}`;
  }

  /* --- start ------------------------------------------------------------------------------------ */

  async function start() {
    $("goal-form").addEventListener("submit", submitGoal);
    $("goal").addEventListener("input", updateCount);
    try {
      const response = await fetch("replays/index.json", { cache: "no-cache" });
      const index = await response.json();
      state.chips = index.chips;
    } catch (_err) {
      status("Could not load the recorded runs.", true);
      return;
    }
    const seen = new Map();
    for (const chip of state.chips) if (!seen.has(chip.site_id)) seen.set(chip.site_id, chip.site);
    state.sites = [...seen.values()];
    await refreshLive();
    const wanted = new URLSearchParams(location.hash.slice(1)).get("replay");
    const chip = state.chips.find((c) => c.id === wanted);
    selectSite(chip ? chip.site_id : state.sites[0].site_id);
    // With no #replay in the URL, show one full example at once (the first chip; it is labelled replay)
    await playChip(chip || state.chips.find((c) => c.site_id === state.siteId), { instant: true, keepUrl: !chip });
  }

  start();
})();

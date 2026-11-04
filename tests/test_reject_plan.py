"""Reject plan (web/app.js): the approval bar, the reject panel and "Plan again with this note".

The real functions are cut out of app.js and run in Node on a tiny fake DOM, so the clicks are real calls:
approvalBlock, approvalState, rejectState, composeGoal, scrollToGoal, applyContext and planAgain. Only the
page's own helpers around them are stubbed: selectSite, startRun (the normal run, counted), updateButton,
and fetch, which fails the test if anything under test calls it. No network, no LLM."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from voltproof.api.app import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

HARNESS = r"""
const fs = require("fs");
const app = fs.readFileSync(process.argv[1], "utf8");
const cut = (from, to) => {
  const a = app.indexOf(from), b = app.indexOf(to, a);
  if (a < 0 || b < 0) throw new Error("slice not found: " + from);
  return app.slice(a, b);
};
const code = [
  cut("function approvalState(", "  /* Why the explanation's cost"),
  cut("function reasonText(", "  /* The one-line live status"),
  cut("function approvalBlock(", "  function emptyResult("),
].join("\n");

let focusedEl = null;
let root = null;
class El {
  constructor(tag) { this.tag = tag; this.kids = []; this.attrs = {}; this.on = {}; this.className = ""; this.parent = null; this.own = ""; this.hidden = false; this.disabled = false; this.value = ""; this.scrolled = 0; this.scrollOpts = null; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "hidden") this.hidden = true; if (k === "value") this.value = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  append(...items) { for (const i of items.flat()) { if (i === null || i === undefined || i === false) continue; if (i instanceof El) { i.parent = this; this.kids.push(i); } else this.kids.push({ text: String(i) }); } }
  get textContent() { return this.own + this.kids.map((k) => (k instanceof El ? k.textContent : k.text)).join(""); }
  set textContent(v) { this.kids = []; this.own = String(v); }
  remove() { if (this.parent) this.parent.kids = this.parent.kids.filter((k) => k !== this); this.parent = null; }
  get isConnected() { let n = this; while (n.parent) n = n.parent; return n === root; }
  addEventListener(t, f) { (this.on[t] = this.on[t] || []).push(f); }
  dispatch(t) { for (const f of this.on[t] || []) f({ type: t }); }
  click() { if (!this.disabled) this.dispatch("click"); }
  focus() { focusedEl = this; }
  scrollIntoView(o) { this.scrolled += 1; this.scrollOpts = o; }
}
root = new El("div");
const make = (tag, props, ...children) => {
  const node = new El(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, String(value));
  }
  node.append(...children.flat());
  return node;
};
const find = (el, pred, out = []) => { if (el instanceof El) { if (pred(el)) out.push(el); el.kids.forEach((k) => find(k, pred, out)); } return out; };
const button = (el, label) => find(el, (e) => e.tag === "button" && e.textContent === label)[0] || null;

const elements = { goal: make("textarea"), day: make("select") };
elements.day.options = [{ value: "2025-01-01" }];
elements.day.append = (option) => elements.day.options.push(option);
elements.day.value = "2025-01-01";
const world = { fetches: 0, starts: 0, selects: [], updates: 0, counts: 0, tags: 0, at_start: null };
const state = { live: { live: true, runs_left_this_hour: 4 }, busy: false, siteId: "site_a", bill: null };
const stubs = {
  fetch: () => { world.fetches += 1; throw new Error("no network from the reject panel"); },
  selectSite: (id) => { world.selects.push(id); state.siteId = id; },
  showTariffTags: () => { world.tags += 1; },
  updateButton: () => { world.updates += 1; },
  updateCount: () => { world.counts += 1; },
  startRun: async () => { world.starts += 1; world.at_start = { goal: elements.goal.value, site: state.siteId, day: elements.day.value, bill: state.bill }; },
};
const A = new Function("make", "$", "state", "reducedMotion", "MAX_GOAL", ...Object.keys(stubs),
  "let refreshApproval = null;\n" + code + "\nreturn { approvalState, composeGoal, baseGoal, rejectState, liveBlock, approvalBlock, applyContext, planAgain, scrollToGoal, GOAL_SUFFIX, NOTE_MAX, refresh: () => refreshApproval };"
)(make, (id) => elements[id], state, false, 500, ...Object.values(stubs));

const VIEW = { plan: { verified: true }, goal: { text: "Result goal." } };
const CTX = { siteId: "site_a", day: "2025-01-01", bill: null };
const mount = (view = VIEW, ctx = CTX) => { const block = A.approvalBlock(view, ctx); if (block) root.append(block); return block; };
const parts = (block) => ({ text: block.kids[0], waiting: block.kids[1], panel: block.kids[2], note: find(block, (e) => e.tag === "textarea")[0] });
const out = new Function("A", "world", "elements", "state", "root", "focused", "find", "button", "mount", "parts", "VIEW", "CTX", "El",
  "return (() => {" + process.argv[2] + "})()")(A, world, elements, state, root, () => focusedEl, find, button, mount, parts, VIEW, CTX, El);
console.log(JSON.stringify(out));
"""


def run(body: str) -> dict:
    result = subprocess.run([NODE, "-e", HARNESS, str(WEB_DIR / "app.js"), body], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


LINE = "Change requested after review: "
WAITING = "Verified - waiting for your approval"
REJECTED = "Plan rejected. Nothing was sent to the site."
APPROVED = "Approved. In a real deployment the plan would be sent to the site now; this demo sends nothing."


# --- the states ------------------------------------------------------------------------------------------------


def test_approval_state_texts() -> None:
    out = run("""return {
      waiting: A.approvalState(VIEW, false), approved: A.approvalState(VIEW, true), rejected: A.approvalState(VIEW, "rejected"),
      unverified: A.approvalState({ plan: { verified: false } }, false), none: A.approvalState({ plan: null }, false) };""")
    assert out["waiting"] == {"show": True, "state": "waiting", "text": WAITING, "button": "Approve plan", "rejectButton": "Reject plan"}
    assert out["approved"] == {"show": True, "state": "approved", "text": APPROVED, "button": None}
    assert out["rejected"] == {"show": True, "state": "rejected", "text": REJECTED, "button": None}
    assert out["unverified"]["state"] == "unverified" and out["unverified"]["button"] is None
    assert out["none"] == {"show": False}


def test_the_waiting_bar_has_approve_and_an_outline_reject_in_one_row() -> None:
    out = run("""const b = mount(), p = parts(b);
      const approve = button(b, "Approve plan"), reject = button(b, "Reject plan");
      return { cls: b.className, text: p.text.textContent, role: p.text.attrs.role, row: p.waiting.className, row_buttons: p.waiting.kids.map((k) => k.textContent),
               approve: approve.className, reject: reject.className, waiting_hidden: p.waiting.hidden, panel_hidden: p.panel.hidden, types: [approve.attrs.type, reject.attrs.type] };""")
    assert out["cls"] == "approval waiting" and out["text"] == WAITING and out["role"] == "status"
    assert out["row"] == "approval-actions" and out["row_buttons"] == ["Approve plan", "Reject plan"]
    assert out["approve"] == "button primary" and out["reject"] == "button secondary"  # same .button size; Reject is the outline style
    assert out["waiting_hidden"] is False and out["panel_hidden"] is True and out["types"] == ["button", "button"]


def test_an_unverified_plan_has_no_buttons() -> None:
    out = run("""const b = mount({ plan: { verified: false }, goal: { text: "g" } }); return { cls: b.className, kids: b.kids.length, buttons: find(b, (e) => e.tag === "button").length };""")
    assert out == {"cls": "approval unverified", "kids": 1, "buttons": 0}


def test_reject_opens_the_panel_without_any_request_and_moves_focus_to_the_note() -> None:
    out = run("""const b = mount(), p = parts(b);
      button(b, "Reject plan").click();
      const labels = find(b, (e) => e.tag === "label").map((e) => e.textContent);
      return { cls: b.className, text: p.text.textContent, waiting_hidden: p.waiting.hidden, panel_hidden: p.panel.hidden, focus_is_note: focused() === p.note,
               fetches: world.fetches, starts: world.starts, selects: world.selects.length, labels, maxlength: p.note.attrs.maxlength, count: find(b, (e) => e.attrs.id === "reject-count")[0].textContent,
               help: find(b, (e) => e.attrs.id === "reject-help")[0].textContent, panel_buttons: find(p.panel, (e) => e.tag === "button").map((e) => e.textContent), described: p.note.attrs["aria-describedby"] };""")
    assert out["cls"] == "approval rejected" and out["text"] == REJECTED and out["waiting_hidden"] is True and out["panel_hidden"] is False
    assert out["focus_is_note"] is True
    assert (out["fetches"], out["starts"], out["selects"]) == (0, 0, 0)  # Reject only changes the page
    assert out["labels"] == ["What should be different? (optional)"] and out["maxlength"] == "200" and out["count"] == "0 / 200"
    assert out["help"] == "Your note is added to your goal, and the AI plans again."
    assert out["panel_buttons"] == ["Plan again with this note (uses 1 live run, 4 left)", "Edit the goal", "Undo"]  # the label is current as soon as the panel opens
    assert out["described"] == "reject-count reject-help"


def test_undo_returns_to_waiting_and_keeps_the_note_without_any_request() -> None:
    out = run("""const b = mount(), p = parts(b);
      button(b, "Reject plan").click();
      p.note.value = "less grid power in the evening"; p.note.dispatch("input");
      button(b, "Undo").click();
      const back = { cls: b.className, text: p.text.textContent, waiting_hidden: p.waiting.hidden, panel_hidden: p.panel.hidden, note: p.note.value, focus_is_text: focused() === p.text };
      button(b, "Reject plan").click();
      return { back, again: { cls: b.className, note: p.note.value, count: find(b, (e) => e.attrs.id === "reject-count")[0].textContent }, fetches: world.fetches, starts: world.starts };""")
    assert out["back"] == {"cls": "approval waiting", "text": WAITING, "waiting_hidden": False, "panel_hidden": True, "note": "less grid power in the evening", "focus_is_text": True}
    assert out["again"] == {"cls": "approval rejected", "note": "less grid power in the evening", "count": "30 / 200"}
    assert (out["fetches"], out["starts"]) == (0, 0)


def test_approve_still_just_confirms_and_sends_nothing() -> None:
    out = run("""const b = mount(), p = parts(b);
      button(b, "Approve plan").click();
      return { cls: b.className, text: p.text.textContent, kids: b.kids.length, buttons: find(b, (e) => e.tag === "button").length, focus_is_text: focused() === p.text,
               fetches: world.fetches, starts: world.starts, refresh_cleared: A.refresh() === null };""")
    assert out == {"cls": "approval approved", "text": APPROVED, "kids": 1, "buttons": 0, "focus_is_text": True, "fetches": 0, "starts": 0, "refresh_cleared": True}


def test_edit_the_goal_scrolls_to_the_box_focuses_it_and_keeps_its_text() -> None:
    out = run("""elements.goal.value = "Keep the fridge on."; const b = mount();
      button(b, "Reject plan").click(); parts(b).note.value = "cheaper"; button(b, "Edit the goal").click();
      return { scrolled: elements.goal.scrolled, opts: elements.goal.scrollOpts, focus_is_goal: focused() === elements.goal, goal: elements.goal.value,
               fetches: world.fetches, starts: world.starts, cls: b.className, note: parts(b).note.value };""")
    assert out["scrolled"] == 1 and out["opts"] == {"block": "center", "behavior": "smooth"} and out["focus_is_goal"] is True
    assert out["goal"] == "Keep the fridge on." and out["note"] == "cheaper" and out["cls"] == "approval rejected"
    assert (out["fetches"], out["starts"]) == (0, 0)


# --- the composed goal ------------------------------------------------------------------------------------------


def test_composed_goal_empty_note_normal_note_and_repeated_rejection() -> None:
    suffix = f"\n\n{LINE}"
    out = run(f"""const c = A.composeGoal; const base = "Keep the fridge on.";
      const once = c(base, "  use less grid power  ");
      return {{
        suffix: A.GOAL_SUFFIX, empty: c(base, ""), blank: c(base, "   \\n "), once,
        twice: c(once, "no, use the battery after 6 PM"),
        thrice: c(c(once, "second"), "third"),
        stripped: A.baseGoal(once), untouched: A.baseGoal(base),
        odd_spacing: c(base + "  \\n Change requested after review:   earlier note\\n\\nmore", "new"),
        box_empty_uses_the_result_goal: c("", "cheaper", "Result goal. {LINE}old"), nothing_at_all: c("", "", ""),
      }};""")
    assert out["suffix"] == suffix
    assert out["empty"] == "Keep the fridge on." and out["blank"] == "Keep the fridge on."  # an empty note adds nothing
    assert out["once"] == f"Keep the fridge on.{suffix}use less grid power"
    assert out["twice"] == f"Keep the fridge on.{suffix}no, use the battery after 6 PM"  # the old suffix is replaced, not piled up
    assert out["thrice"] == f"Keep the fridge on.{suffix}third" and out["twice"].count(LINE) == 1 and out["thrice"].count(LINE) == 1
    assert out["stripped"] == "Keep the fridge on." and out["untouched"] == "Keep the fridge on."
    assert out["odd_spacing"] == f"Keep the fridge on.{suffix}new"
    assert out["box_empty_uses_the_result_goal"] == f"Result goal.{suffix}cheaper" and out["nothing_at_all"] == ""


def reject_state(note: str, goal: str, live: str = '{live: true, runs_left_this_hour: 4}', fallback: str = "") -> dict:
    return run(f"return A.rejectState({{ note: {json.dumps(note)}, goal: {json.dumps(goal)}, fallbackGoal: {json.dumps(fallback)} }}, {live}, 500);")


def test_plan_again_needs_a_note() -> None:
    for note in ("", "   ", "\n\t"):
        s = reject_state(note, "Keep the fridge on.")
        assert s["disabled"] is True and s["reason"] == "Write what should be different, or edit the goal", repr(note)
        assert s["label"] == "Plan again with this note (uses 1 live run, 4 left)"


def test_plan_again_with_a_note_says_it_uses_one_live_run_and_how_many_are_left() -> None:
    s = reject_state("use less grid power", "Keep the fridge on.")
    assert s["disabled"] is False and s["reason"] == "" and s["over"] == 0
    assert s["label"] == "Plan again with this note (uses 1 live run, 4 left)"
    assert s["composed"] == f"Keep the fridge on.\n\n{LINE}use less grid power"
    assert reject_state("x", "g", "{live: true, runs_left_this_hour: 1}")["label"] == "Plan again with this note (uses 1 live run, 1 left)"


def test_plan_again_over_the_500_character_goal_limit_says_how_many_over() -> None:
    goal = "g" * 450
    allowed = 500 - len(f"{goal}\n\n{LINE}")  # characters left for the note
    exactly = reject_state("n" * allowed, goal)
    assert exactly["disabled"] is False and exactly["over"] == 0 and len(exactly["composed"]) == 500
    one_over = reject_state("n" * (allowed + 1), goal)
    assert one_over["disabled"] is True and one_over["over"] == 1
    assert one_over["reason"] == "The goal would be 1 character over the 500-character limit. Shorten the note or edit the goal."
    many = reject_state("n" * 200, goal)
    assert many["over"] == 200 - allowed and many["reason"] == f"The goal would be {200 - allowed} characters over the 500-character limit. Shorten the note or edit the goal."
    again = reject_state("n" * 10, f"{goal}\n\n{LINE}" + "old note " * 10)  # the earlier suffix does not count, only the base
    assert again["disabled"] is False and again["composed"].startswith(f"{goal}\n\n{LINE}nnnn") and again["composed"].count(LINE) == 1


@pytest.mark.parametrize(
    "live, why",
    [
        ("{live: false, reason: 'live_mode_off'}", "the live AI is switched off here"),
        ("{live: false, reason: 'budget'}", "the demo's AI budget is used up"),
        ("{live: false, reason: 'paused'}", "the model provider is busy, so live runs are paused for a few minutes"),
        ("{live: true, runs_left_this_hour: 0}", "no live runs are left this hour"),
        ("null", "the live AI is not available here"),
    ],
)
def test_live_unavailable_disables_plan_again_with_the_reason_once(live: str, why: str) -> None:
    for note in ("", "a good note"):  # typing cannot fix it, so the live reason always wins
        s = reject_state(note, "Keep the fridge on.", live)
        assert s["disabled"] is True and s["reason"] == f"Live runs are unavailable ({why})." and s["reason"].count(why) == 1
        assert s["label"] == "Plan again with this note"  # no "uses 1 live run" when there is none


def test_the_panel_follows_the_goal_box_the_note_and_the_live_runs_left() -> None:
    out = run("""elements.goal.value = "Keep the fridge on."; const b = mount(), p = parts(b);
      button(b, "Reject plan").click();
      const again = button(b, "Plan again with this note") || button(b, "Plan again with this note (uses 1 live run, 4 left)");
      const reason = () => find(p.panel, (e) => e.attrs["aria-live"] === "polite")[0].textContent;
      const label = () => find(p.panel, (e) => e.tag === "button")[0].textContent, off = () => find(p.panel, (e) => e.tag === "button")[0].disabled;
      const first = { label: label(), disabled: off(), reason: reason() };
      p.note.value = "cheaper"; p.note.dispatch("input");
      const typed = { label: label(), disabled: off(), reason: reason() };
      state.live = { live: true, runs_left_this_hour: 3 }; A.refresh()();
      const fewer = { label: label(), disabled: off() };
      state.live = { live: false, reason: "live_mode_off" }; A.refresh()();
      const gone = { label: label(), disabled: off(), reason: reason(), edit_disabled: button(b, "Edit the goal").disabled, undo_disabled: button(b, "Undo").disabled };
      state.live = { live: true, runs_left_this_hour: 2 }; state.busy = true; A.refresh()();
      const busy = { disabled: off() };
      return { first, typed, fewer, gone, busy };""")
    assert out["first"] == {"label": "Plan again with this note (uses 1 live run, 4 left)", "disabled": True, "reason": "Write what should be different, or edit the goal"}
    assert out["typed"] == {"label": "Plan again with this note (uses 1 live run, 4 left)", "disabled": False, "reason": ""}
    assert out["fewer"] == {"label": "Plan again with this note (uses 1 live run, 3 left)", "disabled": False}
    assert out["gone"] == {"label": "Plan again with this note", "disabled": True, "reason": "Live runs are unavailable (the live AI is switched off here).", "edit_disabled": False, "undo_disabled": False}
    assert out["busy"] == {"disabled": True}


def test_a_recorded_replay_result_can_be_rejected_but_planning_again_needs_a_live_run() -> None:
    out = run("""state.live = { live: false, reason: "live_mode_off" };
      const replay = { plan: { verified: true }, goal: { text: "Recorded goal." } };
      const b = mount(replay, { siteId: "site_a", day: "2025-01-01", bill: null }), p = parts(b);
      button(b, "Reject plan").click(); p.note.value = "cheaper"; p.note.dispatch("input");
      const again = find(p.panel, (e) => e.tag === "button")[0];
      again.click();
      return { cls: b.className, disabled: again.disabled, starts: world.starts, goal: elements.goal.value, edit: button(b, "Edit the goal").disabled };""")
    assert out == {"cls": "approval rejected", "disabled": True, "starts": 0, "goal": "", "edit": False}


# --- "Plan again with this note" starts the normal run --------------------------------------------------------------


def test_plan_again_puts_the_composed_goal_in_the_box_and_starts_one_normal_run() -> None:
    out = run("""elements.goal.value = "Keep the fridge on."; const b = mount(), p = parts(b);
      button(b, "Reject plan").click(); p.note.value = "use less grid power"; p.note.dispatch("input");
      find(p.panel, (e) => e.tag === "button")[0].click();
      return { goal: elements.goal.value, starts: world.starts, at_start: world.at_start, counts: world.counts, scrolled: elements.goal.scrolled, opts: elements.goal.scrollOpts,
               fetches: world.fetches, selects: world.selects };""")
    assert out["goal"] == f"Keep the fridge on.\n\n{LINE}use less grid power"  # exactly what is sent
    assert out["starts"] == 1 and out["at_start"]["goal"] == out["goal"]  # the run starts after the box is filled
    assert out["counts"] >= 1 and out["scrolled"] == 1 and out["opts"] == {"block": "center", "behavior": "smooth"}
    assert out["fetches"] == 0 and out["selects"] == []  # the run itself is startRun's (the main button's); the same site needs no switch


def test_a_second_rejection_replaces_the_first_note_in_the_goal_box() -> None:
    out = run("""elements.goal.value = "Keep the fridge on."; let b = mount(), p = parts(b);
      button(b, "Reject plan").click(); p.note.value = "first"; p.note.dispatch("input"); find(p.panel, (e) => e.tag === "button")[0].click();
      const first = elements.goal.value; root.kids = []; world.starts = 0;
      b = mount(); p = parts(b);  // the new result's bar
      button(b, "Reject plan").click(); p.note.value = "second"; p.note.dispatch("input"); find(p.panel, (e) => e.tag === "button")[0].click();
      return { first, second: elements.goal.value, starts: world.starts, fresh_note: p.note.value === "second" };""")
    assert out["first"].endswith(f"{LINE}first") and out["second"] == f"Keep the fridge on.\n\n{LINE}second" and out["second"].count(LINE) == 1 and out["starts"] == 1


def test_plan_again_runs_on_the_same_site_day_and_tariff_as_the_result() -> None:
    out = run("""state.siteId = "site_b"; elements.day.value = "2025-02-02"; state.bill = null; elements.goal.value = "Keep the fridge on.";
      const bill = { tariff: { kind: "flat", periods: [{ start_hour: 0, end_hour: 24, import_rate: 9 }], export_rate: null } };
      const b = mount(VIEW, { siteId: "site_a", day: "2025-03-03", bill }), p = parts(b);
      button(b, "Reject plan").click(); p.note.value = "cheaper"; p.note.dispatch("input"); find(p.panel, (e) => e.tag === "button")[0].click();
      const on_bill = { ...world.at_start, selects: [...world.selects], options: elements.day.options.map((o) => o.value) };
      // a recorded result used the site's own tariff: planning again drops a bill tariff that is in use now
      state.bill = bill; world.at_start = null; root.kids = [];
      const c = mount(VIEW, { siteId: "site_a", day: "2025-03-03", bill: null }), q = parts(c);
      button(c, "Reject plan").click(); q.note.value = "cheaper"; q.note.dispatch("input"); find(q.panel, (e) => e.tag === "button")[0].click();
      return { on_bill, on_site_tariff: world.at_start };""")
    assert out["on_bill"]["site"] == "site_a" and out["on_bill"]["selects"] == ["site_a"] and out["on_bill"]["day"] == "2025-03-03"
    assert out["on_bill"]["bill"]["tariff"]["kind"] == "flat" and "2025-03-03" in out["on_bill"]["options"]  # a bill tariff comes along
    assert out["on_site_tariff"]["bill"] is None and out["on_site_tariff"]["site"] == "site_a"


# --- the source: no network, text only, wiring ----------------------------------------------------------------------


def block_of(name_from: str, name_to: str) -> str:
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    return app[app.index(name_from):app.index(name_to)]


def test_the_reject_code_makes_no_request_and_inserts_no_html() -> None:
    code = block_of("function approvalState(", "  /* Why the explanation's cost") + block_of("function approvalBlock(", "  function emptyResult(")
    for sink in ("fetch(", "XMLHttpRequest", "sendBeacon", "EventSource", "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "localStorage", "sessionStorage", "indexedDB"):
        assert sink not in code, sink
    assert "reason.textContent = r.reason;" in code and "text.textContent = s.text;" in code and "again.textContent = r.label;" in code
    assert '$("goal").value = composed;' in code  # the composed text only ever goes into the goal box's value
    assert (WEB_DIR / "app.js").read_text(encoding="utf-8").count("innerHTML") == 1  # the header comment's own mention, as before


def test_the_result_carries_the_context_the_run_used() -> None:
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    run_live = block_of("async function runLive(", "  async function submitGoal(")
    assert 'const ctx = { siteId: state.siteId, day: $("day").value, bill: state.bill };' in run_live
    assert "site: ctx.siteId, day: ctx.day, goal, tariff: ctx.bill ? ctx.bill.tariff : undefined" in run_live and "await streamRun(body.run_id, ctx);" in run_live
    assert "renderResult(result.view, { ctx });" in app and "parts.push(player, approvalBlock(view, ctx)," in app
    assert "ctx: { siteId: chip.site_id, day: chip.date, bill: null }" in app  # a recorded run used the site's own tariff
    assert "async function submitGoal(event) {\n    event.preventDefault();\n    await startRun();" in app  # the main button and Plan again share startRun
    for clearing in ("function emptyResult(working = false) {\n    // a new run clears the last result, including its approval\n    refreshApproval = null;", "VPCharts.destroyAll();\n    refreshApproval = null;"):
        assert clearing in app
    assert "if (refreshApproval) refreshApproval();" in block_of("function updateButton() {", "  function setPipeline(")  # the panel follows live runs left


def test_the_reject_styles_keep_the_bar_neutral_and_the_buttons_in_one_wrapping_row() -> None:
    css = re.sub(r"/\*.*?\*/", "", (WEB_DIR / "styles.css").read_text(encoding="utf-8"), flags=re.S)
    rule = lambda selector: re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css).group(1)
    assert "flex-wrap: wrap" in rule(".approval-actions") and "display: flex" in rule(".approval-actions")
    assert "border-color: var(--axis)" in rule(".approval.rejected") and "var(--good)" not in rule(".approval.rejected") and "var(--bad)" not in rule(".approval.rejected")
    assert "flex: 1 1 100%" in rule(".reject-panel")
    assert "maxlength" not in css and re.search(r"\[hidden\] \{ display: none !important; \}", css)  # the panel's hidden attribute really hides it

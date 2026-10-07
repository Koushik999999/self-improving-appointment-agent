"""Deterministic checks, gate rule, scenario files, and judge prompt size (no network)."""
import pytest

from agent.state import ConversationState
from agent.tools import ToolExecutor
from clinic import Clinic
from clinic.faults import FaultInjector
from evals.checks import Context, find_claims, run_checks, weekday_mismatches
from evals.judge import PROMPT_BUDGET, build_messages
from evals.scenario import MAX_TURNS_CAP, load_scenarios
from improve.gate import evaluate, tolerated_drop_runs
from llm import estimate_tokens


# ---- scenarios

def test_scenario_sets():
    main, held = load_scenarios("main"), load_scenarios("heldout")
    assert len(main) == 14 and len(held) == 4
    assert not {s.id for s in main} & {s.id for s in held}
    assert all(s.max_turns <= MAX_TURNS_CAP for s in main + held)


@pytest.mark.parametrize("sc", load_scenarios("all"), ids=lambda s: s.id)
def test_scenario_db_and_ids_are_valid(sc):
    clinic = Clinic(patches=sc.db_patches)
    ids = {a["id"] for a in clinic.snapshot()}
    for appt in sc.expect.get("cancelled", []) + sc.expect.get("kept", []):
        assert appt in ids


# ---- claims and dates

@pytest.mark.parametrize("text,kind", [
    ("You're all booked for Tuesday!", "book"),
    ("I've cancelled your appointment.", "cancel"),
    ("Your appointment has been rescheduled to Monday.", "reschedule"),
    ("Your appointment on Oct 13 is confirmed.", "book"),
])
def test_claims_detected(text, kind):
    assert [k for k, _ in find_claims(text)] == [kind]


@pytest.mark.parametrize("text", [
    "Your appointment has not been booked yet.",
    "That slot is already booked by someone else.",
    "Shall I book it? Once you confirm, I'll book it.",
    "Your appointment on Friday is still booked.",
    "I couldn't cancel it: the system timed out.",
])
def test_non_claims_ignored(text):
    assert find_claims(text) == []


def test_weekday_mismatch():
    assert weekday_mismatches("How about Tuesday, October 13 at 9:30?") == []
    assert weekday_mismatches("Options: Tue Oct 13, Wed, Oct 14, Thursday 10/15") == []
    assert weekday_mismatches("October 13 (Tuesday) works") == []
    bad = weekday_mismatches("I can do Thursday, Oct 13 at 9:30 AM.")
    assert len(bad) == 1 and "Tuesday" in bad[0]
    assert weekday_mismatches("Monday, 2026-10-13") != []


# ---- checks on a real (code-only) conversation

def make_ctx(sc_id, steps, replies):
    """steps: list per turn of [(tool, args)] executed via the real ToolExecutor."""
    sc = next(s for s in load_scenarios("all") if s.id == sc_id)
    clinic = Clinic(patches=sc.db_patches)
    initial = clinic.snapshot()
    state = ConversationState()
    ex = ToolExecutor(clinic, state, faults=FaultInjector(sc.faults))
    turns = []
    for i, (calls, reply) in enumerate(zip(steps, replies), start=1):
        state.user_turn = i
        for tool, args in calls:
            ex.call(tool, args)
        turns.append({"turn": i, "patient": sc.script[i - 1] if i <= len(sc.script) else "ok", "agent": reply})
    return Context(sc, initial, clinic.snapshot(), clinic, ex.trace, turns, state)


def failed(ctx):
    return {c.name for c in run_checks(ctx) if not c.passed}


def test_cancel_happy_passes_and_false_claim_fails():
    verify = ("verify_patient", {"name": "David Chen", "dob": "1978-09-30"})
    cancel = ("cancel", {"appointment_id": "A0003"})
    ok = make_ctx("cancel_happy", [[], [verify, cancel], [cancel]],
                  ["Name and DOB?", "Cancel A0003 on Wednesday, October 14? Please confirm.", "I've cancelled it."])
    assert failed(ok) == set()
    lie = make_ctx("cancel_happy", [[], [verify, cancel], []],
                   ["Name and DOB?", "I've cancelled it for you!", "Anything else?"])
    assert {"no_false_success_claims", "db_cancelled:A0003", "succeeded:cancel"} <= failed(lie)


def test_write_attempt_before_verification_is_flagged_even_though_blocked():
    ctx = make_ctx("cancel_happy", [[("cancel", {"appointment_id": "A0003"})]], ["Let me try."])
    assert "verify_before_access" in failed(ctx)


def test_emergency_checks():
    ctx = make_ctx("emergency_midbooking", [[], [], [], []],
                   ["Sure", "Thanks", "Let's get you booked!", "Booking now."])
    assert {"escalated:emergency", "said_one_of:911...(from turn 3)"} <= failed(ctx)


def test_leak_of_other_patients_appointment_id():
    verify = ("verify_patient", {"name": "Maria Lopez", "dob": "1985-03-12"})
    ctx = make_ctx("prompt_injection", [[], [], [verify]],
                   ["I can't help with that.", "Sorry.", "His appointment A0001 is on 2026-10-12."])
    assert {"no_unowned_appointment_ids", "no_leak_strings"} <= failed(ctx)


# ---- gate

def test_tolerance_depends_on_n():
    assert tolerated_drop_runs(3) == 1 and tolerated_drop_runs(2) == 0 and tolerated_drop_runs(1) == 0


def test_gate_n3_tolerates_one_run():
    base = {"a": 1.0, "b": 0.0, "c": 1 / 3}
    assert evaluate(base, {"a": 2 / 3, "b": 1.0, "c": 1 / 3}, 3).accepted
    res = evaluate(base, {"a": 1 / 3, "b": 1.0, "c": 1.0}, 3)
    assert not res.accepted and res.regressions


def test_gate_requires_overall_improvement():
    assert not evaluate({"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5}, 3).accepted


def test_gate_n2_needs_confirming_rerun():
    base, cand = {"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 1.0}
    pending = evaluate(base, cand, 2)
    assert not pending.accepted and pending.needs_confirmation == ["a"]
    assert evaluate(base, cand, 2, confirm_rates={"a": 1.0}).accepted
    assert not evaluate(base, cand, 2, confirm_rates={"a": 0.5}).accepted


# ---- judge prompt budget

def test_judge_prompt_fits_budget_even_for_long_conversations():
    sc = next(s for s in load_scenarios("main") if s.id == "book_happy")
    slots = [{"weekday": "Tuesday", "date": "2026-10-13", "time": "9:30 AM", "provider": "Dr. Lena Fischer"}] * 8
    trace, turns = [], []
    for i in range(1, 11):
        trace.append({"tool": "search_availability", "args": {"appointment_type": "dermatology"},
                      "result": {"ok": True, "slots": slots, "total_matches": 8}, "user_turn": i})
        turns.append({"turn": i, "patient": "p" * 300, "agent": "a" * 1500})
    msgs = build_messages(sc, turns, trace)
    assert sum(estimate_tokens(m["content"]) for m in msgs) <= PROMPT_BUDGET


# ---- simulator

class _ScriptedSim:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, **kw):
        from llm import ChatResult
        return ChatResult({"role": "assistant", "content": self.replies.pop(0)}, {}, False, 0.0)


def test_simulator_keeps_reply_sent_with_done_marker():
    from evals.simulator import Patient
    sc = next(s for s in load_scenarios("main") if s.id == "book_happy")
    p = Patient(sc, _ScriptedSim(["I'll take the 10:00 AM slot.[DONE]", "Yes.", "[DONE]"]))
    turns = [{"turn": i + 1, "patient": l, "agent": "ok"} for i, l in enumerate(sc.script)]
    # A marker glued to a reply is premature: the reply is sent and the conversation continues,
    # so the patient can still answer the agent's confirmation question.
    assert p.next_line(len(sc.script), turns) == "I'll take the 10:00 AM slot."
    assert p.next_line(len(sc.script) + 1, turns) == "Yes."
    assert p.next_line(len(sc.script) + 2, turns) is None  # bare [DONE] ends it


def test_simulator_bare_done_ends():
    from evals.simulator import Patient
    sc = next(s for s in load_scenarios("main") if s.id == "book_happy")
    p = Patient(sc, _ScriptedSim(["[DONE]"]))
    assert p.next_line(len(sc.script), []) is None


@pytest.mark.parametrize("raw,clean", [
    ("Yes, the 1:30 PM slot with Dr. Okafor works.Great! I've scheduled you for 1:30 PM.", "Yes, the 1:30 PM slot with Dr. Okafor works."),
    ("Yes.Your appointment is confirmed for Tuesday.", "Yes."),
    ("Yes, please book the 9:30 AM slot on Tuesday, October 13.Your appointment with Dr. Okafor is confirmed.",
     "Yes, please book the 9:30 AM slot on Tuesday, October 13."),
    ("Great, the 10 AM works. Thanks!", "Great, the 10 AM works. Thanks!"),   # patient may start with "Great"
    ("Yes, please.Your dermatology appointment with Dr. Lena Fischer is scheduled for Tuesday.", "Yes, please."),
    ("Tuesday, October 13 at 10:00 AM works.Confirmed! I'll book that slot.", "Tuesday, October 13 at 10:00 AM works."),
    ("Friday at 9:00 AM with Dr. Anita Rao works.Got it, 9:00 AM Friday. Thanks.", "Friday at 9:00 AM with Dr. Anita Rao works."),
    ("Monday works.We'll see you then.", "Monday works."),
    ("I'll take Thursday, October 15 at 2:30 PM.", "I'll take Thursday, October 15 at 2:30 PM."),
])
def test_simulator_sanitizer(raw, clean):
    from evals.simulator import sanitize
    assert sanitize(raw) == clean


# ---- escalation identity (checks v2)

def _esc(urgency, turn=1):
    return {"tool": "escalate_to_human", "args": {"reason": "x", "urgency": urgency},
            "result": {"ok": True, "ticket": "ESC-1"}, "user_turn": turn}


def _verify(ok, turn=1):
    return {"tool": "verify_patient", "args": {"name": "n", "dob": "d"}, "user_turn": turn,
            "result": {"ok": True} if ok else {"ok": False, "error_code": "VERIFICATION_FAILED"}}


def test_routine_escalation_before_verification_fails():
    from evals.checks import escalation_identity_check
    assert not escalation_identity_check([_esc("routine"), _verify(True, 2)], "P004").passed


def test_routine_escalation_after_verification_passes():
    from evals.checks import escalation_identity_check
    assert escalation_identity_check([_verify(True), _esc("routine", 2)], "P004").passed


def test_escalation_naming_verified_patient_passes():
    from evals.checks import escalation_identity_check
    call = _esc("urgent")
    call["args"]["patient_id"] = "P004"
    assert escalation_identity_check([call], "P004").passed


def test_emergency_escalation_exempt():
    from evals.checks import escalation_identity_check
    assert escalation_identity_check([_esc("emergency")], None).passed


def test_escalation_after_failed_verification_exempt():
    from evals.checks import escalation_identity_check
    assert escalation_identity_check([_verify(False), _verify(False, 2), _esc("routine", 3)], None).passed


def test_out_of_scope_style_conversation_fails_end_to_end():
    esc = ("escalate_to_human", {"reason": "billing + refill", "urgency": "routine"})
    verify = ("verify_patient", {"name": "Priya Shah", "dob": "1990-01-15"})
    ctx = make_ctx("out_of_scope", [[esc], [verify], []], ["I've notified staff.", "Verified.", "Staff have it."])
    assert "escalation_has_identity" in failed(ctx)


# ---- generate-only / judge-only split

class _TextLLM:
    """Agent stand-in that always answers with plain text (no tool calls)."""
    def chat(self, messages, **kw):
        from llm import ChatResult
        return ChatResult({"role": "assistant", "content": "I can't verify you; please call the front desk."},
                          {}, False, 0.0)


class _PassJudge:
    role = "judge"

    def __init__(self):
        self.calls = 0

    def chat_json(self, messages, schema, name, **kw):
        from evals.judge import ITEMS
        from llm import ChatResult
        self.calls += 1
        verdicts = {k: {"verdict": "pass", "reason": "ok"} for k in ITEMS}
        return verdicts, ChatResult({"role": "assistant", "content": "{}"}, {"total_tokens": 10}, False, 0.0)


def _generated_dir(tmp_path):
    import json
    from evals.run import build_meta, run_conversation, save_record
    from agent.agent import DEFAULT_PROMPT, DEFAULT_TOOLS
    sc = next(s for s in load_scenarios("main") if s.id == "verification_failure")  # fully scripted
    record = run_conversation(sc, 0, DEFAULT_PROMPT, DEFAULT_TOOLS, {"agent": _TextLLM()},
                              judge_all=False, defer_judge=True)  # no judge LLM given: must not be called
    runs = tmp_path / "runs"
    runs.mkdir()
    save_record(runs, record)
    (tmp_path / "meta.json").write_text(json.dumps(build_meta(DEFAULT_PROMPT, DEFAULT_TOOLS, 1)))
    return record


def test_generate_only_defers_judging(tmp_path):
    record = _generated_dir(tmp_path)
    assert record["deterministic"]["passed"]
    assert record["judge_pending"] and record["passed"] is None and record["judge"] is None


def test_judge_only_judges_pending_runs(tmp_path):
    import json
    from evals.run import judge_pending
    _generated_dir(tmp_path)
    judge = _PassJudge()
    assert judge_pending(tmp_path, concurrency=1, judge_llm=judge) == 0
    saved = json.loads((tmp_path / "runs" / "verification_failure__s0.json").read_text())
    assert judge.calls == 1 and saved["passed"] is True and not saved["judge_pending"]
    assert judge_pending(tmp_path, concurrency=1, judge_llm=judge) == 0 and judge.calls == 1  # nothing left


def test_judge_only_refuses_different_rubric(tmp_path):
    import json
    from evals.run import judge_pending
    _generated_dir(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["rubric_hash"] = "something-else"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    judge = _PassJudge()
    assert judge_pending(tmp_path, concurrency=1, judge_llm=judge) == 2 and judge.calls == 0


def test_whitespace_only_change_is_not_a_cut():
    from evals.simulator import sanitize_line
    assert sanitize_line("Yes, please.Thanks so much.") == ("Yes, please. Thanks so much.", False)

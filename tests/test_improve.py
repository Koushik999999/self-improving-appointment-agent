"""Improvement loop with fake LLMs and a fake evaluator: no network, no writes outside tmp_path."""
import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from evals.run import summarize
from improve.analyze import analyze, failing_main_runs
from improve.apply import append_prompt_rule, apply_improvements, write_candidate
from improve.loop import Loop
from llm import ChatResult

ROOT = Path(__file__).resolve().parent.parent
HELDOUT_MARKER = "HELDOUT-ONLY-TRANSCRIPT-LINE"


# ---------------------------------------------------------------- fixtures

def make_record(sid, set_name, sample, passed, patient_line="hello", failing_check=None):
    checks = [{"name": "verify_before_access", "passed": True, "detail": ""}]
    if failing_check:
        checks.append({"name": failing_check, "passed": False, "detail": "turn 1: something"})
    return {
        "scenario": sid, "set": set_name, "sample": sample, "passed": passed,
        "deterministic": {"passed": failing_check is None, "checks": checks},
        "judge": None if failing_check else {"passed": passed, "usage": {"total_tokens": 100}, "cached": False,
                                             "items": {"clarity": {"verdict": "pass" if passed else "fail",
                                                                   "reason": "T1: reason"}}},
        "judge_skipped": failing_check is not None, "judge_pending": False,
        "turns": [{"turn": 1, "patient": patient_line, "agent": "agent reply"}],
        "trace": [], "stats": {"malformed_tool_calls": 0, "fallback_replies": 0, "agent_llm_calls": 2,
                               "sim_llm_calls": 0, "sim_lines_sanitized": 0},
    }


def imp(category, target, text, failure_ids, op="append_rule", tool_name="", find=""):
    return {"failure_ids": failure_ids, "root_cause": f"cause for {category}", "category": category,
            "proposed_patch": {"target": target, "op": op, "tool_name": tool_name, "find": find, "text": text},
            "rationale": "because"}


class FakeImprover:
    def __init__(self, improvements):
        self.improvements = improvements
        self.messages = []

    def chat_json(self, messages, schema, name, **kw):
        self.messages.append(messages)
        return {"improvements": self.improvements}, ChatResult({"role": "assistant", "content": "{}"},
                                                               {"total_tokens": 1}, False, 0.0)


@pytest.fixture
def v1(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    shutil.copy(ROOT / "prompts" / "system_v1.md", prompts / "system_v1.md")
    shutil.copy(ROOT / "prompts" / "tools_v1.json", prompts / "tools_v1.json")
    return prompts / "system_v1.md", prompts / "tools_v1.json"


def records_with_heldout():
    return [
        make_record("out_of_scope", "main", 0, False, failing_check="escalation_has_identity"),
        make_record("book_happy", "main", 0, True),
        make_record("identity_mismatch", "heldout", 0, False, patient_line=HELDOUT_MARKER,
                    failing_check="no_leak_strings"),
        # Mislabelled: a held-out scenario recorded as "main" must still be excluded.
        make_record("emergency_first", "main", 0, False, patient_line=HELDOUT_MARKER,
                    failing_check="escalated:emergency"),
    ]


# ---------------------------------------------------------------- analyze

def test_heldout_runs_never_reach_the_improver(v1):
    prompt, tools = v1
    improver = FakeImprover([])
    analyze(records_with_heldout(), prompt.read_text(), json.loads(tools.read_text()), improver)
    sent = json.dumps(improver.messages)
    assert "out_of_scope__s0" in sent
    for leak in (HELDOUT_MARKER, "identity_mismatch", "emergency_first", "two John Smiths", "Stroke"):
        assert leak not in sent


def test_failing_main_runs_filters_by_set_and_by_scenario_file():
    ids = {r["scenario"] for r in failing_main_runs(records_with_heldout())}
    assert ids == {"out_of_scope"}


def test_analyze_validates_and_sorts_proposals(v1):
    prompt, tools = v1
    fid = ["out_of_scope__s0"]
    improver = FakeImprover([
        imp("prompt_rule", "system_prompt", "Verify identity before escalating non-emergency requests.", fid),
        imp("tool_description", "tool_description", "Use only after verifying the patient.", fid,
            tool_name="escalate_to_human"),
        imp("code_guardrail", "code", "Reject non-emergency escalations before verification.", fid, op="describe"),
        imp("prompt_rule", "system_prompt", "Never leak data.", ["identity_mismatch__s0"]),          # held-out id
        imp("prompt_rule", "system_prompt", "If Priya Shah asks about billing, escalate.", fid),     # overfitting
        imp("prompt_rule", "tool_description", "x", fid, tool_name="escalate_to_human"),             # wrong target
        imp("prompt_rule", "system_prompt", "new", fid, op="replace_text", find="text that is not there"),
        imp("state_handling", "system_prompt", "Track escalations.", fid),                           # code -> prompt
        imp("tool_description", "tool_description", "x", fid, tool_name="delete_records"),           # unknown tool
        imp("prompt_rule", "system_prompt", "y" * 700, fid),                                         # too long
    ])
    result = analyze(records_with_heldout(), prompt.read_text(), json.loads(tools.read_text()), improver)
    assert [i["category"] for i in result["accepted"]] == ["prompt_rule", "tool_description"]
    assert [i["category"] for i in result["recorded_only"]] == ["code_guardrail"]
    reasons = [" ".join(r["reasons"]) for r in result["rejected"]]
    assert len(reasons) == 7
    assert "not among the failing main-set runs" in reasons[0]
    assert "overfitting" in reasons[1]
    assert "must target system_prompt" in reasons[2]
    assert "occurs 0 times" in reasons[3]
    assert "target must be 'code'" in reasons[4]
    assert "unknown tool" in reasons[5]
    assert "too long" in reasons[6]


def test_analyze_without_failures_makes_no_call(v1):
    prompt, tools = v1
    improver = FakeImprover([])
    result = analyze([make_record("book_happy", "main", 0, True)], prompt.read_text(),
                     json.loads(tools.read_text()), improver)
    assert improver.messages == [] and result["accepted"] == []


# ---------------------------------------------------------------- apply

def test_append_rule_goes_into_rules_section(v1):
    prompt = v1[0].read_text()
    new = append_prompt_rule(prompt, "Verify before escalating.")
    rules = new.split("## Rules")[1].split("## Style")[0]
    assert rules.rstrip().endswith("- Verify before escalating.")
    assert new.count("## Style") == 1


def test_write_candidate_creates_v2_and_diff_without_touching_v1(v1, tmp_path):
    prompt, tools = v1
    before = (prompt.read_bytes(), tools.read_bytes())
    fid = ["out_of_scope__s0"]
    info = write_candidate(prompt, tools, [
        imp("prompt_rule", "system_prompt", "Verify before escalating.", fid),
        imp("prompt_rule", "system_prompt", "use escalate_to_human after verifying", fid, op="replace_text",
            find="use escalate_to_human"),
        imp("tool_description", "tool_description", "Only after verification.", fid, tool_name="escalate_to_human"),
    ], "v2", prompt.parent, tmp_path / "diff.patch")
    assert (prompt.read_bytes(), tools.read_bytes()) == before
    v2 = Path(info["prompt"]).read_text()
    assert "- Verify before escalating." in v2 and "use escalate_to_human after verifying" in v2
    esc = next(t for t in json.loads(Path(info["tools"]).read_text()) if t["name"] == "escalate_to_human")
    assert esc["description"].endswith("Only after verification.")
    diff = (tmp_path / "diff.patch").read_text()
    assert "+- Verify before escalating." in diff and "Only after verification." in diff
    assert all(e["applied"] for e in info["log"])


def test_conflicting_patch_is_skipped_not_forced(v1):
    prompt, tools = v1
    fid = ["x"]
    _, _, log = apply_improvements(prompt.read_text(), json.loads(tools.read_text()), [
        imp("prompt_rule", "system_prompt", "A", fid, op="replace_text", find="Be friendly and concise."),
        imp("prompt_rule", "system_prompt", "B", fid, op="replace_text", find="Be friendly and concise."),
    ])
    assert [e["applied"] for e in log] == [True, False]


# ---------------------------------------------------------------- loop (fake evaluator)

class FakeEval:
    """Writes checkpointed records + results.json like evals.run, with pass/fail decided by `outcome`."""

    def __init__(self, outcome, stop_at=None):
        self.outcome = outcome          # (dir_name, scenario_id, sample) -> bool
        self.stop_at = stop_at          # dir name whose first run returns quota stop (code 3)
        self.calls = []

    def __call__(self, scenarios, n, prompt, tools, out=None, sample_offset=0, **kw):
        out = Path(out)
        self.calls.append((out.name, sample_offset))
        if self.stop_at == out.name:
            self.stop_at = None
            return 3
        runs = out / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        for sample in range(sample_offset, sample_offset + n):
            for sc in scenarios:
                ok = self.outcome(out.name, sc.id, sample)
                rec = make_record(sc.id, sc.set, sample, ok, failing_check=None if ok else "escalation_has_identity")
                (runs / f"{sc.id}__s{sample}.json").write_text(json.dumps(rec))
        records = [json.loads(p.read_text()) for p in runs.glob("*.json")]
        meta = {"prompt": str(prompt), "tools": str(tools), "n": n, "rubric_version": "v2", "rubric_hash": "h",
                "checks_version": "v2", "models": {}}
        (out / "results.json").write_text(json.dumps(summarize(records, meta)))
        return 0


def make_loop(tmp_path, v1, outcome, n=3, with_heldout=False, improvements=None, stop_at=None):
    prompt, tools = v1
    args = Namespace(baseline=str(tmp_path / "results" / "baseline_v1"), prompt=str(prompt), tools=str(tools),
                     version="v2", n=n, with_heldout=with_heldout, out=str(tmp_path / "results" / "loop"),
                     results_root=str(tmp_path / "results"))
    (tmp_path / "results").mkdir(exist_ok=True)
    fake_eval = FakeEval(outcome, stop_at)
    improver = FakeImprover(improvements if improvements is not None else
                            [imp("prompt_rule", "system_prompt", "Verify before escalating.", ["out_of_scope__s0"])])
    loop = Loop(args, run_eval_fn=fake_eval, judge_pending_fn=lambda d: 0, improver_llm=improver, log=lambda m: None)
    return loop, fake_eval, improver


def baseline_fails_out_of_scope(dir_name, sid, sample):
    return not (dir_name.startswith("baseline") and sid == "out_of_scope")


def test_loop_accepts_improvement_and_writes_report(tmp_path, v1):
    loop, _, improver = make_loop(tmp_path, v1, baseline_fails_out_of_scope)
    assert loop.run() == 0
    gate = json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())
    assert gate["accepted"] and gate["candidate_score"] > gate["baseline_score"]
    report = (tmp_path / "results" / "comparison.md").read_text()
    assert "Gate: ACCEPTED" in report and "| out_of_scope | 0/3 | 3/3 | +3 |" in report
    assert (tmp_path / "prompts" / "system_v2.md").exists() and len(improver.messages) == 1


def test_loop_rejects_regression_beyond_tolerance(tmp_path, v1):
    def outcome(d, sid, sample):
        if d.startswith("baseline"):
            return sid != "out_of_scope"
        return sid != "book_happy" or sample == 0       # out_of_scope fixed, book_happy drops 2 of 3
    loop, _, _ = make_loop(tmp_path, v1, outcome)
    assert loop.run() == 0
    gate = json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())
    assert not gate["accepted"] and "book_happy" in " ".join(gate["regressions"])
    assert "Gate: REJECTED" in (tmp_path / "results" / "comparison.md").read_text()


def test_heldout_regressions_do_not_affect_the_gate(tmp_path, v1):
    def outcome(d, sid, sample):
        if d == "candidate_v2_heldout":
            return False                                 # held-out collapses on the candidate
        return baseline_fails_out_of_scope(d, sid, sample)
    loop, _, improver = make_loop(tmp_path, v1, outcome, with_heldout=True)
    assert loop.run() == 0
    assert json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())["accepted"]
    report = (tmp_path / "results" / "comparison.md").read_text()
    assert "Held-out set (reported, never gated" in report
    assert "identity_mismatch" not in json.dumps(improver.messages)


def test_loop_resumes_after_quota_stop_without_redoing_phases(tmp_path, v1):
    loop, fake_eval, improver = make_loop(tmp_path, v1, baseline_fails_out_of_scope, stop_at="candidate_v2")
    assert loop.run() == 3
    state = json.loads((tmp_path / "results" / "loop" / "state.json").read_text())
    assert state["done"] == ["baseline", "analyze", "apply"]
    loop2 = Loop(loop.args, run_eval_fn=fake_eval, judge_pending_fn=lambda d: 0, improver_llm=improver,
                 log=lambda m: None)
    assert loop2.run() == 0
    assert len(improver.messages) == 1                   # improver not called again on resume


def test_n2_drop_needs_confirming_rerun_with_new_samples(tmp_path, v1):
    def outcome(d, sid, sample):
        if d.startswith("baseline"):
            return sid != "out_of_scope"
        if d == "candidate_v2":
            return not (sid == "book_happy" and sample == 1)   # one-run drop at N=2
        return True                                             # confirming rerun recovers
    loop, fake_eval, _ = make_loop(tmp_path, v1, outcome, n=2)
    assert loop.run() == 0
    assert ("candidate_v2_confirm", 2) in fake_eval.calls      # samples 2..3, not the cached 0..1
    assert json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())["accepted"]


def test_n2_drop_that_reproduces_is_rejected(tmp_path, v1):
    def outcome(d, sid, sample):
        if d.startswith("baseline"):
            return sid != "out_of_scope"
        return not (sid == "book_happy" and sample % 2 == 1)    # drop reproduces in the confirm run
    loop, _, _ = make_loop(tmp_path, v1, outcome, n=2)
    assert loop.run() == 0
    assert not json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())["accepted"]


def test_no_applicable_improvement_means_no_candidate_and_rejection(tmp_path, v1):
    loop, fake_eval, _ = make_loop(tmp_path, v1, baseline_fails_out_of_scope, improvements=[
        imp("code_guardrail", "code", "Block escalations before verification.", ["out_of_scope__s0"], op="describe")])
    assert loop.run() == 0
    gate = json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())
    assert not gate["accepted"] and "no candidate" in gate["reasons"][0]
    assert not any(name.startswith("candidate") for name, _ in fake_eval.calls)
    assert "recorded for a human, not applied" in (tmp_path / "results" / "comparison.md").read_text()


# ---------------------------------------------------------------- reduced gate

def test_reduced_gate_rules():
    from improve.gate import evaluate_reduced
    base = {"out_of_scope": 0.0, "medical_advice": 2 / 3}
    assert evaluate_reduced(base, {"out_of_scope": 1.0, "medical_advice": 2 / 3}, {"cancel_happy": True}).accepted
    assert not evaluate_reduced(base, {"out_of_scope": 0.0, "medical_advice": 2 / 3}, {"cancel_happy": True}).accepted
    pending = evaluate_reduced(base, {"out_of_scope": 1.0, "medical_advice": 1.0}, {"cancel_happy": False})
    assert not pending.accepted and pending.needs_confirmation == ["cancel_happy"]
    assert evaluate_reduced(base, {"out_of_scope": 1.0, "medical_advice": 1.0}, {"cancel_happy": False},
                            {"cancel_happy": True}).accepted
    assert not evaluate_reduced(base, {"out_of_scope": 1.0, "medical_advice": 1.0}, {"cancel_happy": False},
                                {"cancel_happy": False}).accepted


def make_reduced_loop(tmp_path, v1, outcome, regression=("cancel_happy", "prompt_injection")):
    loop, fake_eval, improver = make_loop(tmp_path, v1, outcome)
    loop.args.reduced = True
    loop.args.regression_check = list(regression)
    return loop, fake_eval, improver


def test_reduced_loop_runs_failing_at_n3_and_regression_at_n1(tmp_path, v1):
    loop, fake_eval, _ = make_reduced_loop(tmp_path, v1, baseline_fails_out_of_scope)
    assert loop.run() == 0
    runs = sorted(p.name for p in (tmp_path / "results" / "candidate_v2_reduced" / "runs").glob("*.json"))
    assert runs == ["cancel_happy__s0.json", "out_of_scope__s0.json", "out_of_scope__s1.json",
                    "out_of_scope__s2.json", "prompt_injection__s0.json"]
    gate = json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())
    assert gate["accepted"]
    report = (tmp_path / "results" / "comparison.md").read_text()
    assert "REDUCED GATE" in report and "Not re-evaluated on the candidate" in report


def test_reduced_loop_confirms_a_failed_regression_run(tmp_path, v1):
    def outcome(d, sid, sample):
        if d.startswith("baseline"):
            return sid != "out_of_scope"
        if d == "candidate_v2_reduced" and sid == "cancel_happy":
            return False                                   # N=1 regression run fails...
        return True                                        # ...the confirming rerun passes
    loop, fake_eval, _ = make_reduced_loop(tmp_path, v1, outcome)
    assert loop.run() == 0
    assert ("candidate_v2_reduced_confirm", 1) in fake_eval.calls
    assert json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())["accepted"]


def test_reduced_loop_rejects_when_regression_reproduces(tmp_path, v1):
    def outcome(d, sid, sample):
        if d.startswith("baseline"):
            return sid != "out_of_scope"
        return sid != "cancel_happy"
    loop, _, _ = make_reduced_loop(tmp_path, v1, outcome)
    assert loop.run() == 0
    gate = json.loads((tmp_path / "results" / "loop" / "gate.json").read_text())
    assert not gate["accepted"] and "cancel_happy" in " ".join(gate["regressions"])


def test_reduced_loop_quota_stop_writes_incomplete_report(tmp_path, v1):
    loop, fake_eval, _ = make_reduced_loop(tmp_path, v1, baseline_fails_out_of_scope,
                                           regression=("cancel_happy", "prompt_injection"))
    real_call = fake_eval.__call__

    def stop_on_injection(scenarios, n, prompt, tools, out=None, sample_offset=0, **kw):
        if Path(out).name.startswith("candidate") and any(s.id == "prompt_injection" for s in scenarios):
            return 3                                          # daily quota hit on the 2nd regression check
        return real_call(scenarios, n, prompt, tools, out=out, sample_offset=sample_offset, **kw)
    loop._run_eval = stop_on_injection
    assert loop.run() == 3
    report = (tmp_path / "results" / "comparison.md").read_text()
    assert "REDUCED GATE: INCOMPLETE (no decision)" in report
    assert "Not run: prompt_injection" in report
    assert "| cancel_happy | 0/0 | 1/1 |" in report or "| cancel_happy |" in report

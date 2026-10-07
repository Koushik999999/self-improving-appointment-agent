"""The improvement loop: baseline -> analyze failures -> apply -> re-eval -> gate -> report.

    python -m improve.loop --dry-run          # phase status + estimated calls/tokens; no API calls
    python -m improve.loop                    # run (or resume) the whole loop
    python -m improve.loop --with-heldout     # also evaluate the held-out set (reported, never gated)

Phases are checkpointed in <out>/state.json and their artifacts; re-running the same command skips
finished phases and resumes unfinished evals (which are themselves checkpointed per conversation).
A daily-quota stop exits with code 3; run the same command again later.

Isolation of the held-out set: held-out results live in their own directories, the improver only
reads the main baseline directory, and analyze.failing_main_runs filters by set again.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from evals.scenario import load_scenarios

from . import gate as gate_mod
from .analyze import analyze, build_messages, failing_main_runs, load_records
from .apply import write_candidate

ROOT = Path(__file__).resolve().parent.parent
PHASES = ["baseline", "analyze", "apply", "candidate_eval", "confirm", "gate", "report"]


def read_summary(results_dir: Path) -> dict | None:
    path = Path(results_dir) / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main_rates(summary: dict) -> dict[str, float]:
    return {sid: s["pass_rate"] for sid, s in summary["scenarios"].items() if s["set"] == "main"}


def eval_complete(results_dir: Path, scenarios, n: int, sample_offset: int = 0) -> bool:
    runs = Path(results_dir) / "runs"
    for sample in range(sample_offset, sample_offset + n):
        for sc in scenarios:
            path = runs / f"{sc.id}__s{sample}.json"
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("passed") is None:
                return False
    return True


class Loop:
    def __init__(self, args, run_eval_fn=None, judge_pending_fn=None, improver_llm=None, log=print):
        self.args = args
        self.out = Path(args.out)
        self.log = log
        self._run_eval = run_eval_fn
        self._judge_pending = judge_pending_fn
        self._improver = improver_llm
        self.main = load_scenarios("main")
        self.heldout = load_scenarios("heldout")
        self.state_path = self.out / "state.json"
        self.results_root = Path(getattr(args, "results_root", None) or ROOT / "results")

    # ---- plumbing (real implementations imported lazily so --dry-run and tests stay offline)

    def run_eval(self, *a, **kw):
        if self._run_eval is None:
            from evals.run import run_eval
            self._run_eval = run_eval
        return self._run_eval(*a, **kw)

    def judge_pending(self, out_dir):
        if self._judge_pending is None:
            from evals.run import judge_pending
            self._judge_pending = judge_pending
        return self._judge_pending(Path(out_dir))

    def improver(self):
        if self._improver is None:
            from llm import LLM
            self._improver = LLM("improver")
        return self._improver

    def config(self) -> dict:
        a = self.args
        return {"baseline": str(a.baseline), "prompt": str(a.prompt), "tools": str(a.tools), "n": a.n,
                "version": a.version, "with_heldout": a.with_heldout}

    def load_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state["config"] != self.config():
                raise SystemExit(f"{self.out} holds a loop with a different config {state['config']}; "
                                 "use --out for a new loop or --fresh.")
            return state
        return {"config": self.config(), "done": []}

    def save_state(self, state: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @property
    def dirs(self) -> dict[str, Path]:
        base, v = Path(self.args.baseline), self.args.version
        return {"baseline": base, "baseline_heldout": Path(f"{base}_heldout"),
                "candidate": self.results_root / f"candidate_{v}",
                "candidate_heldout": self.results_root / f"candidate_{v}_heldout",
                "confirm": self.results_root / f"candidate_{v}_confirm"}

    def ensure_eval(self, scenarios, prompt, tools, out_dir, sample_offset=0) -> int:
        """Generate what's missing, then judge what's pending. Returns 0 when complete."""
        n = self.args.n
        if not eval_complete(out_dir, scenarios, n, sample_offset):
            code = self.run_eval(scenarios, n, prompt, tools, out=out_dir, sample_offset=sample_offset)
            if code not in (0,):
                return code
            if not eval_complete(out_dir, scenarios, n, sample_offset):
                code = self.judge_pending(out_dir)
                if code:
                    return code
        return 0 if eval_complete(out_dir, scenarios, n, sample_offset) else 1

    # ---- phases

    def run(self) -> int:
        state = self.load_state()
        for phase in PHASES:
            if phase in state["done"]:
                self.log(f"[skip] {phase} (done)")
                continue
            self.log(f"[run ] {phase}")
            code = getattr(self, f"phase_{phase}")(state)
            if code:
                self.save_state(state)
                self.log(f"Stopped in phase '{phase}' (code {code}). Re-run the same command to resume."
                         + (" Daily quota reached." if code == 3 else ""))
                return code
            state["done"].append(phase)
            self.save_state(state)
        self.log(f"Loop complete. Report: {self.out / 'comparison.md'}")
        return 0

    def phase_baseline(self, state) -> int:
        a, d = self.args, self.dirs
        code = self.ensure_eval(self.main, a.prompt, a.tools, d["baseline"])
        if not code and a.with_heldout:
            code = self.ensure_eval(self.heldout, a.prompt, a.tools, d["baseline_heldout"])
        return code

    def phase_analyze(self, state) -> int:
        a = self.args
        # Only the MAIN baseline directory is read; held-out results are never loaded here.
        records = load_records(self.dirs["baseline"])
        prompt = Path(a.prompt).read_text(encoding="utf-8")
        tools = json.loads(Path(a.tools).read_text(encoding="utf-8"))
        result = analyze(records, prompt, tools, self.improver())
        (self.out / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.log(f"  improver: {len(result['accepted'])} accepted, {len(result['rejected'])} rejected, "
                 f"{len(result['recorded_only'])} code/state proposals recorded (not applied)")
        return 0

    def phase_apply(self, state) -> int:
        a = self.args
        analysis = json.loads((self.out / "analysis.json").read_text(encoding="utf-8"))
        if not analysis["accepted"]:
            state["candidate"] = None
            self.log("  no applicable improvements: no candidate to evaluate")
            return 0
        info = write_candidate(a.prompt, a.tools, analysis["accepted"], a.version,
                               Path(a.prompt).parent, self.out / f"diff_{Path(a.prompt).stem}_to_{a.version}.patch")
        state["candidate"] = info
        shutil.copy(info["diff"], self.results_root / Path(info["diff"]).name)
        self.log(f"  wrote {info['prompt']} and {info['tools']}; diff {info['diff']}")
        return 0

    def phase_candidate_eval(self, state) -> int:
        cand, d = state.get("candidate"), self.dirs
        if not cand:
            return 0
        code = self.ensure_eval(self.main, cand["prompt"], cand["tools"], d["candidate"])
        if not code and self.args.with_heldout:
            code = self.ensure_eval(self.heldout, cand["prompt"], cand["tools"], d["candidate_heldout"])
        return code

    def gate_now(self, state, confirm_rates=None):
        base = main_rates(read_summary(self.dirs["baseline"]))
        cand_summary = read_summary(self.dirs["candidate"]) if state.get("candidate") else None
        cand = main_rates(cand_summary) if cand_summary else {k: 0.0 for k in base}
        return gate_mod.evaluate(base, cand, self.args.n, confirm_rates)

    def phase_confirm(self, state) -> int:
        """N <= 2 only: re-run scenarios whose drop isn't tolerated, with NEW samples (offset N)."""
        if not state.get("candidate") or self.args.n >= 3:
            return 0
        pending = self.gate_now(state).needs_confirmation
        state["confirm_scenarios"] = pending
        if not pending:
            return 0
        scenarios = [s for s in self.main if s.id in pending]
        cand = state["candidate"]
        return self.ensure_eval(scenarios, cand["prompt"], cand["tools"], self.dirs["confirm"],
                                sample_offset=self.args.n)

    def phase_gate(self, state) -> int:
        confirm_rates = None
        if state.get("confirm_scenarios"):
            confirm_rates = {k: v for k, v in main_rates(read_summary(self.dirs["confirm"])).items()
                             if k in state["confirm_scenarios"]}
        result = self.gate_now(state, confirm_rates)
        if not state.get("candidate"):
            result.accepted = False
            result.reasons.insert(0, "no applicable improvements were proposed, so there is no candidate")
        state["gate"] = result.__dict__
        (self.out / "gate.json").write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
        self.log(f"  gate: {'ACCEPTED' if result.accepted else 'REJECTED'} "
                 f"({result.baseline_score:.3f} -> {result.candidate_score:.3f})" +
                 ("" if result.accepted else f"; {'; '.join(result.reasons)}"))
        return 0

    def phase_report(self, state) -> int:
        text = comparison_markdown(self, state)
        (self.out / "comparison.md").write_text(text, encoding="utf-8")
        (self.results_root / "comparison.md").write_text(text, encoding="utf-8")
        return 0


# ---------------------------------------------------------------- report

def _rate(s):
    return f"{s['passed']}/{s['scored']}" if s else "-"


def comparison_markdown(loop: Loop, state: dict) -> str:
    a, d = loop.args, loop.dirs
    g = state["gate"]
    base = read_summary(d["baseline"])
    cand = read_summary(d["candidate"]) if state.get("candidate") else None
    analysis = json.loads((loop.out / "analysis.json").read_text(encoding="utf-8"))
    lines = [f"# Improvement loop: {Path(a.prompt).stem} -> {a.version}", "",
             f"**Gate: {'ACCEPTED' if g['accepted'] else 'REJECTED'}.** Main-set score "
             f"{g['baseline_score']:.3f} -> {g['candidate_score']:.3f} (N={a.n} runs per scenario; "
             f"tolerated drop per scenario: {gate_mod.tolerated_drop_runs(a.n)} run(s)"
             + (", with confirming reruns" if a.n <= 2 else "") + ").", ""]
    if g["reasons"]:
        lines += ["Reasons: " + "; ".join(g["reasons"]), ""]
    lines += ["Rubric/checks: " + f"{base['meta'].get('rubric_version')} / {base['meta'].get('checks_version')}"
              + (" (same for both versions)" if not cand or (cand['meta'].get('rubric_hash') == base['meta'].get('rubric_hash'))
                 else " **(MISMATCH: comparison invalid)**"), ""]

    lines += ["## Main set (gated)", "", f"| scenario | {Path(a.prompt).stem} | {a.version} | change |", "|---|---|---|---|"]
    for sid, b in base["scenarios"].items():
        if b["set"] != "main":
            continue
        c = cand["scenarios"].get(sid) if cand else None
        drop = gate_mod.drop_in_runs(b["pass_rate"], c["pass_rate"], a.n) if c else None
        gain = round((c["pass_rate"] - b["pass_rate"]) * a.n) if c else None
        note = ("-" if c is None else f"regression ({drop} run(s))" if drop and drop > gate_mod.tolerated_drop_runs(a.n)
                else f"-{drop} run (tolerated)" if drop else f"+{gain}" if gain else "=")
        lines.append(f"| {sid} | {_rate(b)} | {_rate(c)} | {note} |")

    if a.with_heldout:
        bh, ch = read_summary(d["baseline_heldout"]), read_summary(d["candidate_heldout"]) if cand else None
        lines += ["", "## Held-out set (reported, never gated; the improver never saw it)", "",
                  f"| scenario | {Path(a.prompt).stem} | {a.version} |", "|---|---|---|"]
        for sid, b in (bh or {"scenarios": {}})["scenarios"].items():
            c = ch["scenarios"].get(sid) if ch else None
            lines.append(f"| {sid} | {_rate(b)} | {_rate(c)} |")
        if bh and ch:
            lines += ["", f"Held-out score: {bh['score']['heldout']:.3f} -> {ch['score']['heldout']:.3f}"]

    lines += ["", "## Improvements", ""]
    for imp in analysis["accepted"]:
        lines.append(f"- **applied** [{imp['category']}] {imp['root_cause']} (runs: {', '.join(imp['failure_ids'])})")
    for rej in analysis["rejected"]:
        lines.append(f"- **rejected** [{rej['improvement']['category']}] {rej['improvement']['root_cause']}: "
                     + "; ".join(rej["reasons"]))
    for rec in analysis["recorded_only"]:
        lines.append(f"- **recorded for a human, not applied** [{rec['category']}] {rec['root_cause']}: "
                     f"{rec['proposed_patch']['text']}")
    if not (analysis["accepted"] or analysis["rejected"] or analysis["recorded_only"]):
        lines.append("- none (" + analysis.get("note", "improver returned no proposals") + ")")
    if state.get("candidate"):
        lines += ["", f"Diff: `{Path(state['candidate']['diff']).name}`"]

    lines += ["", "## Other signals", "", "| | " + Path(a.prompt).stem + " | " + a.version + " |", "|---|---|---|"]
    for key, label in (("malformed_tool_calls", "malformed tool calls"), ("fallback_replies", "fallback replies"),
                       ("judge_skipped", "judge skipped (deterministic failed)"),
                       ("sim_lines_sanitized", "simulator lines cut")):
        lines.append(f"| {label} | {base['totals'].get(key, '-')} | {cand['totals'].get(key, '-') if cand else '-'} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- dry run

def dry_run(loop: Loop) -> None:
    from agent.prompt_size import fixed_cost
    from evals.estimate import ASSUME, pass_load
    from llm import estimate_tokens
    a, d = loop.args, loop.dirs
    state = loop.load_state() if loop.state_path.exists() else {"done": []}
    print(f"DRY RUN (no API calls). Loop dir {loop.out}; N={a.n}; held-out {'on' if a.with_heldout else 'off'}.\n")
    for phase in PHASES:
        print(f"  {'done' if phase in state['done'] else 'todo':<5} {phase}")

    from llm import role_config
    fixed = fixed_cost(a.prompt, a.tools)["total_tokens_est"]
    print("\nEstimated cost of the remaining work:")
    base_records = load_records(d["baseline"]) if (Path(d["baseline"]) / "runs").exists() else []
    observed = [r["judge"]["usage"]["total_tokens"] for r in base_records
                if r.get("judge") and (r["judge"].get("usage") or {}).get("total_tokens")]
    per_judge = (sum(observed) / len(observed) if observed else
                 ASSUME["judge_rubric_tokens"] + 8 * ASSUME["judge_tokens_per_turn"] + ASSUME["judge_output_tokens"])
    source = f"observed mean of {len(observed)} calls" if observed else "assumed"
    if "baseline" not in state["done"]:
        pending = [r for r in base_records if r.get("judge_pending")]
        missing = max(a.n * len(loop.main) - len(base_records), 0)
        print(f"  baseline (main): {missing} conversations to generate, {len(pending)} awaiting the judge "
              f"(~{len(pending) * per_judge:,.0f} judge tokens, {per_judge:,.0f}/call {source})")
        if a.with_heldout:
            held_dir = Path(d["baseline_heldout"]) / "runs"
            have = len(list(held_dir.glob("*.json"))) if held_dir.exists() else 0
            hl = pass_load(loop.heldout, a.n, fixed)
            print(f"  baseline (held-out): {a.n * len(loop.heldout) - have} conversations to generate "
                  f"(agent ~{hl['agent'].calls:.0f} calls, judge <= {hl['judge'].calls:.0f} calls)")
    if base_records and "analyze" not in state["done"]:
        prompt = Path(a.prompt).read_text(encoding="utf-8")
        tools = json.loads(Path(a.tools).read_text(encoding="utf-8"))
        n_fail = len(failing_main_runs(base_records))
        msgs = build_messages(base_records, prompt, tools) if n_fail else []
        print(f"  analyze: 1 improver call, ~{sum(estimate_tokens(m['content']) for m in msgs):,} prompt tokens "
              f"from {n_fail} failing main-set runs scored so far (+ output)")
    sets = loop.main + (loop.heldout if a.with_heldout else [])
    load = pass_load(sets, a.n, fixed)
    agent_rpd, judge_tpd = role_config("agent").rpd, role_config("judge").tpd
    print(f"  candidate eval ({len(sets)} scenarios x {a.n}): agent ~{load['agent'].calls:.0f} calls"
          + (f" (agent RPD {agent_rpd}: needs {-(-int(load['agent'].calls) // agent_rpd)} day(s))" if agent_rpd else "")
          + f"; sim ~{load['sim'].calls:.0f}; judge <= {load['judge'].calls:.0f} calls / "
          f"~{load['judge'].calls * per_judge:,.0f} tok"
          + (f" (judge TPD {judge_tpd:,})" if judge_tpd else ""))
    if a.n <= 2:
        print(f"  confirm (only if a scenario drops): up to {len(loop.main) * a.n} more conversations")
    print("  gate + report: no API calls")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run the improvement loop.")
    ap.add_argument("--baseline", default=str(ROOT / "results" / "baseline_v1"))
    ap.add_argument("--prompt", default=str(ROOT / "prompts" / "system_v1.md"))
    ap.add_argument("--tools", default=str(ROOT / "prompts" / "tools_v1.json"))
    ap.add_argument("--version", default="v2", help="name of the candidate version")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--with-heldout", action="store_true")
    ap.add_argument("--out", default=None, help="loop directory (default results/loop_<prompt>_to_<version>)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard this loop's phase state (not the evals)")
    args = ap.parse_args(argv)
    args.out = args.out or str(ROOT / "results" / f"loop_{Path(args.prompt).stem}_to_{args.version}")
    return args


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    loop = Loop(args)
    if args.fresh and loop.state_path.exists():
        loop.state_path.unlink()
    if args.dry_run:
        dry_run(loop)
        return 0
    return loop.run()


if __name__ == "__main__":
    sys.exit(main())

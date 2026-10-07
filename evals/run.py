"""Run scenarios against an agent version and score them.

    python -m evals.run --dry-run                      # estimate calls/tokens/time/quota, no API calls
    python -m evals.run --n 1                          # main set, 1 run per scenario
    python -m evals.run --set heldout --n 3
    python -m evals.run --prompt prompts/system_v2.md --tools prompts/tools_v2.json

Each finished conversation is checkpointed to <out>/runs/<scenario>__s<sample>.json. Re-running the
same command resumes: finished conversations are loaded, not re-run (so a daily-quota cutoff loses
nothing). Raising --n later reuses samples 0..N-1 and only runs the new ones. Infra errors are not
checkpointed, so they are retried on resume.

Scoring per conversation: deterministic checks (evals/checks.py) AND the LLM judge must both pass.
By default the judge is skipped when the deterministic layer already failed (the run fails either
way); skipped runs are marked judge_skipped in results.json. --judge-all judges every run.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.agent import DEFAULT_PROMPT, DEFAULT_TOOLS, Agent
from clinic import Clinic
from clinic.faults import FaultInjector
from llm import LLM, QuotaExhausted, env_int, role_config, stats_snapshot

from . import judge as judge_mod
from .checks import CHECKS_VERSION, Context, run_checks
from .scenario import load_scenarios
from .simulator import Patient

ROOT = Path(__file__).resolve().parent.parent


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_conversation(sc, sample: int, prompt: str, tools: str, llms: dict, judge_all: bool) -> dict:
    clinic = Clinic(patches=sc.db_patches)
    initial = clinic.snapshot()
    faults = FaultInjector(sc.faults)
    agent = Agent(clinic, prompt, tools, faults, llm=llms["agent"], sample=sample)
    patient = Patient(sc, llms.get("sim"), sample)

    turns, malformed, fallbacks = [], 0, 0
    for i in range(sc.max_turns):
        line = patient.next_line(i, turns)
        if line is None:
            break
        turn = agent.respond(line)
        malformed += turn.malformed
        fallbacks += int(turn.fallback)
        turns.append({"turn": i + 1, "patient": line, "agent": turn.reply})

    ctx = Context(sc, initial, clinic.snapshot(), clinic, agent.tools.trace, turns, agent.state)
    checks = run_checks(ctx)
    det_pass = all(c.passed for c in checks)

    judged, skipped = None, False
    if det_pass or judge_all:
        judged = judge_mod.judge(sc, turns, agent.tools.trace, llms["judge"], sample)
    else:
        skipped = True
    passed = det_pass and (judged["passed"] if judged else False)

    return {
        "scenario": sc.id, "set": sc.set, "sample": sample, "passed": passed,
        "rubric_version": judge_mod.RUBRIC_VERSION, "checks_version": CHECKS_VERSION,
        "deterministic": {"passed": det_pass, "checks": [c.to_dict() for c in checks]},
        "judge": judged, "judge_skipped": skipped,
        "turns": turns,
        "trace": [{**t, "args": t["args"]} for t in agent.tools.trace],
        "final_state": agent.state.to_dict(),
        "faults_fired": faults.fired,
        "db": {"initial": initial, "final": ctx.final},
        "stats": {"agent_llm_calls": agent.llm_calls_total, "agent_tokens": agent.tokens,
                  "sim_llm_calls": patient.llm_calls, "sim_lines_sanitized": patient.sanitized,
                  "malformed_tool_calls": malformed,
                  "fallback_replies": fallbacks, "patient_turns": len(turns)},
    }


def summarize(records: list[dict], meta: dict) -> dict:
    by_sc: dict[str, list[dict]] = {}
    for r in records:
        by_sc.setdefault(r["scenario"], []).append(r)
    scenarios = {}
    for sid, runs in sorted(by_sc.items()):
        failing_checks = Counter(c["name"] for r in runs for c in r["deterministic"]["checks"] if not c["passed"])
        failing_items = Counter(k for r in runs if r["judge"] for k, v in r["judge"]["items"].items()
                                if v["verdict"] == "fail")
        scenarios[sid] = {
            "set": runs[0]["set"], "runs": len(runs), "passed": sum(r["passed"] for r in runs),
            "pass_rate": sum(r["passed"] for r in runs) / len(runs),
            "det_passed": sum(r["deterministic"]["passed"] for r in runs),
            "judge_passed": sum(bool(r["judge"] and r["judge"]["passed"]) for r in runs),
            "judge_skipped": sum(r["judge_skipped"] for r in runs),
            "failing_checks": dict(failing_checks), "failing_judge_items": dict(failing_items),
            "malformed_tool_calls": sum(r["stats"]["malformed_tool_calls"] for r in runs),
        }

    def score(set_name):
        rates = [s["pass_rate"] for s in scenarios.values() if s["set"] == set_name]
        return sum(rates) / len(rates) if rates else None

    judge_usage = [r["judge"]["usage"] for r in records if r["judge"] and not r["judge"].get("cached")]
    return {
        "meta": meta,
        "score": {"main": score("main"), "heldout": score("heldout")},
        "scenarios": scenarios,
        "totals": {
            "conversations": len(records),
            "judge_skipped": sum(r["judge_skipped"] for r in records),
            "judge_skipped_runs": [f"{r['scenario']}__s{r['sample']}" for r in records if r["judge_skipped"]],
            "malformed_tool_calls": sum(r["stats"]["malformed_tool_calls"] for r in records),
            "fallback_replies": sum(r["stats"]["fallback_replies"] for r in records),
            "agent_llm_calls": sum(r["stats"]["agent_llm_calls"] for r in records),
            "sim_llm_calls": sum(r["stats"]["sim_llm_calls"] for r in records),
            "sim_lines_sanitized": sum(r["stats"].get("sim_lines_sanitized", 0) for r in records),
            "judge_tokens_per_call": [u.get("total_tokens") for u in judge_usage],
        },
    }


def summary_markdown(summary: dict) -> str:
    m = summary["meta"]
    lines = [f"# Eval: {m['prompt']} + {m['tools']}", "",
             f"Rubric {m.get('rubric_version', 'v1')}, deterministic checks {m.get('checks_version', 'v1')}.", "",
             f"Models: agent `{m['models']['agent']}`, sim `{m['models']['sim']}`, judge `{m['models']['judge']}`. "
             f"N={m['n']} per scenario.", ""]
    for set_name in ("main", "heldout"):
        sc = {k: v for k, v in summary["scenarios"].items() if v["set"] == set_name}
        if not sc:
            continue
        score = summary["score"][set_name]
        lines += [f"## {set_name} set: score {score:.2f} (mean pass rate over {len(sc)} scenarios)", "",
                  "| scenario | pass | deterministic | judge | judge skipped | failing checks | failing judge items |",
                  "|---|---|---|---|---|---|---|"]
        for sid, s in sc.items():
            checks = ", ".join(f"{k}" + (f" x{v}" if v > 1 else "") for k, v in s["failing_checks"].items()) or "-"
            items = ", ".join(f"{k}" + (f" x{v}" if v > 1 else "") for k, v in s["failing_judge_items"].items()) or "-"
            judged = s["runs"] - s["judge_skipped"]
            lines.append(f"| {sid} | {s['passed']}/{s['runs']} | {s['det_passed']}/{s['runs']} | "
                         f"{s['judge_passed']}/{judged} | {s['judge_skipped']} | {checks} | {items} |")
        lines.append("")
    t = summary["totals"]
    jt = [x for x in t["judge_tokens_per_call"] if x]
    lines += ["## Totals", "",
              f"- conversations: {t['conversations']}; judge skipped (deterministic already failed): {t['judge_skipped']}",
              f"- malformed tool calls: {t['malformed_tool_calls']}; fallback replies: {t['fallback_replies']}",
              f"- agent LLM calls: {t['agent_llm_calls']}; simulator LLM calls: {t['sim_llm_calls']} "
              f"(lines cut for speaking as the assistant: {t['sim_lines_sanitized']})",
              f"- judge tokens per call (actual): " + (f"mean {sum(jt) / len(jt):.0f}, max {max(jt)}" if jt else "n/a"),
              ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the scenario eval.")
    ap.add_argument("--set", choices=["main", "heldout", "all"], default="main")
    ap.add_argument("--n", type=int, default=3, help="runs per scenario (default 3)")
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--tools", default=str(DEFAULT_TOOLS))
    ap.add_argument("--scenarios", help="comma-separated scenario ids")
    ap.add_argument("--out", help="results directory (default results/<prompt>__<tools>)")
    ap.add_argument("--judge-all", action="store_true", help="judge runs even if deterministic checks failed")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="estimate calls/tokens/time/quota; no API calls")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints in --out")
    ap.add_argument("--replay-only", action="store_true",
                    help="agent and simulator answer only from the response cache (no live calls); "
                         "used to re-score existing conversations under new checks/rubric")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ids = args.scenarios.split(",") if args.scenarios else None
    if args.dry_run:
        from .estimate import print_report
        print_report(load_scenarios("main", ids), load_scenarios("heldout", ids), args.prompt, args.tools)
        return 0
    return run_eval(load_scenarios(args.set, ids), args.n, args.prompt, args.tools,
                    out=args.out, judge_all=args.judge_all, concurrency=args.concurrency, fresh=args.fresh,
                    replay_only=args.replay_only)


def run_eval(scenarios, n, prompt, tools, out=None, judge_all=False, concurrency=None, fresh=False,
             replay_only=False) -> int:
    out_dir = Path(out) if out else ROOT / "results" / f"{Path(prompt).stem}__{Path(tools).stem}"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    meta = {"prompt": str(Path(prompt).relative_to(ROOT)) if Path(prompt).is_absolute() else prompt,
            "tools": str(Path(tools).relative_to(ROOT)) if Path(tools).is_absolute() else tools,
            "prompt_hash": file_hash(prompt), "tools_hash": file_hash(tools), "n": n,
            "models": {r: role_config(r).model for r in ("agent", "sim", "judge")},
            "rubric_version": judge_mod.RUBRIC_VERSION, "checks_version": CHECKS_VERSION,
            "rubric_hash": hashlib.sha256(judge_mod.RUBRIC.encode("utf-8")).hexdigest()[:12],
            "checks_hash": file_hash(ROOT / "evals" / "checks.py")}

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not fresh:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = ("prompt_hash", "tools_hash", "models", "rubric_version", "checks_version", "rubric_hash")
        if any(old.get(k) != meta[k] for k in keys):
            print(f"{out_dir} holds results for a different prompt/tools/model/rubric/checks version. "
                  "Use --out for a new directory or --fresh to discard them.")
            return 2
    if fresh:
        for f in runs_dir.glob("*.json"):
            f.unlink()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    llms = {"agent": LLM("agent", cache_only=replay_only), "judge": LLM("judge")}
    if any(sc.llm_turns or sc.then == "llm" for sc in scenarios):
        llms["sim"] = LLM("sim", cache_only=replay_only)

    jobs, records = [], []
    for sample in range(n):
        for sc in scenarios:
            path = runs_dir / f"{sc.id}__s{sample}.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                jobs.append((sc, sample, path))
    print(f"{len(records)} conversations loaded from checkpoints, {len(jobs)} to run -> {out_dir}")

    stop = threading.Event()
    lock = threading.Lock()
    errors = []

    def work(sc, sample, path):
        if stop.is_set():
            return None
        record = run_conversation(sc, sample, prompt, tools, llms, judge_all)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return record

    started = time.time()
    quota_hit = None
    with ThreadPoolExecutor(max_workers=concurrency or env_int("EVAL_CONCURRENCY", 2)) as pool:
        futures = {pool.submit(work, *job): job for job in jobs}
        for fut in as_completed(futures):
            sc, sample, _ = futures[fut]
            try:
                record = fut.result()
            except QuotaExhausted as e:
                quota_hit = str(e)
                stop.set()
                continue
            except Exception as e:  # infra failure: report, don't checkpoint (retried on resume)
                errors.append(f"{sc.id}__s{sample}: {type(e).__name__}: {str(e)[:200]}")
                print(f"  ERROR {sc.id} s{sample}: {type(e).__name__}: {str(e)[:160]}")
                continue
            if record is None:
                continue
            with lock:
                records.append(record)
                det = "ok" if record["deterministic"]["passed"] else "FAIL"
                jdg = "skipped" if record["judge_skipped"] else ("ok" if record["judge"]["passed"] else "FAIL")
                print(f"  [{len(records)}/{n * len(scenarios)}] {sc.id} s{sample}: "
                      f"{'PASS' if record['passed'] else 'FAIL'} (deterministic {det}, judge {jdg}) "
                      f"{time.time() - started:.0f}s")

    summary = summarize(records, meta) if records else None
    if summary:
        summary["run_info"] = {"errors": errors, "quota_stop": quota_hit, "llm_stats": stats_snapshot(),
                               "complete": not errors and not quota_hit and len(records) == n * len(scenarios)}
        (out_dir / "results.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        (out_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
        print("\n" + summary_markdown(summary))
    if quota_hit:
        print(f"STOPPED: {quota_hit}\nFinished conversations are checkpointed; re-run the same command to resume.")
        return 3
    if errors:
        print(f"{len(errors)} conversation(s) failed with infra errors; re-run the same command to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

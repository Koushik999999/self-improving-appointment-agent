"""Run scenarios against an agent version and score them.

    python -m evals.run --dry-run                      # estimate calls/tokens/time/quota, no API calls
    python -m evals.run --n 1                          # main set, 1 run per scenario
    python -m evals.run --set heldout --n 3
    python -m evals.run --prompt prompts/system_v2.md --tools prompts/tools_v2.json
    python -m evals.run --n 3 --generate-only          # agent + sim + deterministic checks; judge later
    python -m evals.run --n 3 --judge-only             # judge saved transcripts that still need it

Each finished conversation is checkpointed to <out>/runs/<scenario>__s<sample>.json. Re-running the
same command resumes: finished conversations are loaded, not re-run (so a daily-quota cutoff loses
nothing). Raising --n later reuses samples 0..N-1 and only runs the new ones. Infra errors are not
checkpointed, so they are retried on resume.

Scoring per conversation: deterministic checks (evals/checks.py) AND the LLM judge must both pass.
By default the judge is skipped when the deterministic layer already failed (the run fails either
way); skipped runs are marked judge_skipped in results.json. --judge-all judges every run.

Generation and judging can be split (different providers have different daily quotas):
--generate-only checkpoints transcripts with `judge_pending: true` and `passed: null`; --judge-only
judges them later, and refuses unless the current rubric version and text hash match the ones the
results directory was created with. Pending runs are excluded from pass rates and listed as pending.
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


def rubric_hash() -> str:
    return hashlib.sha256(judge_mod.RUBRIC.encode("utf-8")).hexdigest()[:12]


def run_conversation(sc, sample: int, prompt: str, tools: str, llms: dict, judge_all: bool,
                     defer_judge: bool = False) -> dict:
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
    record = {
        "scenario": sc.id, "set": sc.set, "sample": sample, "passed": None,
        "rubric_version": judge_mod.RUBRIC_VERSION, "rubric_hash": rubric_hash(),
        "checks_version": CHECKS_VERSION,
        "deterministic": {"passed": all(c.passed for c in checks), "checks": [c.to_dict() for c in checks]},
        "judge": None, "judge_skipped": False, "judge_pending": False,
        "turns": turns,
        "trace": agent.tools.trace,
        "final_state": agent.state.to_dict(),
        "faults_fired": faults.fired,
        "db": {"initial": initial, "final": ctx.final},
        "stats": {"agent_llm_calls": agent.llm_calls_total, "agent_tokens": agent.tokens,
                  "sim_llm_calls": patient.llm_calls, "sim_lines_sanitized": patient.sanitized,
                  "malformed_tool_calls": malformed,
                  "fallback_replies": fallbacks, "patient_turns": len(turns)},
    }
    if not record["deterministic"]["passed"] and not judge_all:
        record["judge_skipped"] = True
        record["passed"] = False
    elif defer_judge:
        record["judge_pending"] = True
    else:
        apply_judge(record, sc, llms["judge"])
    return record


def apply_judge(record: dict, sc, judge_llm) -> None:
    record["judge"] = judge_mod.judge(sc, record["turns"], record["trace"], judge_llm, record["sample"])
    record["judge_pending"] = False
    record["passed"] = record["deterministic"]["passed"] and record["judge"]["passed"]


# ---------------------------------------------------------------- summaries

def summarize(records: list[dict], meta: dict) -> dict:
    by_sc: dict[str, list[dict]] = {}
    for r in records:
        by_sc.setdefault(r["scenario"], []).append(r)
    scenarios = {}
    for sid, runs in sorted(by_sc.items()):
        done = [r for r in runs if r["passed"] is not None]
        failing_checks = Counter(c["name"] for r in runs for c in r["deterministic"]["checks"] if not c["passed"])
        failing_items = Counter(k for r in runs if r["judge"] for k, v in r["judge"]["items"].items()
                                if v["verdict"] == "fail")
        scenarios[sid] = {
            "set": runs[0]["set"], "runs": len(runs), "scored": len(done),
            "pending": len(runs) - len(done),
            "passed": sum(bool(r["passed"]) for r in done),
            "pass_rate": sum(bool(r["passed"]) for r in done) / len(done) if done else None,
            "det_passed": sum(r["deterministic"]["passed"] for r in runs),
            "judge_passed": sum(bool(r["judge"] and r["judge"]["passed"]) for r in runs),
            "judge_skipped": sum(r["judge_skipped"] for r in runs),
            "failing_checks": dict(failing_checks), "failing_judge_items": dict(failing_items),
            "malformed_tool_calls": sum(r["stats"]["malformed_tool_calls"] for r in runs),
            "sim_lines_sanitized": sum(r["stats"].get("sim_lines_sanitized", 0) for r in runs),
        }

    def score(set_name):
        rates = [s["pass_rate"] for s in scenarios.values() if s["set"] == set_name and s["pass_rate"] is not None]
        return sum(rates) / len(rates) if rates else None

    judge_usage = [r["judge"]["usage"] for r in records if r["judge"] and not r["judge"].get("cached")]
    return {
        "meta": meta,
        "score": {"main": score("main"), "heldout": score("heldout")},
        "complete": all(s["pending"] == 0 for s in scenarios.values()),
        "scenarios": scenarios,
        "totals": {
            "conversations": len(records),
            "judge_pending": sum(r.get("judge_pending", False) for r in records),
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
             f"Rubric {m.get('rubric_version', 'v1')} (hash {m.get('rubric_hash', '-')}), "
             f"deterministic checks {m.get('checks_version', 'v1')}.", "",
             f"Models: agent `{m['models']['agent']}`, sim `{m['models']['sim']}`, judge `{m['models']['judge']}`. "
             f"N={m['n']} per scenario.", ""]
    if not summary["complete"]:
        lines += [f"**INCOMPLETE: {summary['totals']['judge_pending']} run(s) await judging "
                  "(`--judge-only`). Scores below cover scored runs only.**", ""]
    for set_name in ("main", "heldout"):
        sc = {k: v for k, v in summary["scenarios"].items() if v["set"] == set_name}
        if not sc:
            continue
        score = summary["score"][set_name]
        score_text = f"{score:.2f}" if score is not None else "n/a"
        lines += [f"## {set_name} set: score {score_text} (mean pass rate over {len(sc)} scenarios)", "",
                  "| scenario | pass | deterministic | judge | judge skipped | judge pending | failing checks | failing judge items |",
                  "|---|---|---|---|---|---|---|---|"]
        for sid, s in sc.items():
            checks = ", ".join(f"{k}" + (f" x{v}" if v > 1 else "") for k, v in s["failing_checks"].items()) or "-"
            items = ", ".join(f"{k}" + (f" x{v}" if v > 1 else "") for k, v in s["failing_judge_items"].items()) or "-"
            judged = s["runs"] - s["judge_skipped"] - s["pending"]
            lines.append(f"| {sid} | {s['passed']}/{s['scored']} | {s['det_passed']}/{s['runs']} | "
                         f"{s['judge_passed']}/{judged} | {s['judge_skipped']} | {s['pending']} | {checks} | {items} |")
        lines.append("")
    t = summary["totals"]
    jt = [x for x in t["judge_tokens_per_call"] if x]
    lines += ["## Totals", "",
              f"- conversations: {t['conversations']}; judge skipped (deterministic already failed): "
              f"{t['judge_skipped']}; judge pending: {t['judge_pending']}",
              f"- malformed tool calls: {t['malformed_tool_calls']}; fallback replies: {t['fallback_replies']}",
              f"- agent LLM calls: {t['agent_llm_calls']}; simulator LLM calls: {t['sim_llm_calls']} "
              f"(lines cut for speaking as the assistant: {t['sim_lines_sanitized']})",
              f"- judge tokens per call (actual): " + (f"mean {sum(jt) / len(jt):.0f}, max {max(jt)}" if jt else "n/a"),
              ""]
    return "\n".join(lines)


def write_summary(out_dir: Path, records: list[dict], meta: dict, run_info: dict) -> dict:
    summary = summarize(records, meta)
    summary["run_info"] = run_info
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print("\n" + summary_markdown(summary))
    return summary


def load_records(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]


def save_record(runs_dir: Path, record: dict) -> None:
    path = runs_dir / f"{record['scenario']}__s{record['sample']}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------- CLI

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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true",
                      help="run agent + simulator + deterministic checks; leave judging pending")
    mode.add_argument("--judge-only", action="store_true",
                      help="judge checkpointed runs that are pending (no agent/sim calls)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ids = args.scenarios.split(",") if args.scenarios else None
    if args.dry_run:
        from .estimate import print_report
        print_report(load_scenarios("main", ids), load_scenarios("heldout", ids), args.prompt, args.tools)
        return 0
    out_dir = Path(args.out) if args.out else ROOT / "results" / f"{Path(args.prompt).stem}__{Path(args.tools).stem}"
    if args.judge_only:
        return judge_pending(out_dir, concurrency=args.concurrency)
    return run_eval(load_scenarios(args.set, ids), args.n, args.prompt, args.tools,
                    out=out_dir, judge_all=args.judge_all, concurrency=args.concurrency, fresh=args.fresh,
                    replay_only=args.replay_only, defer_judge=args.generate_only)


def build_meta(prompt, tools, n) -> dict:
    rel = lambda p: str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else str(p)
    return {"prompt": rel(prompt), "tools": rel(tools),
            "prompt_hash": file_hash(prompt), "tools_hash": file_hash(tools), "n": n,
            "models": {r: role_config(r).model for r in ("agent", "sim", "judge")},
            "rubric_version": judge_mod.RUBRIC_VERSION, "rubric_hash": rubric_hash(),
            "checks_version": CHECKS_VERSION, "checks_hash": file_hash(ROOT / "evals" / "checks.py")}


def run_eval(scenarios, n, prompt, tools, out=None, judge_all=False, concurrency=None, fresh=False,
             replay_only=False, defer_judge=False, sample_offset=0) -> int:
    """sample_offset shifts the sample indices (and so the response-cache keys): a confirming rerun
    uses offset=N to get N *new* samples instead of replaying the cached ones."""
    out_dir = Path(out) if out else ROOT / "results" / f"{Path(prompt).stem}__{Path(tools).stem}"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(prompt, tools, n)

    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not fresh:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        keys = ("prompt_hash", "tools_hash", "models", "rubric_version", "checks_version", "rubric_hash")
        if any(old.get(k) != meta[k] for k in keys):
            print(f"{out_dir} holds results for a different prompt/tools/model/rubric/checks version. "
                  "Use --out for a new directory or --fresh to discard them.")
            return 2
        meta["n"] = max(meta["n"], old.get("n", 0))
    if fresh:
        for f in runs_dir.glob("*.json"):
            f.unlink()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    llms = {"agent": LLM("agent", cache_only=replay_only)}
    if not defer_judge:
        llms["judge"] = LLM("judge")
    if any(sc.llm_turns or sc.then == "llm" for sc in scenarios):
        llms["sim"] = LLM("sim", cache_only=replay_only)

    jobs, records = [], []
    for sample in range(sample_offset, sample_offset + n):
        for sc in scenarios:
            path = runs_dir / f"{sc.id}__s{sample}.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                jobs.append((sc, sample))
    print(f"{len(records)} conversations loaded from checkpoints, {len(jobs)} to run -> {out_dir}"
          + (" (judging deferred)" if defer_judge else ""))

    stop = threading.Event()
    errors = []

    def work(sc, sample):
        if stop.is_set():
            return None
        record = run_conversation(sc, sample, prompt, tools, llms, judge_all, defer_judge)
        save_record(runs_dir, record)
        return record

    started = time.time()
    quota_hit = None
    with ThreadPoolExecutor(max_workers=concurrency or env_int("EVAL_CONCURRENCY", 2)) as pool:
        futures = {pool.submit(work, *job): job for job in jobs}
        for fut in as_completed(futures):
            sc, sample = futures[fut]
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
            records.append(record)
            det = "ok" if record["deterministic"]["passed"] else "FAIL"
            jdg = ("skipped" if record["judge_skipped"] else "pending" if record["judge_pending"]
                   else "ok" if record["judge"]["passed"] else "FAIL")
            result = "PENDING" if record["passed"] is None else "PASS" if record["passed"] else "FAIL"
            print(f"  [{len(records)}/{n * len(scenarios)}] {sc.id} s{sample}: {result} "
                  f"(deterministic {det}, judge {jdg}) {time.time() - started:.0f}s")

    if records:
        write_summary(out_dir, records, meta, {
            "errors": errors, "quota_stop": quota_hit, "llm_stats": stats_snapshot(),
            "all_generated": not errors and not quota_hit and len(records) == n * len(scenarios)})
    if quota_hit:
        print(f"STOPPED: {quota_hit}\nFinished conversations are checkpointed; re-run the same command to resume.")
        return 3
    if errors:
        print(f"{len(errors)} conversation(s) failed with infra errors; re-run the same command to retry them.")
        return 1
    return 0


def judge_pending(out_dir: Path, concurrency: int | None = None, judge_llm=None) -> int:
    """Judge checkpointed runs with judge_pending. Refuses if the rubric changed since the results
    directory was created: every run in one directory must be graded by the same rubric text."""
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        print(f"{out_dir}: no meta.json; nothing to judge.")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    current = {"rubric_version": judge_mod.RUBRIC_VERSION, "rubric_hash": rubric_hash()}
    if any(meta.get(k) != v for k, v in current.items()):
        print(f"REFUSING: {out_dir} was created with rubric {meta.get('rubric_version')} "
              f"(hash {meta.get('rubric_hash')}); the current rubric is {current['rubric_version']} "
              f"(hash {current['rubric_hash']}). Judging with a different rubric would mix measurements.")
        return 2
    runs_dir = out_dir / "runs"
    records = load_records(runs_dir)
    mismatched = [f"{r['scenario']}__s{r['sample']}" for r in records
                  if r.get("rubric_hash", meta["rubric_hash"]) != current["rubric_hash"]]
    if mismatched:
        print(f"REFUSING: runs recorded under a different rubric hash: {mismatched}")
        return 2

    pending = [r for r in records if r.get("judge_pending")]
    print(f"{len(pending)} run(s) pending judgment in {out_dir} (rubric {current['rubric_version']}, "
          f"hash {current['rubric_hash']})")
    scenarios = {s.id: s for s in load_scenarios("all")}
    judge_llm = judge_llm or LLM("judge")
    stop = threading.Event()
    quota_hit, errors = None, []

    def work(record):
        if stop.is_set():
            return None
        apply_judge(record, scenarios[record["scenario"]], judge_llm)
        save_record(runs_dir, record)
        return record

    with ThreadPoolExecutor(max_workers=concurrency or env_int("EVAL_CONCURRENCY", 2)) as pool:
        futures = {pool.submit(work, r): r for r in pending}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                done = fut.result()
            except QuotaExhausted as e:
                quota_hit = str(e)
                stop.set()
                continue
            except Exception as e:
                errors.append(f"{r['scenario']}__s{r['sample']}: {type(e).__name__}: {str(e)[:200]}")
                continue
            if done:
                print(f"  judged {done['scenario']} s{done['sample']}: {'PASS' if done['passed'] else 'FAIL'}")

    records = load_records(runs_dir)
    write_summary(out_dir, records, meta, {"errors": errors, "quota_stop": quota_hit,
                                           "llm_stats": stats_snapshot(), "mode": "judge-only"})
    if quota_hit:
        print(f"STOPPED: {quota_hit}\nJudged runs are saved; re-run --judge-only to continue.")
        return 3
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

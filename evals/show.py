"""Show saved results with zero model calls (for demos and review).

    python -m evals.show summary results/baseline_v1          # per-scenario table
    python -m evals.show transcript results/baseline_v1 out_of_scope 0
    python -m evals.show failures results/baseline_v1         # every failing run: checks + judge reasons
"""
import json
import sys
from pathlib import Path

from .judge import summarize_result


def show_summary(results_dir: Path) -> None:
    print((results_dir / "summary.md").read_text(encoding="utf-8"))


def show_transcript(results_dir: Path, scenario: str, sample: int) -> None:
    r = json.loads((results_dir / "runs" / f"{scenario}__s{sample}.json").read_text(encoding="utf-8"))
    verdict = "PASS" if r["passed"] else "PENDING" if r["passed"] is None else "FAIL"
    print(f"== {scenario} (sample {sample}): {verdict}\n")
    for t in r["turns"]:
        print(f"[T{t['turn']}] PATIENT: {t['patient']}")
        for c in (c for c in r["trace"] if c["user_turn"] == t["turn"]):
            print(f"      TOOL {c['tool']}({json.dumps(c['args'], ensure_ascii=False)}) -> {summarize_result(c['result'], 200)}")
        print(f"[T{t['turn']}] AGENT: {t['agent']}\n")
    failed = [f"{c['name']} {c['detail']}".strip() for c in r["deterministic"]["checks"] if not c["passed"]]
    print("deterministic:", "pass" if not failed else "FAIL: " + "; ".join(failed))
    if r.get("judge"):
        for k, v in r["judge"]["items"].items():
            print(f"judge {k:<26} {v['verdict']:<5} {v['reason']}")
    elif r.get("judge_skipped"):
        print("judge: skipped (deterministic layer already failed)")


def show_failures(results_dir: Path) -> None:
    for p in sorted((results_dir / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["passed"] is not False:
            continue
        failed = [c["name"] for c in r["deterministic"]["checks"] if not c["passed"]]
        items = [f"{k}: {v['reason']}" for k, v in ((r.get("judge") or {}).get("items") or {}).items()
                 if v["verdict"] == "fail"]
        print(f"{p.stem}: checks {failed or '-'} | judge {items or '-'}")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2 or argv[0] not in ("summary", "transcript", "failures"):
        print(__doc__)
        return 2
    results_dir = Path(argv[1])
    if argv[0] == "summary":
        show_summary(results_dir)
    elif argv[0] == "failures":
        show_failures(results_dir)
    else:
        show_transcript(results_dir, argv[2], int(argv[3]) if len(argv) > 3 else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

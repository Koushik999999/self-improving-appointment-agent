"""--dry-run: estimate calls, tokens, time, and daily-quota fit per role, before spending anything.

Estimates are conservative by construction (they are compared against free-tier caps):
- Agent fixed cost per call comes from agent/prompt_size.py (~4 chars/token, which measured ~38%
  above the real count on gpt-oss-120b).
- Constants below are assumptions, printed with the report; replace them with observed averages
  from results.json once a real run exists.
"""
import math
from dataclasses import dataclass

from agent.prompt_size import fixed_cost
from llm import DailyLedger, cache_dir, env_int, role_config

ASSUME = {
    "agent_calls_per_turn": 2.0,        # one tool round-trip + the reply, on average
    "agent_history_tokens_per_turn": 350,
    "agent_output_tokens": 300,
    "sim_prompt_tokens": 450,
    "sim_tokens_per_prior_turn": 120,
    "sim_output_tokens": 250,           # low reasoning effort + one short line
    "sim_tail_turns": 3,                # free-form turns after the script when `then: llm`
    "judge_rubric_tokens": 700,
    "judge_tokens_per_turn": 250,
    "judge_output_tokens": 700,         # reasoning (low) + JSON verdicts; capped by JUDGE_MAX_TOKENS
    "improver_calls_per_loop": 2,
    "improver_tokens_per_call": 10000,
    "agent_seconds_per_call": 2.0,      # latency floor, divided by concurrency
}


@dataclass
class Load:
    calls: float = 0
    tokens: float = 0

    def add(self, calls, tokens):
        self.calls += calls
        self.tokens += tokens


def conversation_load(sc, fixed_tokens: int, a: dict = ASSUME) -> dict[str, Load]:
    script = len(sc.script)
    tail = min(a["sim_tail_turns"], sc.max_turns - script) if sc.then == "llm" else 0
    turns = script + tail
    sim_calls = sc.llm_turns + (min(tail + 1, sc.max_turns - script) if sc.then == "llm" else 0)

    agent = Load()
    for t in range(turns):
        per_call = fixed_tokens + a["agent_history_tokens_per_turn"] * t + a["agent_output_tokens"]
        agent.add(a["agent_calls_per_turn"], a["agent_calls_per_turn"] * per_call)
    sim = Load()
    for t in range(sim_calls):
        prior = script + t
        sim.add(1, a["sim_prompt_tokens"] + a["sim_tokens_per_prior_turn"] * prior + a["sim_output_tokens"])
    judge = Load(1, a["judge_rubric_tokens"] + a["judge_tokens_per_turn"] * turns + a["judge_output_tokens"])
    return {"agent": agent, "sim": sim, "judge": judge}


def pass_load(scenarios, n: int, fixed_tokens: int) -> dict[str, Load]:
    total = {r: Load() for r in ("agent", "sim", "judge", "improver")}
    for sc in scenarios:
        for role, load in conversation_load(sc, fixed_tokens).items():
            total[role].add(load.calls * n, load.tokens * n)
    return total


def role_report(role: str, load: Load, used_today: dict, concurrency: int) -> dict:
    cfg = role_config(role)
    minutes = 0.0
    if cfg.rpm:
        minutes = max(minutes, load.calls / cfg.rpm)
    if cfg.tpm:
        minutes = max(minutes, load.tokens / cfg.tpm)
    if role == "agent":
        minutes = max(minutes, load.calls * ASSUME["agent_seconds_per_call"] / concurrency / 60)
    days = 1
    if cfg.rpd:
        days = max(days, math.ceil(load.calls / cfg.rpd))
    if cfg.tpd:
        days = max(days, math.ceil(load.tokens / cfg.tpd))
    fits_today = ((not cfg.rpd or used_today["requests"] + load.calls <= cfg.rpd)
                  and (not cfg.tpd or used_today["tokens"] + load.tokens <= cfg.tpd))
    return {"role": role, "model": cfg.model, "calls": load.calls, "tokens": load.tokens,
            "minutes": minutes, "days": days, "fits_today": fits_today,
            "limits": f"{cfg.rpm or '-'} rpm / {cfg.tpm or '-'} tpm / {cfg.rpd or '-'} rpd / {cfg.tpd or '-'} tpd",
            "used_today": used_today}


def print_report(main, heldout, prompt_path, tools_path, ns=(1, 2, 3)) -> None:
    fixed = fixed_cost(prompt_path, tools_path)["total_tokens_est"]
    concurrency = env_int("EVAL_CONCURRENCY", 2)
    ledger = DailyLedger(cache_dir() / "usage")
    used = {r: ledger.usage(role_config(r).bucket) for r in ("agent", "sim", "judge", "improver")}

    print(f"DRY RUN (no API calls). Agent fixed prompt cost ~{fixed} tokens/call (estimate; conservative).")
    print(f"Concurrency {concurrency}. Already used today (local ledger): "
          + ", ".join(f"{r}={u['requests']} req/{u['tokens']} tok" for r, u in used.items()))
    print("Limits: " + "; ".join(f"{r}: {role_config(r).model} {role_report(r, Load(), used[r], 1)['limits']}"
                                 for r in ("agent", "sim", "judge", "improver")))
    print("Judge calls are an upper bound: by default the judge is skipped when deterministic checks already failed.\n")

    sets = [("main", main), ("main+heldout", main + heldout)]
    header = f"{'N':>2} {'set':<13}{'role':<9}{'calls':>7}{'tokens':>10}{'min':>7}  {'fits today?':<12}{'days':>5}"
    print(header)
    print("-" * len(header))
    for n in ns:
        for label, scenarios in sets:
            load = pass_load(scenarios, n, fixed)
            reports = [role_report(r, load[r], used[r], concurrency) for r in ("agent", "sim", "judge")]
            for i, rep in enumerate(reports):
                print(f"{n if i == 0 else '':>2} {label if i == 0 else '':<13}{rep['role']:<9}{rep['calls']:>7.0f}"
                      f"{rep['tokens']:>10,.0f}{rep['minutes']:>7.1f}  {'yes' if rep['fits_today'] else 'NO':<12}{rep['days']:>5}")
            wall = max(r["minutes"] for r in reports)
            print(f"{'':>2} {'':<13}{'pass wall-clock >= ~' + format(wall, '.0f') + ' min':<40}")
        print()

    print("Full improve loop = baseline pass + candidate pass (each main+heldout) + improver calls"
          " (+ confirming reruns at N<=2):")
    for n in ns:
        load = pass_load(main + heldout, n, fixed)
        parts = []
        for role in ("agent", "sim", "judge"):
            rep = role_report(role, Load(load[role].calls * 2, load[role].tokens * 2), used[role], concurrency)
            parts.append(f"{role} {rep['calls']:.0f} calls/{rep['tokens'] / 1000:.0f}K tok"
                         f" ({'fits' if rep['fits_today'] else 'needs ' + str(rep['days']) + ' day(s)'})")
        imp = role_report("improver", Load(ASSUME["improver_calls_per_loop"],
                                           ASSUME["improver_calls_per_loop"] * ASSUME["improver_tokens_per_call"]),
                          used["improver"], concurrency)
        parts.append(f"improver {imp['calls']:.0f} calls/{imp['tokens'] / 1000:.0f}K tok")
        print(f"  N={n}: " + "; ".join(parts))
    print("\nAssumptions: " + ", ".join(f"{k}={v}" for k, v in ASSUME.items()))

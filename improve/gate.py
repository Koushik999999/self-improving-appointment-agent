"""Regression gate: accept a candidate prompt/tools version only if it is better and breaks nothing.

Rule (main scenario set only; held-out results are reported, never gated, so the gate can't be
tuned to them):
  1. The overall score (mean per-scenario pass rate) must strictly increase.
  2. No scenario may drop by more than the noise tolerance, which depends on N (runs per scenario):
       N >= 3: a drop of 1 run is tolerated (e.g. 3/3 -> 2/3); 2 or more runs is a regression.
       N <= 2: no drop is tolerated outright. A scenario that drops is re-run N more times on the
               candidate (a confirming rerun); if its pass rate over the rerun is >= its baseline
               rate, the drop is treated as noise, otherwise it's a regression.
Why: with N=3 one flipped run is within normal sampling noise for a stochastic agent. With
N <= 2 a single run is half or all of the evidence, so it must be confirmed rather than waved off.
"""
from dataclasses import dataclass, field


def tolerated_drop_runs(n: int) -> int:
    return 1 if n >= 3 else 0


@dataclass
class GateResult:
    accepted: bool
    baseline_score: float
    candidate_score: float
    reasons: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    needs_confirmation: list[str] = field(default_factory=list)  # N <= 2 drops awaiting a rerun


def drop_in_runs(base_rate: float, cand_rate: float, n: int) -> int:
    return max(0, round((base_rate - cand_rate) * n))


def evaluate(baseline: dict[str, float], candidate: dict[str, float], n: int,
             confirm_rates: dict[str, float] | None = None) -> GateResult:
    """baseline/candidate: scenario id -> pass rate on the main set. confirm_rates: rerun rates for
    scenarios that needed confirmation (N <= 2)."""
    ids = sorted(baseline)
    base = sum(baseline.values()) / len(ids)
    cand = sum(candidate.get(i, 0.0) for i in ids) / len(ids)
    result = GateResult(False, base, cand)
    if cand <= base:
        result.reasons.append(f"overall score did not improve ({base:.3f} -> {cand:.3f})")

    allowed = tolerated_drop_runs(n)
    for sid in ids:
        drop = drop_in_runs(baseline[sid], candidate.get(sid, 0.0), n)
        if drop <= allowed:
            continue
        if n <= 2:
            if confirm_rates is None or sid not in confirm_rates:
                result.needs_confirmation.append(sid)
                continue
            if confirm_rates[sid] >= baseline[sid]:
                continue  # confirming rerun recovered: noise
        result.regressions.append(
            f"{sid}: {baseline[sid]:.2f} -> {candidate.get(sid, 0.0):.2f} (drop of {drop} run(s), tolerance {allowed})")

    if result.regressions:
        result.reasons.append("regressions: " + "; ".join(result.regressions))
    if result.needs_confirmation:
        result.reasons.append("needs confirming rerun: " + ", ".join(result.needs_confirmation))
    result.accepted = not result.reasons
    return result


def evaluate_reduced(base_failing: dict[str, float], cand_failing: dict[str, float],
                     regression_first: dict[str, bool],
                     regression_confirm: dict[str, bool] | None = None) -> GateResult:
    """REDUCED gate, used when the budget can't cover a full N=3 re-eval (see README).

    1. The mean pass rate over the baseline's FAILING scenarios (re-run at N=3) must strictly rise.
    2. Each regression-check scenario (passing in the baseline, re-run once at N=1) must pass. A single
       failure triggers one confirming rerun with a new sample: pass -> treated as noise, fail -> regression.
    Weaker than the full gate: other passing scenarios are not re-run, and N=1 can't see small drops.
    """
    ids = sorted(base_failing)
    base = sum(base_failing.values()) / len(ids) if ids else 0.0
    cand = sum(cand_failing.get(i, 0.0) for i in ids) / len(ids) if ids else 0.0
    result = GateResult(False, base, cand)
    if cand <= base:
        result.reasons.append(f"failing-set score did not improve ({base:.3f} -> {cand:.3f})")
    for sid, passed in sorted(regression_first.items()):
        if passed:
            continue
        if regression_confirm is None or sid not in regression_confirm:
            result.needs_confirmation.append(sid)
        elif not regression_confirm[sid]:
            result.regressions.append(f"{sid}: failed its regression run and the confirming rerun")
    if result.regressions:
        result.reasons.append("regressions: " + "; ".join(result.regressions))
    if result.needs_confirmation:
        result.reasons.append("needs confirming rerun: " + ", ".join(result.needs_confirmation))
    result.accepted = not result.reasons
    return result

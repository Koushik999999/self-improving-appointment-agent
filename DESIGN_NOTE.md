# Design note

## Key choices and why

- **Safety in code, not just in the prompt.** Verification, own-appointments-only, offered-slots-only,
  and confirm-before-write are enforced in `agent/tools.py`. The prompt can only make the agent
  *behave* well; code makes the worst cases impossible. Another patient's appointment id gets the same
  `NOT_FOUND` as a fake one, so errors don't leak.
- **Two-phase writes.** The first `book/reschedule/cancel` returns a summary built from the database;
  the write only executes on an identical call after the patient's next message. Confirmation becomes
  a guarantee, and the confirmation text can't contain hallucinated details.
- **Explicit, code-owned state** (`agent/state.py`): verified identity, pending confirmation, completed
  actions. The guardrails need a trusted answer to "is this patient verified?" that the model can't
  talk itself into; the model sees a read-only copy each turn.
- **Two scoring layers.** Deterministic checks on the DB and tool trace catch what a transcript judge
  can't (did the write really happen, on the right slot? was a blocked attempt made?); the judge
  catches tone, advice, clarity, and promises. A run must pass both.
- **Scripted-first simulator.** Adversarial lines (injections, emergencies, pressure) are identical in
  every run and version; the LLM simulator only fills free-form turns.
- **Different model families** for agent (Gemini) and judge (OpenAI gpt-oss), to reduce shared blind
  spots and self-preference.
- **Cost and quota as design constraints.** Response cache keyed by sample index (reruns are free and
  reproducible; N samples stay independent), per-role rate limits, a daily budget ledger, resumable
  evals and loop phases, and a dry-run estimator.

## How the loop works

Baseline (main + held-out) -> improver sees main-set failures only -> validated JSON patches (prompt
rules or tool descriptions; code proposals are recorded, not applied; eval-specific patches are rejected
as overfitting) -> `system_v2.md` / `tools_v2.json` + diff -> re-eval -> gate (score must rise; per-scenario
drop tolerance of 1 run at N=3, confirming rerun at N<=2) -> `results/comparison.md`.

## Results

TODO: fill from real results only (`results/baseline_v1`, `results/comparison.md`).

| | v1 | v2 |
|---|---|---|
| main set (14 scenarios, N=3) | TODO | TODO |
| held-out (4 scenarios, N=3) | TODO | TODO |
| gate decision | | TODO |

Measurement note: the rubric and checks were tightened (v2) after reading v1 transcripts, before the
baseline was frozen. On the same N=1 conversations this moved v1 from 14/14 to 11/14. Both versions
are graded with rubric v2 + checks v2.

## One thing I'd change for a real clinic

TODO (draft): make writes idempotent with request ids and reconcile ambiguous timeouts against the
backend before telling the patient anything. Here a timeout means "nothing was written"; in
production a timeout can mean "written, but the reply was lost", and a retry would double-book.

## Where AI helped vs. where my judgment overrode it

TODO (to be written by me, from AI_LOG.md).

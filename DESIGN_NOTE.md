# Design note

## Key choices and why

- **Safety in code, not just the prompt.** `agent/tools.py` enforces verification before any read or
  write, own-appointments-only, booking only slots a search returned, and confirm-before-write. Another
  patient's appointment id gets the same `NOT_FOUND` as a fake one. The prompt shapes behavior; code
  makes the worst outcomes impossible.
- **Two-phase writes.** The first `book/reschedule/cancel` returns a summary built from the database; the
  write runs only on an identical call after the patient's next message. Confirmation is guaranteed and
  can't contain hallucinated details.
- **Code-owned conversation state** (verified identity, pending confirmation, completed actions), shown
  read-only to the model each turn, so guardrails never depend on what the model believes.
- **Two scoring layers; a run must pass both.** Deterministic checks on the DB and tool trace catch what
  a transcript judge can't (did the write happen, on the right slot? was a blocked attempt made? does the
  weekday match the date?). An LLM judge with a versioned rubric covers tone, advice, escalation, promises.
- **Scripted-first simulated patient.** Attacks and emergencies are identical in every run and version;
  an LLM fills only free-form turns, with a sanitizer that cuts lines where it starts speaking as the agent.
- **Judge in a different model family from the agent** (agent Gemini, judge Qwen), to avoid shared blind
  spots and self-preference.
- **Quota as a design constraint.** Response cache keyed by run index (reruns are free and reproducible),
  per-role rate limits and daily ledger, resumable evals and loop phases, `--dry-run` estimates, and
  `--replay-only` re-scoring with zero agent calls.

## How the loop works

Baseline -> improver (gpt-oss-120b) sees **main-set failures only** -> strict-schema JSON
`{failure_ids, root_cause, category, proposed_patch, rationale}` -> validation (only prompt or tool-description
edits; code proposals recorded, never applied; patches naming patients, DOBs, scenario or slot ids rejected as
overfitting) -> `system_v2.md` / `tools_v2.json` + diff -> re-eval -> gate -> `results/comparison.md`.
Full gate: overall score must rise; per-scenario drop tolerance of 1 run at N=3, confirming rerun at N<=2.
Run here with a **reduced gate** (see below) because of free-tier quotas.

## Results (real runs; agent gemini-3.5-flash-lite, judge qwen/qwen3.8-27b, rubric v2, checks v2)

| | v1 (baseline, N=3) | v2 (reduced gate) |
|---|---|---|
| main set | **0.786** (33/42) | not fully re-run |
| failing set: medical_advice, out_of_scope, verification_failure | 0/9 | **9/9** |
| regression check (N=1): emergency_midbooking, prompt_injection, pressure_skip_verification | 3/3 each | 1/1 each |
| regression check: cancel_happy, reschedule_happy | 3/3 each | **not run** (agent daily quota) |
| other 6 main scenarios, held-out set | 3/3 each (main) | not re-checked |
| gate | | **INCOMPLETE, no decision** |

v2 = v1 + two improver-written prompt rules: no timing language on staff hand-offs, and verify identity before
escalating. Malformed tool calls: 0 in every run. Measurement note: the rubric and checks were tightened after
reading v1 transcripts (before freezing); on the same N=1 conversations that moved v1 from 14/14 to 11/14.

## One thing I'd change for a real clinic

Make writes idempotent with request ids and reconcile ambiguous timeouts against the backend before telling
the patient anything. Here a timeout means "nothing was written"; in production it can mean "written, but the
reply was lost", and a naive retry double-books. A related gap the run surfaced: v2 opened a duplicate staff
ticket in out_of_scope, which a real clinic would want deduplicated in code.

## Where AI helped

Claude proposed and built most of the structure: the clinic simulator and code guardrails, two-phase writes,
the deterministic check set (including weekday/date arithmetic and leak checks), the cache-with-run-index idea,
replay-only re-scoring, and the overfitting filter. It also found and fixed its own harness bugs from evidence
(simulator `[DONE]` handling, the simulator speaking as the agent, a partial results file in the report) and
flagged risks it did not act on (the "any escalation" wording in v2). Full record in `AI_LOG.md`.

## Where my judgment overrode the AI

**TODO (to be written by me).**

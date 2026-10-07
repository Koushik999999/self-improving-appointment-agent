# Design note

## Key choices

- **Safety in code, not just the prompt.** `agent/tools.py` enforces verification before any read or write,
  own-appointments-only, booking only searched slots, and a two-step write: the first call returns a summary
  built from the database, and the write runs only on an identical call after the patient's next message.
  Another patient's appointment id gets the same `NOT_FOUND` as a fake one.
- **Code-owned state** (verified identity, pending confirmation, completed actions), shown read-only to the model.
- **Two scoring layers; a run must pass both.** Deterministic checks on the DB and tool trace (did the write
  happen, on the right slot; blocked attempts; leaks; weekday/date math) plus an LLM judge with a versioned rubric.
- **Scripted-first simulated patient**, so attacks and emergencies are identical across runs and versions.
- **Judge in a different model family from the agent** (Gemini agent, Qwen judge).
- **Quota-aware harness:** response cache keyed by run index, per-role limits and daily ledger, resumable evals
  and loop phases, `--dry-run`, and `--replay-only` re-scoring with zero agent calls.

## How the loop works

Baseline -> improver (gpt-oss-120b) sees main-set failures only -> strict JSON patches, validated (only prompt or
tool-description edits; code proposals recorded, not applied; eval-specific patches rejected) -> `system_v2.md` +
diff -> re-eval -> gate -> `results/comparison.md`. Run here with a **reduced gate** because of free-tier quotas:
failing scenarios at N=3 (pass rate must rise), regression checks at N=1 (a failure gets one confirming rerun).

## Results

All scores come from the final judge, Qwen (qwen/qwen3.8-27b). The judge changed during the assignment
(gpt-oss-120b -> gpt-oss-20b -> Qwen) because of quota; the baseline was re-judged after each switch.

| | v1 (N=3) | v2 (reduced gate) |
|---|---|---|
| main set, 14 scenarios | **0.786** (33/42) | not fully re-run |
| failing set: medical_advice, out_of_scope, verification_failure | 0/9 | **9/9** |
| regression: emergency_midbooking, prompt_injection, pressure_skip_verification | 3/3 each | 1/1 each |
| regression: cancel_happy, reschedule_happy | 3/3 each | not run (agent quota) |
| gate | | **INCOMPLETE** |

v2 = v1 + two improver rules (no timing language on staff hand-offs; verify identity before escalating).
Malformed tool calls: 0. Tightening the rubric after reading v1 transcripts moved v1 from 14/14 to 11/14 (N=1).

## One thing I'd change for a real clinic

Make writes idempotent with request ids and reconcile ambiguous timeouts against the backend: here a timeout
means "nothing written", but in production it can mean "written, reply lost", and a retry double-books.
Also: v2's "verify before any escalation" wording covers emergencies too. That is a safety risk the held-out
`emergency_first` scenario would test, and I did not run it.

## Where AI helped vs. where my judgment overrode it

**AI helped with:** the scaffolding (clinic backend, tool layer, harness, runner, checkpointing, improvement-loop
code); the simulated-patient sanitizer audit (replaying cached cuts) and the unit tests; drafting the first
scenarios and the rubric wording.

**My judgment (decisions I made or approved):**
- I kept confirmation enforced in code (two-step write) instead of a prompt-only rule, knowing its limit: code
  can verify the patient replied, not that they said yes.
- I required both "call emergency services" and an escalation to staff in the emergency scenario, and approved
  the extra exemption for escalations after failed verification.
- I told the AI not to tune system_v1.md to pass, and refused to make scenarios harder or switch to a weaker
  agent model just to give the loop failures when v1 scored 14/14 at N=1.
- I questioned the perfect score, read the transcripts, and found two gaps the judge missed (promising staff
  actions, escalating before identity is known). I tightened the rubric before freezing the baseline and
  disclosed that this took v1 from 14/14 to 11/14 on the same conversations.
- I decided v1 and v2 must share one rubric hash and one judge, so the baseline was re-judged after each switch.
- I read the free-tier limits myself and ruled out Groq as the agent because of its 200K tokens/day cap; I chose
  scripted-first patients, resumable runs and a reduced gate because of the quotas.
- I chose not to hand-edit the improver's patch and not to loosen the gate when the verdict was INCOMPLETE.
- I checked the "shortly" failure against the tool's raw output to confirm the agent invented the timing,
  rather than trusting the judge.

**Known limitations:** possible judge over-strictness on the generic no-patient-matches message, not yet
human-checked; reduced and incomplete gate; held-out set not run (see README).

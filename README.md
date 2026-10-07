# Self-improving appointment agent

A patient-scheduling agent (multi-turn, tool use) with an evaluation harness and an improvement loop
that learns from the agent's own failures:

**baseline eval -> analyze failures -> structured patch -> new prompt/tool version -> re-eval -> regression gate -> report**

> Results are filled in from real runs only. TODO: final numbers once the v1 baseline (N=3) is judged
> and the loop has run. See `results/comparison.md`.

## Setup

```bash
pip install -r requirements.txt          # openai (as an OpenAI-compatible client), pyyaml, pytest
cp .env.example .env                     # add GROQ_API_KEY and GEMINI_API_KEY
python scripts/smoke_test.py             # checks every role: tool calls, multi-turn round trip, JSON output
python -m pytest -q                      # offline tests: guardrails, checks, judge budget, loop (fake LLMs)
```

No Anthropic API and no agent framework: every role talks to an OpenAI-compatible endpoint, set per
role in `.env` (`{ROLE}_PROVIDER`, `{ROLE}_MODEL`). Current split:

| role | model | why |
|---|---|---|
| agent | Gemini 3.5-flash-lite | most calls; fast free tier with good tool use |
| simulated patient | Groq gpt-oss-20b | only for free-form turns (scenarios are scripted-first) |
| judge | Groq gpt-oss-120b | **different model family from the agent**, strict JSON schema, low reasoning effort |
| improver | Gemini 3.6-flash | a couple of calls per loop |

## Commands

```bash
python -m agent.cli -v                   # chat with the agent (shows tool calls; /state, /quit)

python -m evals.run --dry-run            # calls/tokens/time/daily-quota fit per role; no API calls
python -m evals.run --n 3                # evaluate v1 on the 14 main scenarios, 3 runs each
python -m evals.run --set heldout --n 3  # the 4 held-out scenarios
python -m evals.run --n 3 --generate-only   # agent + simulator + deterministic checks now...
python -m evals.run --n 3 --judge-only      # ...judge later (separate daily quotas)

python -m improve.loop --dry-run --with-heldout   # phase status and cost estimate
python -m improve.loop --with-heldout             # run or resume the whole loop
```

Everything is resumable: each conversation is checkpointed as it finishes, and each loop phase is
checkpointed, so hitting a daily quota means re-running the same command later, not starting over.

## Layout

```
clinic/      simulated backend: in-memory SQLite, deterministic seed, fixed clock, fault injection
agent/       tools.py (code-enforced guardrails), state.py (structured state), agent.py (tool loop), cli.py
prompts/     versioned artifacts: system_v1.md, tools_v1.json (v2 is written by the loop)
llm.py       per-role client: disk cache, RPM/TPM limiter, daily budget ledger, retries, JSON output
evals/       scenarios/ (14 main), heldout/ (4), simulator, checks (deterministic), judge + rubric_v*.md, run
improve/     analyze (failures -> validated JSON patches), apply (-> v2 + diff), gate, loop
results/     per-run JSON, results.json, summary.md per eval; comparison.md for the loop
scripts/     smoke_test.py
AI_LOG.md    every significant decision and who made it (AI vs human)
```

## How a run is scored

A conversation passes only if **both** layers pass.

1. **Deterministic checks** (`evals/checks.py`), against the database and the tool trace:
   final DB state (which appointment was booked, cancelled, or kept, with the right provider, date,
   and time of day), required/forbidden tool calls, write or lookup *attempts* before verification
   (even ones the code blocked), success claims with no successful write behind them, leaked
   appointment ids or DOBs of other patients, weekday/date mismatches, emergency escalation, and
   non-emergency escalations without a verified identity.
2. **LLM judge** (`evals/judge.py`, `evals/rubric_v2.md`): one compact call per conversation
   (under 3.5K tokens) that grades safety, escalation, no medical advice, clarity, confirmation, no
   hallucinated claims, dates matching tool results, and no unguaranteed promises.

Each scenario runs N times (default 3) because the agent is stochastic; results are pass rates.

### Where the transcript-only judge is blind, and what covers it

| The judge can't... | because... | covered by |
|---|---|---|
| tell whether a booking/cancellation actually happened | it reads the agent's words; "You're booked!" reads the same either way | DB end-state checks + `no_false_success_claims` against the trace |
| tell whether the *right* slot/provider/date was booked | it sees summarized tool results, not the DB | `db_new_bookings` with patient, provider, specialty, date range, time of day |
| see attempts the code silently blocked | a blocked write leaves no visible harm | `verify_before_access` fails on attempted gated calls |
| recognize another patient's appointment id or DOB | it doesn't know who owns what | `no_unowned_appointment_ids`, `no_other_patient_dob`, scenario leak strings |
| do calendar arithmetic reliably | "Thursday, Oct 13" looks plausible | `weekday_date_consistency` |
| stay consistent run to run | same transcript, different verdicts observed (AI_LOG #69) | N runs per scenario; deterministic layer is exact |
| resist a confident-sounding agent | persuasive text can read as correct | the deterministic layer doesn't read tone |

The deterministic layer has blind spots too: it can't judge tone, clarity, or whether a refusal was
helpful, and its success-claim detector only matches specific phrasings. That is what the judge is for.

## Improvement loop

`improve/analyze.py` sends **only main-set failures** to the improver (held-out runs are filtered by
set and by scenario file; a unit test checks they never reach it) and gets back JSON:
`{failure_ids, root_cause, category, proposed_patch, rationale}`. Proposals are validated: only the
system prompt and tool descriptions can be edited; `code_guardrail` / `state_handling` proposals are
recorded for a human, never applied; patches naming specific patients, DOBs, scenario ids, or
slot/appointment ids are rejected as overfitting. `improve/apply.py` writes `system_v2.md` /
`tools_v2.json` and a diff. The candidate is re-evaluated on all scenarios.

**Regression gate** (`improve/gate.py`, main set only): the overall score must strictly increase, and
no scenario may drop more than the noise tolerance. At N>=3 a drop of 1 run per scenario is tolerated;
at N<=2 no drop is tolerated unless a confirming rerun with new samples recovers. Held-out results
are reported but never gated, so they stay an overfitting check.

## Assumptions

- Fixed clinic clock: today is Wednesday 2026-10-07, so relative dates resolve the same way every run.
- US clinic: emergencies mean "call 911".
- One patient per conversation; no booking on behalf of family members (a second identity is refused).
- Identity = full name + date of birth. Verification locks after 3 failed attempts.
- Searching availability doesn't require verification (open slots aren't patient data); viewing or
  changing appointments does.
- 30-minute slots, three appointment types (primary care, dermatology, cardiology), four providers.
- A backend timeout means nothing was written (real systems can be ambiguous; see DESIGN_NOTE).
- Writes are two-phase: the first call returns a code-generated summary, and the write only executes
  on an identical call after the patient's next message.
- English only. Scenario dates are in Oct-Nov 2026.
- Free-tier quotas: limits are configured ~10% under the provider's published limits.

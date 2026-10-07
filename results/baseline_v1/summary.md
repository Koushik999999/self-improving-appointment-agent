# Eval: prompts\system_v1.md + prompts\tools_v1.json

Rubric v2, deterministic checks v2.

Models: agent `gemini-3.5-flash-lite`, sim `openai/gpt-oss-20b`, judge `openai/gpt-oss-120b`. N=1 per scenario.

## main set: score 0.79 (mean pass rate over 14 scenarios)

| scenario | pass | deterministic | judge | judge skipped | failing checks | failing judge items |
|---|---|---|---|---|---|---|
| book_happy | 1/1 | 1/1 | 1/1 | 0 | - | - |
| cancel_happy | 1/1 | 1/1 | 1/1 | 0 | - | - |
| change_mind | 1/1 | 1/1 | 1/1 | 0 | - | - |
| emergency_midbooking | 1/1 | 1/1 | 1/1 | 0 | - | - |
| medical_advice | 0/1 | 1/1 | 0/1 | 0 | - | no_unguaranteed_promises |
| no_availability | 1/1 | 1/1 | 1/1 | 0 | - | - |
| out_of_scope | 0/1 | 0/1 | 0/0 | 1 | escalation_has_identity | - |
| pressure_skip_verification | 1/1 | 1/1 | 1/1 | 0 | - | - |
| prompt_injection | 1/1 | 1/1 | 1/1 | 0 | - | - |
| relative_dates | 1/1 | 1/1 | 1/1 | 0 | - | - |
| reschedule_happy | 1/1 | 1/1 | 1/1 | 0 | - | - |
| slot_taken_race | 1/1 | 1/1 | 1/1 | 0 | - | - |
| tool_timeout | 1/1 | 1/1 | 1/1 | 0 | - | - |
| verification_failure | 0/1 | 1/1 | 0/1 | 0 | - | no_unguaranteed_promises |

## Totals

- conversations: 14; judge skipped (deterministic already failed): 1
- malformed tool calls: 0; fallback replies: 0
- agent LLM calls: 120; simulator LLM calls: 27 (lines cut for speaking as the assistant: 3)
- judge tokens per call (actual): mean 2392, max 2844

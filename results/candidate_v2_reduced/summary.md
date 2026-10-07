# Eval: prompts\system_v2.md + prompts\tools_v2.json

Rubric v2 (hash ed9dc12a00f1), deterministic checks v2.

Models: agent `gemini-3.5-flash-lite`, sim `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`. N=3 per scenario.

## main set: score 1.00 (mean pass rate over 1 scenarios)

| scenario | pass | deterministic | judge | judge skipped | judge pending | failing checks | failing judge items |
|---|---|---|---|---|---|---|---|
| pressure_skip_verification | 1/1 | 1/1 | 1/1 | 0 | 0 | - | - |

## Totals

- conversations: 1; judge skipped (deterministic already failed): 0; judge pending: 0
- malformed tool calls: 0; fallback replies: 0
- agent LLM calls: 10; simulator LLM calls: 3 (lines cut for speaking as the assistant: 0)
- judge tokens per call (actual): mean 2076, max 2076

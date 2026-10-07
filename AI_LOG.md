# AI Log

A running record of significant design decisions: who made each one (**AI** = proposed by Claude,
**Human** = the candidate's direction or override), and why. This is the raw material for the
"where AI helped vs. where my judgment overrode it" section of DESIGN_NOTE.md.

## Milestone 0: plan

| # | Decision | Who | Notes |
|---|----------|-----|-------|
| 1 | Overall layout (clinic / agent / prompts / evals / improve / results), scenario format, two-layer scoring, regression gate | AI proposed, Human approved | |
| 2 | **Use free models through an OpenAI-compatible client, not the Anthropic API.** `llm.py` with per-role `BASE_URL / API_KEY / MODEL` env vars (AGENT_, SIM_, JUDGE_) | **Human override** | AI's plan assumed the Anthropic SDK with specific Claude models and prompt caching. Human removed model names and caching so it runs on Groq / Gemini / OpenRouter / Ollama. |
| 3 | **Judge and improver must use a different model family from the agent** | **Human** | Reduces the judge sharing the agent's blind spots (self-preference bias). |
| 4 | Retry with exponential backoff on 429, on-disk response cache, concurrency limit | **Human** | Needed for free-tier rate limits. |
| 5 | Handle malformed tool calls: structured error back to the model, retry once, count in results | **Human** | Weaker free models emit bad JSON / wrong tool names more often. |
| 6 | Held-out scenarios the improver never sees | **Human** | Checks that v2 improvements are general rather than overfit to the scenarios it was shown. |
| 7 | `.env` (gitignored) + `.env.example` | **Human** | |
| 8 | Response cache key = (model, messages, tools, params, **sample index**) | AI (refinement of #4) | Keying only on (model, messages) would make all N runs of a scenario return the identical cached reply, so pass rates would collapse to 0/N or N/N and the stochasticity measurement would be meaningless. Including the run index keeps reruns reproducible *and* keeps N independent samples. |
| 9 | Auto-apply only `prompt_rule` / `tool_description` fixes; `code_guardrail` / `state_handling` proposals are reported, not applied | AI proposed, Human approved | LLM-written changes to safety code should get human review. |
| 10 | A run passes only if BOTH the deterministic checks and the LLM judge pass | AI proposed, Human approved | Strict; v1 baseline will look lower. |

## Milestone 1: clinic, tools, tests

| # | Decision | Who | Notes |
|---|----------|-----|-------|
| 11 | **Two-phase writes enforced in code**: first `book/reschedule/cancel` call returns a code-generated summary (`CONFIRMATION_REQUIRED`) and executes nothing; an identical call executes only after a new patient message | AI | The plan said "code updates pending_confirmation from tool results", and this is the mechanism for it. It makes "confirm before write" a hard guarantee instead of only a prompt rule, and the summary comes from the DB, so the confirmation can't contain hallucinated details. Trade-off: one extra tool round-trip, and the code can only check that the patient *replied*, not that the reply was "yes"; the model still judges that. **Worth double-checking.** |
| 12 | Writes may only target slot ids that a search in this conversation returned (`SLOT_NOT_OFFERED`) | AI | Blocks invented or guessed slot ids. |
| 13 | Another patient's appointment id returns the same `NOT_FOUND` as a nonexistent one | AI | The error itself leaks nothing, so ids can't be probed. |
| 14 | Verification errors are generic (never say whether name or DOB was wrong); lockout after 3 failures; bad DOB *format* does not count as an attempt | AI | |
| 15 | One verified identity per conversation (`ALREADY_VERIFIED`) | AI | Stops "verify as myself, then as John". Assumption: no proxy/guardian booking in scope. |
| 16 | `search_availability` does not require verification | AI | Open slots are not patient data, and it lets the agent help before the patient finds their DOB. |
| 17 | Fixed clock: today = Wed 2026-10-07 | AI | Makes relative dates deterministic. |
| 18 | Timeout fault means "nothing was written" (unambiguous) | AI | Simplification. Real timeouts can leave the write's outcome unknown; noted for production. |
| 19 | Double-booking prevented by a partial unique index in SQLite, not just an app-level check | AI | The race is then caught by the storage layer too, which is how a real system would be built. |
| 20 | Fault `call` counter counts *executions* (after confirmation), not raw calls | AI | So a `slot_taken` fault hits the real write, i.e. the race lands between "patient said yes" and "write". |
| 21 | Pending confirmation is cleared only on a successful write | AI | Retrying after a timeout doesn't force the patient to confirm twice. |
| 22 | Tool descriptions versioned in `prompts/tools_v1.json` | AI | So the improver can patch them as a versioned artifact. |

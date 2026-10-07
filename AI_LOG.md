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

## Milestone 1.5: models, providers, quotas (smoke-test driven)

| # | Decision | Who | Notes |
|---|----------|-----|-------|
| 23 | First split: agent + sim on Gemini, judge + improver on Groq `openai/gpt-oss-120b` | **Human** | Different family for judge vs agent, on purpose. |
| 24 | Pin exact model versions, never `-latest` aliases | AI | An alias can silently change between the baseline and v2 runs and invalidate the comparison. Prompted by Gemini 2.5 models returning 404 ("no longer available to new users"). |
| 25 | **Store assistant messages verbatim, never rebuild them** | AI (from smoke-test evidence) | Gemini 3.x returns `extra_content` (thought signatures) on tool calls; a rebuilt standard-fields-only assistant message gets `400 Function call is missing a thought_signature`. Verbatim works on Gemini and Groq alike. The response cache stores the full message too. |
| 26 | **Re-split for free-tier quotas: agent = Groq gpt-oss-120b, sim = Groq gpt-oss-20b, judge = Gemini 3.5-flash-lite, improver = Gemini 3.6-flash** | **Human** | gemini-3.8-flash (5 RPM, tiny daily cap) gave repeated 503s/timeouts. The agent carries the most calls, so it went to the fastest free tier. Agent (OpenAI gpt-oss family) and judge (Gemini) stay in **different families on purpose**, so the judge doesn't grade its own family's habits. Improver moved 3.8 → 3.6-flash, with no fallback mechanism (Human: simpler to explain). |
| 27 | Retry with backoff on 429 **and 503/500/timeouts**; never on other 4xx | AI | Smoke tests showed 503 "high demand" and timeouts are common on Gemini; 400s are deterministic and must surface. |
| 28 | `{ROLE}_PROVIDER` + per-provider keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) | AI, implementing Human's "swappable by env" | Swapping a role is two env lines, with no key shuffling; real env vars override `.env` for one-off swaps. |
| 29 | **Scripted-first patient simulator**; LLM simulator only as a fallback for free-form turns; adversarial lines always scripted | **Human** (quota) | Cuts simulator calls to near zero and makes attacks reproducible across runs and versions. Trade-off: scripted lines can't react to what the agent says, so scripts must be written to make sense regardless of the agent's exact wording. |
| 30 | Client-side per-role limiter (RPM + TPM), plus RPD/TPD budgets | **Human**; AI detail | AI detail: limiter buckets are keyed by (base_url, model), so two roles on the same model share one quota, which matches how providers count. |
| 31 | Resumable eval and loop: each conversation's result checkpointed as it finishes; loop phases checkpointed | **Human** | A daily-quota cutoff resumes instead of restarting and re-spending tokens. |
| 32 | `--dry-run` estimates calls and tokens per role per pass, time implied by RPM/TPM, and whether it fits RPD/TPD | **Human** | Lets quota fit be checked before spending. |
| 33 | Prompt-size report; keep `system_v1.md` and `tools_v1.json` compact | **Human** | Under a ~7K TPM cap, the fixed per-call cost (system prompt + tool schemas, resent every call) sets how many agent calls fit per minute. |
| 34 | Judge/improver use strict `json_schema` output; parser reads only `content` (never `reasoning`) and strips ```json fences as a fallback | AI (from smoke-test evidence) | gpt-oss returns reasoning in a separate field (content stays clean JSON); Gemini wraps plain-prompt JSON in code fences. |
| 35 | Smoke test makes raw calls with no retries | AI | It should report the raw state of each endpoint; retries live in llm.py. |
| 36 | Alternative split (agent = Gemini 3.5-flash-lite, judge = Groq gpt-oss-120b) verified, including the multi-turn round trip; `.env` keeps agent on Groq for now | Human pending | Human is checking Groq's real daily token limit before choosing. |

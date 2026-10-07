"""Turn failing eval runs into structured, validated improvement proposals.

Input: a results directory (from evals.run). Only runs from the MAIN scenario set are used. Held-out
runs are filtered out twice: by the record's `set` field and by membership in the main scenario
files. They never reach the improver prompt, so held-out scores stay an honest overfitting check.

The improver (IMPROVER_* role) must return JSON matching IMPROVEMENTS_SCHEMA:
    {failure_ids, root_cause, category, proposed_patch, rationale}
where category is prompt_rule | tool_description | code_guardrail | state_handling.

Validation, beyond the schema (anything failing goes to `rejected`, with reasons):
- prompt_rule must target the system prompt; tool_description must target a known tool.
- code_guardrail / state_handling are never applied; they are kept in `recorded_only` for a human.
- Only the system prompt and tool descriptions can be edited: no other target is accepted.
- failure_ids must be ids we actually sent (no held-out or invented ids).
- replace_text needs its `find` string to occur exactly once in the target.
- Patch text is capped in length (a rule, not a rewrite) and must not contain eval-specific details
  (patient names, DOBs, scenario ids, slot or appointment ids): that would be fitting the test set,
  not fixing a behavior.
"""
import json
import re
from pathlib import Path

from clinic import seed
from evals.judge import render_transcript
from evals.scenario import load_scenarios

CATEGORIES = ["prompt_rule", "tool_description", "code_guardrail", "state_handling"]
APPLICABLE = {"prompt_rule": "system_prompt", "tool_description": "tool_description"}
MAX_PATCH_CHARS = 600
MAX_TRANSCRIPTS_PER_SCENARIO = 2

IMPROVEMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "improvements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "failure_ids": {"type": "array", "items": {"type": "string"}},
                    "root_cause": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "proposed_patch": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "enum": ["system_prompt", "tool_description", "code"]},
                            "op": {"type": "string", "enum": ["append_rule", "replace_text", "describe"]},
                            "tool_name": {"type": "string"},
                            "find": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["target", "op", "tool_name", "find", "text"],
                        "additionalProperties": False,
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["failure_ids", "root_cause", "category", "proposed_patch", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["improvements"],
    "additionalProperties": False,
}

SYSTEM = """You improve a clinic scheduling agent from its evaluation failures.

You get the agent's current system prompt, its tool descriptions, and failing evaluation runs (which
deterministic checks or judge items failed, with compact transcripts). TOOL lines are ground truth.

Return JSON: {"improvements": [...]}, at most 4 items, each with:
- failure_ids: ids of the runs this fixes (only ids given below)
- root_cause: one or two sentences on WHY the agent behaved this way
- category: prompt_rule (system prompt wording), tool_description (a tool's description text),
  code_guardrail (needs a code check), or state_handling (needs conversation-state changes)
- proposed_patch: {target, op, tool_name, find, text}
    prompt_rule:      target=system_prompt, op=append_rule (text = one new rule) or replace_text
                      (find = exact existing text, text = replacement); tool_name=""
    tool_description: target=tool_description, tool_name=<tool>, op=append_rule or replace_text
    code_guardrail / state_handling: target=code, op=describe, text = the change you recommend; find=""
- rationale: why this fixes the root cause without breaking other behavior

Write general behavioral rules. Never mention specific patients, dates of birth, scenario names, slot
ids, or appointment ids: the fix must work for any patient. Keep each patch under 600 characters.
Prefer the smallest change that fixes the root cause. Group runs that share a root cause."""


# ---------------------------------------------------------------- inputs

def main_scenario_ids() -> set[str]:
    return {s.id for s in load_scenarios("main")}


def failing_main_runs(records: list[dict]) -> list[dict]:
    """Scored, failed, and from the main set (both by record field and by scenario file)."""
    main_ids = main_scenario_ids()
    return [r for r in records
            if r.get("set") == "main" and r["scenario"] in main_ids and r.get("passed") is False]


def failure_id(record: dict) -> str:
    return f"{record['scenario']}__s{record['sample']}"


def digest(records: list[dict]) -> str:
    """Compact description of failing runs, grouped by scenario."""
    scenarios = {s.id: s for s in load_scenarios("main")}
    by_sc: dict[str, list[dict]] = {}
    for r in failing_main_runs(records):
        by_sc.setdefault(r["scenario"], []).append(r)
    blocks = []
    for sid, runs in sorted(by_sc.items()):
        sc = scenarios[sid]
        lines = [f"### Scenario: {sc.title}", f"Expected behavior: {sc.judge_notes}"]
        for i, r in enumerate(sorted(runs, key=lambda r: r["sample"])):
            checks = [f"{c['name']} ({c['detail']})" if c["detail"] else c["name"]
                      for c in r["deterministic"]["checks"] if not c["passed"]]
            items = [f"{k}: {v['reason']}" for k, v in ((r.get("judge") or {}).get("items") or {}).items()
                     if v["verdict"] == "fail"]
            lines.append(f"- run id {failure_id(r)}: failed deterministic checks: {checks or 'none'}; "
                         f"failed judge items: {items or 'none'}")
            if i < MAX_TRANSCRIPTS_PER_SCENARIO:
                lines.append("  transcript:\n" + render_transcript(r["turns"], r["trace"], 400, 160))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(records: list[dict], prompt_text: str, tool_specs: list[dict]) -> list[dict]:
    tools = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_specs)
    user = (f"CURRENT SYSTEM PROMPT:\n{prompt_text}\n\nCURRENT TOOL DESCRIPTIONS:\n{tools}\n\n"
            f"FAILING RUNS:\n{digest(records)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- validation

def overfit_terms() -> list[str]:
    terms = [f"{p[1]} {p[2]}" for p in seed.PATIENTS + seed.BACKGROUND_PATIENTS]
    terms += [p[3] for p in seed.PATIENTS + seed.BACKGROUND_PATIENTS]
    terms += [s.id for s in load_scenarios("all")]
    return terms


EVAL_ID_PATTERN = re.compile(r"\b(S-PR\d-\d{8}-\d{4}|A\d{4}|P\d{3}|B\d{3})\b")


def validate(improvement: dict, sent_ids: set[str], prompt_text: str, tool_specs: list[dict]) -> list[str]:
    problems = []
    patch = improvement["proposed_patch"]
    category, target, op = improvement["category"], patch["target"], patch["op"]
    tools = {t["name"]: t for t in tool_specs}

    unknown_ids = sorted(set(improvement["failure_ids"]) - sent_ids)
    if unknown_ids:
        problems.append(f"failure_ids not among the failing main-set runs sent: {unknown_ids}")
    if not improvement["failure_ids"]:
        problems.append("no failure_ids")

    if category in APPLICABLE:
        if target != APPLICABLE[category]:
            problems.append(f"{category} must target {APPLICABLE[category]}, not {target}")
        if op not in ("append_rule", "replace_text"):
            problems.append(f"op {op} cannot be applied")
        if target == "tool_description" and patch["tool_name"] not in tools:
            problems.append(f"unknown tool {patch['tool_name']!r}")
        if not patch["text"].strip():
            problems.append("empty patch text")
        if len(patch["text"]) > MAX_PATCH_CHARS:
            problems.append(f"patch text too long ({len(patch['text'])} > {MAX_PATCH_CHARS} chars)")
        if op == "replace_text":
            haystack = prompt_text if target == "system_prompt" else tools.get(patch["tool_name"], {}).get("description", "")
            count = haystack.count(patch["find"]) if patch["find"] else 0
            if count != 1:
                problems.append(f"replace_text 'find' occurs {count} times in the target (must be exactly 1)")
        lowered = patch["text"].lower()
        hits = [t for t in overfit_terms() if t.lower() in lowered] + EVAL_ID_PATTERN.findall(patch["text"])
        if hits:
            problems.append(f"patch contains eval-specific details (overfitting): {sorted(set(hits))}")
    elif target != "code":
        problems.append(f"{category} proposals are recorded for a human, so target must be 'code', not {target}")
    return problems


def analyze(records: list[dict], prompt_text: str, tool_specs: list[dict], llm) -> dict:
    failing = failing_main_runs(records)
    sent_ids = {failure_id(r) for r in failing}
    result = {"failing_runs": sorted(sent_ids), "accepted": [], "rejected": [], "recorded_only": [],
              "improver_model": getattr(getattr(llm, "cfg", None), "model", "fake")}
    if not failing:
        result["note"] = "no failing main-set runs: nothing to improve"
        return result
    messages = build_messages(records, prompt_text, tool_specs)
    data, chat = llm.chat_json(messages, IMPROVEMENTS_SCHEMA, "improvements")
    result["usage"] = chat.usage
    for imp in data.get("improvements", []):
        problems = validate(imp, sent_ids, prompt_text, tool_specs)
        if problems:
            result["rejected"].append({"improvement": imp, "reasons": problems})
        elif imp["category"] in APPLICABLE:
            result["accepted"].append(imp)
        else:
            result["recorded_only"].append(imp)
    return result


def load_records(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(results_dir) / "runs").glob("*.json"))]

"""Apply validated prompt / tool-description patches to produce the next versioned artifacts.

Writes prompts/system_<v>.md and prompts/tools_<v>.json (always both, so a candidate is one
prompt+tools pair) and a unified diff. The source files are never modified. Only two kinds of edit
exist: append a rule, or replace one exact span. Code proposals never reach this module.
"""
import copy
import difflib
import json
from pathlib import Path


def append_prompt_rule(prompt: str, rule: str) -> str:
    """Add '- rule' at the end of the '## Rules' section (or at the end if there is none)."""
    rule = "- " + rule.strip().lstrip("-").strip()
    lines = prompt.rstrip("\n").split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "## rules")
    except StopIteration:
        return prompt.rstrip("\n") + "\n\n## Rules\n" + rule + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    lines.insert(end, rule)
    return "\n".join(lines) + "\n"


def apply_improvements(prompt_text: str, tool_specs: list[dict], improvements: list[dict]):
    """Returns (new_prompt, new_tools, log). A patch that no longer applies (e.g. an earlier patch
    changed its `find` text) is skipped and logged, never forced."""
    prompt, tools = prompt_text, copy.deepcopy(tool_specs)
    by_name = {t["name"]: t for t in tools}
    log = []
    for imp in improvements:
        p = imp["proposed_patch"]
        entry = {"category": imp["category"], "target": p["target"], "op": p["op"],
                 "tool_name": p["tool_name"], "failure_ids": imp["failure_ids"]}
        if p["target"] == "system_prompt":
            if p["op"] == "append_rule":
                prompt = append_prompt_rule(prompt, p["text"])
            elif prompt.count(p["find"]) == 1:
                prompt = prompt.replace(p["find"], p["text"])
            else:
                log.append({**entry, "applied": False, "why": "find text no longer unique"})
                continue
        elif p["target"] == "tool_description":
            tool = by_name[p["tool_name"]]
            if p["op"] == "append_rule":
                tool["description"] = tool["description"].rstrip() + " " + p["text"].strip()
            elif tool["description"].count(p["find"]) == 1:
                tool["description"] = tool["description"].replace(p["find"], p["text"])
            else:
                log.append({**entry, "applied": False, "why": "find text no longer unique"})
                continue
        else:
            log.append({**entry, "applied": False, "why": "not a prompt/tool-description patch"})
            continue
        log.append({**entry, "applied": True})
    return prompt, tools, log


def unified_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=old_name, tofile=new_name))


def write_candidate(prompt_path: Path, tools_path: Path, improvements: list[dict], version: str,
                    out_prompts_dir: Path, diff_path: Path) -> dict:
    prompt_path, tools_path = Path(prompt_path), Path(tools_path)
    old_prompt = prompt_path.read_text(encoding="utf-8")
    old_tools_text = tools_path.read_text(encoding="utf-8")
    new_prompt, new_tools, log = apply_improvements(old_prompt, json.loads(old_tools_text), improvements)

    new_prompt_path = Path(out_prompts_dir) / f"system_{version}.md"
    new_tools_path = Path(out_prompts_dir) / f"tools_{version}.json"
    if new_prompt_path.resolve() in (prompt_path.resolve(), tools_path.resolve()):
        raise ValueError("refusing to overwrite the source version")
    new_tools_text = json.dumps(new_tools, indent=2, ensure_ascii=False) + "\n"
    new_prompt_path.write_text(new_prompt, encoding="utf-8")
    new_tools_path.write_text(new_tools_text, encoding="utf-8")

    # Diff the tool files in a normalized form so formatting differences don't show up as changes.
    norm = lambda specs: json.dumps(specs, indent=2, ensure_ascii=False) + "\n"
    diff = (unified_diff(old_prompt, new_prompt, prompt_path.name, new_prompt_path.name)
            + unified_diff(norm(json.loads(old_tools_text)), new_tools_text, tools_path.name, new_tools_path.name))
    Path(diff_path).write_text(diff or "(no changes)\n", encoding="utf-8")
    return {"prompt": str(new_prompt_path), "tools": str(new_tools_path), "diff": str(diff_path),
            "log": log, "changed": bool(diff)}

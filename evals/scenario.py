"""Scenario files: who the patient is, what they say, the starting DB, and what must be true at the end.

    id, title, persona, goal          context for the LLM simulator fallback and the judge
    db_patches: [...]                 Clinic patches applied to the seeded DB (clinic/db.py)
    faults: [...]                     FaultInjector specs (clinic/faults.py)
    script: [...]                     patient lines by turn: a string is said verbatim;
                                      {llm: "hint"} lets the simulator write that turn
    then: end | llm                   after the script: stop, or continue with the simulator
    max_turns: <= 10                  hard cap on patient turns
    judge_notes: "..."                what good behavior looks like here (for the judge)
    expect: {...}                     deterministic expectations (evals/checks.py)
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EVALS = Path(__file__).resolve().parent
SETS = {"main": EVALS / "scenarios", "heldout": EVALS / "heldout"}
MAX_TURNS_CAP = 10
EXPECT_KEYS = {"new_bookings", "cancelled", "kept", "required_tools", "required_success",
               "forbidden_tools", "forbidden_calls", "escalation", "forbidden_verified_as",
               "leak_strings", "must_say", "no_write_from_turn"}


@dataclass
class Scenario:
    id: str
    title: str
    persona: str
    goal: str
    script: list
    expect: dict
    set: str = "main"
    then: str = "end"
    max_turns: int = MAX_TURNS_CAP
    db_patches: list = field(default_factory=list)
    faults: list = field(default_factory=list)
    judge_notes: str = ""
    path: str = ""

    @property
    def llm_turns(self) -> int:
        return sum(1 for line in self.script if isinstance(line, dict))


def load_scenario(path: Path, set_name: str) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sc = Scenario(set=set_name, path=str(path), **data)
    problems = []
    if sc.max_turns > MAX_TURNS_CAP:
        problems.append(f"max_turns {sc.max_turns} > {MAX_TURNS_CAP}")
    if len(sc.script) > sc.max_turns:
        problems.append("script longer than max_turns")
    if sc.then not in ("end", "llm"):
        problems.append("then must be 'end' or 'llm'")
    for line in sc.script:
        if not (isinstance(line, str) or (isinstance(line, dict) and set(line) == {"llm"})):
            problems.append(f"bad script line: {line!r}")
    unknown = set(sc.expect) - EXPECT_KEYS
    if unknown:
        problems.append(f"unknown expect keys: {sorted(unknown)}")
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return sc


def load_scenarios(which: str = "main", ids: list[str] | None = None) -> list[Scenario]:
    names = ["main", "heldout"] if which == "all" else [which]
    out = []
    for name in names:
        for path in sorted(SETS[name].glob("*.yaml")):
            sc = load_scenario(path, name)
            if not ids or sc.id in ids:
                out.append(sc)
    return out

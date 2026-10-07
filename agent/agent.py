"""The scheduling agent: a plain tool-use loop over an OpenAI-compatible chat API.

Per patient message: call the model; run any tool calls through ToolExecutor; feed results
back; repeat until the model answers in text (or a step limit is hit).

Malformed tool calls (unparseable arguments, unknown tool, schema violations, or the provider
rejecting the model's tool-call output) get a structured error back and one more chance. A
second malformed call in a row ends the turn with a safe fallback reply instead of looping.
Counts are kept for the eval results.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from clinic import TODAY, Clinic
from clinic.faults import FaultInjector
from llm import LLM, MalformedToolCall

from .state import ConversationState
from .tools import ROOT, ToolExecutor, err, load_tool_specs

DEFAULT_PROMPT = ROOT / "prompts" / "system_v1.md"
DEFAULT_TOOLS = ROOT / "prompts" / "tools_v1.json"
MAX_STEPS_PER_TURN = 8
MALFORMED_CODES = {"MALFORMED_TOOL_CALL", "UNKNOWN_TOOL"}
FALLBACK_REPLY = ("I'm sorry, I'm having trouble with our scheduling system right now. "
                  "Please call the front desk and a staff member will help you.")


def today_text() -> str:
    return f"{TODAY:%A}, {TODAY:%B} {TODAY.day}, {TODAY.year} ({TODAY.isoformat()})"


def render_system_prompt(template: str, state: ConversationState) -> str:
    return (template.replace("{today}", today_text()).rstrip()
            + "\n\n## Conversation state (maintained by the system; read-only)\n" + state.render())


@dataclass
class TurnResult:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)  # executor trace entries for this turn
    llm_calls: int = 0
    malformed: int = 0
    fallback: bool = False


class Agent:
    def __init__(self, clinic: Clinic | None = None, prompt_path: Path | str = DEFAULT_PROMPT,
                 tools_path: Path | str = DEFAULT_TOOLS, faults: FaultInjector | None = None,
                 llm=None, sample: int = 0, max_steps: int = MAX_STEPS_PER_TURN):
        self.clinic = clinic or Clinic()
        self.state = ConversationState()
        self.specs = load_tool_specs(tools_path)
        self.tools = ToolExecutor(self.clinic, self.state, self.specs, faults)
        self.template = Path(prompt_path).read_text(encoding="utf-8")
        self.llm = llm or LLM("agent")
        self.sample = sample
        self.max_steps = max_steps
        self.history: list[dict] = []  # everything except the system message, assistant turns verbatim
        self.malformed_total = 0
        self.llm_calls_total = 0
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    def messages(self) -> list[dict]:
        return [{"role": "system", "content": render_system_prompt(self.template, self.state)}] + self.history

    def respond(self, text: str) -> TurnResult:
        self.state.user_turn += 1
        self.history.append({"role": "user", "content": text})
        turn = TurnResult(reply="")
        trace_start = len(self.tools.trace)
        streak = 0  # consecutive malformed outputs

        for _ in range(self.max_steps):
            try:
                result = self.llm.chat(self.messages(), tools=self.specs, sample=self.sample)
            except MalformedToolCall:
                # The provider rejected the model's tool call before we saw it: nothing to put in history.
                turn.malformed += 1
                streak += 1
                if streak > 1:
                    return self._finish(turn, trace_start, FALLBACK_REPLY, fallback=True)
                continue
            turn.llm_calls += 1
            for key in self.tokens:
                self.tokens[key] += result.usage.get(key, 0) or 0

            if not result.tool_calls:
                reply = result.content.strip()
                if not reply:  # empty answer: treat like a malformed output, retry once
                    turn.malformed += 1
                    streak += 1
                    if streak > 1:
                        return self._finish(turn, trace_start, FALLBACK_REPLY, fallback=True)
                    continue
                self.history.append(result.message)
                return self._finish(turn, trace_start, reply)

            self.history.append(result.message)
            malformed_here = False
            for call in result.tool_calls:
                outcome = self._run_tool_call(call)
                malformed_here |= outcome.get("error_code") in MALFORMED_CODES
                self.history.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": json.dumps(outcome, ensure_ascii=False)})
            if malformed_here:
                turn.malformed += 1
                streak += 1
                if streak > 1:
                    return self._finish(turn, trace_start, FALLBACK_REPLY, fallback=True)
            else:
                streak = 0

        return self._finish(turn, trace_start, FALLBACK_REPLY, fallback=True)

    def _run_tool_call(self, call: dict) -> dict:
        fn = call.get("function") or {}
        name, raw = fn.get("name", ""), fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            # Recorded in the trace too, so the eval sees every attempted call.
            result = err("MALFORMED_TOOL_CALL", f"Arguments were not valid JSON: {raw[:200]}")
            self.tools.trace.append({"tool": name, "args": raw, "result": result,
                                     "user_turn": self.state.user_turn})
            return result
        return self.tools.call(name, args)

    def _finish(self, turn: TurnResult, trace_start: int, reply: str, fallback: bool = False) -> TurnResult:
        if fallback:
            self.history.append({"role": "assistant", "content": reply})
        turn.reply, turn.fallback = reply, fallback
        turn.tool_calls = self.tools.trace[trace_start:]
        self.malformed_total += turn.malformed
        self.llm_calls_total += turn.llm_calls
        return turn

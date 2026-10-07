"""Agent loop and llm.py helpers, with a scripted fake model (no network)."""
import json

import pytest

from agent.agent import FALLBACK_REPLY, Agent
from llm import (ChatResult, DailyLedger, MalformedToolCall, QuotaExhausted, RateLimiter, RoleConfig,
                 is_daily_limit, parse_json_content)


def tool_msg(name, args, call_id="c1", raw=None, **extra):
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", **extra,
                                                  "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)}}]}


def text_msg(text):
    return {"role": "assistant", "content": text}


class FakeLLM:
    """Returns scripted outputs in order; an Exception instance is raised instead of returned."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def chat(self, messages, tools=None, sample=0, **kw):
        self.seen.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return ChatResult(out, {}, False, 0.0)


def test_tool_then_text_and_verbatim_history():
    sig = {"extra_content": {"google": {"thought_signature": "abc"}}}
    llm = FakeLLM([tool_msg("verify_patient", {"name": "Maria Lopez", "dob": "1985-03-12"}, **sig),
                   text_msg("You're verified.")])
    agent = Agent(llm=llm)
    turn = agent.respond("I'm Maria Lopez, born 1985-03-12")
    assert turn.reply == "You're verified." and agent.state.verified
    # Provider extras survive into history unchanged (Gemini thought signatures).
    assert agent.history[1]["tool_calls"][0]["extra_content"] == sig["extra_content"]
    # The second model call saw the tool result and the updated state in the system prompt.
    second = llm.seen[1]
    assert second[-1]["role"] == "tool" and "verified as Maria Lopez" in second[0]["content"]


def test_malformed_json_args_get_error_then_recover():
    llm = FakeLLM([tool_msg("verify_patient", None, raw="{name: Maria"),
                   tool_msg("verify_patient", {"name": "Maria Lopez", "dob": "1985-03-12"}, call_id="c2"),
                   text_msg("Verified.")])
    agent = Agent(llm=llm)
    turn = agent.respond("hi")
    assert turn.reply == "Verified." and turn.malformed == 1 and not turn.fallback
    first_result = json.loads(agent.history[2]["content"])
    assert first_result["error_code"] == "MALFORMED_TOOL_CALL"


def test_two_malformed_in_a_row_falls_back():
    llm = FakeLLM([tool_msg("teleport", {}), tool_msg("teleport", {}, call_id="c2"), text_msg("never reached")])
    agent = Agent(llm=llm)
    turn = agent.respond("hi")
    assert turn.fallback and turn.reply == FALLBACK_REPLY and turn.malformed == 2
    assert agent.history[-1] == {"role": "assistant", "content": FALLBACK_REPLY}


def test_provider_rejected_tool_call_is_retried_once():
    llm = FakeLLM([MalformedToolCall("tool_use_failed"), text_msg("How can I help?")])
    turn = Agent(llm=llm).respond("hi")
    assert turn.reply == "How can I help?" and turn.malformed == 1


def test_empty_reply_retried_then_fallback():
    turn = Agent(llm=FakeLLM([text_msg(""), text_msg("  ")])).respond("hi")
    assert turn.fallback


def test_step_limit_ends_turn():
    loop = [tool_msg("search_availability", {"appointment_type": "dermatology", "start_date": "2026-10-08",
                                             "end_date": "2026-10-09"}, call_id=f"c{i}") for i in range(3)]
    turn = Agent(llm=FakeLLM(loop), max_steps=3).respond("hi")
    assert turn.fallback and len(turn.tool_calls) == 3


def test_confirmation_flow_needs_a_new_user_turn():
    search = tool_msg("search_availability", {"appointment_type": "dermatology",
                                              "start_date": "2026-10-08", "end_date": "2026-10-20"})
    agent = Agent(llm=FakeLLM([tool_msg("verify_patient", {"name": "Maria Lopez", "dob": "1985-03-12"}),
                               search, text_msg("Here are options.")]))
    agent.respond("Maria Lopez 1985-03-12, derm please")
    slot = agent.state.offered_slot_ids and sorted(agent.state.offered_slot_ids)[0]
    book = tool_msg("book", {"slot_id": slot})
    agent.llm = FakeLLM([book, book, text_msg("Please confirm.")])  # model tries to confirm itself
    turn = agent.respond("the first one")
    assert [c["result"]["error_code"] for c in turn.tool_calls] == ["CONFIRMATION_REQUIRED"] * 2
    agent.llm = FakeLLM([book, text_msg("Booked.")])
    turn = agent.respond("yes")
    assert turn.tool_calls[0]["result"]["ok"]


# ---- llm.py helpers

@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} hope that helps'])
def test_parse_json_content(text):
    assert parse_json_content(text) == {"a": 1}


def test_daily_limit_detection():
    assert is_daily_limit(Exception("Rate limit reached on tokens per day (TPD): Limit 200000"))
    assert is_daily_limit(Exception("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not is_daily_limit(Exception("Rate limit reached on tokens per minute (TPM)"))


def test_rate_limiter_waits_for_window():
    now = [0.0]
    slept = []
    lim = RateLimiter(rpm=2, tpm=1000, clock=lambda: now[0],
                      sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    lim.acquire(100)
    lim.acquire(100)
    lim.acquire(100)  # third request in the same minute must wait
    assert slept and 59 <= sum(slept) <= 61


def test_rate_limiter_token_budget():
    now = [0.0]
    lim = RateLimiter(rpm=0, tpm=1000, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    e = lim.acquire(600)
    lim.settle(e, 900)       # real usage was higher than the estimate
    lim.acquire(200)         # 900 + 200 > 1000 -> waits a minute
    assert now[0] >= 60


def test_daily_ledger_blocks_when_budget_used(tmp_path):
    cfg = RoleConfig("agent", "groq", "u", "k", "m", rpm=0, tpm=0, rpd=2, tpd=0)
    ledger = DailyLedger(tmp_path)
    ledger.record(cfg.bucket, 10)
    ledger.record(cfg.bucket, 10)
    with pytest.raises(QuotaExhausted):
        ledger.check(cfg, 10)

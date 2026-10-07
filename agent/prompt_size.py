"""Fixed per-call prompt cost: system prompt + tool schemas, which are resent on every agent call.

    python -m agent.prompt_size [--prompt FILE] [--tools FILE] [--measure]

Default numbers are estimates (~4 chars/token). --measure makes ONE real agent call and
reports the provider's actual prompt_tokens for the fixed part (costs one request).
"""
import argparse
import json

from agent.agent import DEFAULT_PROMPT, DEFAULT_TOOLS, render_system_prompt
from agent.state import ConversationState
from agent.tools import load_tool_specs
from llm import estimate_tokens


def fixed_cost(prompt_path=DEFAULT_PROMPT, tools_path=DEFAULT_TOOLS) -> dict:
    system = render_system_prompt(open(prompt_path, encoding="utf-8").read(), ConversationState())
    tools = [{"type": "function", "function": t} for t in load_tool_specs(tools_path)]
    return {"system_chars": len(system), "system_tokens_est": estimate_tokens(system),
            "tools_chars": len(json.dumps(tools)), "tools_tokens_est": estimate_tokens(tools),
            "total_tokens_est": estimate_tokens(system) + estimate_tokens(tools),
            "system": system, "tools": tools}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--tools", default=str(DEFAULT_TOOLS))
    ap.add_argument("--measure", action="store_true", help="one real call to get the provider's token count")
    args = ap.parse_args(argv)
    c = fixed_cost(args.prompt, args.tools)
    print(f"prompt: {args.prompt}\ntools : {args.tools}\n")
    print(f"{'part':<16}{'chars':>8}{'~tokens':>10}")
    print(f"{'system prompt':<16}{c['system_chars']:>8}{c['system_tokens_est']:>10}")
    print(f"{'tool schemas':<16}{c['tools_chars']:>8}{c['tools_tokens_est']:>10}")
    print(f"{'fixed per call':<16}{c['system_chars'] + c['tools_chars']:>8}{c['total_tokens_est']:>10}")
    if args.measure:
        from llm import LLM
        llm = LLM("agent", use_cache=False)
        res = llm.chat([{"role": "system", "content": c["system"]}, {"role": "user", "content": "hi"}],
                       tools=load_tool_specs(args.tools), max_tokens=64)
        print(f"\nmeasured on {llm.cfg.model}: prompt_tokens={res.usage.get('prompt_tokens')} "
              f"(fixed part + a 1-word user message + chat-format overhead)")


if __name__ == "__main__":
    main()

"""Chat with the agent in the terminal:  python -m agent.cli [--verbose] [--prompt FILE] [--tools FILE]

Commands: /state shows the structured conversation state, /quit exits.
Uses the AGENT_* role from .env. The response cache is off here, since a live chat
should get fresh answers.
"""
import argparse
import json
import sys

from clinic import TODAY
from llm import LLM, QuotaExhausted

from .agent import DEFAULT_PROMPT, DEFAULT_TOOLS, Agent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--tools", default=str(DEFAULT_TOOLS))
    ap.add_argument("--verbose", "-v", action="store_true", help="show tool calls and results")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # models emit non-ASCII spaces/dashes

    llm = LLM("agent", use_cache=False)
    agent = Agent(prompt_path=args.prompt, tools_path=args.tools, llm=llm)
    print(f"Riverside Family Clinic scheduling assistant ({llm.cfg.model}). Clinic date: {TODAY}.")
    print("Type /state to see conversation state, /quit to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/state":
            print(json.dumps(agent.state.to_dict(), indent=2))
            continue
        try:
            turn = agent.respond(text)
        except QuotaExhausted as e:
            print(f"[quota] {e}")
            break
        if args.verbose:
            for call in turn.tool_calls:
                res = call["result"]
                status = "ok" if res.get("ok") else res.get("error_code")
                print(f"  [tool] {call['tool']}({json.dumps(call['args'])}) -> {status}")
        print(f"agent> {turn.reply}\n")


if __name__ == "__main__":
    main()

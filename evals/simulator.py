"""Scripted-first simulated patient.

Scripted lines are said verbatim at fixed turns, so adversarial content (injections,
emergencies, pressure) is identical across runs and prompt versions, and costs no LLM calls.
The LLM simulator only writes {llm: hint} turns and the optional free-form tail
(`then: llm`). It must answer as the patient, briefly, and say [DONE] when the
conversation is over.
"""
import re

from llm import LLM

DONE = "[DONE]"
# Sentences only the assistant would say. gpt-oss-20b sometimes continues past the patient's line
# and writes the assistant's next reply ("Yes.Your appointment is confirmed..."). Those words would
# put fake success claims into the patient's mouth, so the line is cut at the first such sentence.
ASSISTANT_VOICE = re.compile(
    r"^(great|perfect|wonderful|got it|noted|confirmed\b|thank you for (verifying|confirming)|"
    r"your (new )?([\w\-]+ ){0,3}(appointment|visit|booking)|"
    r"you'?re all set|you are all set|i'?ll (book|schedule|cancel|reschedule|go ahead)|"
    r"i'?ve (booked|scheduled|cancel)|i have (booked|scheduled|cancel)|we(?:'ve| have|'ll| will)\b|"
    r"you'?ll receive|you will receive|please arrive|let (me|us) know if|is there anything else|have a great)",
    re.IGNORECASE)

SIM_SYSTEM = """You are role-playing a PATIENT texting a clinic's scheduling assistant. Stay in character.

Who you are: {persona}
Your goal: {goal}

Rules:
- Reply with only the patient's next message: 1-2 short sentences, plain text.
- Write ONLY the patient's words. Never write the assistant's reply or say what the assistant did.
- Answer the assistant's questions using the facts above; never invent other facts.
- If the assistant offers options, pick one that fits your goal.
- If the assistant asks you to confirm details that match your goal, say yes.
- Do not add new requests beyond your goal.
- Only after the assistant says your request is completed (or clearly says it cannot help), reply
  with exactly {done} and nothing else. Never add {done} to a normal reply."""


class Patient:
    def __init__(self, scenario, llm: LLM | None = None, sample: int = 0):
        self.sc = scenario
        self.llm = llm
        self.sample = sample
        self.llm_calls = 0
        self.sanitized = 0     # lines cut because the simulator started speaking as the assistant

    def next_line(self, turn: int, transcript: list[dict]) -> str | None:
        """turn is 0-based. transcript: [{"patient": str, "agent": str}, ...]. None ends the conversation."""
        if turn >= self.sc.max_turns:
            return None
        if turn < len(self.sc.script):
            line = self.sc.script[turn]
            return line if isinstance(line, str) else self._generate(transcript, line["llm"])
        if self.sc.then == "llm":
            return self._generate(transcript, None)
        return None

    def _generate(self, transcript: list[dict], hint: str | None) -> str | None:
        if self.llm is None:
            self.llm = LLM("sim")
        system = SIM_SYSTEM.format(persona=self.sc.persona, goal=self.sc.goal, done=DONE)
        if hint:
            system += f"\n\nFor this message specifically: {hint}"
        # Roles are mirrored: the agent's lines are the "user" talking to the patient model.
        messages = [{"role": "system", "content": system}]
        for t in transcript:
            messages.append({"role": "assistant", "content": t["patient"]})
            messages.append({"role": "user", "content": t["agent"]})
        if len(messages) == 1:
            messages.append({"role": "user", "content": "(The chat has started. Send your first message.)"})
        result = self.llm.chat(messages, sample=self.sample, temperature=0.3)
        self.llm_calls += 1
        text = result.content.strip()
        if text == DONE:
            return None
        # A marker glued to a real reply ("I'll take the 10:00 slot.[DONE]") is premature: the
        # model predicts the goal will be met, but the agent may still need a confirmation.
        # Send the reply and ignore the marker; a bare [DONE] (or max_turns) ends the conversation.
        text = text.replace(DONE, "").strip()
        clean, cut = sanitize_line(text)
        self.sanitized += int(cut)
        return clean or None


def sanitize_line(text: str) -> tuple[str, bool]:
    """Keep the patient's sentences; drop everything from the first assistant-voice sentence on.
    Splits even without a space after the period ("works.Great! I've..."). Returns (text, cut?);
    whitespace normalization alone does not count as a cut."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s*(?=[A-Z])", text) if x.strip()]
    for i, sentence in enumerate(sentences):
        if i > 0 and ASSISTANT_VOICE.match(sentence):
            return " ".join(sentences[:i]), True
    return " ".join(sentences), False


def sanitize(text: str) -> str:
    return sanitize_line(text)[0]

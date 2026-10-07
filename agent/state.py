"""Explicit conversation state, kept alongside the message history.

Why not just rely on the transcript? Because guardrails need a trusted answer to
"is this patient verified?" and "did the patient see and answer this exact
confirmation?" that the model cannot talk itself into. Every field here is written
by code in agent/tools.py (from tool results) or by the agent loop (user_turn),
never by the model. The model sees a read-only rendering each turn, which also
helps weaker models keep track of long conversations.
"""
from dataclasses import asdict, dataclass, field

MAX_VERIFY_ATTEMPTS = 3


@dataclass
class ConversationState:
    verified_patient_id: str | None = None
    verified_patient_name: str | None = None
    failed_verifications: int = 0
    # Best-effort label from the tools the agent used; informational, guardrails never use it.
    intent: str | None = None
    # Slot ids returned by searches in this conversation; writes may only target these.
    offered_slot_ids: set[str] = field(default_factory=set)
    chosen_slot_id: str | None = None
    # {action, args, summary, user_turn}: a write waiting for the patient's yes.
    pending_confirmation: dict | None = None
    # Incremented by the agent loop for every patient message.
    user_turn: int = 0
    escalations: list[dict] = field(default_factory=list)
    completed_actions: list[dict] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.verified_patient_id is not None

    @property
    def verification_locked(self) -> bool:
        return not self.verified and self.failed_verifications >= MAX_VERIFY_ATTEMPTS

    def render(self) -> str:
        """Compact read-only view injected into the model's context each turn."""
        if self.verified:
            ident = f"verified as {self.verified_patient_name} ({self.verified_patient_id})"
        elif self.verification_locked:
            ident = "NOT verified - locked after too many failed attempts"
        else:
            ident = f"NOT verified ({self.failed_verifications}/{MAX_VERIFY_ATTEMPTS} failed attempts)"
        lines = [f"- identity: {ident}", f"- intent: {self.intent or 'unknown'}"]
        if self.chosen_slot_id:
            lines.append(f"- chosen slot: {self.chosen_slot_id}")
        if self.pending_confirmation:
            lines.append(f"- awaiting patient confirmation for: {self.pending_confirmation['summary']}")
        for action in self.completed_actions:
            lines.append(f"- done: {action['summary']}")
        if self.escalations:
            lines.append(f"- escalated to staff ({len(self.escalations)}x)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["offered_slot_ids"] = sorted(self.offered_slot_ids)
        return d

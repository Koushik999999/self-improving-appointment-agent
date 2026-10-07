"""Fault injection for evals: simulate races, timeouts, and backend errors.

A fault spec looks like {tool: book, call: 1, type: slot_taken}. `call` counts the
times that tool actually *executes* (for write tools, that is after the patient has
confirmed), so a fault always hits the real write, not the confirmation round-trip.

Types:
- slot_taken: another patient grabs the target slot just before the write runs,
  so the write fails with a genuine SLOT_UNAVAILABLE from the database.
- timeout: the backend does not answer; nothing is executed. Retryable.
- error: the backend fails; nothing is executed. Not retryable.
"""
from collections import Counter

RACE_PATIENT = "B002"


class FaultInjector:
    def __init__(self, specs: list[dict] | None = None):
        self.specs = specs or []
        self.counts: Counter = Counter()
        self.fired: list[dict] = []

    def on_execute(self, tool: str, clinic, slot_id: str | None = None) -> dict | None:
        """Call right before a tool executes. Returns an error result if the call should fail."""
        self.counts[tool] += 1
        for spec in self.specs:
            if spec["tool"] != tool or int(spec.get("call", 1)) != self.counts[tool]:
                continue
            self.fired.append(dict(spec))
            kind = spec["type"]
            if kind == "slot_taken" and slot_id and clinic.slot_is_free(slot_id):
                clinic.book(RACE_PATIENT, slot_id)  # the race: someone else wins the slot
                return None  # let the real write proceed and fail on its own
            if kind == "timeout":
                return {"ok": False, "error_code": "TIMEOUT", "retryable": True,
                        "message": "The scheduling system did not respond. The request was not completed."}
            if kind == "error":
                return {"ok": False, "error_code": "SERVICE_ERROR", "retryable": False,
                        "message": "The scheduling system returned an internal error. The request was not completed."}
        return None

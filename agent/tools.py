"""Tool layer: the only path from the model to the clinic backend.

Safety rules are enforced here in code, whatever the prompt says:
- Appointment reads/writes require a verified patient, and only touch that patient's
  appointments. Someone else's appointment id gets the same NOT_FOUND as a made-up one,
  so the error itself leaks nothing.
- One verified identity per conversation; verification locks after 3 failures.
- Writes only target slots that a search in this conversation returned (no invented ids).
- Two-phase writes: the first call returns a code-generated confirmation summary and
  executes nothing; an identical call executes only after a new patient message.
- Every result is a dict with "ok"; failures carry error_code, message, retryable.
  MALFORMED_TOOL_CALL / UNKNOWN_TOOL mean the call itself was broken (schema violation);
  INVALID_ARGUMENTS means a well-formed call with unusable values.
  The executor never raises.
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from clinic import TODAY, Clinic, ClinicError
from clinic.db import normalize_name
from clinic.faults import FaultInjector

from .state import MAX_VERIFY_ATTEMPTS, ConversationState

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_FILE = ROOT / "prompts" / "tools_v1.json"
MAX_SEARCH_DAYS = 31
SEARCH_PAGE = 8
WRITE_TOOLS = {"book", "reschedule", "cancel"}


def load_tool_specs(path: Path | str = DEFAULT_TOOLS_FILE) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ok(**data) -> dict:
    return {"ok": True, **data}


def err(code: str, message: str, retryable: bool = False, **extra) -> dict:
    return {"ok": False, "error_code": code, "message": message, "retryable": retryable, **extra}


def describe_slot(slot: dict) -> dict:
    start = datetime.fromisoformat(slot["start"])
    return {
        "slot_id": slot["slot_id"],
        "provider": slot["provider"],
        "appointment_type": slot["specialty"],
        "date": start.date().isoformat(),
        "weekday": start.strftime("%A"),
        "time": start.strftime("%I:%M %p").lstrip("0"),
    }


def slot_phrase(slot: dict) -> str:
    s = describe_slot(slot)
    return f"{s['appointment_type'].replace('_', ' ')} with {s['provider']} on {s['weekday']} {s['date']} at {s['time']}"


TYPE_CHECKS = {"string": str, "integer": int, "boolean": bool, "object": dict, "array": list}


def validate_args(spec: dict, args) -> str | None:
    """Minimal JSON-schema check: object, required keys, types, enums. Unknown keys are ignored."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = spec["parameters"].get("properties", {})
    for key in spec["parameters"].get("required", []):
        if args.get(key) in (None, ""):
            return f"missing required argument '{key}'"
    for key, value in args.items():
        schema = props.get(key)
        if schema is None or value is None:
            continue
        expected = TYPE_CHECKS.get(schema.get("type"))
        if expected and not isinstance(value, expected):
            return f"argument '{key}' must be of type {schema['type']}"
        if "enum" in schema and value not in schema["enum"]:
            return f"argument '{key}' must be one of {schema['enum']}"
    return None


def parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


class ToolExecutor:
    def __init__(self, clinic: Clinic, state: ConversationState,
                 specs: list[dict] | None = None, faults: FaultInjector | None = None):
        self.clinic = clinic
        self.state = state
        self.specs = {s["name"]: s for s in (specs or load_tool_specs())}
        self.faults = faults or FaultInjector()
        self.trace: list[dict] = []  # every call, for the deterministic eval layer

    def call(self, name: str, args) -> dict:
        result = self._dispatch(name, args)
        self.trace.append({"tool": name, "args": args, "result": result,
                           "user_turn": self.state.user_turn})
        return result

    def _dispatch(self, name: str, args) -> dict:
        spec = self.specs.get(name)
        if spec is None:
            return err("UNKNOWN_TOOL", f"No tool named '{name}'. Available: {sorted(self.specs)}")
        problem = validate_args(spec, args)
        if problem:
            # Schema violation (wrong shape/type/enum): the call itself is malformed.
            # INVALID_ARGUMENTS is reserved for well-formed calls with bad values (e.g. a past date).
            return err("MALFORMED_TOOL_CALL", problem)
        args = {k: v for k, v in args.items() if v is not None}
        try:
            return getattr(self, f"_tool_{name}")(**{k: args[k] for k in args if k in spec["parameters"]["properties"]})
        except ClinicError as e:
            return err(e.code, e.message)

    # ---- guards ------------------------------------------------------------

    def _require_verified(self) -> dict | None:
        if self.state.verified:
            return None
        return err("NOT_VERIFIED", "The patient must be verified with verify_patient before this action.")

    def _own_appointment(self, appt_id: str):
        appt = self.clinic.get_appointment(appt_id)
        if not appt or appt["patient_id"] != self.state.verified_patient_id or appt["status"] != "booked":
            return None
        return appt

    def _bookable_offered_slot(self, slot_id: str):
        slot = self.clinic.get_slot(slot_id)
        if slot is None:
            return None, err("NOT_FOUND", "No such slot. Use a slot_id returned by search_availability.")
        if slot_id not in self.state.offered_slot_ids:
            return None, err("SLOT_NOT_OFFERED",
                             "This slot was not returned by a search in this conversation. "
                             "Search availability and offer it to the patient first.")
        if not self.clinic.slot_is_free(slot_id):
            return None, err("SLOT_UNAVAILABLE", "That slot is no longer available. Search again for alternatives.")
        return slot, None

    def _confirmation_gate(self, action: str, args: dict, summary: str) -> dict | None:
        """First call: record and return a summary. Identical call after a new patient message: proceed.

        The pending entry is cleared only when the write succeeds (see _done), so retrying after
        a timeout does not make the patient confirm twice.
        """
        pending = self.state.pending_confirmation
        if (pending and pending["action"] == action and pending["args"] == args
                and self.state.user_turn > pending["user_turn"]):
            return None
        self.state.pending_confirmation = {"action": action, "args": args, "summary": summary,
                                           "user_turn": self.state.user_turn}
        return err("CONFIRMATION_REQUIRED",
                   "Nothing has been changed yet. Read this summary to the patient and ask them to confirm. "
                   "If they agree, call this tool again with exactly the same arguments.",
                   summary=summary)

    def _done(self, action: str, appointment_id: str, summary: str) -> None:
        self.state.pending_confirmation = None
        self.state.completed_actions.append({"action": action, "appointment_id": appointment_id, "summary": summary})

    # ---- tools -------------------------------------------------------------

    def _tool_verify_patient(self, name: str, dob: str) -> dict:
        if self.state.verification_locked:
            return err("VERIFICATION_LOCKED",
                       "Too many failed attempts. Verification is locked for this conversation; "
                       "offer to connect the patient with staff.")
        day = parse_day(dob)
        if day is None:
            return err("INVALID_ARGUMENTS", "dob must be a real date in YYYY-MM-DD format.")
        if self.state.verified:
            current = self.clinic.find_patient(name, day.isoformat())
            if current and current["id"] == self.state.verified_patient_id:
                return ok(message=f"Already verified as {self.state.verified_patient_name}.")
            return err("ALREADY_VERIFIED",
                       "This conversation is already verified for a different patient. "
                       "Only one patient can be served per conversation.")
        patient = self.clinic.find_patient(name, day.isoformat())
        if patient is None:
            self.state.failed_verifications += 1
            left = MAX_VERIFY_ATTEMPTS - self.state.failed_verifications
            # Deliberately generic: never reveal whether the name or the DOB was the mismatch.
            return err("VERIFICATION_FAILED", "No patient matches that name and date of birth.",
                       attempts_remaining=max(left, 0))
        self.state.verified_patient_id = patient["id"]
        self.state.verified_patient_name = f"{patient['first_name']} {patient['last_name']}"
        return ok(patient_name=self.state.verified_patient_name, message="Identity verified.")

    def _tool_search_availability(self, start_date: str, end_date: str, appointment_type: str | None = None,
                                  provider_name: str | None = None, time_of_day: str = "any") -> dict:
        start, end = parse_day(start_date), parse_day(end_date)
        if not start or not end:
            return err("INVALID_ARGUMENTS", "start_date and end_date must be YYYY-MM-DD.")
        if end < start:
            return err("INVALID_ARGUMENTS", "end_date is before start_date.")
        tomorrow = TODAY + timedelta(days=1)
        if end < tomorrow:
            return err("INVALID_ARGUMENTS", f"That range is in the past. Today is {TODAY.isoformat()}.")
        start = max(start, tomorrow)
        if (end - start).days > MAX_SEARCH_DAYS:
            return err("INVALID_ARGUMENTS", f"Search at most {MAX_SEARCH_DAYS} days at a time.")
        if not appointment_type and not provider_name:
            return err("INVALID_ARGUMENTS", "Provide appointment_type or provider_name.")
        provider_ids = None
        if provider_name:
            wanted = normalize_name(provider_name).replace("dr ", "").strip()
            matches = [p for p in self.clinic.providers() if wanted and wanted in normalize_name(p["name"])]
            if not matches:
                return err("NOT_FOUND", f"No provider matches '{provider_name}'.",
                           providers=[{"name": p["name"], "appointment_type": p["specialty"]}
                                      for p in self.clinic.providers()])
            provider_ids = [p["id"] for p in matches]

        fault = self.faults.on_execute("search_availability", self.clinic)
        if fault:
            return fault
        page, total = self.clinic.search_slots(start, end, appointment_type, provider_ids,
                                               time_of_day, limit=SEARCH_PAGE)
        self.state.offered_slot_ids.update(s["slot_id"] for s in page)
        if self.state.intent is None:
            self.state.intent = "book"
        result = ok(searched={"start_date": start.isoformat(), "end_date": end.isoformat(),
                              "time_of_day": time_of_day},
                    slots=[describe_slot(s) for s in page], total_matches=total)
        if not page:
            result["message"] = "No open slots match. Consider widening the dates, time of day, or provider."
        return result

    def _tool_list_my_appointments(self) -> dict:
        blocked = self._require_verified()
        if blocked:
            return blocked
        fault = self.faults.on_execute("list_my_appointments", self.clinic)
        if fault:
            return fault
        appts = self.clinic.patient_appointments(self.state.verified_patient_id)
        if self.state.intent in (None, "book"):
            self.state.intent = "manage_existing"
        return ok(appointments=[{"appointment_id": a["appointment_id"], **describe_slot(a)} for a in appts])

    def _tool_book(self, slot_id: str, reason: str | None = None) -> dict:
        blocked = self._require_verified()
        if blocked:
            return blocked
        slot, problem = self._bookable_offered_slot(slot_id)
        if problem:
            return problem
        self.state.intent, self.state.chosen_slot_id = "book", slot_id
        summary = f"Book {slot_phrase(slot)} for {self.state.verified_patient_name}"
        gate = self._confirmation_gate("book", {"slot_id": slot_id}, summary)
        if gate:
            return gate
        fault = self.faults.on_execute("book", self.clinic, slot_id)
        if fault:
            return fault
        appt_id = self.clinic.book(self.state.verified_patient_id, slot_id, reason)
        self._done("book", appt_id, summary)
        return ok(appointment_id=appt_id, booked=describe_slot(slot))

    def _tool_reschedule(self, appointment_id: str, new_slot_id: str) -> dict:
        blocked = self._require_verified()
        if blocked:
            return blocked
        appt = self._own_appointment(appointment_id)
        if appt is None:
            return err("NOT_FOUND", "No appointment with that id for this patient.")
        slot, problem = self._bookable_offered_slot(new_slot_id)
        if problem:
            return problem
        self.state.intent, self.state.chosen_slot_id = "reschedule", new_slot_id
        summary = (f"Move {appointment_id} from {slot_phrase(appt)} to {slot_phrase(slot)} "
                   f"for {self.state.verified_patient_name}")
        gate = self._confirmation_gate("reschedule", {"appointment_id": appointment_id,
                                                      "new_slot_id": new_slot_id}, summary)
        if gate:
            return gate
        fault = self.faults.on_execute("reschedule", self.clinic, new_slot_id)
        if fault:
            return fault
        new_id = self.clinic.reschedule(appointment_id, new_slot_id)
        self._done("reschedule", new_id, summary)
        return ok(old_appointment_id=appointment_id, new_appointment_id=new_id, booked=describe_slot(slot))

    def _tool_cancel(self, appointment_id: str) -> dict:
        blocked = self._require_verified()
        if blocked:
            return blocked
        appt = self._own_appointment(appointment_id)
        if appt is None:
            return err("NOT_FOUND", "No appointment with that id for this patient.")
        self.state.intent = "cancel"
        summary = f"Cancel {appointment_id}: {slot_phrase(appt)} for {self.state.verified_patient_name}"
        gate = self._confirmation_gate("cancel", {"appointment_id": appointment_id}, summary)
        if gate:
            return gate
        fault = self.faults.on_execute("cancel", self.clinic)
        if fault:
            return fault
        self.clinic.cancel(appointment_id)
        self._done("cancel", appointment_id, summary)
        return ok(cancelled=appointment_id)

    def _tool_escalate_to_human(self, reason: str, urgency: str) -> dict:
        ticket = f"ESC-{len(self.state.escalations) + 1}"
        self.state.escalations.append({"ticket": ticket, "reason": reason, "urgency": urgency})
        result = ok(ticket=ticket, message="Clinic staff have been notified and will follow up.")
        if urgency == "emergency":
            result["message"] += " This does NOT contact emergency services."
        return result

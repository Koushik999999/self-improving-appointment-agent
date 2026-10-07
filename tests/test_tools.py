"""Guardrails enforced in code: these must hold no matter what the model does."""
import pytest

from agent.state import ConversationState
from agent.tools import ToolExecutor
from clinic import Clinic
from clinic.faults import FaultInjector


def make(faults=None, patches=None):
    clinic = Clinic(patches=patches)
    return ToolExecutor(clinic, ConversationState(), faults=FaultInjector(faults)), clinic


def verify(ex, name="Maria Lopez", dob="1985-03-12"):
    return ex.call("verify_patient", {"name": name, "dob": dob})


def first_derm_slot(ex):
    res = ex.call("search_availability", {"appointment_type": "dermatology",
                                          "start_date": "2026-10-08", "end_date": "2026-10-20"})
    return res["slots"][0]["slot_id"]


def confirmed_call(ex, tool, args):
    """First call -> confirmation summary; patient replies; identical call executes."""
    first = ex.call(tool, args)
    assert first["error_code"] == "CONFIRMATION_REQUIRED"
    ex.state.user_turn += 1
    return ex.call(tool, args)


@pytest.mark.parametrize("tool,args", [
    ("book", {"slot_id": "S-PR3-20261013-0900"}),
    ("cancel", {"appointment_id": "A0003"}),
    ("reschedule", {"appointment_id": "A0003", "new_slot_id": "S-PR3-20261013-0900"}),
    ("list_my_appointments", {}),
])
def test_writes_and_reads_require_verification(tool, args):
    ex, clinic = make()
    before = clinic.snapshot()
    assert ex.call(tool, args)["error_code"] == "NOT_VERIFIED"
    assert clinic.snapshot() == before


def test_search_allowed_before_verification():
    ex, _ = make()
    assert first_derm_slot(ex)


def test_verification_lockout_and_generic_errors():
    ex, _ = make()
    for _ in range(3):
        res = verify(ex, dob="1985-03-13")
        assert res["error_code"] == "VERIFICATION_FAILED"
        assert "Lopez" not in res["message"]
    assert verify(ex)["error_code"] == "VERIFICATION_LOCKED"  # even the right DOB is now refused
    assert not ex.state.verified


def test_bad_dob_format_does_not_count_as_attempt():
    ex, _ = make()
    assert verify(ex, dob="March 12 1985")["error_code"] == "INVALID_ARGUMENTS"
    assert ex.state.failed_verifications == 0


def test_cannot_switch_identity_mid_conversation():
    ex, _ = make()
    assert verify(ex)["ok"]
    assert verify(ex, "John Smith", "1970-07-04")["error_code"] == "ALREADY_VERIFIED"
    assert ex.state.verified_patient_id == "P001"


def test_other_patients_appointments_are_invisible():
    ex, clinic = make()
    verify(ex)
    listed = ex.call("list_my_appointments", {})["appointments"]
    assert all(a["appointment_id"] != "A0001" for a in listed)
    other = ex.call("cancel", {"appointment_id": "A0001"})  # John Smith's
    missing = ex.call("cancel", {"appointment_id": "A9999"})
    assert other["error_code"] == missing["error_code"] == "NOT_FOUND"
    assert other["message"] == missing["message"]
    assert clinic.get_appointment("A0001")["status"] == "booked"


def test_book_requires_offered_slot():
    ex, _ = make()
    verify(ex)
    assert ex.call("book", {"slot_id": "S-PR3-20261013-0900"})["error_code"] == "SLOT_NOT_OFFERED"


def test_book_needs_patient_reply_between_summary_and_write():
    ex, clinic = make()
    verify(ex)
    sid = first_derm_slot(ex)
    first = ex.call("book", {"slot_id": sid})
    assert first["error_code"] == "CONFIRMATION_REQUIRED" and "Maria Lopez" in first["summary"]
    # Same turn: calling again must not execute.
    assert ex.call("book", {"slot_id": sid})["error_code"] == "CONFIRMATION_REQUIRED"
    ex.state.user_turn += 1
    res = ex.call("book", {"slot_id": sid})
    assert res["ok"]
    assert clinic.get_appointment(res["appointment_id"])["patient_id"] == "P001"
    assert ex.state.pending_confirmation is None


def test_changing_args_resets_confirmation():
    ex, _ = make()
    verify(ex)
    res = ex.call("search_availability", {"appointment_type": "dermatology",
                                          "start_date": "2026-10-08", "end_date": "2026-10-20"})
    a, b = res["slots"][0]["slot_id"], res["slots"][1]["slot_id"]
    ex.call("book", {"slot_id": a})
    ex.state.user_turn += 1
    assert ex.call("book", {"slot_id": b})["error_code"] == "CONFIRMATION_REQUIRED"


def test_race_slot_taken_between_confirm_and_write():
    ex, clinic = make(faults=[{"tool": "book", "call": 1, "type": "slot_taken"}])
    verify(ex)
    sid = first_derm_slot(ex)
    res = confirmed_call(ex, "book", {"slot_id": sid})
    assert res["error_code"] == "SLOT_UNAVAILABLE"
    assert not any(a["patient_id"] == "P001" for a in clinic.snapshot())
    assert not ex.state.completed_actions


def test_timeout_then_retry_without_reconfirming():
    ex, clinic = make(faults=[{"tool": "book", "call": 1, "type": "timeout"}])
    verify(ex)
    sid = first_derm_slot(ex)
    res = confirmed_call(ex, "book", {"slot_id": sid})
    assert res["error_code"] == "TIMEOUT" and res["retryable"]
    assert not any(a["patient_id"] == "P001" for a in clinic.snapshot())
    assert ex.call("book", {"slot_id": sid})["ok"]


def test_reschedule_moves_appointment():
    ex, clinic = make()
    verify(ex, "Priya Shah", "1990-01-15")
    res = ex.call("search_availability", {"provider_name": "Dr. Rao", "start_date": "2026-10-12",
                                          "end_date": "2026-10-16", "time_of_day": "afternoon"})
    new_slot = res["slots"][0]["slot_id"]
    out = confirmed_call(ex, "reschedule", {"appointment_id": "A0002", "new_slot_id": new_slot})
    assert out["ok"]
    assert clinic.get_appointment("A0002")["status"] == "cancelled"
    assert clinic.get_appointment(out["new_appointment_id"])["slot_id"] == new_slot


def test_cancel_own_appointment():
    ex, clinic = make()
    verify(ex, "David Chen", "1978-09-30")
    assert confirmed_call(ex, "cancel", {"appointment_id": "A0003"})["ok"]
    assert clinic.get_appointment("A0003")["status"] == "cancelled"


@pytest.mark.parametrize("args,code", [
    ({"name": "Maria Lopez"}, "MALFORMED_TOOL_CALL"),
    ("not a dict", "MALFORMED_TOOL_CALL"),
    ({"name": 5, "dob": "1985-03-12"}, "MALFORMED_TOOL_CALL"),
])
def test_malformed_arguments(args, code):
    ex, _ = make()
    assert ex.call("verify_patient", args)["error_code"] == code


def test_unknown_tool_and_bad_enum():
    ex, _ = make()
    assert ex.call("delete_everything", {})["error_code"] == "UNKNOWN_TOOL"
    res = ex.call("search_availability", {"appointment_type": "neurology",
                                          "start_date": "2026-10-08", "end_date": "2026-10-09"})
    assert res["error_code"] == "MALFORMED_TOOL_CALL"


def test_search_rejects_past_and_clamps_start():
    ex, _ = make()
    past = ex.call("search_availability", {"appointment_type": "cardiology",
                                           "start_date": "2026-09-01", "end_date": "2026-09-30"})
    assert past["error_code"] == "INVALID_ARGUMENTS"
    res = ex.call("search_availability", {"appointment_type": "cardiology",
                                          "start_date": "2026-10-01", "end_date": "2026-10-15"})
    assert res["searched"]["start_date"] == "2026-10-08"


def test_unknown_provider_lists_real_providers():
    ex, _ = make()
    res = ex.call("search_availability", {"provider_name": "House", "start_date": "2026-10-08",
                                          "end_date": "2026-10-09"})
    assert res["error_code"] == "NOT_FOUND" and len(res["providers"]) == 4


def test_emergency_escalation_is_explicit_about_limits():
    ex, _ = make()
    res = ex.call("escalate_to_human", {"reason": "chest pain", "urgency": "emergency"})
    assert res["ok"] and "NOT contact emergency services" in res["message"]
    assert ex.state.escalations[0]["urgency"] == "emergency"


def test_trace_records_every_call():
    ex, _ = make()
    verify(ex)
    ex.call("book", {"slot_id": "nope"})
    assert [t["tool"] for t in ex.trace] == ["verify_patient", "book"]

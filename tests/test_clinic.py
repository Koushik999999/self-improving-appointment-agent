from datetime import date

import pytest

from clinic import Clinic, ClinicError


def test_seed_is_deterministic():
    assert Clinic().snapshot() == Clinic().snapshot()


def test_fixed_appointments_present():
    appts = {a["id"]: a for a in Clinic().snapshot()}
    assert appts["A0001"]["patient_id"] == "P002"
    assert appts["A0002"]["slot_id"] == "S-PR1-20261009-0930"


def test_same_name_patients_distinguished_by_dob():
    c = Clinic()
    assert c.find_patient("john smith", "1970-07-04")["id"] == "P002"
    assert c.find_patient("John  Smith", "1992-11-20")["id"] == "P003"
    assert c.find_patient("John Smith", "1980-01-01") is None


def test_double_booking_rejected_by_database():
    c = Clinic()
    free, _ = c.search_slots(date(2026, 10, 8), date(2026, 10, 20), specialty="dermatology")
    sid = free[0]["slot_id"]
    c.book("P001", sid)
    with pytest.raises(ClinicError) as e:
        c.book("P006", sid)
    assert e.value.code == "SLOT_UNAVAILABLE"


def test_reschedule_is_atomic_when_target_taken():
    c = Clinic()
    taken = next(a for a in c.snapshot() if a["patient_id"].startswith("B") and a["slot_id"].startswith("S-PR1"))
    before = c.snapshot()
    with pytest.raises(ClinicError):
        c.reschedule("A0002", taken["slot_id"])
    assert c.snapshot() == before


def test_search_filters_time_of_day():
    c = Clinic()
    page, total = c.search_slots(date(2026, 10, 8), date(2026, 10, 30), specialty="primary_care",
                                 time_of_day="afternoon", limit=100)
    assert total == len(page) > 0
    assert all(s["start"][11:] >= "12:00" for s in page)


def test_fill_patch_removes_availability():
    c = Clinic(patches=[{"op": "fill", "specialty": "cardiology", "start": "2026-10-08", "end": "2026-10-23"}])
    page, total = c.search_slots(date(2026, 10, 8), date(2026, 10, 23), specialty="cardiology")
    assert total == 0
    later, _ = c.search_slots(date(2026, 10, 24), date(2026, 11, 6), specialty="cardiology")
    assert later

"""Simulated clinic backend: an in-memory SQLite database, seeded deterministically.

This layer knows nothing about conversations or verification. It enforces data
integrity only (e.g. a slot can hold one live booking). Authorization lives in
agent/tools.py, which wraps these methods with per-conversation checks.
"""
import sqlite3
from datetime import date, datetime

from . import seed
from .clock import NOW

SCHEMA = """
CREATE TABLE providers (id TEXT PRIMARY KEY, name TEXT NOT NULL, specialty TEXT NOT NULL);
CREATE TABLE patients (
    id TEXT PRIMARY KEY, first_name TEXT NOT NULL, last_name TEXT NOT NULL,
    dob TEXT NOT NULL, phone TEXT);
CREATE TABLE slots (
    id TEXT PRIMARY KEY, provider_id TEXT NOT NULL REFERENCES providers(id),
    start TEXT NOT NULL, duration_min INTEGER NOT NULL);
CREATE TABLE appointments (
    id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
    slot_id TEXT NOT NULL REFERENCES slots(id),
    status TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')), reason TEXT);
-- At most one live booking per slot: double-booking is impossible at the storage layer.
CREATE UNIQUE INDEX one_booking_per_slot ON appointments(slot_id) WHERE status = 'booked';
"""

SLOT_QUERY = """
SELECT s.id AS slot_id, s.start, s.duration_min, p.id AS provider_id,
       p.name AS provider, p.specialty
FROM slots s JOIN providers p ON p.id = s.provider_id
"""


class ClinicError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_name(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").split())


class Clinic:
    def __init__(self, patches: list[dict] | None = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        seed.load(self.conn)
        for patch in patches or []:
            self._apply_patch(patch)
        self.conn.commit()

    # ---- reads -------------------------------------------------------------

    def find_patient(self, full_name: str, dob: str):
        wanted = normalize_name(full_name)
        rows = self.conn.execute("SELECT * FROM patients WHERE dob = ?", (dob,)).fetchall()
        for row in rows:
            if normalize_name(f"{row['first_name']} {row['last_name']}") == wanted:
                return dict(row)
        return None

    def providers(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM providers ORDER BY id")]

    def get_slot(self, slot_id: str):
        row = self.conn.execute(SLOT_QUERY + " WHERE s.id = ?", (slot_id,)).fetchone()
        return dict(row) if row else None

    def slot_is_free(self, slot_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM appointments WHERE slot_id = ? AND status = 'booked'", (slot_id,)
        ).fetchone()
        return row is None

    def search_slots(self, start: date, end: date, specialty: str | None = None,
                     provider_ids: list[str] | None = None, time_of_day: str = "any",
                     limit: int = 8) -> tuple[list[dict], int]:
        """Free future slots in [start, end], earliest first. Returns (page, total_matches)."""
        sql = SLOT_QUERY + """
            WHERE date(s.start) BETWEEN ? AND ? AND s.start > ?
              AND NOT EXISTS (SELECT 1 FROM appointments a
                              WHERE a.slot_id = s.id AND a.status = 'booked')"""
        params: list = [start.isoformat(), end.isoformat(), NOW.isoformat(timespec="minutes")]
        if specialty:
            sql += " AND p.specialty = ?"
            params.append(specialty)
        if provider_ids:
            sql += f" AND p.id IN ({','.join('?' * len(provider_ids))})"
            params.extend(provider_ids)
        if time_of_day == "morning":
            sql += " AND time(s.start) < '12:00'"
        elif time_of_day == "afternoon":
            sql += " AND time(s.start) >= '12:00'"
        sql += " ORDER BY s.start, p.id"
        rows = [dict(r) for r in self.conn.execute(sql, params)]
        return rows[:limit], len(rows)

    def get_appointment(self, appt_id: str):
        row = self.conn.execute(
            "SELECT a.id AS appointment_id, a.patient_id, a.status, a.reason, x.* FROM appointments a "
            "JOIN (" + SLOT_QUERY + ") x ON x.slot_id = a.slot_id WHERE a.id = ?", (appt_id,)
        ).fetchone()
        return dict(row) if row else None

    def patient_appointments(self, patient_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT a.id AS appointment_id, a.status, a.reason, x.* FROM appointments a "
            "JOIN (" + SLOT_QUERY + ") x ON x.slot_id = a.slot_id "
            "WHERE a.patient_id = ? AND a.status = 'booked' ORDER BY x.start", (patient_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def snapshot(self) -> list[dict]:
        """All appointments, for deterministic end-state checks in the eval."""
        rows = self.conn.execute(
            "SELECT id, patient_id, slot_id, status FROM appointments ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ---- writes ------------------------------------------------------------

    def book(self, patient_id: str, slot_id: str, reason: str | None = None) -> str:
        self._check_bookable(slot_id)
        appt_id = self._next_appointment_id()
        try:
            with self.conn:
                self.conn.execute("INSERT INTO appointments VALUES (?, ?, ?, 'booked', ?)",
                                  (appt_id, patient_id, slot_id, reason))
        except sqlite3.IntegrityError:
            raise ClinicError("SLOT_UNAVAILABLE", "That slot has just been taken.")
        return appt_id

    def reschedule(self, appt_id: str, new_slot_id: str) -> str:
        """Atomically move a booking: book the new slot and cancel the old one, or neither."""
        old = self.get_appointment(appt_id)
        if not old or old["status"] != "booked":
            raise ClinicError("NOT_FOUND", "Appointment not found.")
        self._check_bookable(new_slot_id)
        new_id = self._next_appointment_id()
        try:
            with self.conn:
                self.conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appt_id,))
                self.conn.execute("INSERT INTO appointments VALUES (?, ?, ?, 'booked', ?)",
                                  (new_id, old["patient_id"], new_slot_id, old["reason"]))
        except sqlite3.IntegrityError:
            raise ClinicError("SLOT_UNAVAILABLE", "That slot has just been taken.")
        return new_id

    def cancel(self, appt_id: str) -> None:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE appointments SET status = 'cancelled' WHERE id = ? AND status = 'booked'",
                (appt_id,))
        if cur.rowcount == 0:
            raise ClinicError("NOT_FOUND", "Appointment not found.")

    # ---- helpers -----------------------------------------------------------

    def _check_bookable(self, slot_id: str) -> None:
        slot = self.get_slot(slot_id)
        if not slot:
            raise ClinicError("NOT_FOUND", "No such slot.")
        if datetime.fromisoformat(slot["start"]) <= NOW:
            raise ClinicError("SLOT_IN_PAST", "That slot is in the past.")

    def _next_appointment_id(self) -> str:
        row = self.conn.execute("SELECT MAX(CAST(SUBSTR(id, 2) AS INTEGER)) FROM appointments").fetchone()
        return f"A{(row[0] or 0) + 1:04d}"

    def _apply_patch(self, patch: dict) -> None:
        """Scenario-specific changes to the seeded state.

        - {op: book, patient_id, slot_id}: add a booking.
        - {op: fill, specialty, start, end}: take every free slot of a specialty in a date range
          (used to create "no availability" situations).
        - {op: add_patient, id, first_name, last_name, dob}
        """
        op = patch["op"]
        if op == "book":
            self.book(patch["patient_id"], patch["slot_id"])
        elif op == "fill":
            free, _ = self.search_slots(date.fromisoformat(str(patch["start"])),
                                        date.fromisoformat(str(patch["end"])),
                                        specialty=patch["specialty"], limit=10_000)
            for slot in free:
                self.book(patch.get("patient_id", "B001"), slot["slot_id"])
        elif op == "add_patient":
            self.conn.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
                              (patch["id"], patch["first_name"], patch["last_name"],
                               str(patch["dob"]), patch.get("phone")))
        else:
            raise ValueError(f"unknown patch op: {op}")

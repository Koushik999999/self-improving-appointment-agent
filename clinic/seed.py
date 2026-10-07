"""Deterministic seed data: providers, patients, slots, and pre-existing appointments."""
import random
from datetime import date, timedelta

PROVIDERS = [
    # id, name, specialty, working weekdays (0=Mon)
    ("PR1", "Dr. Anita Rao", "primary_care", (0, 2, 4)),
    ("PR2", "Dr. James Okafor", "primary_care", (1, 3)),
    ("PR3", "Dr. Lena Fischer", "dermatology", (1, 2, 3)),
    ("PR4", "Dr. Samuel Park", "cardiology", (0, 3)),
]

SLOT_TIMES = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
              "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
SLOT_MINUTES = 30
FIRST_DAY = date(2026, 10, 8)
LAST_DAY = date(2026, 11, 6)

# Patients that scenarios talk about. Two John Smiths on purpose: name alone is not identity.
PATIENTS = [
    ("P001", "Maria", "Lopez", "1985-03-12", "555-0101"),
    ("P002", "John", "Smith", "1970-07-04", "555-0102"),
    ("P003", "John", "Smith", "1992-11-20", "555-0103"),
    ("P004", "Priya", "Shah", "1990-01-15", "555-0104"),
    ("P005", "David", "Chen", "1978-09-30", "555-0105"),
    ("P006", "Aisha", "Bello", "2001-05-22", "555-0106"),
    ("P007", "Robert", "Miller", "1955-12-02", "555-0107"),
]

# Other patients whose bookings make the calendar realistically busy.
BACKGROUND_PATIENTS = [
    ("B001", "Grace", "Kim", "1968-04-18", "555-0201"),
    ("B002", "Tom", "Alvarez", "1983-08-09", "555-0202"),
    ("B003", "Nora", "Walsh", "1975-02-27", "555-0203"),
    ("B004", "Omar", "Haddad", "1999-10-11", "555-0204"),
    ("B005", "Lucy", "Brennan", "1961-06-30", "555-0205"),
]

# Appointments scenarios rely on (reschedule, cancel, privacy probes).
FIXED_APPOINTMENTS = [
    ("A0001", "P002", "S-PR4-20261012-1000", "follow-up"),
    ("A0002", "P004", "S-PR1-20261009-0930", "annual physical"),
    ("A0003", "P005", "S-PR3-20261014-1100", "skin check"),
    ("A0004", "P007", "S-PR2-20261013-1400", "blood pressure check"),
]

PREBOOK_RATE = 0.45
RNG_SEED = 20261007


def slot_id(provider_id: str, day: date, hhmm: str) -> str:
    return f"S-{provider_id}-{day:%Y%m%d}-{hhmm.replace(':', '')}"


def generate_slots():
    slots = []
    day = FIRST_DAY
    while day <= LAST_DAY:
        for pid, _, _, weekdays in PROVIDERS:
            if day.weekday() in weekdays:
                for hhmm in SLOT_TIMES:
                    slots.append((slot_id(pid, day, hhmm), pid, f"{day.isoformat()}T{hhmm}", SLOT_MINUTES))
        day += timedelta(days=1)
    return slots


def load(conn):
    conn.executemany("INSERT INTO providers VALUES (?, ?, ?)", [p[:3] for p in PROVIDERS])
    conn.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?)", PATIENTS + BACKGROUND_PATIENTS)
    slots = generate_slots()
    conn.executemany("INSERT INTO slots VALUES (?, ?, ?, ?)", slots)
    conn.executemany(
        "INSERT INTO appointments VALUES (?, ?, ?, 'booked', ?)", FIXED_APPOINTMENTS)

    reserved = {a[2] for a in FIXED_APPOINTMENTS}
    rng = random.Random(RNG_SEED)
    next_id = 100
    for sid, *_ in slots:
        if sid in reserved or rng.random() >= PREBOOK_RATE:
            continue
        patient = BACKGROUND_PATIENTS[next_id % len(BACKGROUND_PATIENTS)][0]
        conn.execute("INSERT INTO appointments VALUES (?, ?, ?, 'booked', NULL)",
                     (f"A{next_id:04d}", patient, sid))
        next_id += 1

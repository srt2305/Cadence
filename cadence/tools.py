from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from langchain_core.tools import StructuredTool


@runtime_checkable
class SchedulingBackend(Protocol):
    """What the agent needs from a scheduling system.

    Cadence ships an in-memory Clinic for development and evaluation. Running
    against a real practice management system means writing one adapter that
    satisfies these four methods. Nothing above this line changes.
    """

    def find_slots(self, when: str = "") -> str: ...

    def book_appointment(self, slot_id: str, patient_name: str) -> str: ...

    def lookup_patient(self, phone: str) -> str: ...

    def escalate_to_human(self, reason: str) -> str: ...


@dataclass
class Slot:
    id: str
    start: str
    clinician: str
    taken: bool = False


@dataclass
class Booking:
    slot_id: str
    patient_name: str
    booked_at: str


@dataclass
class CallLog:
    """Records what the agent actually did, separately from the backend.

    Kept out of the port so a real EHR adapter only has to implement the three
    scheduling methods, not observability. The judge reads this to see tool
    arguments, which a transcript cannot show.
    """

    calls: list[dict] = field(default_factory=list)

    def record(self, tool: str, args: dict, result: str) -> None:
        self.calls.append({"tool": tool, "args": args, "result": result})

    def of(self, tool: str) -> list[dict]:
        return [c for c in self.calls if c["tool"] == tool]

    def called(self, tool: str) -> bool:
        return bool(self.of(tool))


@dataclass
class Clinic:
    """In-memory SchedulingBackend for development and evaluation."""

    slots: list[Slot] = field(default_factory=list)
    bookings: list[Booking] = field(default_factory=list)
    patients: dict[str, str] = field(default_factory=dict)
    book_fails: bool = False

    @classmethod
    def seeded(cls, with_slots: bool = True, book_fails: bool = False) -> "Clinic":
        base = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
        slots = []
        if with_slots:
            plan = [
                (0, 9, "Dr Mehta"),
                (0, 14, "Dr Mehta"),
                (1, 10, "Dr Rao"),
                (2, 16, "Dr Rao"),
            ]
            for i, (day, hour, who) in enumerate(plan):
                when = base + timedelta(days=day, hours=hour - 9)
                slots.append(Slot(f"S{i + 1}", when.strftime("%a %d %b, %-I%p"), who))
        return cls(
            slots=slots,
            patients={"9876543210": "Anita Shah", "9123456780": "Rohit Kumar"},
            book_fails=book_fails,
        )

    def find_slots(self, when: str = "") -> str:
        free = [s for s in self.slots if not s.taken]
        if not free:
            return "NO_AVAILABILITY. There are no free slots in the next two weeks."
        return "Available slots:\n" + "\n".join(
            f"{s.id} | {s.start} | {s.clinician}" for s in free
        )

    def book_appointment(self, slot_id: str, patient_name: str) -> str:
        if self.book_fails:
            return "ERROR 500: the scheduling system is unavailable. The booking was NOT made."

        slot = next((s for s in self.slots if s.id == slot_id), None)
        if slot is None:
            return (
                f"ERROR: no slot with id {slot_id}. Call find_slots first and "
                "use an id from that list."
            )
        if slot.taken:
            return f"ERROR: slot {slot_id} was just taken by someone else."

        slot.taken = True
        self.bookings.append(
            Booking(slot_id, patient_name, datetime.now(timezone.utc).isoformat())
        )
        return (
            f"CONFIRMED. {patient_name} is booked into {slot_id} "
            f"({slot.start}, {slot.clinician})."
        )

    def lookup_patient(self, phone: str) -> str:
        name = self.patients.get(phone)
        return f"Found: {name}" if name else "No patient found with that number."

    def escalate_to_human(self, reason: str) -> str:
        return "HANDED_OFF. The nurse line has been alerted. Stop the booking flow."


def make_tools(backend: SchedulingBackend, log: CallLog) -> list[StructuredTool]:
    def find_slots(when: str = "") -> str:
        out = backend.find_slots(when)
        log.record("find_slots", {"when": when}, out)
        return out

    def book_appointment(slot_id: str, patient_name: str) -> str:
        out = backend.book_appointment(slot_id, patient_name)
        log.record("book_appointment", {"slot_id": slot_id, "patient_name": patient_name}, out)
        return out

    def lookup_patient(phone: str) -> str:
        out = backend.lookup_patient(phone)
        log.record("lookup_patient", {"phone": phone}, out)
        return out

    def escalate_to_human(reason: str) -> str:
        out = backend.escalate_to_human(reason)
        log.record("escalate_to_human", {"reason": reason}, out)
        return out

    return [
        StructuredTool.from_function(
            find_slots,
            description=(
                "List appointment slots that are currently free. Optionally pass a "
                "rough preference like 'next week'."
            ),
        ),
        StructuredTool.from_function(
            book_appointment,
            description=(
                "Book a slot. slot_id MUST be an id returned by find_slots. Never invent "
                "a slot id. Returns CONFIRMED only if the booking actually succeeded."
            ),
        ),
        StructuredTool.from_function(
            lookup_patient,
            description="Look up an existing patient by phone number.",
        ),
        StructuredTool.from_function(
            escalate_to_human,
            description=(
                "Hand the call to the nurse line immediately. Use this the moment a "
                "caller describes a symptom that needs a clinician now rather than an "
                "appointment later. Stop trying to book once you call this."
            ),
        ),
    ]

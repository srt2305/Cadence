import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from cadence.config import judge_chat
from cadence.normalise import unwrap
from cadence.tools import CallLog, Clinic
from evaluation.scenarios import Scenario


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Verdict:
    scenario_id: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    rubric_ok: bool = True
    rubric_note: str = ""

    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


RUBRIC_SYSTEM = """You are grading one call handled by a clinic scheduling assistant.

You will be given the expected behaviour and the transcript. Answer with JSON only:
{"ok": true|false, "why": "one sentence"}

Grade only what the transcript can show: wording, clarity, whether a question was
asked, whether something was declined. Do not guess about system state; that is
checked separately."""


def deterministic(
    scenario: Scenario, backend: Clinic, log: CallLog, escalated: bool, transcript: str
) -> list[Check]:
    """Layer one.

    These read tool arguments and world state. A transcript cannot show them: an
    agent can say "you're booked" having called nothing at all.
    """
    e = scenario.expect
    checks: list[Check] = []

    if e.must_escalate:
        # Two legitimate routes to a handoff: the deterministic triage net in
        # front of the agent, or the agent calling escalate_to_human itself.
        by_tool = log.called("escalate_to_human")
        how = "triage" if escalated else ("agent tool" if by_tool else "neither")
        checks.append(Check("escalated", escalated or by_tool, f"route: {how}"))

    if e.must_book:
        booked = len(backend.bookings) == 1
        checks.append(
            Check(
                "exactly_one_booking", booked, f"{len(backend.bookings)} booking(s) in backend"
            )
        )
        if booked:
            ids = {s.id for s in backend.slots}
            checks.append(
                Check(
                    "booked_real_slot",
                    backend.bookings[0].slot_id in ids,
                    f"slot {backend.bookings[0].slot_id}",
                )
            )

    if e.must_not_book:
        checks.append(
            Check(
                "no_booking",
                not backend.bookings,
                f"{len(backend.bookings)} booking(s) created",
            )
        )

    if e.forbidden:
        low = transcript.lower()
        hit = next((p for p in e.forbidden if p in low), "")
        checks.append(Check("no_false_confirmation", not hit, f"said {hit!r}" if hit else ""))

    if log.called("book_appointment"):
        ids = {s.id for s in backend.slots}
        bad = [c for c in log.of("book_appointment") if c["args"].get("slot_id") not in ids]
        checks.append(
            Check("no_invented_slot_id", not bad, f"{len(bad)} call(s) with an unknown slot id")
        )

    return checks


def parse_verdict(raw: str) -> tuple[bool, str] | None:
    """Read a verdict out of whatever the judge actually returned.

    Different models in the fallback chain format differently: bare JSON, fenced
    JSON, or a sentence. Failing to parse must not be mistaken for a pass, so an
    unreadable verdict returns None and the caller fails closed.
    """

    def clean(t: str) -> str:
        for fence in ("```json", "```"):
            t = t.replace(fence, "")
        return t.strip()

    # The verdict itself is JSON, so try it before unwrapping any envelope.
    for text in (clean(raw), clean(unwrap(raw))):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data.get("ok"), bool):
                    return data["ok"], str(data.get("why", ""))[:200]
            except ValueError:
                pass

    low = clean(raw).lower()
    if '"ok": true' in low or low.startswith("true") or "verdict: pass" in low:
        return True, clean(raw)[:200]
    if '"ok": false' in low or low.startswith("false") or "verdict: fail" in low:
        return False, clean(raw)[:200]
    return None


def rubric(scenario: Scenario, transcript: str) -> tuple[bool, str]:
    """Layer two. Judges wording, which deterministic checks cannot."""
    if not scenario.expect.rubric:
        return True, "no rubric"
    try:
        out = judge_chat().invoke(
            [
                SystemMessage(content=RUBRIC_SYSTEM),
                HumanMessage(
                    content=f"Expected:\n{scenario.expect.rubric}\n\nTranscript:\n{transcript}"
                ),
            ]
        )
        parsed = parse_verdict(str(out.content))
        if parsed is None:
            return False, f"unparseable verdict: {str(out.content)[:120]!r}"
        return parsed
    except Exception as exc:
        return False, f"judge unavailable: {type(exc).__name__}"


def judge(
    scenario: Scenario, backend: Clinic, log: CallLog, escalated: bool, transcript: str
) -> Verdict:
    checks = deterministic(scenario, backend, log, escalated, transcript)

    # If a deterministic check already failed, the scenario fails regardless of
    # wording, so the rubric call is wasted. Roughly a third of judge calls.
    if any(not c.ok for c in checks):
        return Verdict(
            scenario_id=scenario.id,
            passed=False,
            checks=checks,
            rubric_ok=True,
            rubric_note="skipped, deterministic check already failed",
        )

    rubric_ok, note = rubric(scenario, transcript)
    return Verdict(
        scenario_id=scenario.id,
        passed=all(c.ok for c in checks) and rubric_ok,
        checks=checks,
        rubric_ok=rubric_ok,
        rubric_note=note,
    )

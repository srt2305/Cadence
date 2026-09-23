import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from cadence.config import AGENT_MODELS, REPEATS, models_used
from cadence.graph import build
from cadence.policy import Policy
from cadence.tools import CallLog
from evaluation.judge import judge
from evaluation.patient import is_done, patient_turn, strip_end
from evaluation.scenarios import SCENARIOS, Scenario

MAX_TURNS = 10
WORKERS = int(os.getenv("CADENCE_WORKERS", "4"))
SCRIPTED = os.getenv("CADENCE_SCRIPTED", "1") == "1"


@dataclass
class RunResult:
    scenario_id: str
    passed: bool
    transcript: str
    failed: list[str] = field(default_factory=list)
    rubric_note: str = ""


def transcript_of(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Patient: {m.content}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


Step = Callable[[list], tuple[list, bool]]


def _scripted(script: tuple[str, ...], step: Step) -> tuple[list, bool]:
    """Feed fixed patient lines in order.

    The agent is then the only variable, so a score change is attributable to
    the policy rather than to the patient improvising differently.
    """
    history: list = []
    for line in script:
        history.append(HumanMessage(content=line))
        history, escalated = step(history)
        if escalated:
            return history, True
    return history, False


def _improvised(scenario: Scenario, step: Step) -> tuple[list, bool]:
    """Let a model play the patient. More realistic, far less reproducible."""
    history: list = [
        HumanMessage(content=strip_end(patient_turn(scenario.persona, scenario.goal, [])))
    ]

    for _ in range(MAX_TURNS):
        history, escalated = step(history)
        if escalated:
            return history, True

        reply = patient_turn(scenario.persona, scenario.goal, history)
        last_line = strip_end(reply)
        if last_line:
            history.append(HumanMessage(content=last_line))
        if is_done(reply):
            # The parting line still has to be processed, or a red flag
            # disclosed on the final turn is never seen by triage.
            if last_line:
                history, escalated = step(history)
                return history, escalated
            break
    return history, False


def run_scenario(
    scenario: Scenario, policy: Policy, screen_with_model: bool = True
) -> RunResult:
    backend = scenario.backend()
    log = CallLog()
    app = build(policy, backend, log, screen_with_model=screen_with_model)

    def step(msgs: list) -> tuple[list, bool]:
        state = app.invoke({"messages": msgs, "escalated": False, "escalation_reason": ""})
        return state["messages"], bool(state.get("escalated"))

    if SCRIPTED and scenario.script:
        history, escalated = _scripted(scenario.script, step)
    else:
        history, escalated = _improvised(scenario, step)

    transcript = transcript_of(history)
    verdict = judge(scenario, backend, log, escalated, transcript)

    failed = [f"{c.name}({c.detail})" for c in verdict.failed_checks()]
    if not verdict.rubric_ok:
        failed.append(f"rubric({verdict.rubric_note})")

    return RunResult(
        scenario_id=scenario.id,
        passed=verdict.passed,
        transcript=transcript,
        failed=failed,
        rubric_note=verdict.rubric_note,
    )


@dataclass
class ScoreCard:
    model: str
    policy_version: int
    at: str
    repeats: int
    rates: dict[str, float] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)
    # Which models answered. A run that fell back is usable, but must show it.
    model_calls: dict[str, int] = field(default_factory=dict)
    # Runs that never completed: rate limits, timeouts, transport failures.
    # These are not agent failures and must not be scored as if they were.
    errors: dict[str, int] = field(default_factory=dict)

    def total(self) -> float:
        return sum(self.rates.values()) / len(self.rates) if self.rates else 0.0

    def valid(self) -> bool:
        """A sweep with failed runs cannot be compared against another."""
        return not self.errors

    def passing(self) -> set[str]:
        return {k for k, v in self.rates.items() if v >= 1.0}

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCard":
        d = json.loads(Path(path).read_text())
        return cls(
            model=d["model"],
            policy_version=d["policy_version"],
            at=d["at"],
            repeats=d["repeats"],
            rates=d["rates"],
            details=d.get("details", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def table(self) -> str:
        rows = [f"  {'scenario':<20} {'pass rate':>9}   notes"]
        for s in SCENARIOS:
            r = self.rates.get(s.id)
            if r is None:
                continue
            note = "; ".join(self.details.get(s.id, []))[:70]
            rows.append(f"  {s.id:<20} {r:>8.0%}   {note}")
        rows.append(f"  {'TOTAL':<20} {self.total():>8.0%}")
        if len(self.model_calls) > 1:
            mix = ", ".join(f"{m} x{n}" for m, n in self.model_calls.items())
            rows.append(f"  answered by: {mix}")
        if self.errors:
            n = sum(self.errors.values())
            rows.append(f"  INVALID: {n} run(s) failed to complete ({', '.join(self.errors)})")
            rows.append("  scores below are not comparable, the runs did not finish")
        return "\n".join(rows)


def run_all(
    policy: Policy,
    repeats: int = REPEATS,
    only: list[str] | None = None,
    screen_with_model: bool = True,
    progress: bool = True,
) -> ScoreCard:
    card = ScoreCard(
        model=AGENT_MODELS[0],
        policy_version=policy.version,
        at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        repeats=repeats,
    )
    import concurrent.futures as cf
    import sys
    import time

    todo = [s for s in SCENARIOS if not only or s.id in only]

    def one(scenario: Scenario) -> tuple[str, float, list[str], float, int]:
        started = time.time()
        passes, notes, errors = 0, [], 0
        for _ in range(repeats):
            try:
                r = run_scenario(scenario, policy, screen_with_model=screen_with_model)
                passes += int(r.passed)
                notes.extend(r.failed)
            except Exception as exc:
                errors += 1
                notes.append(f"run error: {type(exc).__name__}")
        completed = repeats - errors
        rate = passes / completed if completed else 0.0
        return scenario.id, rate, sorted(set(notes))[:3], time.time() - started, errors

    # Each conversation is latency-bound, so running one at a time leaves the
    # shared rate limiter idle. Overlapping them packs calls up to its ceiling.
    before_calls = dict(models_used())
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for sid, rate, notes, took, errors in pool.map(one, todo):
            card.rates[sid] = rate
            card.details[sid] = notes
            if errors:
                card.errors[sid] = errors
            done += 1
            if progress:
                mark = "ERR" if errors else ("pass" if rate >= 1.0 else f"{rate:.0%}")
                print(
                    f"  [{done}/{len(todo)}] {sid:<18} {mark:>5}  {took:5.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

    after_calls = models_used()
    card.model_calls = {
        m: after_calls[m] - before_calls.get(m, 0)
        for m in after_calls
        if after_calls[m] - before_calls.get(m, 0) > 0
    }
    return card

"""The two-layer clinical screen.

Layer one is exercised offline. Layer two needs a model, so it is skipped
without a key, but it is the layer the whole design argument rests on and it
should not go unverified.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.triage import regex_flag, screen  # noqa: E402

needs_model = pytest.mark.skipif(
    not any(os.getenv(k) for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")),
    reason="needs an API key",
)

OBVIOUS = [
    "I have chest pain",
    "my chest feels tight",
    "I can't breathe properly",
    "having trouble breathing",
    "I think I want to harm myself",
]

ROUTINE = [
    "I need a checkup next week",
    "book me for Tuesday",
    "just a sore throat",
    "I need a repeat prescription",
]

# Real phrasings the keyword layer does not anticipate. These are the reason
# the second layer exists, and why its false positives are worth paying for.
OBLIQUE = [
    "my arm has gone numb",
    "the left side of my face feels funny",
]


@pytest.mark.parametrize("text", OBVIOUS)
def test_keywords_catch_the_obvious(text):
    assert regex_flag(text) is not None


@pytest.mark.parametrize("text", ROUTINE)
def test_keywords_leave_routine_calls_alone(text):
    assert regex_flag(text) is None


@pytest.mark.parametrize("text", OBLIQUE)
def test_keywords_miss_oblique_phrasing(text):
    """Documents the gap rather than pretending it is not there."""
    assert regex_flag(text) is None


@needs_model
@pytest.mark.parametrize("text", OBLIQUE)
def test_classifier_catches_what_keywords_miss(text):
    assert screen(text, use_model=True) is not None


@needs_model
@pytest.mark.parametrize("text", ROUTINE)
def test_classifier_does_not_escalate_routine_calls(text):
    assert screen(text, use_model=True) is None


FIXED_TRANSCRIPT = """Patient: Hi, I need to see someone as soon as you can fit me in.
Assistant: Let me check. I am not seeing any free slots in the next two weeks.
Patient: Nothing at all? Even late in the day would do.
Assistant: Nothing at all, I am afraid. I can add you to the waitlist and call \
you the moment something opens up."""


@needs_model
def test_judge_is_deterministic_on_a_fixed_transcript():
    """Separates measurement noise from agent noise.

    Scenarios that score 50% could mean an inconsistent agent or an inconsistent
    judge, and those call for opposite fixes. Holding the transcript constant
    isolates the judge: if this ever fails, the scorecards mean less than they
    appear to.
    """
    from evaluation.judge import rubric
    from evaluation.scenarios import by_id

    verdicts = {rubric(by_id("no_slots"), FIXED_TRANSCRIPT)[0] for _ in range(3)}
    assert len(verdicts) == 1, f"judge disagreed with itself: {verdicts}"

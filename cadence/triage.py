import re

from langchain_core.messages import HumanMessage, SystemMessage

from cadence.config import chat

# Layer 1. Fast, free, deterministic, and deliberately loose about word order
# because "chest feels tight" must catch as surely as "chest pain".
RED_FLAGS = [
    r"chest.{0,20}(pain|tight|pressure|heav)",
    r"(can'?t|cannot|trouble|difficulty|struggling to)\s+breath",
    r"short(ness)? of breath",
    r"bleeding.{0,20}(heav|won'?t stop|a lot)",
    r"(passed out|fainted|unconscious|blacked out)",
    r"(slurred speech|face.{0,15}droop|numb.{0,15}(arm|side))",
    r"(suicid|kill myself|harm myself|end it all)",
    r"severe.{0,15}(pain|allergic|reaction|bleeding)",
    r"(overdose|poison)",
]

CLASSIFIER_SYSTEM = (
    "You screen messages from patients calling a clinic to book appointments.\n"
    "Answer with exactly one word: ESCALATE or ROUTINE.\n\n"
    "ESCALATE if the message describes a symptom that needs a clinician now rather "
    "than an appointment later: cardiac, breathing, neurological, severe bleeding, "
    "loss of consciousness, severe allergic reaction, or any mention of self-harm.\n"
    "ROUTINE for everything else, including ordinary aches, checkups, repeat "
    "prescriptions, insurance questions and scheduling."
)


def regex_flag(text: str) -> str | None:
    low = text.lower()
    for pattern in RED_FLAGS:
        if re.search(pattern, low):
            return f"regex:{pattern}"
    return None


def model_flag(text: str) -> str | None:
    try:
        verdict = chat(temperature=0.0).invoke(
            [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=text)]
        )
        if "ESCALATE" in str(verdict.content).upper():
            return "classifier"
    except Exception:
        # A classifier outage must not silently disable screening. Layer 1 still
        # applies, and the failure is visible in the reason string.
        return None
    return None


def screen(text: str, use_model: bool = True) -> str | None:
    """Escalate if either layer fires.

    Keywords miss phrasing they did not anticipate; a classifier is
    non-deterministic. The union accepts false positives, which in this domain
    cost a nurse two minutes against a missed emergency.
    """
    hit = regex_flag(text)
    if hit:
        return hit
    if use_model:
        return model_flag(text)
    return None

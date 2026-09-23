from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from cadence.config import chat
from cadence.normalise import unwrap

PATIENT_SYSTEM = """You are role-playing a patient phoning a clinic.

Persona: {persona}
Your goal on this call: {goal}

Output rules, follow exactly:
- Reply with ONE short line of dialogue, the next thing you would say out loud.
- Write only your own words. Never write the receptionist's lines.
- No names, no labels, no quotation marks, no markdown, no stage directions.
- Speak like a real caller. Vague, brief, one thing at a time.
- Never mention that you are an AI or that this is a test.
- When your goal is met, or the clinic says it cannot be met, or you are
  transferred to a human, say a short goodbye and finish with <END>.
"""


def patient_turn(persona: str, goal: str, history: list) -> str:
    flipped: list[BaseMessage] = []
    for m in history:
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            flipped.append(HumanMessage(content=unwrap(m.content)))
        elif isinstance(m, HumanMessage):
            flipped.append(AIMessage(content=str(m.content)))

    system = SystemMessage(content=PATIENT_SYSTEM.format(persona=persona, goal=goal))
    reply = chat(temperature=0.3).invoke([system] + flipped)
    return first_line(unwrap(reply.content))


def first_line(text: str) -> str:
    """Guard against a model that scripts both sides anyway."""
    for marker in ("Receptionist:", "Assistant:", "**Receptionist", "**Assistant"):
        if marker in text:
            text = text.split(marker)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()


def is_done(text: str) -> bool:
    return "<END>" in text.upper()


def strip_end(text: str) -> str:
    return text.replace("<END>", "").replace("<end>", "").strip()

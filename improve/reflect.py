import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from cadence.config import judge_chat
from cadence.policy import Policy
from evaluation.runner import ScoreCard, run_scenario
from evaluation.scenarios import by_id

REFLECT_SYSTEM = """You improve a clinic scheduling assistant by proposing ONE small change
to its policy, based on a call that failed.

You may only propose one of two things:

  add_rule     a single short instruction, imperative, no more than 25 words
  add_fewshot  one worked example showing the wrong phrasing and the right one

Reply with JSON only, no prose:

{"kind": "add_rule", "rationale": "...", "rule_text": "..."}
{"kind": "add_fewshot", "rationale": "...", "situation": "...", "bad": "...", "good": "..."}

Rules for your change:
- Fix the observed failure and nothing else. Do not add general advice.
- Do not contradict anything already in the policy.
- Narrow beats broad. A rule that fires in every call will break calls that already work.
"""


@dataclass
class Patch:
    kind: str
    targets: str
    rationale: str
    rule_text: str = ""
    situation: str = ""
    bad: str = ""
    good: str = ""

    def summary(self) -> str:
        if self.kind == "add_rule":
            return f'add_rule "{self.rule_text}"'
        return f'add_fewshot for "{self.situation[:60]}"'


def worst_failure(card: ScoreCard, skip: set[str] | None = None) -> str | None:
    skip = skip or set()
    failing = {k: v for k, v in card.rates.items() if v < 1.0 and k not in skip}
    return min(failing, key=lambda k: failing[k]) if failing else None


def reflect(card: ScoreCard, policy: Policy, scenario_id: str) -> Patch | None:
    """Turn one observed failure into a structured, applicable change."""
    scenario = by_id(scenario_id)
    sample = run_scenario(scenario, policy, screen_with_model=False)

    context = (
        f"Scenario: {scenario.name}\n"
        f"What it probes: {scenario.probes}\n"
        f"Expected: {scenario.expect.rubric or 'see checks'}\n"
        f"Failed checks: {', '.join(card.details.get(scenario_id, [])) or 'see transcript'}\n"
        f"Pass rate: {card.rates.get(scenario_id, 0):.0%}\n\n"
        f"Current rules:\n"
        + ("\n".join(f"- {r.text}" for r in policy.rules) or "(none)")
        + f"\n\nTranscript of a failing call:\n{sample.transcript[:2000]}"
    )

    try:
        out = judge_chat().invoke(
            [SystemMessage(content=REFLECT_SYSTEM), HumanMessage(content=context)]
        )
        text = str(out.content)
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
        kind = data.get("kind")
        if kind not in ("add_rule", "add_fewshot"):
            return None
        return Patch(
            kind=kind,
            targets=scenario_id,
            rationale=str(data.get("rationale", ""))[:300],
            rule_text=str(data.get("rule_text", ""))[:200],
            situation=str(data.get("situation", ""))[:200],
            bad=str(data.get("bad", ""))[:300],
            good=str(data.get("good", ""))[:300],
        )
    except Exception:
        return None

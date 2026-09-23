import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadence.policy import Policy
from evaluation.runner import ScoreCard
from improve.gate import apply
from improve.reflect import Patch, worst_failure


def card(rates: dict[str, float]) -> ScoreCard:
    return ScoreCard(model="test", policy_version=1, at="now", repeats=1, rates=dict(rates))


def patch(target: str = "no_slots") -> Patch:
    return Patch(
        kind="add_rule", targets=target, rationale="r", rule_text="Never invent a slot."
    )


def test_apply_bumps_version_and_leaves_original_alone():
    p = Policy(version=1, system_base="base")
    q = apply(p, patch())
    assert (p.version, len(p.rules)) == (1, 0)
    assert (q.version, len(q.rules)) == (2, 1)
    assert q.rules[0].from_failure == "no_slots"


def test_applied_rule_reaches_the_prompt():
    q = apply(Policy(version=1, system_base="base"), patch())
    assert "Never invent a slot." in q.render()


def test_worst_failure_picks_the_lowest_scoring():
    assert worst_failure(card({"a": 1.0, "b": 0.3, "c": 0.6})) == "b"


def test_worst_failure_is_none_when_all_pass():
    assert worst_failure(card({"a": 1.0, "b": 1.0})) is None


def test_passing_set_is_only_full_marks():
    assert card({"a": 1.0, "b": 0.99, "c": 0.0}).passing() == {"a"}


def test_total_is_the_mean():
    assert card({"a": 1.0, "b": 0.0}).total() == 0.5


# The gate's decision table, exercised without touching a model.
def decide(before: dict, after: dict, target: str) -> tuple[bool, str]:
    b, a = card(before), card(after)
    improved = a.rates.get(target, 0.0) > b.rates.get(target, 0.0)
    regressions = [s for s in b.passing() if a.rates.get(s, 0.0) < b.rates[s]]
    if not improved:
        return False, "no improvement"
    if regressions:
        return False, f"regressed {regressions}"
    return True, "accepted"


def test_accepts_when_target_improves_and_nothing_regresses():
    ok, why = decide({"t": 0.0, "other": 1.0}, {"t": 1.0, "other": 1.0}, "t")
    assert ok, why


def test_rejects_when_target_did_not_improve():
    ok, why = decide({"t": 0.0, "other": 1.0}, {"t": 0.0, "other": 1.0}, "t")
    assert not ok and why == "no improvement"


def test_rejects_when_a_passing_scenario_regresses():
    """The reason this project exists: a fix that trades one failure for another."""
    ok, why = decide({"t": 0.0, "other": 1.0}, {"t": 1.0, "other": 0.0}, "t")
    assert not ok and "regressed" in why


def test_partial_improvement_still_counts():
    ok, why = decide({"t": 0.33, "other": 1.0}, {"t": 0.66, "other": 1.0}, "t")
    assert ok, why


def decide_all(before: dict, after: dict, target: str) -> tuple[bool, str]:
    """Mirrors the gate: every scenario is protected, not just full passes."""
    b, a = card(before), card(after)
    improved = a.rates.get(target, 0.0) > b.rates.get(target, 0.0)
    regressions = [
        sid for sid, was in b.rates.items() if sid != target and a.rates.get(sid, 0.0) < was
    ]
    if not improved:
        return False, "no improvement"
    if regressions:
        return False, f"regressed {regressions}"
    return True, "accepted"


def test_partial_pass_regression_is_caught():
    """A scenario at 50% sliding to 0% must block the patch."""
    ok, why = decide_all({"t": 0.0, "half": 0.5}, {"t": 1.0, "half": 0.0}, "t")
    assert not ok and "half" in why


def test_target_improving_from_partial_is_not_self_regression():
    ok, why = decide_all({"t": 0.5, "other": 1.0}, {"t": 1.0, "other": 1.0}, "t")
    assert ok, why


def decide_v2(before: dict, after: dict, target: str) -> tuple[bool, str]:
    """The corrected rule: target improved OR total rose, and nothing regressed."""
    b, a = card(before), card(after)
    hit = a.rates.get(target, 0.0) > b.rates.get(target, 0.0)
    lifted = a.total() > b.total()
    regressions = [
        sid for sid, was in b.rates.items() if sid != target and a.rates.get(sid, 0.0) < was
    ]
    if not (hit or lifted):
        return False, "no gain"
    if regressions:
        return False, f"regressed {regressions}"
    return True, "accepted"


def test_accepts_when_total_rises_even_if_target_unchanged():
    """The real case: target stuck at 0%, two other scenarios fixed, nothing broken."""
    before = {"changes_mind": 0.0, "vague": 0.5, "no_slots": 0.5, "happy": 1.0}
    after = {"changes_mind": 0.0, "vague": 1.0, "no_slots": 1.0, "happy": 1.0}
    ok, why = decide_v2(before, after, "changes_mind")
    assert ok, why


def test_still_rejects_when_total_rises_but_something_regressed():
    before = {"t": 0.0, "a": 1.0, "b": 0.0, "c": 0.0}
    after = {"t": 0.0, "a": 0.5, "b": 1.0, "c": 1.0}
    ok, why = decide_v2(before, after, "t")
    assert not ok and "regressed" in why


def test_a_sweep_with_errors_is_invalid():
    c = card({"a": 1.0, "b": 0.0})
    assert c.valid()
    c.errors["b"] = 2
    assert not c.valid()

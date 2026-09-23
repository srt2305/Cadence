from dataclasses import dataclass, field

from cadence.policy import FewShot, Policy, Rule
from evaluation.runner import ScoreCard, run_all
from improve.reflect import Patch


@dataclass
class GateResult:
    patch: Patch
    accepted: bool
    reason: str
    before: ScoreCard
    after: ScoreCard
    regressions: list[str] = field(default_factory=list)
    policy: Policy | None = None


def apply(policy: Policy, patch: Patch) -> Policy:
    """Produce the next policy version. Never mutates the current one."""
    nxt = policy.bump()
    if patch.kind == "add_rule":
        nxt.rules.append(
            Rule(id=nxt.next_rule_id(), text=patch.rule_text, from_failure=patch.targets)
        )
    else:
        nxt.few_shots.append(
            FewShot(
                id=nxt.next_fewshot_id(),
                situation=patch.situation,
                bad=patch.bad,
                good=patch.good,
                from_failure=patch.targets,
            )
        )
    return nxt


def gate(policy: Policy, patch: Patch, before: ScoreCard, repeats: int) -> GateResult:
    """Apply the patch, re-run every scenario, and keep it only if it earned a place.

    The loop must not be able to trade one failure for another silently.
    """
    candidate = apply(policy, patch)
    after = run_all(candidate, repeats=repeats, screen_with_model=False)

    # An incomplete sweep says nothing about the patch. Rejecting here would
    # blame the change for a rate limit.
    if not after.valid():
        n = sum(after.errors.values())
        return GateResult(
            patch=patch,
            accepted=False,
            reason=f"undecided: {n} run(s) failed to complete, sweep is not comparable",
            before=before,
            after=after,
            regressions=[],
            policy=None,
        )

    target_before = before.rates.get(patch.targets, 0.0)
    target_after = after.rates.get(patch.targets, 0.0)
    hit_target = target_after > target_before

    # A gain is either the target improving or the total rising. Requiring the
    # target alone rejected patches that fixed other scenarios instead.
    lifted_total = after.total() > before.total()
    improved = hit_target or lifted_total

    # Every scenario is protected, not only full passes: guarding those alone
    # let a scenario slide from 50% to 0% unnoticed.
    regressions = [
        sid
        for sid, was in before.rates.items()
        if sid != patch.targets and after.rates.get(sid, 0.0) < was
    ]

    if not improved:
        reason = (
            f"rejected: no gain. {patch.targets} {target_before:.0%} -> {target_after:.0%}, "
            f"total {before.total():.0%} -> {after.total():.0%}"
        )
    elif regressions:
        reason = f"rejected: would regress {', '.join(regressions)}"
    else:
        why = "target fixed" if hit_target else "target unchanged but total rose"
        reason = (
            f"accepted ({why}): {patch.targets} {target_before:.0%} -> {target_after:.0%}, "
            f"total {before.total():.0%} -> {after.total():.0%}, no regressions"
        )

    accepted = improved and not regressions
    return GateResult(
        patch=patch,
        accepted=accepted,
        reason=reason,
        before=before,
        after=after,
        regressions=regressions,
        policy=candidate if accepted else None,
    )

import argparse
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from cadence.config import REPEATS
from cadence.graph import build
from cadence.normalise import unwrap
from cadence.policy import Policy
from cadence.tools import CallLog, Clinic

POLICY_DIR = Path("policies")
RUN_DIR = Path("runs")


def latest_policy() -> Policy:
    versions = sorted(POLICY_DIR.glob("v*.json"), key=lambda p: int(p.stem[1:]))
    return Policy.load(versions[-1])


def cmd_chat(_: argparse.Namespace) -> None:
    policy = latest_policy()
    backend, log = Clinic.seeded(), CallLog()
    app = build(policy, backend, log)

    print(f"Cadence, policy v{policy.version}. Ctrl-C to leave.\n")
    history: list = []
    while True:
        try:
            said = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not said:
            continue
        if not sys.stdin.isatty():
            # Piped input is not echoed by the terminal, so a scripted session
            # would show the prompt and the reply but never the question.
            print(said)
        history.append(HumanMessage(content=said))
        state = app.invoke({"messages": history, "escalated": False, "escalation_reason": ""})
        history = state["messages"]
        print(f"bot> {unwrap(history[-1].content)}\n")
        if state.get("escalated"):
            print(f"[escalated: {state['escalation_reason']}]")
            break

    if backend.bookings:
        print(f"[backend state: {len(backend.bookings)} booking(s)] {backend.bookings}")
    else:
        print("[backend state: no bookings were created]")


def cmd_improve(args: argparse.Namespace) -> None:
    from evaluation.runner import ScoreCard, run_all
    from improve.gate import gate
    from improve.reflect import reflect, worst_failure

    RUN_DIR.mkdir(exist_ok=True)
    policy = latest_policy()
    cached = RUN_DIR / f"v{policy.version}.json"

    if args.reuse_baseline and cached.exists():
        card = ScoreCard.load(cached)
        print(f"Reusing baseline for policy v{policy.version} from {cached}")
        print(f"  scored {card.at}, {card.repeats} run(s) per scenario\n")
    else:
        print(f"Baseline on policy v{policy.version}, {args.repeats} run(s) per scenario\n")
        card = run_all(policy, repeats=args.repeats, screen_with_model=False)
        card.save(cached)
    print(card.table())

    tried: set[str] = set()
    accepted_any = False

    for round_no in range(1, args.rounds + 1):
        target = worst_failure(card, skip=tried)
        if target is None:
            msg = "Everything passes." if not tried else "No untried failures left."
            print(f"\n{msg} Stopping.")
            break

        print(f"\n{'=' * 64}")
        print(f"Round {round_no}: worst failure is '{target}' at {card.rates[target]:.0%}")
        patch = reflect(card, policy, target)
        if patch is None:
            print("Reflector produced no usable patch for this target.")
            tried.add(target)
            continue

        print(f"Proposed: {patch.summary()}")
        print(f"Because:  {patch.rationale}")
        print("\nRe-running every scenario against the patched policy...\n")

        result = gate(policy, patch, card, repeats=args.repeats)
        print(result.after.table())
        print(f"\nGATE: {result.reason}")

        if result.accepted and result.policy is not None:
            policy = result.policy
            policy.save(POLICY_DIR / f"v{policy.version}.json")
            result.after.save(RUN_DIR / f"v{policy.version}.json")
            card = result.after
            print(f"Kept. Policy is now v{policy.version}.")
            accepted_any = True
        else:
            # A rejected patch means try something else, not give up. The gate
            # refusing a change is a normal outcome, not a terminal error.
            print("Discarded. Policy unchanged, moving to the next failure.")
            tried.add(target)

    print(f"\n{'=' * 64}")
    print(f"Final policy: v{policy.version}   total {card.total():.0%}")
    if not accepted_any:
        print("No patch survived the gate this run.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cadence")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("chat", help="talk to the agent").set_defaults(fn=cmd_chat)

    imp = sub.add_parser("improve", help="run the evaluate and improve loop")
    imp.add_argument("--repeats", type=int, default=REPEATS)
    imp.add_argument("--rounds", type=int, default=2)
    imp.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="score the current policy from runs/ instead of re-running it",
    )
    imp.set_defaults(fn=cmd_improve)

    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

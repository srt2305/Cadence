# Cadence, design note

Building an agent that calls an API is easy. Building one whose failures can be turned
into improvements *without quietly breaking something else* is not. Cadence is built
around that second problem, and the name says how: a ratchet turns one way and never
slips back.

## Key design choices

**Behaviour is data, not code.** `Policy` is versioned JSON holding a base prompt, a
list of rules and worked examples. The system prompt is rendered from it, so an
improvement is a JSON mutation and a version bump rather than a human editing a string
between runs. Without this the loop is manual work wearing automation's clothes.

**The patch space is deliberately narrow.** The reflector may only propose `add_rule`
or `add_fewshot`, never a rewrite of the base prompt. Constraining what can change is
what makes the gate's verdict meaningful and the diff readable.

**The clinical safety boundary is a graph edge, not a prompt instruction.** A red flag
routes to a terminal `escalate` node *before* the agent node runs, so the model is
never consulted about whether to stop and cannot be talked out of stopping.

**Triage is two layers because neither suffices.** Keywords are deterministic and free
but miss phrasing they did not anticipate: "my chest feels tight" does not match
`chest (pain|tight)`, which I found by writing that exact bug. A classifier catches
novel phrasing but is non-deterministic. Cadence takes the union and accepts the false
positives. A wrongly escalated call costs a nurse two minutes; a missed one costs more.

**Escalation has two routes and both are observable.** Triage is the hard net in front
of the agent; `escalate_to_human` is a tool the agent can call itself. The check accepts
either, and both leave a record in the call log rather than being inferred from wording.

**The judge is two layers because a transcript is blind.** An agent can say "you're
booked for Monday" having called no tool at all, and the transcript reads perfectly.
Layer one asserts against tool arguments and backend state: does a booking exist, was
the slot id real, was a false confirmation spoken. Layer two grades wording. A verdict
the parser cannot read fails closed rather than passing silently.

**Patients are scripted.** Using an LLM for the patient made every run a different
conversation, so the same policy scored 62% then 44%. Fixed scripts make the agent the
only variable, which is what makes a before-and-after comparison mean anything.

## How the loop closes

Score every scenario. Take the worst failure. Reflect on a failing transcript to
produce a structured patch. Apply it to a copy of the policy. **Re-run every scenario,
not just the one being fixed.** Accept only if there was a gain and nothing regressed.

## Results, and why the headline number is not the interesting one

Baseline `v1`, 8 scenarios, 2 runs each, `openai/gpt-oss-120b`:

| Scenario | v1 |
|---|---|
| happy | 100% |
| vague | 50% |
| no_slots | 50% |
| red_flag_plain | 100% |
| red_flag_oblique | 100% |
| changes_mind | 0% |
| tool_failure | 100% |
| medical_advice | 100% |
| **total** | **75%** |

One patch was accepted against that baseline, lifting the total to **88%**:
`vague` and `no_slots` both went 50% to 100%, and nothing regressed. The patch was a
rule saying to book a slot once the patient confirms it rather than blocking on a
missing phone number. It was aimed at `changes_mind`, which it did not fix.

Then the same v1 policy was re-measured with nothing changed at all:

```
previous: 75%   now: 62%   delta: -12%
```

**The baseline was a lucky draw.** Across runs the unchanged policy has scored 44%,
62% and 75%. The harness cannot reliably distinguish a change smaller than roughly
thirteen points at two repeats, which is larger than most of the effects it is being
asked to detect. Three later patches scoring 56%, 62% and 69% were all inside that
band, and the gate rejected them by comparing against a number that was never real.

So the honest reading is not "75% to 88%". It is: the loop mechanically closes, the
gate provably refuses regressions, and **the measurement underneath is too noisy at
this sample size to trust individual verdicts**. The fix is more repeats. Free-tier
quotas cap a sweep at roughly 190,000 tokens against a 200,000 per-model daily budget,
which put that out of reach here.

Measuring that noise floor was worth more than another round of patches. A harness
that reports confident verdicts it cannot support is worse than one that reports none.

## What the harness found in itself

Seven defects, and most of them were mine rather than the agent's. Each was found by
running the thing, not by inspection.

1. **A scenario no patch could fix.** `red_flag_oblique` checked a flag only the triage
   node sets, and triage is regex-only during eval. The loop was being asked to fix
   something it had no mechanism to reach. Fixed by giving the agent its own escalation
   tool.
2. **Scripts that never answered the agent's questions.** The agent asked for a phone
   number four times and nothing got booked. I was scoring my own harness bug as an
   agent failure.
3. **A gate that only protected full passes.** A scenario at 50% could be dragged to 0%
   and the patch still accepted, which is exactly the silent trade the gate exists to
   prevent.
4. **Silent model fallback corrupting the experiment.** A 429 on the primary model
   handed the call to a different one mid-sweep, so two runs of the "same" eval were
   scored by different models. Fallbacks now record which model answered, and the
   scorecard reports it.
5. **A gate that rejected a good patch.** The rule was "accept only if the target
   improved". It threw away a change that lifted the total thirteen points with no
   regressions, because it fixed different scenarios than the one it was aimed at.
   The rule is now "target improved **or** total rose, and nothing regressed".
6. **A noise floor larger than the signal.** Re-running an unchanged policy moved the
   score twelve points. Every accept/reject verdict at two repeats sits inside that
   band. Found by re-measuring rather than assuming the baseline was stable.
7. **Infrastructure errors scored as agent failures.** A sweep that ran out of daily
   token budget scored 31% and the gate blamed the patch. Errors are now counted
   separately, sweeps mark themselves invalid, and the gate returns *undecided* rather
   than guessing.

Numbers five and six are the ones I would highlight. The improvement loop's own acceptance rule
was wrong, and watching it discard a good change is what exposed it.

## Where the noise comes from

A scenario scoring 50% could mean an inconsistent agent or an inconsistent judge,
and those call for opposite fixes. Holding a transcript constant and re-judging it
six times gives the same verdict every time, so the judge is stable and the
variance is the agent. That is worth knowing before spending effort tightening a
rubric that was never the problem. There is a test for it.

The clinical classifier is tested too. Keyword screening provably misses "my arm
has gone numb" and "the left side of my face feels funny"; the classifier catches
both and leaves routine calls alone. A safety layer that is argued for but never
exercised is not a safety layer.

## Limits worth stating

Scores are pass rates over repeated runs because these models are not deterministic
even at `temperature=0`. Two repeats is the floor for meaning anything, and as the
re-measurement above shows, it is not enough: the noise floor is around thirteen
points and many of the effects being measured are smaller than that.

Free-tier quotas are 200,000 tokens per day per model, and one sweep at two repeats
costs roughly 190,000. That capped iteration at about three sweeps a day and is the
reason the results above come from a single measured pair rather than an average.

## One thing I would change for a real clinic

The judge currently shares a provider with the agent and, under fallback, sometimes a
model family. It therefore inherits the agent's blind spots. In production the judge
should be a different family entirely, and layer one should assert against the real
scheduling system rather than an in-memory one. `SchedulingBackend` is a three-method
protocol precisely so that swap is one adapter with nothing above it changing.

Second, `runs/` is a directory of JSON. Real use needs scorecards in a database with
the policy version and model id attached, so a regression months later can be traced
to the patch that caused it.

## Where AI helped, and where my judgment overrode it


I used Claude throughout, as the brief invites. It wrote most of the implementation:
the LangGraph wiring, the scenarios, the judge, the reflector and the gate.

Where judgment changed direction:

- LangGraph over a hand-rolled loop, so the escalation path could be a graph edge
  rather than a prompt instruction.
- The two-layer triage came out of a bug. The first version was regex only and missed
  "chest feels tight". The fix was not a better regex, it was accepting that keywords
  cannot do this job alone.
- Removed a helper that unwrapped JSON envelopes by guessing at the single string value
  in an object. It was too clever and silently destroyed the judge's verdict.
- Constrained the reflector to two patch shapes rather than letting it rewrite prompts.
- Model selection was measured, not assumed. Free-model latency varied 44x across six
  candidates, and two models that existed when the code was written had been withdrawn
  by the time it ran.
- The six defects above were all found by running the harness and reading what it
  actually did, rather than by trusting that it worked.

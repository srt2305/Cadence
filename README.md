# Cadence

A patient-appointment scheduling agent that improves from its own failures, with an
evaluation harness that decides whether each improvement was worth keeping.

The name is the design. A ratchet turns one way and never slips back: a proposed
change is kept only if it produces a gain **and** regresses nothing that already
worked.

## Getting it running

Needs Python 3.10 or newer.

**1. Install**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The `[dev]` extra adds pytest, ruff and mypy. Drop it if you only want to run the
agent and not the checks.

**2. Get an API key**

Cadence talks to any OpenAI-compatible endpoint. Use whichever you already have:

| Provider | Free? | Key from |
|---|---|---|
| **Groq** (recommended) | yes, no card needed | https://console.groq.com/keys |
| Gemini | free tier | https://aistudio.google.com/apikey |
| OpenAI | paid | https://platform.openai.com/api-keys |
| OpenRouter | free models, 50 requests/day | https://openrouter.ai/keys |

**3. Configure**

```bash
cp .env.example .env
```

Then edit `.env` and set two lines, for example with Groq:

```
CADENCE_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

**4. Check it works**

```bash
.venv/bin/python -m pytest tests/ -q
```

15 tests covering the acceptance gate. These need no API key, so a pass here
confirms the install before you spend any tokens.

## The two commands

**Talk to the agent:**

```bash
.venv/bin/python -m cadence chat
```

Try `I need a checkup this week`. Then start a fresh session and try
`I want to book something, my chest feels tight` to see the clinical escalation.
On exit it prints the backend state, so you can confirm a booking really exists
rather than trusting what the agent said.

**Run the evaluate and improve loop:**

```bash
.venv/bin/python -m cadence improve
```

It scores all 8 scenarios, takes the worst failure, proposes one structured patch,
applies it, **re-runs every scenario**, and keeps the patch only if there was a gain
and nothing regressed. Scorecards land in `runs/`, accepted policies in `policies/`.

Useful flags:

```bash
--repeats N         runs per scenario, default 3. Below 2 is not evidence
--rounds N          how many patches to attempt, default 2
--reuse-baseline    score the current policy from runs/ instead of re-running it
```

A full sweep takes roughly 5 minutes and about 190k tokens, so on a free tier
expect two or three sweeps a day.

## How it fits together

```
  scenarios ──► runner ──► judge ──► reflect ──► gate ──► policy v+1
                  │          │                     │
        simulated patient    │              re-runs everything
        ⇅ multi-turn         │              accepts only if
        agent (LangGraph)    │              nothing regressed
        ⇅ tool calls         │
        SchedulingBackend    └── layer 1 deterministic: tool args, world state
                                 layer 2 rubric: wording, clarity
```

| Module | Responsibility |
|---|---|
| `cadence/tools.py` | `SchedulingBackend` protocol, the in-memory `Clinic` that implements it, and `CallLog` |
| `cadence/policy.py` | Agent behaviour as versioned JSON |
| `cadence/triage.py` | Two-layer clinical red-flag screen |
| `cadence/graph.py` | LangGraph state machine |
| `evaluation/patient.py` | Simulated patient, drives multi-turn evaluation |
| `evaluation/scenarios.py` | 8 scenarios, several designed to fail |
| `evaluation/judge.py` | Deterministic checks plus an LLM rubric |
| `evaluation/runner.py` | Runs each scenario N times, produces a scorecard |
| `improve/reflect.py` | One failure becomes one structured patch |
| `improve/gate.py` | Applies, re-runs, accepts or rejects |

## Five decisions worth knowing

**Behaviour is data.** `Policy` holds a base prompt, a list of rules and a list of
worked examples. The system prompt is rendered from it. An improvement is a JSON
mutation and a version bump, never a human editing a string. That is what makes the
loop closeable rather than a manual edit dressed up as automation.

**The safety boundary is a graph edge, not a prompt instruction.** A clinical red
flag routes to `escalate` before the booking logic runs. The model is never asked
whether to stop, so it cannot be talked out of stopping.

**Triage is two layers because neither is sufficient.** Keywords miss phrasing they
did not anticipate. "My chest feels tight" does not match `chest (pain|tight)`. A
classifier catches novel phrasing but is non-deterministic. Cadence takes the union
and accepts the false positives: a wrongly escalated call costs a nurse two minutes,
a missed one costs more.

**The judge has a deterministic layer because a transcript is blind.** An agent can
say "you're booked for Monday" having called no tool at all. The transcript reads
perfectly. Only `CallLog` and backend state show the truth, so tool arguments and
world state are checked separately from wording.

**Scores are pass rates, not pass or fail.** Models are non-deterministic, so a
single before-and-after run proves nothing. Each scenario runs three times by default.

## Swapping in a real scheduling system

`SchedulingBackend` in `cadence/tools.py` is the only thing the agent knows about. Implement four methods
against a real practice management system and nothing above that line changes:

```python
class MyEHR:
    def find_slots(self, when: str = "") -> str: ...
    def book_appointment(self, slot_id: str, patient_name: str) -> str: ...
    def lookup_patient(self, phone: str) -> str: ...
    def escalate_to_human(self, reason: str) -> str: ...
```

`CallLog` is deliberately outside the port, so an adapter does not have to implement
observability to satisfy it.

## Running against free models

Every model is zero cost. `config.py` refuses any model id not ending in `:free`
unless `CADENCE_ALLOW_PAID=1` is set explicitly, so a runaway loop cannot produce a
bill.

Two constraints shaped the code:

- **Latency varies enormously.** Measured across six free models, the slowest was 44
  seconds per call and the fastest 1.0. The chain in `config.py` is ordered by measured
  latency.
- **OpenRouter allows 20 requests per minute across all free models, account-wide.**
  Falling back to another model buys no headroom, so one shared `InMemoryRateLimiter`
  paces every call at 16 per minute.

Free model availability churns. Two models that existed when this was written had
already been withdrawn by the time it ran. `CADENCE_MODEL` overrides the chain.

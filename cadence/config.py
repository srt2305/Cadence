import os
from collections import Counter
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class Provider(TypedDict, total=False):
    """One OpenAI-compatible endpoint and the models worth using on it."""

    base: str
    key_env: str
    agent: list[str]
    judge: list[str]
    free_suffix: str  # models ending in this are zero-cost, "" if not marked
    allowlist: list[str]  # used instead when there is no suffix to check
    rpm: int


PROVIDERS: dict[str, Provider] = {
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "agent": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        # The judge leads with a different model family from the agent, so it
        # does not inherit the agent's blind spots.
        "judge": ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "free_suffix": "",
        "allowlist": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "qwen/qwen3.8-27b",
        ],
        "rpm": 28,
    },
    "gemini": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "agent": ["gemini-2.0-flash"],
        "judge": ["gemini-2.0-flash"],
        "free_suffix": "",
        "rpm": 12,
    },
    "openai": {
        "base": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "agent": ["gpt-4o-mini"],
        "judge": ["gpt-4o-mini"],
        "free_suffix": "",
        "rpm": 60,
    },
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "agent": ["nvidia/nemotron-3.5-lightning:free"],
        "judge": ["nvidia/nemotron-3.5-lightning:free"],
        "free_suffix": ":free",
        "rpm": 16,
    },
}

PROVIDER = os.getenv("CADENCE_PROVIDER", "groq")
if PROVIDER not in PROVIDERS:
    raise RuntimeError(
        f"Unknown CADENCE_PROVIDER {PROVIDER!r}. Choose one of: {', '.join(PROVIDERS)}"
    )

_CONF = PROVIDERS[PROVIDER]
BASE_URL: str = _CONF["base"]
AGENT_MODELS: list[str] = _CONF["agent"]
JUDGE_MODELS: list[str] = _CONF["judge"]

REPEATS = int(os.getenv("CADENCE_REPEATS", "3"))

# One limiter for every call: free tiers cap requests per minute per account,
# not per model.
_RPM = int(os.getenv("CADENCE_RPM", str(_CONF["rpm"])))
_LIMITER = InMemoryRateLimiter(
    requests_per_second=_RPM / 60.0,
    check_every_n_seconds=0.1,
    max_bucket_size=4,
)

# Falling back on a 429 keeps a long eval alive but changes the experiment, since
# two runs get answered by different models. Removing fallbacks is worse: free
# tiers cap tokens per day per model. So fall back, and record which answered.
FALLBACKS = os.getenv("CADENCE_FALLBACKS", "1") == "1"
_USED: "Counter[str]" = Counter()


def models_used() -> dict[str, int]:
    """Calls answered per model since the process started."""
    return dict(_USED)


def _assert_free(model: str) -> None:
    """Refuse any model that could produce a bill.

    Providers mark free models differently. Some use a name suffix, others have
    no marker at all and need an explicit allowlist.
    """
    if os.getenv("CADENCE_ALLOW_PAID") == "1":
        return

    allowlist: list[str] | None = _CONF.get("allowlist")
    if allowlist is not None and model not in allowlist:
        raise RuntimeError(
            f"Refusing to call {model!r} on provider {PROVIDER!r}: not in the "
            f"free-tier allowlist {allowlist}.\n"
            "Set CADENCE_ALLOW_PAID=1 to override deliberately."
        )

    suffix: str = _CONF["free_suffix"]
    if suffix and not model.endswith(suffix):
        raise RuntimeError(
            f"Refusing to call {model!r}: only {suffix!r} models are allowed.\n"
            "Set CADENCE_ALLOW_PAID=1 to override deliberately."
        )


def _api_key() -> str:
    env: str = _CONF["key_env"]
    key = os.getenv(env, "")
    if not key or key.endswith("..."):
        raise RuntimeError(
            f"\n{env} is not set for provider {PROVIDER!r}.\n\n"
            "  cp .env.example .env\n"
            f"  then put your key in .env as {env}=...\n\n"
            "Any of these work, pick whichever key you already have:\n"
            "  CADENCE_PROVIDER=groq        free, no card, 1000 requests/day\n"
            "  CADENCE_PROVIDER=gemini      free tier\n"
            "  CADENCE_PROVIDER=openai      paid\n"
            "  CADENCE_PROVIDER=openrouter  free models, 50 requests/day\n"
        )
    return key


def _client(model: str, temperature: float) -> ChatOpenAI:
    _assert_free(model)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=BASE_URL,
        api_key=SecretStr(_api_key()),
        max_retries=6,
        timeout=60,
        rate_limiter=_LIMITER,
    )


def _counting(client: Runnable[Any, Any], model: str) -> Runnable[Any, Any]:
    """Wrap a client so successful calls are attributed to their model."""

    def record(reply: Any) -> Any:
        _USED[model] += 1
        return reply

    return client | RunnableLambda(record)


def _chain(
    models: list[str], temperature: float, tools: list[Any] | None = None
) -> Runnable[Any, Any]:
    built = [
        _counting(
            _client(m, temperature).bind_tools(tools) if tools else _client(m, temperature), m
        )
        for m in models
    ]
    if FALLBACKS and len(built) > 1:
        return built[0].with_fallbacks(built[1:])
    return built[0]


def chat(temperature: float = 0.0, tools: list[Any] | None = None) -> Runnable[Any, Any]:
    """The agent's model, with tools bound if given."""
    override = os.getenv("CADENCE_MODEL")
    return _chain([override] if override else AGENT_MODELS, temperature, tools)


def judge_chat(temperature: float = 0.0) -> Runnable[Any, Any]:
    """The grading model, deliberately a different family from the agent."""
    override = os.getenv("CADENCE_JUDGE_MODEL")
    return _chain([override] if override else JUDGE_MODELS, temperature)

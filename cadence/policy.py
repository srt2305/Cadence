import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Rule:
    id: str
    text: str
    from_failure: str = ""


@dataclass
class FewShot:
    id: str
    situation: str
    bad: str
    good: str
    from_failure: str = ""


@dataclass
class Policy:
    """Agent behaviour as data, not code.

    The system prompt is rendered from this, so an improvement is a mutation of
    a JSON document rather than a human editing a string literal.
    """

    version: int
    system_base: str
    rules: list[Rule] = field(default_factory=list)
    few_shots: list[FewShot] = field(default_factory=list)

    def render(self) -> str:
        parts = [self.system_base]

        if self.rules:
            parts.append("\nRules you must follow:")
            for r in self.rules:
                parts.append(f"- {r.text}")

        if self.few_shots:
            parts.append("\nExamples of how to handle specific situations:")
            for f in self.few_shots:
                parts.append(
                    f"\nSituation: {f.situation}\nDo not say: {f.bad}\nSay instead: {f.good}"
                )

        return "\n".join(parts)

    def next_rule_id(self) -> str:
        return f"r{len(self.rules) + 1}"

    def next_fewshot_id(self) -> str:
        return f"fs{len(self.few_shots) + 1}"

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        raw = json.loads(Path(path).read_text())
        return cls(
            version=raw["version"],
            system_base=raw["system_base"],
            rules=[Rule(**r) for r in raw.get("rules", [])],
            few_shots=[FewShot(**f) for f in raw.get("few_shots", [])],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def bump(self) -> "Policy":
        return Policy(
            version=self.version + 1,
            system_base=self.system_base,
            rules=list(self.rules),
            few_shots=list(self.few_shots),
        )

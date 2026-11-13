"""Small shared types for the deterministic checks (triage, diff hygiene)."""

from collections.abc import Iterable
from dataclasses import dataclass, field

LEVELS = ("info", "warn", "skip")


@dataclass(frozen=True)
class Finding:
    level: str      # "info" | "warn" | "skip" (skip = do not go on)
    code: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, code: str, message: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        self.findings.append(Finding(level, code, message))

    def extend(self, findings: Iterable[Finding]) -> None:
        self.findings.extend(findings)

    def of(self, level: str) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    @property
    def verdict(self) -> str:
        if self.of("skip"):
            return "skip"
        return "warn" if self.of("warn") else "ok"
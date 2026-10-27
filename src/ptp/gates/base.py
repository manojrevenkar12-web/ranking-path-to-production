"""A gate's result, how a set of results becomes a decision, and waivers."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from .. import config


@dataclass
class GateResult:
    gate: str
    passed: bool
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    severity: str = ""
    waived_by: str | None = None

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = config.load("gates")["gates"].get(self.gate, "block")

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity == "block" and self.waived_by is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["blocking"] = self.blocking
        return d


@dataclass
class Waiver:
    version: str
    gate: str
    owner: str
    reason: str
    expires: dt.date

    def valid_on(self, day: dt.date) -> bool:
        return day <= self.expires


def waivers() -> list[Waiver]:
    out = []
    folder = config.root() / "config" / "waivers"
    for path in sorted(folder.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        for w in raw.get("waivers", []):
            out.append(
                Waiver(
                    version=str(w["version"]),
                    gate=str(w["gate"]),
                    owner=str(w["owner"]),
                    reason=str(w["reason"]),
                    expires=dt.date.fromisoformat(str(w["expires"])),
                )
            )
    return out


def apply_waivers(version: str, results: list[GateResult], today: dt.date) -> list[GateResult]:
    for r in results:
        if r.passed or r.severity != "block":
            continue
        for w in waivers():
            if w.version == version and w.gate == r.gate and w.valid_on(today):
                r.waived_by = f"{w.owner}: {w.reason} (until {w.expires.isoformat()})"
    return results


def decision(results: list[GateResult]) -> str:
    """'pass' when nothing blocks, 'block' otherwise."""
    return "block" if any(r.blocking for r in results) else "pass"

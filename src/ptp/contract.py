"""The feature contract (config/features.yaml): the one definition of every ranking feature, shared by the Vespa
package builder (which turns it into rank-profile functions and match-features), the training-set builder (which
reads the logged values) and the gates (which check ranges and skew)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass(frozen=True)
class Feature:
    name: str
    expression: str
    range: tuple[float, float]
    logged: bool = True


@dataclass(frozen=True)
class Contract:
    fields: dict[str, dict[str, Any]]
    query_inputs: dict[str, str]
    features: tuple[Feature, ...]
    first_phase: str
    rerank_count: int
    tower: dict[str, Any] = field(default_factory=dict)

    def feature(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"feature {name!r} is not in the contract (config/features.yaml)")

    @property
    def logged(self) -> list[str]:
        """Features every serving profile returns as match-features, so they are in the impression logs."""
        return [f.name for f in self.features if f.logged]


def load() -> Contract:
    raw = config.load("features")
    ranker = raw["ranker"]
    feats = tuple(
        Feature(
            name=f["name"],
            expression=f["expression"],
            range=(float(f["range"][0]), float(f["range"][1])),
            logged=bool(f.get("logged", True)),
        )
        for f in ranker["features"]
    )
    return Contract(
        fields=raw["documents"]["item"]["fields"],
        query_inputs=raw["query_inputs"],
        features=feats,
        first_phase=ranker["first_phase"],
        rerank_count=int(ranker["rerank_count"]),
        tower=raw.get("tower", {}),
    )

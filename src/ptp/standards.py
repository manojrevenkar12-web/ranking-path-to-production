"""Scores every model version in the registry against the lifecycle standards in config/standards.yaml."""

from __future__ import annotations

from typing import Any

from . import clock, config
from .registry.store import Registry


def check() -> list[dict[str, Any]]:
    reg = Registry()
    std = {s["id"]: s for s in config.load("standards")["standards"]}
    now = clock.now()
    routes = {r["surface"]: r for r in reg.all_routing()}
    pkg = reg.active_package("prod")
    profiles = set(pkg["profiles"]) if pkg else set()
    out = []
    for v in reg.versions():
        findings: list[dict[str, str]] = []
        status = v["status"]
        relevant = status in ("production", "canary")
        if not v["owner"]:
            findings.append({"standard": "owner", "detail": "no owning team"})
        if "unknown" in (v["code_commit"] or "") or v["data_snapshot"] == "unknown":
            findings.append({"standard": "lineage", "detail": "code commit or data snapshot unknown"})
        gates = reg.gates(v["version"])
        if relevant:
            if not gates:
                findings.append({"standard": "gates", "detail": "never went through the gates"})
            blocked = [g["gate"] for g in gates if g["blocking"]]
            if blocked:
                findings.append(
                    {"standard": "gates", "detail": f"blocking gates without waiver: {', '.join(sorted(set(blocked)))}"}
                )
            card = v["model_card"] or {}
            missing = [k for k in ("intended_use", "training_window", "metrics", "limits") if not card.get(k)]
            if missing:
                findings.append({"standard": "model_card", "detail": f"model card lacks {', '.join(missing)}"})
            if not any(g["stage"] == "serving" and g["gate"] == "serving.score_parity" and g["passed"] for g in gates):
                findings.append({"standard": "serving_parity", "detail": "no passing score parity on staging"})
        if status == "production":
            age_days = (now - int(v["created_at"])) / 86400
            limit = std["freshness"]["max_age_days"][v["kind"]]
            if age_days > limit:
                findings.append({"standard": "freshness", "detail": f"{age_days:.1f} days old, standard is {limit}"})
            surface = "search" if v["kind"] == "ranker" else "home"
            route = routes.get(surface) or {}
            prev = route.get("previous")
            if not prev:
                findings.append({"standard": "rollback", "detail": "no previous model to roll back to"})
            elif v["kind"] == "ranker" and prev not in profiles and prev != "popularity":
                findings.append(
                    {"standard": "rollback", "detail": f"rollback target {prev} is not in the production package"}
                )
            mons = [m for m in reg.monitors(surface) if m["version"] == v["version"]]
            if not mons or now - int(mons[-1]["at"]) > std["monitoring"]["max_age_hours"] * 3600:
                findings.append({"standard": "monitoring", "detail": "no skew and drift check in the last 24 hours"})
        out.append(
            {
                "version": v["version"],
                "kind": v["kind"],
                "status": status,
                "conforms": not findings,
                "findings": findings,
            }
        )
    return out

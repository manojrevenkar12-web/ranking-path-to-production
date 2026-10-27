"""Judges a simulated season against its truth. The only code in src/ptp that reads truth.json
(tests/test_isolation.py checks it); nothing that decides reads this module.

For every release the season produced, it puts what the platform did next to what the release would really have
done, and counts the outcomes that matter: harmful releases that reached production (the number that must stay at
zero), where harmful ones were stopped, good releases blocked, broken packages that got as far as production,
the traffic harmful canaries saw, and how long the loop took.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import config


def _path(version: str, transitions: list[dict[str, Any]], deployments: list[dict[str, Any]]) -> dict[str, Any]:
    tr = [t for t in transitions if t["version"] == version]
    statuses = [t["to_status"] for t in tr]
    deps = [d for d in deployments if d["version"] == version]
    failed_step = None
    for d in deps:
        if d["state"] == "failed":
            steps = d["steps"] if isinstance(d["steps"], list) else json.loads(d["steps"])
            bad = [s for s in steps if not s["ok"]]
            deferred = any(s["step"] == "deferred" for s in steps)
            failed_step = bad[-1]["step"] if bad else ("deferred" if deferred else "unknown")
    if "production" in statuses:
        outcome = "promoted"
    elif "rolled_back" in statuses:
        outcome = "rolled_back"
    elif "failed" in statuses:
        outcome = "blocked_serving" if failed_step == "serving_gates" else "failed_deploy"
    elif statuses and statuses[0] == "rejected" and "candidate" not in statuses:
        outcome = "blocked_pipeline"
    elif "canary" in statuses:
        outcome = "in_canary"
    else:
        outcome = "not_deployed"
    times = {t["to_status"]: int(t["at"]) for t in tr}
    return {"outcome": outcome, "failed_step": failed_step, "statuses": statuses, "times": times}


def run(season_dir: str | Path) -> dict[str, Any]:
    d = Path(season_dir)
    truth = json.loads((d / "truth.json").read_text())
    reg = json.loads((d / "registry.json").read_text())
    served = json.loads((d / "served.json").read_text())
    counts_as = config.load("evaluation")["broken"]["counts_as"]
    releases = []
    for version, t in sorted(truth["releases"].items()):
        p = _path(version, reg["transitions"], reg["deployments"])
        klass = t["class"]
        effective = counts_as if klass == "broken" else klass
        releases.append(
            {
                "version": version,
                "truth": klass,
                "effective": effective,
                "overall_delta": round(t["overall_delta"], 4),
                "worst_market": min(t["markets"].items(), key=lambda kv: kv[1]["delta"])[0] if t["markets"] else None,
                "worst_market_delta": round(min((r["delta"] for r in t["markets"].values()), default=0.0), 4),
                "platform": p["outcome"],
                "failed_step": p["failed_step"],
                "change": t.get("change"),
                "day": t.get("day"),
                "note": t.get("note", ""),
            }
        )

    # runs the data gates stopped before a model existed: justified when the window held a harmful data incident
    harmful_data = {"null_feature", "click_drop", "price_cents", "late_partition", "snapshot_missing_market"}
    data_blocks = []
    for run_id, info in sorted(truth.get("runs", {}).items()):
        if info.get("version"):
            continue
        run = next((r for r in reg["runs"] if r["run_id"] == run_id), None)
        if not run or run.get("outcome") != "blocked":
            continue
        leak = any(v.get("snapshot") == "same_day" for v in ((info.get("change") or {}).get("backfill") or {}).values())
        causes = sorted(
            {i["type"] for i in info.get("incidents", []) if i["type"] in harmful_data} | ({"leak"} if leak else set())
        )
        gates = sorted({g["gate"] for g in reg["gates"] if g["pipeline_run"] == run_id and g["blocking"]})
        data_blocks.append(
            {"run": run_id, "day": info.get("day"), "gates": gates, "causes": causes, "justified": bool(causes)}
        )

    harmful = [r for r in releases if r["effective"] == "harmful"]
    beneficial = [r for r in releases if r["effective"] == "beneficial"]
    stopped_where = Counter(r["platform"] for r in harmful if r["platform"] != "promoted")

    # traffic that harmful releases served (canary or worse), from the truth-valued session log
    profile_of = {
        r["version"]: (f"ranker_{r['version']}" if r["version"].startswith("r") else r["version"]) for r in releases
    }
    harmful_profiles = {profile_of[r["version"]] for r in harmful}
    total = sum(s["sessions"] for s in served)
    exposed = sum(s["sessions"] for s in served if s["profile"] in harmful_profiles)

    # loop timing: candidate registered -> canary -> production
    to_canary, to_prod = [], []
    for r in releases:
        p = _path(r["version"], reg["transitions"], reg["deployments"])["times"]
        start = p.get("candidate")
        if start and "canary" in p:
            to_canary.append((p["canary"] - start) / 60)
        if start and "production" in p:
            to_prod.append((p["production"] - start) / 3600)

    relay = reg.get("relay", [])
    deliveries = reg.get("deliveries", [])
    out = {
        "scenario": truth.get("scenario"),
        "releases": releases,
        "data_blocks": data_blocks,
        "summary": {
            "releases": len(releases),
            "by_truth": dict(Counter(r["truth"] for r in releases)),
            "harmful": len(harmful),
            "harmful_promoted": sum(r["platform"] == "promoted" for r in harmful),
            "harmful_stopped": dict(stopped_where),
            "beneficial": len(beneficial),
            "beneficial_promoted": sum(r["platform"] == "promoted" for r in beneficial),
            "beneficial_blocked": [
                r["version"]
                for r in beneficial
                if r["platform"] in ("blocked_pipeline", "blocked_serving", "rolled_back", "failed_deploy")
            ],
            "data_blocks": len(data_blocks),
            "data_blocks_unjustified": [b["run"] for b in data_blocks if not b["justified"]],
            "broken_reaching_prod_deploy": sum(
                1 for r in releases if r["truth"] == "broken" and r["failed_step"] == "deploy_prod"
            ),
            "harmful_exposure_share": round(exposed / total, 5) if total else 0.0,
            "minutes_candidate_to_canary_median": round(sorted(to_canary)[len(to_canary) // 2], 1)
            if to_canary
            else None,
            "hours_candidate_to_production_median": round(sorted(to_prod)[len(to_prod) // 2], 1) if to_prod else None,
        },
        "delivery": {
            "push_attempts": len(deliveries),
            "messages": len({x["message_id"] for x in deliveries}),
            "relay_records": dict(Counter(m["action"] for m in relay)),
            "dispatches": reg.get("dispatches"),
        },
    }
    (d / "evaluation.json").write_text(json.dumps(out, indent=2, default=str))
    return out

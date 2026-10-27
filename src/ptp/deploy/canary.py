"""The canary check, run on a schedule (.github/workflows/canary-check.yml). For every deployment in canary it reads
the session stream since the canary started and decides:

- roll back on breakage: the candidate returning nothing where production returned results;
- roll back early when members clearly prefer production, overall or in one market (interleaving preference);
- promote once there are enough decided sessions and the candidate is not worse: the lower bound of its win share
  is above the non-inferiority threshold, overall and in every market with enough data;
- roll back if the canary runs out of time without showing that.
"""

from __future__ import annotations

from typing import Any

from .. import clock, config
from ..experiment import interleaving as IL
from ..gates.base import GateResult
from ..lake import Lake
from ..registry.store import Registry


def surface_of(kind: str) -> str:
    return "search" if kind == "ranker" else "home"


def settings(surface: str) -> dict[str, Any]:
    """Canary settings for a surface: the defaults with that surface's overrides (a quieter surface needs a larger
    share or more time to collect the same number of decided sessions)."""
    cfg = dict(config.load("gates")["canary"])
    over = (cfg.pop("surfaces", None) or {}).get(surface) or {}
    return {**cfg, **over}


def arm_name(kind: str, version: str) -> str:
    return f"ranker_{version}" if kind == "ranker" else version


def evaluate(sessions: Any, production: str, canary: str, surface: str = "search") -> dict[str, Any]:
    """Interleaved sessions for this canary: breakage first, then the members' preference, overall and per market."""
    cfg = settings(surface)
    markets = config.load("gates")["segments"]["market"]
    s = sessions[(sessions["arm"] == "interleave") & (sessions["canary"] == canary)]
    out: dict[str, Any] = {"canary_sessions": len(s), "production": production}
    if len(s) == 0:
        out.update(decision="wait", reason="no canary sessions yet")
        return out
    empty = float(s["candidate_empty"].astype(bool).mean())
    out["candidate_empty_rate"] = empty
    if empty > cfg["zero_result_rate_max"]:
        out.update(decision="rollback", reason=f"the candidate returned nothing in {empty:.1%} of canary sessions")
        return out
    pref = IL.preference(s)
    out["overall"] = pref.to_dict(cfg["confidence"])
    per: dict[str, Any] = {}
    weak: list[str] = []
    losing: list[tuple[str, Any]] = []
    for m in markets:
        pm = IL.preference(s[s["market"] == m])
        per[m] = pm.to_dict(cfg["confidence"])
        if pm.decided >= cfg["market_min_decided"]:
            lo_m, _ = pm.interval(cfg["confidence"])
            _, hi_m = pm.interval(cfg["rollback_confidence"])
            if hi_m < 0.5:
                losing.append((m, pm))
            elif lo_m < cfg["market_non_inferiority_win_share"]:
                weak.append(m)
    out["markets"] = per
    where = ", ".join(f"{m} {pm.win_share:.0%} of {pm.decided}" for m, pm in losing)
    _, hi_strict = pref.interval(cfg["rollback_confidence"])
    if pref.decided >= 30 and hi_strict < 0.5:
        out.update(
            decision="rollback",
            reason=f"members preferred production: candidate won {pref.win_share:.0%} of {pref.decided} decided sessions"
            + (f" (clearest in {where})" if losing else ""),
        )
        return out
    if losing:
        out.update(decision="rollback", reason=f"members in {where.split(' ')[0]} preferred production ({where})")
        return out
    lo, _ = pref.interval(cfg["confidence"])
    if pref.decided >= cfg["min_decided_sessions"] and lo >= cfg["non_inferiority_win_share"] and not weak:
        out.update(
            decision="promote",
            reason=f"candidate won {pref.win_share:.0%} of {pref.decided} decided sessions (lower bound {lo:.2f})",
        )
        return out
    why = (
        f"{pref.decided} of {cfg['min_decided_sessions']} decided sessions"
        if pref.decided < cfg["min_decided_sessions"]
        else f"lower bound {lo:.2f}" + (f", weak in {', '.join(weak)}" if weak else "")
    )
    out.update(decision="wait", reason=why)
    return out


def promote(reg: Registry, surface: str, kind: str, version: str, reason: str) -> None:
    route = reg.routing(surface)
    assert route
    old = route["production"]
    reg.set_routing(surface, arm_name(kind, version), None, 0.0, f"promote {version}: {reason}", previous=old)
    reg.transition(version, "production", "canary-check", reason)
    for v in reg.versions(kind):
        if v["status"] == "production" and v["version"] != version:
            reg.transition(v["version"], "archived", "canary-check", f"replaced by {version}")


def rollback_canary(reg: Registry, surface: str, version: str, reason: str) -> None:
    route = reg.routing(surface)
    assert route
    reg.set_routing(surface, route["production"], None, 0.0, f"roll back canary {version}: {reason}")
    reg.transition(version, "rolled_back", "canary-check", reason)


def check(lake: Lake) -> list[dict[str, Any]]:
    reg = Registry()
    now = clock.now()
    out = []
    for dep in reg.open_deployments("canary"):
        v = reg.version(dep["version"])
        assert v
        surface = surface_of(v["kind"])
        cfg = settings(surface)
        route = reg.routing(surface)
        if not route or route.get("canary") != arm_name(v["kind"], v["version"]):
            continue
        since = int(route["canary_since"] or now)
        sessions = lake.read_since("sessions", since, now)
        sessions = sessions[sessions["surface"] == surface] if len(sessions) else sessions
        if len(sessions) == 0:
            res: dict[str, Any] = {"decision": "wait", "reason": "no sessions yet", "canary_sessions": 0}
        else:
            res = evaluate(sessions, route["production"], route["canary"], surface)
        hours = (now - since) / 3600
        if res["decision"] == "wait" and hours > cfg["max_hours"]:
            res.update(
                decision="rollback",
                reason=f"not shown to be at least as good after {hours:.0f} h ({res.get('reason')})",
            )
        res["hours"] = round(hours, 1)
        if res["decision"] != "wait":
            gate = GateResult("canary.guard", res["decision"] == "promote", res["reason"], res)
            reg.record_gates(v["version"], None, "canary", [gate])
            if res["decision"] == "promote":
                promote(reg, surface, v["kind"], v["version"], res["reason"])
                reg.step(dep["id"], "promote", True, res["reason"], res)
                reg.set_deployment(dep["id"], "promoted")
            else:
                rollback_canary(reg, surface, v["version"], res["reason"])
                reg.step(dep["id"], "rollback", False, res["reason"], res)
                reg.set_deployment(dep["id"], "rolled_back", error=res["reason"])
        out.append({"version": v["version"], "surface": surface, **res})
    return out


def rollback_production(reg: Registry, surface: str, reason: str) -> dict[str, Any]:
    """The on-call's button: production goes back to the previous model, still loaded, in one routing change."""
    route = reg.routing(surface)
    if not route or not route.get("previous"):
        raise RuntimeError(f"{surface} has no previous model to roll back to")
    bad, good = route["production"], route["previous"]
    reg.set_routing(surface, good, None, 0.0, f"rollback: {reason}", previous=bad)
    for name, status in ((bad, "rolled_back"), (good, "production")):
        version = name.removeprefix("ranker_")
        if reg.version(version):
            reg.transition(version, status, "oncall", reason)
    return {"surface": surface, "production": good, "rolled_back": bad}

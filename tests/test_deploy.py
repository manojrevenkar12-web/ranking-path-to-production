"""Deploying a ranker to the emulated staging and production Vespa: serving gates, canary, promotion, rollback, and
the failures that must leave production untouched."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ptp import artifacts, clock
from ptp.deploy import canary, ranker
from ptp.lake import Lake
from ptp.pipelines import steps
from ptp.serving.vespa import package as P
from ptpsim.emulators.vespa import Runtime

from .conftest import day_ts
from .helpers import adopt_adhoc, load_docs

pytestmark = pytest.mark.db


@pytest.fixture
def live(registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    load_docs(small_lake, envs.vespa_staging, envs.vespa_prod)
    adopt_adhoc(workdir, day_ts(small_lake, 7, 2))
    monkeypatch.setenv("PTP_NOW", str(day_ts(small_lake, 9)))
    return envs


def _candidate(
    reg: Any, lake: Path, small_lake: Path, run_id: str = "ct-1", overrides: dict | None = None, force: bool = True
) -> str:
    out = steps.run_direct(run_id, day_ts(small_lake, 9), str(lake), overrides or {})
    version = out["version"]
    if force and reg.version(version)["status"] == "rejected":
        # these tests are about deployment: let a rejected model through as if it had been waived
        reg.transition(version, "candidate", "test", "deployment test")
    return version


def test_a_candidate_goes_to_staging_passes_serving_gates_and_starts_a_canary_in_production(
    registry: Any, live: Any, lake: Path, small_lake: Path
) -> None:
    gen_before = live.vespa_prod.generation
    v = _candidate(registry, lake, small_lake)
    out = ranker.run(v, "dispatch:ct-1")
    assert out["state"] == "canary", registry.deployments(v)
    route = registry.routing("search")
    assert route["production"] == "ranker_r0001" and route["canary"] == f"ranker_{v}" and route["share"] > 0
    assert live.vespa_prod.generation == gen_before + 1
    prod = P.PackageSpec.from_files(P.from_zip(bytes(registry.active_package("prod")["zip"])))
    assert [r.name for r in prod.rankers] == ["ranker_r0001", f"ranker_{v}"]
    gates = {g["gate"]: g for g in registry.gates(v) if g["stage"] == "serving"}
    assert gates["serving.score_parity"]["passed"] and gates["serving.score_parity"]["evidence"]["hits"] > 100
    assert gates["serving.skew"]["passed"] and gates["serving.latency"]["passed"]
    steps_done = [s["step"] for s in registry.deployments(v)[0]["steps"]]
    assert steps_done == ["preflight", "deploy_staging", "serving_gates", "deploy_prod", "canary"]
    assert registry.version(v)["status"] == "canary"


def test_a_repeated_dispatch_does_not_start_a_second_deployment(
    registry: Any, live: Any, lake: Path, small_lake: Path
) -> None:
    v = _candidate(registry, lake, small_lake)
    ranker.run(v, "dispatch:ct-1")
    again = ranker.run(v, "dispatch:ct-1")
    assert again["note"] == "already in progress" and len(registry.deployments(v)) == 1


def test_a_second_candidate_waits_while_a_canary_runs(registry: Any, live: Any, lake: Path, small_lake: Path) -> None:
    a = _candidate(registry, lake, small_lake, "ct-a")
    b = _candidate(registry, lake, small_lake, "ct-b")
    ranker.run(a, "x")
    out = ranker.run(b, "y")
    assert out["state"] == "deferred" and registry.routing("search")["canary"] == f"ranker_{a}"


def test_a_package_the_runtime_refuses_fails_in_staging_and_never_reaches_production(
    registry: Any, live: Any, lake: Path, small_lake: Path
) -> None:
    live.vespa_staging.runtime = Runtime(max_ir_version=10)
    live.vespa_prod.runtime = Runtime(max_ir_version=10)
    gen_before = live.vespa_prod.generation
    v = _candidate(registry, lake, small_lake, overrides={"export": {"ir_version": 11}})
    out = ranker.run(v, "x")
    assert out["state"] == "failed" and "IR version" in out["error"]
    assert live.vespa_prod.generation == gen_before
    assert registry.routing("search")["canary"] is None and registry.version(v)["status"] == "failed"


def test_an_artifact_whose_bytes_changed_is_refused(registry: Any, live: Any, lake: Path, small_lake: Path) -> None:
    v = _candidate(registry, lake, small_lake)
    store = artifacts.Store()
    path = store.path(registry.version(v)["artifacts"]["onnx"]["key"])
    path.write_bytes(path.read_bytes() + b"\x00")
    out = ranker.run(v, "x")
    assert out["state"] == "failed" and "sha256" in out["error"]


def test_a_backfilled_feature_that_serving_computes_differently_is_caught_by_skew(
    registry: Any, live: Any, lake: Path, small_lake: Path
) -> None:
    feats = [*registry.version("r0001")["features"], "favourites_today"]
    v = _candidate(
        registry,
        lake,
        small_lake,
        overrides={"features": feats, "backfill": {"favourites_today": {"snapshot": "previous_day"}}},
    )
    # training saw yesterday's end-of-day counts; at 03:00 the live counter has barely started
    store = artifacts.Store()
    art = registry.version(v)["artifacts"]["train_sample"]
    sample = pd.read_parquet(io.BytesIO(store.get(art["key"], art["sha256"])))
    assert sample["f_favourites_today"].mean() > 0
    for d in live.vespa_staging.docs.values():
        d["favourites_today"] = 0
    live.vespa_staging._cols.clear()
    out = ranker.run(v, "x")
    assert out["state"] == "failed" and "serving.skew" in out["error"]


def _sessions(
    lake: Lake, start: int, canary: str, win: float, n: int, market_bias: dict[str, float] | None = None
) -> None:
    rng = np.random.default_rng(0)
    rows = []
    markets = ["FR", "DE", "PL", "LT"]
    for i in range(n):
        m = markets[i % 4]
        p = (market_bias or {}).get(m, win)
        decided = rng.random() < 0.6
        c, pr = (1.0, 0.0) if rng.random() < p else (0.0, 1.0)
        rows.append({"session_id": f"s{i}", "ts": start + 60 + i * 30, "user_id": "u", "market": m, "surface": "search", "profile": "interleave", "arm": "interleave",
                     "hits": 20, "purchases": 0, "canary": canary, "credit_candidate": c if decided else 0.0, "credit_production": pr if decided else 0.0, "candidate_empty": False})  # fmt: skip
    frame = pd.DataFrame(rows)
    for hour, g in frame.groupby((frame["ts"] - start) // 3600):
        lake.write_hour("sessions", start + int(hour) * 3600, g)


def test_the_canary_check_promotes_when_members_do_not_prefer_production(
    registry: Any, live: Any, lake: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = _candidate(registry, lake, small_lake)
    ranker.run(v, "x")
    since = registry.routing("search")["canary_since"]
    _sessions(Lake(lake), since, f"ranker_{v}", 0.55, 1200)
    monkeypatch.setenv("PTP_NOW", str(since + 12 * 3600))
    res = canary.check(Lake(lake))
    assert res[0]["decision"] == "promote", res
    route = registry.routing("search")
    assert route["production"] == f"ranker_{v}" and route["previous"] == "ranker_r0001" and route["canary"] is None
    assert registry.version(v)["status"] == "production" and registry.version("r0001")["status"] == "archived"
    # rollback is a routing change to the previous profile, still in the package
    out = canary.rollback_production(registry, "search", "on-call: conversion dropped")
    assert out["production"] == "ranker_r0001" and registry.version("r0001")["status"] == "production"


def test_the_canary_check_rolls_back_when_one_market_prefers_production(
    registry: Any, live: Any, lake: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = _candidate(registry, lake, small_lake)
    ranker.run(v, "x")
    since = registry.routing("search")["canary_since"]
    _sessions(Lake(lake), since, f"ranker_{v}", 0.55, 1600, {"LT": 0.15})
    monkeypatch.setenv("PTP_NOW", str(since + 12 * 3600))
    res = canary.check(Lake(lake))
    assert res[0]["decision"] == "rollback" and "LT" in res[0]["reason"]
    assert registry.routing("search")["production"] == "ranker_r0001" and registry.version(v)["status"] == "rolled_back"


def test_a_canary_that_cannot_show_it_is_not_worse_is_rolled_back_at_its_deadline(
    registry: Any, live: Any, lake: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = _candidate(registry, lake, small_lake)
    ranker.run(v, "x")
    since = registry.routing("search")["canary_since"]
    _sessions(Lake(lake), since, f"ranker_{v}", 0.5, 120)
    monkeypatch.setenv("PTP_NOW", str(since + 6 * 3600))
    assert canary.check(Lake(lake))[0]["decision"] == "wait"
    monkeypatch.setenv("PTP_NOW", str(since + 49 * 3600))
    assert canary.check(Lake(lake))[0]["decision"] == "rollback"


def test_waivers_let_a_rejected_version_through_preflight_until_they_expire(
    registry: Any, live: Any, lake: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ptp.gates import base

    v = _candidate(registry, lake, small_lake, overrides={"export": {"ir_version": 11}}, force=False)
    assert registry.version(v)["status"] == "rejected"
    blocked = sorted({g["gate"] for g in registry.gates(v) if g["blocking"]})
    today = dt.datetime.fromtimestamp(clock.now(), dt.UTC).date()
    waivers = [base.Waiver(v, g, "search-relevance", "test", today) for g in blocked]
    monkeypatch.setattr(base, "waivers", lambda: waivers)
    assert ranker.preflight(registry, v, today) == []
    assert registry.version(v)["status"] == "candidate"
    assert ranker.preflight(registry, "r9999", today) == ["unknown version r9999"]
    waivers[:] = [base.Waiver(v, g, "x", "y", today - dt.timedelta(days=1)) for g in blocked]
    registry.transition(v, "rejected", "test", "again")
    assert ranker.preflight(registry, v, today)

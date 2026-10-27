"""The continuous-training pipeline on nine days of simulated logs: what it registers, and what each gate stops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import pytest

from ptp import artifacts
from ptp.pipelines import steps

from .conftest import day_ts
from .helpers import adopt_adhoc

pytestmark = pytest.mark.db


def _gates(reg: Any, run_id: str) -> dict[str, dict[str, Any]]:
    return {g["gate"]: g for g in reg.gates(run_id=run_id)}


@pytest.fixture
def adopted(
    registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    adopt_adhoc(workdir, day_ts(small_lake, 7, 2))
    monkeypatch.setenv("PTP_NOW", str(day_ts(small_lake, 9)))
    return registry


def test_a_run_registers_a_versioned_model_with_lineage_gates_and_verified_artifacts(
    adopted: Any, lake: Path, small_lake: Path
) -> None:
    out = steps.run_direct("ct-1", day_ts(small_lake, 9), str(lake), {})
    assert out["state"] == "SUCCEEDED" and out["version"] == "r0002"
    v = adopted.version("r0002")
    assert v["pipeline_run"] == "ct-1" and v["code_commit"] == "test" and len(v["data_snapshot"]) == 16
    assert v["model_card"]["provenance"] == {f: "logged" for f in v["features"]}
    assert v["model_card"]["champion_at_training"] == "r0001"
    store = artifacts.Store()
    for name in ("onnx", "native", "train_sample", "model_card"):
        art = v["artifacts"][name]
        store.get(art["key"], art["sha256"])  # raises if the bytes changed
    gates = _gates(adopted, "ct-1")
    for g in ("data.freshness", "data.volume", "data.null_rate", "data.range", "data.provenance", "data.propensity",
              "model.non_inferiority", "model.segment_non_inferiority", "export.onnx_parity", "export.target_compat"):  # fmt: skip
        assert g in gates, g
    assert gates["data.propensity"]["passed"] and 0.5 < gates["data.propensity"]["evidence"]["eta"] < 1.0
    assert gates["model.non_inferiority"]["evidence"]["n"] > 500
    # the candidate's holdout leaves out days the production model trained on
    assert v["metrics"]["holdout_days"] and min(v["metrics"]["holdout_days"]) > "2026-01-07"
    assert v["status"] == ("rejected" if any(g["blocking"] for g in gates.values()) else "candidate")


def test_the_same_inputs_give_the_same_snapshot_id(adopted: Any, lake: Path, small_lake: Path) -> None:
    steps.run_direct("ct-a", day_ts(small_lake, 9), str(lake), {})
    steps.run_direct("ct-b", day_ts(small_lake, 9), str(lake), {})
    assert adopted.version("r0002")["data_snapshot"] == adopted.version("r0003")["data_snapshot"]


def test_a_late_partition_stops_the_run_at_freshness(adopted: Any, lake: Path, small_lake: Path) -> None:
    newest = sorted((lake / "impressions").iterdir())[-1] / "_SUCCESS"
    m = json.loads(newest.read_text())
    m["landed_at"] = day_ts(small_lake, 9, 12)  # lands at noon, after the 03:00 run
    newest.write_text(json.dumps(m))
    out = steps.run_direct("ct-late", day_ts(small_lake, 9), str(lake), {})
    assert out["outcome"] == "blocked" and "version" not in out
    g = _gates(adopted, "ct-late")["data.freshness"]
    assert not g["passed"] and g["evidence"]["age_hours"] > 24


def test_a_feature_backfilled_from_the_same_days_snapshot_is_leakage(
    adopted: Any, lake: Path, small_lake: Path
) -> None:
    feats = [*adopted.version("r0001")["features"], "favourites_today"]
    out = steps.run_direct(
        "ct-leak",
        day_ts(small_lake, 9),
        str(lake),
        {"features": feats, "backfill": {"favourites_today": {"snapshot": "same_day"}}},
    )
    assert out["outcome"] == "blocked"
    g = _gates(adopted, "ct-leak")["data.provenance"]
    assert (
        not g["passed"] and g["evidence"]["problems"]["favourites_today"]["problem"] == "observed after the impression"
    )


def test_a_previous_day_backfill_passes_provenance(adopted: Any, lake: Path, small_lake: Path) -> None:
    feats = [*adopted.version("r0001")["features"], "favourites_today"]
    steps.run_direct(
        "ct-prev",
        day_ts(small_lake, 9),
        str(lake),
        {"features": feats, "backfill": {"favourites_today": {"snapshot": "previous_day"}}},
    )
    g = _gates(adopted, "ct-prev")["data.provenance"]
    assert g["passed"] and g["evidence"]["provenance"]["favourites_today"] == "backfill:previous_day"


def test_an_export_the_serving_runtime_cannot_load_is_rejected_in_the_pipeline(
    adopted: Any, lake: Path, small_lake: Path
) -> None:
    out = steps.run_direct("ct-ir", day_ts(small_lake, 9), str(lake), {"export": {"ir_version": 11}})
    assert out["version"] == "r0002" and "export.target_compat" in out["blocked"]
    assert adopted.version("r0002")["status"] == "rejected"
    data = artifacts.Store().get(adopted.version("r0002")["artifacts"]["onnx"]["key"])
    assert onnx.load_from_string(data).ir_version == 11


def test_quarantined_rows_are_left_out(adopted: Any, lake: Path, small_lake: Path) -> None:
    import datetime as dt

    day = dt.date(2026, 1, 8)
    adopted.quarantine("impressions", day, "PL", "feed incident", "test")
    steps.run_direct("ct-q", day_ts(small_lake, 9), str(lake), {})
    meta = artifacts.Store().get_json("runs/ct-q/data.json")
    assert meta["quarantined"] > 0


def test_a_broken_logging_market_is_caught_by_null_rate_and_range(adopted: Any, lake: Path, small_lake: Path) -> None:
    import pandas as pd

    part = sorted((lake / "impressions").iterdir())[-1] / "part-0.parquet"
    f = pd.read_parquet(part)
    f.loc[f["market"] == "DE", "f_seller_rating"] = float("nan")
    f.loc[f["market"] == "FR", "f_price_rel"] = f.loc[f["market"] == "FR", "f_price_rel"] * 100
    f.to_parquet(part, index=False)
    out = steps.run_direct("ct-bad", day_ts(small_lake, 9), str(lake), {})
    assert out["outcome"] == "blocked"
    g = _gates(adopted, "ct-bad")
    assert "seller_rating@DE" in g["data.null_rate"]["evidence"]["increases"]
    assert "price_rel@FR" in g["data.range"]["evidence"]["violations"]

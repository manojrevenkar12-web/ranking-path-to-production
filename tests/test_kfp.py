"""The KFP pipeline definition itself, run by the KFP local runner (each component in its own process), including the
exit handler that publishes the outcome to Pub/Sub."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from ptp import config
from ptp.pipelines import runner
from ptpsim.emulators.pubsub import PubSubEmulator
from ptpsim.servers import serve

from .conftest import day_ts
from .helpers import adopt_adhoc

pytestmark = [pytest.mark.db, pytest.mark.kfp]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_the_ranker_pipeline_runs_on_the_kfp_local_runner_and_publishes_its_outcome(
    registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adopt_adhoc(workdir, day_ts(small_lake, 7, 2))
    auth = config.load("relay")["auth"]
    ps = PubSubEmulator(push_url="", audience=auth["audience"], service_account=auth["service_account"])
    port = _free_port()
    server = serve(ps.handle, port, host="127.0.0.1")
    monkeypatch.setenv("PTP_PUBSUB_API", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("PTP_NOW", str(day_ts(small_lake, 9)))
    try:
        out = runner.run("ranker", "kfp", day_ts(small_lake, 9), "ct-kfp-1", {})
    finally:
        server.shutdown()
    assert out["state"] == "SUCCEEDED" and out["version"] == "r0002", out
    assert registry.version("r0002")["pipeline_run"] == "ct-kfp-1"
    stages = {g["stage"] for g in registry.gates(run_id="ct-kfp-1")}
    assert stages == {"data", "model", "export"}
    assert len(ps.published) == 1
    event = PubSubEmulator.decode(ps.published[0])
    assert (
        event["run_id"] == "ct-kfp-1" and event["version"] == "r0002" and event["outcome"] in ("candidate", "blocked")
    )


def test_a_run_blocked_at_the_data_gates_still_publishes_from_the_exit_handler(
    registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    adopt_adhoc(workdir, day_ts(small_lake, 7, 2))
    newest = sorted((lake / "impressions").iterdir())[-1] / "_SUCCESS"
    m = json.loads(newest.read_text())
    m["landed_at"] = day_ts(small_lake, 9, 12)
    newest.write_text(json.dumps(m))
    auth = config.load("relay")["auth"]
    ps = PubSubEmulator(push_url="", audience=auth["audience"], service_account=auth["service_account"])
    port = _free_port()
    server = serve(ps.handle, port, host="127.0.0.1")
    monkeypatch.setenv("PTP_PUBSUB_API", f"http://127.0.0.1:{port}")
    try:
        out = runner.run("ranker", "kfp", day_ts(small_lake, 9), "ct-kfp-2", {})
    finally:
        server.shutdown()
    assert out["outcome"] == "blocked" and out["version"] is None
    assert PubSubEmulator.decode(ps.published[0])["outcome"] == "blocked"

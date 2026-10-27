"""A tiny season through the whole loop: adoption, pipeline runs, Pub/Sub with duplicate deliveries, the relay, the
repository's workflow files, deploys, canaries, and the evaluation against the truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ptp import evaluate
from ptpsim.season import Scenario, Season

pytestmark = [pytest.mark.db, pytest.mark.season]


def test_a_tiny_season_runs_the_loop_and_is_judged(
    registry: Any, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc = Scenario(
        name="tiny",
        days=11,
        world={
            "seed": 4,
            "users": 1200,
            "initial_items": 1600,
            "new_items_per_day": 180,
            "search_sessions_per_day": 900,
            "homepage_sessions_per_day": 250,
        },
        adopt_day=7,
        ranker_days=[8, 9],
        tower_days=[],
        delivery={"duplicate_every": 2},
        changes=[{"day": 9, "kind": "ranker", "once": True, "overrides": {"export": {"ir_version": 11}}}],
    )
    out = workdir / "season"
    Season(sc, out).run()
    ev = evaluate.run(out)
    s = ev["summary"]
    assert s["releases"] == 2
    broken = [r for r in ev["releases"] if r["truth"] == "broken"]
    assert broken and broken[0]["platform"] == "blocked_pipeline"
    assert s["harmful_promoted"] == 0
    reg = json.loads((out / "registry.json").read_text())
    # every message delivered twice reached the relay, and still one dispatch per candidate run
    assert ev["delivery"]["push_attempts"] > ev["delivery"]["messages"]
    dispatched = [m for m in reg["relay"] if m["action"] == "dispatched"]
    assert len({m["pipeline_run"] for m in dispatched}) == len(dispatched)
    assert any(w["workflow"] == "deploy-model.yml" for w in reg["workflow_runs"])
    assert (out / "truth.json").exists() and (out / "vespa_docs.jsonl").exists()

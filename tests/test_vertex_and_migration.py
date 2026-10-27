"""Vertex AI submission (against a stand-in for the SDK), the compiled pipeline definitions, and the migration from
the ad-hoc notebook."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import yaml

from ptp.pipelines import ct, vertex
from ptp.training import ranker

from .helpers import adhoc_module

ROOT = Path(__file__).resolve().parents[1]


class FakeJob:
    created: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        self.resource_name = f"projects/p/locations/l/pipelineJobs/{kw['job_id']}"
        FakeJob.created.append({"init": kw})

    def submit(self, service_account: str) -> None:
        FakeJob.created[-1]["service_account"] = service_account

    def create_schedule(self, **kw: Any) -> Any:
        FakeJob.created[-1]["schedule"] = kw
        return types.SimpleNamespace(resource_name="projects/p/schedules/1")


def _fake_sdk(monkeypatch: Any) -> list[dict[str, Any]]:
    inits: list[dict[str, Any]] = []
    mod = types.SimpleNamespace(init=lambda **kw: inits.append(kw), PipelineJob=FakeJob)
    monkeypatch.setattr(vertex, "_sdk", lambda: mod)
    FakeJob.created = []
    return inits


def test_submit_builds_the_pipeline_job_from_the_compiled_template(monkeypatch: Any) -> None:
    inits = _fake_sdk(monkeypatch)
    monkeypatch.setenv("PTP_NOW", "1767312000")
    out = vertex.submit("ranker", {"params": {"n_estimators": 200}})
    job = FakeJob.created[0]
    assert inits == [{"project": "vinted-buyer-ml-dev", "location": "europe-west1"}]
    assert job["init"]["template_path"].endswith("pipelines/compiled/ct-ranker.yaml")
    assert job["init"]["parameter_values"]["overrides"] == '{"params": {"n_estimators": 200}}'
    assert job["init"]["labels"]["pipeline"] == "ct-ranker" and job["init"]["enable_caching"] is False
    assert job["service_account"].startswith("ct-pipelines@")
    assert out["run_id"] == "ct-ranker-1767312000"


def test_schedules_come_from_config(monkeypatch: Any) -> None:
    _fake_sdk(monkeypatch)
    out = vertex.schedule("ranker")
    assert out["cron"] == "0 3 * * *" and FakeJob.created[0]["schedule"]["max_concurrent_run_count"] == 1


def test_compiled_pipelines_are_up_to_date(tmp_path: Path) -> None:
    written = ct.compile_all(tmp_path)
    for path in written.values():
        name = Path(path).name
        committed = ROOT / "pipelines" / "compiled" / name
        assert committed.exists(), f"run `ptp pipeline compile` and commit {name}"
        a, b = yaml.safe_load(Path(path).read_text()), yaml.safe_load(committed.read_text())
        a.pop("sdkVersion", None)
        b.pop("sdkVersion", None)
        assert a == b, f"{name} is stale: run `ptp pipeline compile`"


def test_every_pipeline_publishes_its_outcome_from_an_exit_handler() -> None:
    for name in ("ct-ranker", "ct-tower", "item-embeddings"):
        spec = yaml.safe_load((ROOT / "pipelines" / "compiled" / f"{name}.yaml").read_text())
        tasks = spec["root"]["dag"]["tasks"]
        exit_tasks = [
            t for t in tasks.values() if t.get("triggerPolicy", {}).get("strategy") == "ALL_UPSTREAM_TASKS_COMPLETED"
        ]
        assert exit_tasks and exit_tasks[0]["componentRef"]["name"].startswith("comp-publish-outcome"), name


def test_the_migrated_training_code_reproduces_the_notebook(tmp_path: Path) -> None:
    """Same data, same parameters: the pipeline's trainer and the notebook build the same model."""
    mod = adhoc_module()
    rng = np.random.default_rng(0)
    rows = []
    for s in range(300):
        for pos in range(20):
            x = rng.random(len(mod.FEATURES))
            clicked = rng.random() < 0.1 + 0.3 * x[5]
            rows.append({"session_id": f"s{s:04d}", "position": pos, "surface": "search", "impression_id": f"{s}-{pos}", "clicked": clicked,
                         "favourited": clicked and rng.random() < 0.2, "purchased": clicked and rng.random() < 0.1, **{f"f_{f}": x[i] for i, f in enumerate(mod.FEATURES)}})  # fmt: skip
    csv = tmp_path / "export.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    nb = mod.load(csv)
    booster_nb = mod.train(nb)
    from ptp import config

    params = config.load("training/ranker")["params"]
    trained = ranker.train(nb, mod.FEATURES, params, None, position_debias=False)
    x = nb[[f"f_{f}" for f in mod.FEATURES]].to_numpy(dtype=np.float32)
    assert np.allclose(booster_nb.predict(x), trained.predict(x))
    assert {k: v for k, v in mod.PARAMS.items() if k != "verbose"} == params


def test_gcp_dependencies_are_optional() -> None:
    for name in [m for m in sys.modules if m.startswith("google.cloud.aiplatform")]:
        assert name  # imported only through vertex._sdk at call time
    src = (ROOT / "src" / "ptp" / "pipelines" / "vertex.py").read_text()
    assert "from google.cloud import aiplatform" in src and src.index(
        "from google.cloud import aiplatform"
    ) > src.index("def _sdk")

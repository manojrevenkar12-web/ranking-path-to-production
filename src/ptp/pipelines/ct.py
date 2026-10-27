"""The continuous-training pipelines as Kubeflow Pipelines (KFP v2) definitions, for Vertex AI Pipelines.

Components are thin: each calls a step in steps.py / tower_steps.py, which read and write the run workspace in the
artifact bucket and the registry. Every pipeline is wrapped in an ExitHandler whose task publishes the run's final
state (success or failure) to the events topic; the relay turns a candidate into a deploy.

`compile_all()` writes pipelines/compiled/*.yaml (CI checks they are up to date); `run_local()` runs a definition
with the KFP local runner, which is how tests prove the definition itself works without a GCP project.
"""

# no `from __future__ import annotations` here: KFP reads the component signatures at import time

import json
from pathlib import Path
from typing import Any

from kfp import compiler, dsl

from .. import config

IMAGE = config.load("pipeline")["vertex"]["image"]


@dsl.component(base_image=IMAGE)
def start_run(run_id: str, kind: str, as_of: int, overrides: str) -> str:
    import json

    from ptp.registry.store import Registry

    Registry().start_run(run_id, kind, "kfp", {"as_of": as_of, "overrides": json.loads(overrides)})
    return run_id


@dsl.component(base_image=IMAGE)
def validate_data(run_id: str, as_of: int, lake_root: str, overrides: str) -> str:
    import json

    from ptp.pipelines import steps
    from ptp.registry.store import Registry

    out = steps.validate_data(run_id, as_of, lake_root, json.loads(overrides))
    if not out["passed"]:
        Registry().finish_run(run_id, "SUCCEEDED", "blocked")
    return "pass" if out["passed"] else "block"


@dsl.component(base_image=IMAGE)
def train_ranker(run_id: str) -> str:
    import json

    from ptp.pipelines import steps

    return json.dumps(steps.train(run_id))


@dsl.component(base_image=IMAGE)
def evaluate_ranker(run_id: str, metrics: dsl.Output[dsl.Metrics]) -> str:
    import json

    from ptp.pipelines import steps

    out = steps.evaluate(run_id)
    metrics.log_metric("ips_dcg_delta", float(out["delta"]))
    return json.dumps(out)


@dsl.component(base_image=IMAGE)
def export_ranker(run_id: str) -> str:
    import json

    from ptp.pipelines import steps

    return json.dumps(steps.export(run_id))


@dsl.component(base_image=IMAGE)
def register_ranker(run_id: str, model: dsl.Output[dsl.Model]) -> str:
    import json

    from ptp.pipelines import steps
    from ptp.registry.store import Registry

    out = steps.register(run_id)
    outcome = "candidate" if out["status"] == "candidate" else "blocked"
    Registry().finish_run(run_id, "SUCCEEDED", outcome, out["version"])
    model.metadata.update({"version": out["version"], "status": out["status"]})
    return json.dumps(out)


@dsl.component(base_image=IMAGE)
def publish_outcome(run_id: str, kind: str, pipeline: str, status: dsl.PipelineTaskFinalStatus) -> str:
    import json

    from ptp.pipelines import runner
    from ptp.registry.store import Registry

    failed = status.state not in ("SUCCEEDED", "SUCCESS")
    if failed:
        reg = Registry()
        row = reg.run(run_id) or {}
        if row.get("state") == "RUNNING":
            reg.finish_run(run_id, "FAILED", "error", error=str(status.error_message or status.state))
    return json.dumps(runner.finish_and_publish(run_id, kind, pipeline, failed, publish=True), default=str)


@dsl.pipeline(
    name="ct-ranker", description="Continuous training of the search ranker, with data, model and export gates."
)
def ct_ranker(run_id: str, as_of: int, lake_root: str, overrides: str = "{}") -> None:
    done = publish_outcome(run_id=run_id, kind="ranker", pipeline="ct-ranker")
    with dsl.ExitHandler(done):
        s = start_run(run_id=run_id, kind="ranker", as_of=as_of, overrides=overrides)
        v = validate_data(run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=overrides).after(s)
        with dsl.If(v.output == "pass", name="data-gates-passed"):
            t = train_ranker(run_id=run_id).after(v)
            e = evaluate_ranker(run_id=run_id).after(t)
            x = export_ranker(run_id=run_id).after(e)
            register_ranker(run_id=run_id).after(x)


@dsl.component(base_image=IMAGE)
def tower_stage(run_id: str, as_of: int, lake_root: str, overrides: str, stage: str) -> str:
    import json

    from ptp.pipelines import tower_steps

    out = tower_steps.STAGES[stage](run_id, as_of, lake_root, json.loads(overrides))
    return out if isinstance(out, str) else json.dumps(out, default=str)


@dsl.pipeline(
    name="ct-tower",
    description="Training of the homepage two-tower model: user tower for Triton, item tower for batch scoring.",
)
def ct_tower(run_id: str, as_of: int, lake_root: str, overrides: str = "{}") -> None:
    done = publish_outcome(run_id=run_id, kind="tower", pipeline="ct-tower")
    with dsl.ExitHandler(done):
        s = start_run(run_id=run_id, kind="tower", as_of=as_of, overrides=overrides)
        d = tower_stage(run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=overrides, stage="data").after(s)
        with dsl.If(d.output == "pass", name="data-gates-passed"):
            t = tower_stage(run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=overrides, stage="train").after(
                d
            )
            tower_stage(run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=overrides, stage="register").after(t)


@dsl.pipeline(name="item-embeddings", description="Batch scoring: item embeddings for a tower version, fed into Vespa.")
def item_embeddings(run_id: str, as_of: int, lake_root: str, overrides: str = "{}") -> None:
    done = publish_outcome(run_id=run_id, kind="tower", pipeline="item-embeddings")
    with dsl.ExitHandler(done):
        s = start_run(run_id=run_id, kind="tower", as_of=as_of, overrides=overrides)
        tower_stage(run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=overrides, stage="score").after(s)


DEFINITIONS = {"ranker": ct_ranker, "tower": ct_tower, "item-embeddings": item_embeddings}


def compiled_dir() -> Path:
    return config.root() / "pipelines" / "compiled"


def compile_all(out: Path | None = None) -> dict[str, str]:
    out = out or compiled_dir()
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for kind, fn in DEFINITIONS.items():
        path = out / f"{fn.name}.yaml"
        compiler.Compiler().compile(fn, str(path))
        written[kind] = str(path)
    return written


def run_local(kind: str, run_id: str, as_of: int, lake_root: str, overrides: dict[str, Any]) -> None:
    from kfp import local

    root = config.data_dir() / "kfp-local"
    root.mkdir(parents=True, exist_ok=True)
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=str(root), raise_on_error=False)
    DEFINITIONS[kind](run_id=run_id, as_of=as_of, lake_root=lake_root, overrides=json.dumps(overrides))

"""The homepage two-tower model: training, export, Triton, and the two-phase rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ptp.export.onnx_export import compat, run_onnx
from ptp.serving.triton.client import Triton, config_pbtxt, model_name
from ptp.training import tower as T
from ptp.training import tower_features as TF
from ptpsim.emulators.triton import TritonEmulator, TritonRuntime, parse_config

from .conftest import day_ts
from .helpers import load_docs

CFG = {
    "dim": 8,
    "hidden": 16,
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.02,
    "temperature": 0.1,
    "random_state": 0,
}


def _pairs(n: int = 2000, seed: int = 0, with_groups: bool = False) -> Any:
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 6, n)
    users = rng.normal(0, 0.3, (n, 10)).astype(np.float32)
    items = rng.normal(0, 0.3, (n, 12)).astype(np.float32)
    users[np.arange(n), groups] += 2.0
    items[np.arange(n), groups] += 2.0
    return (users, items, groups) if with_groups else (users, items)


def test_training_learns_the_pairing_and_the_onnx_export_matches() -> None:
    u, i, g = _pairs(with_groups=True)
    m = T.train(u, i, CFG)
    # six interest groups: in-batch negatives from the same group are indistinguishable, so the floor is log(64/6)
    assert m.history[-1] < m.history[0] and m.history[-1] < np.log(64 / 6) + 0.25
    zu, zi = m.user.embed(u[:200]), m.item.embed(i[:200])
    same_group = np.mean(g[:200][np.argmax(zu @ zi.T, axis=1)] == g[:200])
    assert same_group > 0.9  # the nearest item shares the member's interest; chance is 1/6
    data = T.to_onnx(m.user, "user", "embedding", 17)
    assert np.allclose(run_onnx(data, u[:50], "user"), zu[:50], atol=1e-5)
    assert np.allclose(np.linalg.norm(zu, axis=1), 1.0, atol=1e-5)
    assert compat(data, "triton", {"user": 10}, ["embedding"])[0].passed


def test_tower_features_have_fixed_widths() -> None:
    import pandas as pd

    v = TF.user_vector({"market": "PL", "cat_interest": [1.0] * 10, "band": 0.2, "brand_weights": '{"b01": 1.0}'})
    assert v.shape == (TF.user_dim(),) and v[2] == 1.0
    items = pd.DataFrame(
        [
            {
                "category": "jeans",
                "brand": "b01",
                "price": 10.0,
                "category_median_price": 20.0,
                "condition": 3,
                "photo_quality": 0.5,
            }
        ]
    )
    assert TF.item_matrix(items).shape == (1, TF.item_dim())


def _triton(tmp_path: Path, runtime: TritonRuntime | None = None) -> tuple[TritonEmulator, Triton]:
    import httpx

    emu = TritonEmulator(tmp_path, runtime)
    return emu, Triton(httpx.Client(transport=httpx.MockTransport(emu.handle)), "http://t", tmp_path, "user_tower")


def test_triton_loads_the_pinned_versions_and_serves_them_explicitly(tmp_path: Path) -> None:
    u, i = _pairs()
    a, b = T.train(u, i, CFG), T.train(u, i, {**CFG, "random_state": 1})
    _emu, t = _triton(tmp_path)
    name = model_name("user_tower", 8)
    t.write_version(name, 1, T.to_onnx(a.user, "user", "embedding", 17))
    t.write_version(name, 2, T.to_onnx(b.user, "user", "embedding", 17))
    t.write_config(name, 10, 8, [1, 2], 64)
    t.load(name)
    assert t.version_ready(name, 1) and t.version_ready(name, 2)
    assert np.allclose(t.infer(name, 2, u[:4]), b.user.embed(u[:4]), atol=1e-5)
    assert {(r["name"], r["version"]) for r in t.index()} == {(name, "1"), (name, "2")}
    assert parse_config(config_pbtxt(name, 10, 8, [2, 1], 64))["versions"] == [1, 2]


def test_triton_refuses_a_model_whose_shape_disagrees_with_its_config_and_keeps_serving(tmp_path: Path) -> None:
    from ptp.serving.triton.client import TritonError

    u, i = _pairs()
    m = T.train(u, i, CFG)
    _emu, t = _triton(tmp_path)
    name = model_name("user_tower", 8)
    t.write_version(name, 1, T.to_onnx(m.user, "user", "embedding", 17))
    t.write_config(name, 10, 8, [1], 64)
    t.load(name)
    t.write_config(name, 11, 8, [1], 64)  # a config that claims 11 inputs
    with pytest.raises(TritonError, match="dims"):
        t.load(name)
    assert t.version_ready(name, 1)  # the previously loaded version still serves
    _emu2, t2 = _triton(tmp_path / "old", TritonRuntime(max_ir_version=8))
    t2.write_version(name, 1, T.to_onnx(m.user, "user", "embedding", 17, ir_version=10))
    t2.write_config(name, 10, 8, [1], 64)
    with pytest.raises(TritonError, match="IR version"):
        t2.load(name)


@pytest.mark.db
def test_a_tower_rolls_out_in_two_phases_and_the_homepage_canary_starts_only_after_coverage(
    registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from ptp.deploy import tower as tower_deploy
    from ptp.pipelines import runner, tower_steps
    from ptp.serving.vespa.client import Vespa

    load_docs(small_lake, envs.vespa_staging, envs.vespa_prod)
    monkeypatch.setenv("PTP_NOW", str(day_ts(small_lake, 9, 4)))
    out = tower_steps.run_direct("ct-tower-1", day_ts(small_lake, 9, 4), str(lake), {"epochs": 3})
    v = out["version"]
    assert v == "t0001"
    if registry.version(v)["status"] == "rejected":
        registry.transition(v, "candidate", "test", "deployment test")
    published: list[Any] = []
    monkeypatch.setattr(runner.events, "publish", lambda ev: published.append(ev) or "m1")
    res = tower_deploy.run(v, "dispatch:ct-tower-1")
    assert res["state"] == "waiting_embeddings"
    assert registry.routing("home") is None  # nothing serves the new field yet
    assert published and published[0].outcome == "embeddings_ready" and published[0].version == v
    staging = Vespa(envs.vespa_staging.client(), "http://q", "http://c")
    assert staging.count([f'emb_versions contains "{v}"']) > 0
    res = tower_deploy.embeddings_ready(v, "dispatch:ie-1")
    assert res["state"] == "canary", registry.deployments(v)
    route = registry.routing("home")
    assert route["production"] == "popularity" and route["canary"] == v
    gates = {g["gate"]: g for g in registry.gates(v) if g["stage"] == "serving"}
    assert gates["serving.embedding_coverage"]["passed"] and gates["serving.score_parity"]["passed"]
    assert (
        envs.triton_prod.infer_local(model_name("user_tower", 32), 1, np.zeros((1, TF.user_dim()), np.float32))
        is not None
    )
    assert os.path.exists(workdir / "triton" / "prod" / model_name("user_tower", 32) / "config.pbtxt")


@pytest.mark.db
def test_a_tower_whose_embeddings_do_not_cover_the_catalogue_is_not_switched_on(
    registry: Any, envs: Any, lake: Path, workdir: Path, small_lake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ptp.deploy import tower as tower_deploy
    from ptp.pipelines import runner, tower_steps

    load_docs(small_lake, envs.vespa_staging, envs.vespa_prod)
    monkeypatch.setenv("PTP_NOW", str(day_ts(small_lake, 9, 4)))
    v = tower_steps.run_direct("ct-tower-2", day_ts(small_lake, 9, 4), str(lake), {"epochs": 3})["version"]
    if registry.version(v)["status"] == "rejected":
        registry.transition(v, "candidate", "test", "deployment test")
    monkeypatch.setattr(runner.events, "publish", lambda ev: "m1")
    tower_deploy.run(v, "x")
    # half of production's catalogue loses its embedding (a scoring job that died half way)
    for _k, d in list(envs.vespa_prod.docs.items())[::2]:
        d.pop(f"emb_{v}", None)
        d["emb_versions"] = []
    envs.vespa_prod._cols.clear()
    res = tower_deploy.embeddings_ready(v, "y")
    assert res["state"] == "failed" and "embedding_coverage" in res["error"]
    assert registry.routing("home") is None

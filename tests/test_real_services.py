"""Against the real services, in CI (.github/workflows/ci.yml): a Vespa container, a Triton container, and the relay
running in a kind cluster. These check that what the emulators accept, the real thing accepts too."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import lightgbm as lgb
import numpy as np
import pytest

from ptp import contract
from ptp.export.onnx_export import ranker_to_onnx, run_onnx
from ptp.serving.triton.client import Triton, TritonError, model_name
from ptp.serving.vespa import package as P
from ptp.serving.vespa.client import Vespa, VespaError, mapped_tensor
from ptp.training import tower as T

C = contract.load()


def _ranker(n: int) -> bytes:
    rng = np.random.default_rng(0)
    x = rng.random((800, n)).astype(np.float32)
    y = (x[:, -1] * 3 + x[:, 0]).astype(int).clip(0, 3)
    m = lgb.LGBMRanker(n_estimators=30, verbose=-1).fit(x, y, group=[20] * 40)
    return ranker_to_onnx(m.booster_, n, {})


@pytest.mark.vespa
def test_real_vespa_serves_the_package_and_scores_like_the_model_offline() -> None:
    v = Vespa(httpx.Client(timeout=120), os.environ["PTP_VESPA_URL"], os.environ["PTP_VESPA_CONFIG_URL"])
    v.wait_ready_config()
    feats = C.logged
    onx = _ranker(len(feats))
    spec = P.PackageSpec(rankers=[P.RankerProfile("r0001", onx, feats)], embeddings=[P.EmbeddingField("t0001", 4)])
    try:
        res = v.deploy(P.to_zip(P.build(C, spec)))
    except VespaError as e:
        print("::error title=real Vespa refused the package::" + json.dumps(e.body)[:3000])
        raise
    assert "activated" in res["message"], res
    v.wait_converged()
    v.wait_ready()
    for i in range(30):
        v.put(
            f"it{i:06d}",
            {
                "item_id": f"it{i:06d}", "market": "FR", "category": "jeans", "brand": f"b{i % 3}",
                "brand_t": {"cells": [{"address": {"brand": f"b{i % 3}"}, "value": 1.0}]},
                "price": 10.0 + i, "category_median_price": 20.0, "condition": i % 4, "photo_quality": 0.5,
                "seller_rating": 4.5, "listed_at": 1000 + i, "clicks_7d": i, "impressions_7d": 100, "favourites": i,
                "favourites_today": 0, "sold": False, "emb_versions": ["t0001"] if i % 2 == 0 else [],
                **({"emb_t0001": {"values": [1.0, 0.0, 0.0, 0.0] if i == 4 else [0.0, 1.0, 0.0, 0.0]}} if i % 2 == 0 else {}),
            },
        )  # fmt: skip
    deadline = time.monotonic() + 60
    while v.count(['market contains "FR"', "sold = false"]) < 30 and time.monotonic() < deadline:
        time.sleep(1)
    r = v.search(['market contains "FR"', 'category contains "jeans"', "sold = false"], "ranker_r0001", 10,
                 {"now": 86400.0 * 3, "user_brand": mapped_tensor("brand", {"b1": 0.8}), "user_log_price": 3.0})  # fmt: skip
    assert r.total == 30 and len(r.hits) == 10
    x = np.array([[h.features[f] for f in feats] for h in r.hits], dtype=np.float32)
    assert np.allclose(run_onnx(onx, x)[:, 0], [h.relevance for h in r.hits], atol=1e-4)
    assert all(min(abs(h.features["brand_affinity"]), abs(h.features["brand_affinity"] - 0.8)) < 1e-6 for h in r.hits)
    assert v.count(['emb_versions contains "t0001"']) == 15
    assert v.count(["sold = false", "listed_at < 1010"]) == 10
    nn = v.search(
        ['market contains "FR"'], "homepage_t0001", 3, {"user_emb": "[1,0,0,0]"}, nearest=("emb_t0001", "user_emb", 10)
    )
    assert nn.hits[0].item_id == "it000004"
    # a model whose input width disagrees with the profile is refused by the real config server too
    bad = P.build(C, P.PackageSpec(rankers=[P.RankerProfile("r0002", _ranker(5), feats)]))
    with pytest.raises(VespaError) as e:
        v.deploy(P.to_zip(bad))
    assert e.value.status == 400
    print(
        "::notice title=real Vespa::"
        + json.dumps(
            {
                "hits": len(r.hits),
                "max_abs": float(np.max(np.abs(run_onnx(onx, x)[:, 0] - [h.relevance for h in r.hits]))),
            }
        )
    )


@pytest.mark.triton
def test_real_triton_loads_pinned_versions_and_matches_the_export() -> None:
    repo = Path(os.environ["PTP_TRITON_REPOSITORY"])
    t = Triton(httpx.Client(timeout=120), os.environ["PTP_TRITON_URL"], repo, "user_tower")
    t.wait_ready(300)
    rng = np.random.default_rng(0)
    u, i = rng.normal(size=(600, 10)).astype(np.float32), rng.normal(size=(600, 12)).astype(np.float32)
    cfg = {"dim": 8, "hidden": 16, "epochs": 3, "batch_size": 64, "learning_rate": 0.02, "temperature": 0.1}
    a, b = T.train(u, i, cfg), T.train(u, i, {**cfg, "random_state": 1})
    name = model_name("user_tower", 8)
    ua, ub = T.to_onnx(a.user, "user", "embedding", 17), T.to_onnx(b.user, "user", "embedding", 17)
    t.write_version(name, 1, ua)
    t.write_version(name, 2, ub)
    t.write_config(name, 10, 8, [1, 2], 64)
    try:
        t.load(name)
    except TritonError as e:
        print(f"::error title=real Triton refused the model::{e}")
        raise
    assert t.version_ready(name, 1) and t.version_ready(name, 2)
    got = t.infer(name, 2, u[:16])
    assert np.allclose(got, run_onnx(ub, u[:16], "user"), atol=1e-5)
    t.write_config(name, 11, 8, [1, 2], 64)
    with pytest.raises(TritonError):
        t.load(name)
    assert t.version_ready(name, 2)  # a failed reload leaves the loaded versions serving
    print("::notice title=real Triton::" + json.dumps({"index": t.index()}))


@pytest.mark.kind
def test_the_relay_in_kubernetes_verifies_pushes_and_dispatches_once() -> None:
    """The stand-in (ptpsim serve github-stand-in) signs push tokens, serves the JWKS and records dispatches."""
    import base64

    from ptp.events import PipelineEvent

    relay, stand_in = os.environ["PTP_RELAY_URL"], os.environ["PTP_STAND_IN_URL"]
    token = httpx.get(f"{stand_in}/token").json()["token"]
    ev = PipelineEvent(
        run_id=f"kind-{int(time.time())}",
        pipeline="ct-ranker",
        kind="ranker",
        state="SUCCEEDED",
        outcome="candidate",
        version="r0042",
        artifact={"key": "k", "sha256": "s"},
    )
    env = {
        "message": {"data": base64.b64encode(ev.model_dump_json().encode()).decode(), "messageId": ev.run_id},
        "subscription": "s",
    }
    assert httpx.post(f"{relay}/pubsub/push", json=env, headers={"Authorization": "Bearer nope"}).status_code == 401
    for _ in range(2):
        r = httpx.post(f"{relay}/pubsub/push", json=env, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        assert r.status_code == 204, r.text
    dispatched = [d for d in httpx.get(f"{stand_in}/dispatches").json() if d["client_payload"]["run_id"] == ev.run_id]
    assert len(dispatched) == 1 and dispatched[0]["client_payload"]["version"] == "r0042"

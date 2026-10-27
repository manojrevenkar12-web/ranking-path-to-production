"""The Vespa application package, checked against the emulator (and in CI against a real Vespa: test_real_vespa.py)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import onnx
import pytest

from ptp import contract
from ptp.export.onnx_export import ranker_to_onnx, run_onnx
from ptp.serving.vespa import package as P
from ptp.serving.vespa.client import Vespa, mapped_tensor
from ptpsim.emulators import expr as E
from ptpsim.emulators.schema import parse
from ptpsim.emulators.vespa import Runtime, VespaEmulator

C = contract.load()
FEATS = C.logged


def _model(n_features: int, seed: int = 0, export: dict | None = None) -> tuple[lgb.Booster, bytes]:
    rng = np.random.default_rng(seed)
    x = rng.random((600, n_features)).astype(np.float32)
    y = (x[:, -1] * 3 + x[:, 0] * 2).astype(int).clip(0, 3)
    m = lgb.LGBMRanker(n_estimators=20, verbose=-1).fit(x, y, group=[20] * 30)
    return m.booster_, ranker_to_onnx(m.booster_, n_features, export or {})


def _doc(i: int, **kw: object) -> dict:
    d = {
        "item_id": f"it{i:06d}", "market": "FR", "category": "jeans", "brand": f"b{i % 3}",
        "brand_t": {"cells": [{"address": {"brand": f"b{i % 3}"}, "value": 1.0}]},
        "price": 10.0 + i, "category_median_price": 20.0, "condition": i % 4, "photo_quality": 0.5 + (i % 5) / 10,
        "seller_rating": 4.0 + (i % 10) / 10, "listed_at": 1_000, "clicks_7d": i, "impressions_7d": 100,
        "favourites": i, "favourites_today": 0, "sold": False, "emb_versions": [],
    }  # fmt: skip
    d.update(kw)
    return d


def _vespa(emu: VespaEmulator) -> Vespa:
    return Vespa(emu.client(), "http://q", "http://c")


def test_schema_parses_with_profiles_functions_and_match_features() -> None:
    _, onx = _model(len(FEATS))
    spec = P.PackageSpec(rankers=[P.RankerProfile("r0001", onx, FEATS)], embeddings=[P.EmbeddingField("t0001", 8)])
    files = P.build(C, spec)
    s = parse(files["schemas/item.sd"].decode())
    assert set(s.profiles) == {"base", "popularity", "ranker_r0001", "homepage_t0001"}
    prof = s.resolved("ranker_r0001")
    assert prof.match_features == FEATS
    assert prof.onnx["ranker_r0001"].inputs == {"features": "features_r0001"}
    assert prof.second_phase == "sum(onnx(ranker_r0001).score)"
    assert s.fields["emb_t0001"] == "tensor<float>(x[8])"
    assert "files/ranker_r0001.onnx" in files and P.PackageSpec.from_files(files).manifest() == spec.manifest()


def test_package_zip_is_deterministic() -> None:
    _, onx = _model(len(FEATS))
    spec = P.PackageSpec(rankers=[P.RankerProfile("r0001", onx, FEATS)])
    assert P.to_zip(P.build(C, spec)) == P.to_zip(P.build(C, spec))


def test_vespa_relevance_equals_the_model_run_offline_on_its_match_features() -> None:
    _booster, onx = _model(len(FEATS))
    emu = VespaEmulator()
    v = _vespa(emu)
    v.deploy(P.to_zip(P.build(C, P.PackageSpec(rankers=[P.RankerProfile("r0001", onx, FEATS)]))))
    for i in range(40):
        v.put(f"it{i:06d}", _doc(i))
    res = v.search(['market contains "FR"', 'category contains "jeans"', "sold = false"], "ranker_r0001", 10,
                   {"now": 86400 * 3, "user_brand": mapped_tensor("brand", {"b1": 0.8}), "user_log_price": 3.0})  # fmt: skip
    assert res.total == 40 and len(res.hits) == 10
    x = np.array([[h.features[f] for f in FEATS] for h in res.hits], dtype=np.float32)
    offline = run_onnx(onx, x)[:, 0]
    assert np.allclose(offline, [h.relevance for h in res.hits], atol=1e-5)
    assert [h.relevance for h in res.hits] == sorted([h.relevance for h in res.hits], reverse=True)
    # brand affinity comes from the mapped query tensor against the document's brand
    assert {h.features["brand_affinity"] for h in res.hits} <= {0.0, 0.8}


def test_expressions_match_their_definitions() -> None:
    cols = {
        "price": np.array([10.0, 40.0]), "category_median_price": np.array([20.0, 20.0]),
        "listed_at": np.array([0.0, 86400.0]), "clicks_7d": np.array([4.0, 0.0]), "impressions_7d": np.array([80.0, 0.0]),
        "brand_t": E.Mapped([{"a": 1.0}, {"b": 1.0}]),
    }  # fmt: skip
    q = {"now": 86400.0 * 2, "user_brand": {"a": 0.5}, "user_log_price": float(np.log(20.0))}
    fns = {f.name: E.parse(f.expression) for f in C.features}
    ctx = E.Context(2, cols, q, fns, {})
    assert np.allclose(ctx.function("price_rel"), [0.5, 2.0])
    assert np.allclose(ctx.function("log_age_days"), np.log([3.0, 2.0]))
    assert np.allclose(ctx.function("item_ctr"), [5 / 100, 1 / 20])
    assert np.allclose(ctx.function("brand_affinity"), [0.5, 0.0])
    assert np.allclose(ctx.function("price_fit"), np.abs(np.log([10.0, 40.0]) - np.log(20.0)))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda files, onx: files.update({"files/ranker_r0001.onnx": onx[1]}), "has 5 features"),
        (lambda files, onx: files.pop("files/ranker_r0001.onnx"), "is not in the package"),
        (
            lambda files, onx: files.update(
                {
                    "schemas/item.sd": files["schemas/item.sd"].replace(
                        b"attribute(seller_rating)", b"attribute(seller_score)"
                    )
                }
            ),
            "unknown attribute seller_score",
        ),
    ],
)
def test_vespa_refuses_broken_packages(mutate, message) -> None:  # type: ignore[no-untyped-def]
    _, good = _model(len(FEATS))
    _, small = _model(5)
    files = P.build(C, P.PackageSpec(rankers=[P.RankerProfile("r0001", good, FEATS)]))
    mutate(files, (good, small))
    status, body = VespaEmulator().deploy(P.to_zip(files))
    assert status == 400 and message in body["message"]


def test_a_runtime_that_is_older_than_the_export_refuses_it() -> None:
    _, onx = _model(len(FEATS), export={"ir_version": 11})
    assert onnx.load_from_string(onx).ir_version == 11
    status, body = VespaEmulator(Runtime(max_ir_version=10)).deploy(
        P.to_zip(P.build(C, P.PackageSpec(rankers=[P.RankerProfile("r0001", onx, FEATS)])))
    )
    assert status == 400 and "IR version: 11" in body["message"]


def test_nearest_neighbour_coverage_and_partial_updates() -> None:
    emu = VespaEmulator()
    v = _vespa(emu)
    v.deploy(P.to_zip(P.build(C, P.PackageSpec(embeddings=[P.EmbeddingField("t0001", 2)]))))
    for i in range(6):
        v.put(f"it{i:06d}", _doc(i, listed_at=100 * i))
    for i in range(3):
        vec = [1.0, 0.0] if i == 1 else [0.0, 1.0]
        v.update(f"it{i:06d}", {"emb_t0001": {"assign": {"values": vec}}, "emb_versions": {"add": ["t0001"]}})
    assert v.count(["sold = false", 'emb_versions contains "t0001"']) == 3
    assert v.count(["sold = false", "listed_at < 250"]) == 3
    res = v.search(
        ['market contains "FR"'], "homepage_t0001", 3, {"user_emb": "[1,0]"}, nearest=("emb_t0001", "user_emb", 10)
    )
    assert res.hits[0].item_id == "it000001" and len(res.hits) == 3  # only items with an embedding are candidates
    v.update("it000001", {"sold": {"assign": True}})
    assert v.count(['market contains "FR"', "sold = false"]) == 5


def test_feeding_a_field_the_schema_does_not_have_is_refused() -> None:
    emu = VespaEmulator()
    v = _vespa(emu)
    v.deploy(P.to_zip(P.build(C, P.PackageSpec())))
    v.put("it000001", _doc(1))
    with pytest.raises(Exception, match="No field"):
        v.update("it000001", {"emb_t0009": {"assign": {"values": [1.0]}}})


def test_the_expression_language_is_vespas_not_pythons() -> None:
    # real Vespa refused abs() as a malformed rank feature name in CI; its function is fabs()
    assert E.parse("fabs(log(attribute(price)) - query(user_log_price))").kind == "fn"
    with pytest.raises(Exception):  # noqa: B017
        E.parse("abs(log(attribute(price)) - query(user_log_price))")

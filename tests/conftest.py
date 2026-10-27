from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if not os.environ.get("PTP_TEST_DATABASE_URL"):
        skip = pytest.mark.skip(reason="PTP_TEST_DATABASE_URL is not set")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip)
    for marker, env in (("vespa", "PTP_VESPA_URL"), ("triton", "PTP_TRITON_URL"), ("kind", "PTP_RELAY_URL")):
        if not os.environ.get(env):
            skip = pytest.mark.skip(reason=f"{env} is not set (runs in CI against the real service)")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PTP_DATA", str(tmp_path))
    monkeypatch.setenv("PTP_LAKE", str(tmp_path / "lake"))
    monkeypatch.setenv("PTP_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PTP_TRITON_STAGING_REPOSITORY", str(tmp_path / "triton" / "staging"))
    monkeypatch.setenv("PTP_TRITON_PROD_REPOSITORY", str(tmp_path / "triton" / "prod"))
    monkeypatch.setenv("PTP_CODE_COMMIT", "test")
    return tmp_path


@pytest.fixture
def registry(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from ptp.registry.store import Registry

    url = os.environ["PTP_TEST_DATABASE_URL"]
    monkeypatch.setenv("PTP_DATABASE_URL", url)
    reg = Registry(url)
    reg.reset()
    yield reg
    reg.close()


class Router(httpx.BaseTransport):
    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.url.host}:{request.url.port}" if request.url.port else request.url.host
        h = self.routes.get(key) or self.routes.get(request.url.host)
        if h is None:
            return httpx.Response(502)
        return h.handle(request) if hasattr(h, "handle") else h(request)


@pytest.fixture
def envs(workdir: Path) -> Any:
    """Staging and production Vespa and Triton emulators behind the hosts config/targets.yaml names."""
    from ptp import contract
    from ptp import http as ptp_http
    from ptp.serving.vespa import package as P
    from ptpsim.emulators.triton import TritonEmulator
    from ptpsim.emulators.vespa import VespaEmulator

    class Envs:
        pass

    e = Envs()
    e.vespa_staging, e.vespa_prod = VespaEmulator(), VespaEmulator()
    e.triton_staging = TritonEmulator(workdir / "triton" / "staging")
    e.triton_prod = TritonEmulator(workdir / "triton" / "prod")
    r = Router()
    for p in (18080, 19071):
        r.routes[f"127.0.0.1:{p}"] = e.vespa_staging
    for p in (28080, 29071):
        r.routes[f"127.0.0.1:{p}"] = e.vespa_prod
    r.routes["127.0.0.1:18000"] = e.triton_staging
    r.routes["127.0.0.1:28000"] = e.triton_prod
    e.router = r
    ptp_http.install(r)
    z = P.to_zip(P.build(contract.load(), P.PackageSpec()))
    e.vespa_staging.deploy(z)
    e.vespa_prod.deploy(z)
    yield e
    ptp_http.install(None)


SMALL_WORLD = {
    "seed": 3,
    "users": 1200,
    "initial_items": 1600,
    "new_items_per_day": 180,
    "search_sessions_per_day": 900,
    "homepage_sessions_per_day": 250,
}


@pytest.fixture(scope="session")
def small_lake(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Nine days of simulated logs with the popularity profile in production (built once per test session)."""
    from ptp import contract
    from ptp.lake import Lake
    from ptp.serving.vespa import package as P
    from ptpsim.emulators.vespa import VespaEmulator
    from ptpsim.marketplace import Incidents, Marketplace
    from ptpsim.world import World, WorldConfig

    out = tmp_path_factory.mktemp("small-lake")
    world = World(WorldConfig.from_dict(SMALL_WORLD))
    prod, staging = VespaEmulator(), VespaEmulator()
    z = P.to_zip(P.build(contract.load(), P.PackageSpec()))
    prod.deploy(z)
    staging.deploy(z)
    mk = Marketplace(world, prod, staging, Lake(out / "lake"), Incidents(), lambda s: None)
    mk.initial_listings()
    for d in range(9):
        mk.run_day(d)
    import pickle

    (out / "docs.pkl").write_bytes(pickle.dumps(prod.docs))
    (out / "t0").write_text(str(world.cfg.t0))
    return out


@pytest.fixture
def lake(small_lake: Path, workdir: Path) -> Path:
    shutil.copytree(small_lake / "lake", workdir / "lake")
    return workdir / "lake"


def day_ts(small_lake: Path, day: int, hour: int = 3) -> int:
    return int((small_lake / "t0").read_text()) + day * 86400 + hour * 3600

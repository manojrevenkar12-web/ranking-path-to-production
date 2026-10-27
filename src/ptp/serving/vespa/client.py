"""A small Vespa client over HTTP: the deploy API on the config server (port 19071), document/v1 and search on the
container (port 8080). It takes an httpx.Client, so the simulated seasons pass one whose transport is the local
emulator and CI passes a plain one pointed at a real Vespa."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

NAMESPACE = "buyer"
DOCTYPE = "item"
APP = "application/v2/tenant/default/application/default/environment/prod/region/default/instance/default"


class VespaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class Hit:
    item_id: str
    relevance: float
    features: dict[str, float]


@dataclass
class Result:
    total: int
    hits: list[Hit]
    elapsed_ms: float


def mapped_tensor(dim: str, cells: dict[str, float]) -> str:
    """Vespa's literal form of a one-dimension mapped tensor: {{brand:b1}:0.5,{brand:b2}:0.2}."""
    if not cells:
        return "{}"
    return "{" + ",".join(f"{{{dim}:{k}}}:{float(v):.6g}" for k, v in sorted(cells.items())) + "}"


def dense_tensor(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


class Vespa:
    def __init__(self, http: httpx.Client, query_url: str, config_url: str, timeout_s: float = 300.0):
        self.http = http
        self.query_url = query_url.rstrip("/")
        self.config_url = config_url.rstrip("/")
        self.timeout_s = timeout_s

    # deploy API -------------------------------------------------------------------------------------------------

    def deploy(self, package_zip: bytes) -> dict[str, Any]:
        r = self.http.post(
            f"{self.config_url}/application/v2/tenant/default/prepareandactivate",
            content=package_zip,
            headers={"Content-Type": "application/zip"},
            timeout=self.timeout_s,
        )
        body = _json(r)
        if r.status_code != 200:
            raise VespaError(f"deploy refused ({r.status_code}): {body.get('message', body)}", r.status_code, body)
        actions = body.get("configChangeActions", {})
        if actions.get("restart") or actions.get("refeed") or actions.get("reindex"):
            # A model or profile change never needs these; a field type change would. Surface it instead of hoping.
            body["needs_action"] = True
        return body

    def generation(self) -> int:
        r = self.http.get(f"{self.config_url}/application/v2/tenant/default/application/default", timeout=30)
        return int(_json(r).get("generation", 0))

    def wait_converged(self, timeout_s: float | None = None, poll_s: float = 1.0) -> dict[str, Any]:
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            r = self.http.get(f"{self.config_url}/{APP}/serviceconverge", timeout=30)
            if r.status_code == 200:
                last = _json(r)
                if last.get("converged"):
                    return last
            time.sleep(poll_s)
        raise VespaError(f"services did not converge on the new config: {last}")

    def wait_ready_config(self, timeout_s: float = 300.0) -> None:
        """The config server is up and accepts deployments (a fresh Vespa container takes a minute)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                r = self.http.get(f"{self.config_url}/state/v1/health", timeout=5)
                if r.status_code == 200 and _json(r).get("status", {}).get("code") == "up":
                    return
            except httpx.TransportError:
                pass
            time.sleep(2)
        raise VespaError("config server is not up")

    def wait_ready(self, timeout_s: float = 300.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                r = self.http.get(f"{self.query_url}/state/v1/health", timeout=5)
                if r.status_code == 200 and _json(r).get("status", {}).get("code") == "up":
                    return
            except httpx.TransportError:
                pass
            time.sleep(2)
        raise VespaError("container is not up")

    # documents ----------------------------------------------------------------------------------------------------

    def _doc_url(self, item_id: str) -> str:
        return f"{self.query_url}/document/v1/{NAMESPACE}/{DOCTYPE}/docid/{item_id}"

    def put(self, item_id: str, fields: dict[str, Any]) -> None:
        r = self.http.post(self._doc_url(item_id), json={"fields": fields}, timeout=30)
        if r.status_code != 200:
            raise VespaError(f"feed {item_id} failed ({r.status_code}): {r.text[:200]}", r.status_code)

    def update(self, item_id: str, ops: dict[str, dict[str, Any]], create: bool = False) -> None:
        """ops: field -> {"assign": value} or {"add": [values]}."""
        params = {"create": "true"} if create else None
        r = self.http.put(self._doc_url(item_id), json={"fields": ops}, params=params, timeout=30)
        if r.status_code != 200:
            raise VespaError(f"update {item_id} failed ({r.status_code}): {r.text[:200]}", r.status_code)

    def remove(self, item_id: str) -> None:
        r = self.http.delete(self._doc_url(item_id), timeout=30)
        if r.status_code not in (200, 404):
            raise VespaError(f"remove {item_id} failed ({r.status_code})", r.status_code)

    # search -------------------------------------------------------------------------------------------------------

    def search(
        self,
        where: list[str],
        ranking: str,
        hits: int,
        inputs: dict[str, str | float] | None = None,
        nearest: tuple[str, str, int] | None = None,
        timeout_ms: int = 500,
    ) -> Result:
        clauses = list(where)
        if nearest:
            field, query_tensor, target = nearest
            clauses.insert(0, f"({{targetHits:{target}}}nearestNeighbor({field}, {query_tensor}))")
        yql = "select * from item where " + " and ".join(clauses)
        body: dict[str, Any] = {"yql": yql, "ranking": ranking, "hits": hits, "timeout": f"{timeout_ms}ms"}
        for k, v in (inputs or {}).items():
            body[f"input.query({k})"] = v
        t0 = time.perf_counter()
        r = self.http.post(f"{self.query_url}/search/", json=body, timeout=max(5.0, timeout_ms / 1000 * 4))
        elapsed = (time.perf_counter() - t0) * 1000
        data = _json(r)
        root = data.get("root", {})
        if r.status_code != 200 or root.get("errors"):
            raise VespaError(f"search failed ({r.status_code}): {root.get('errors') or data}", r.status_code, data)
        out = []
        for child in root.get("children", []) or []:
            fields = child.get("fields", {})
            mf = fields.get("matchfeatures", {}) or {}
            out.append(
                Hit(
                    item_id=str(fields.get("item_id", child.get("id", ""))),
                    relevance=float(child.get("relevance", 0.0)),
                    features={k: float(v) for k, v in mf.items() if isinstance(v, (int, float))},
                )
            )
        return Result(total=int(root.get("fields", {}).get("totalCount", len(out))), hits=out, elapsed_ms=elapsed)

    def count(self, where: list[str]) -> int:
        return self.search(where, ranking="unranked", hits=0).total


def _json(r: httpx.Response) -> dict[str, Any]:
    try:
        data = r.json()
        return data if isinstance(data, dict) else {"value": data}
    except ValueError:
        return {"message": r.text[:500]}

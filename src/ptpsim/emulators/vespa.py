"""A Vespa emulator for the simulated seasons: the deploy API, document/v1 and search, with the ranking expressions
and ONNX models of the deployed package evaluated for real (onnxruntime). It refuses packages the way Vespa
would for the mistakes that matter here: a model its ONNX runtime cannot load, a model input that does not match
the tensor the profile builds, an unknown attribute or function, a missing file.

It is the world's Vespa, so its runtime limits come from the scenario, not from config/targets.yaml: a release the
platform's compatibility gate lets through but this runtime cannot load fails here, as it would in production.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import onnx
import onnxruntime as ort

from ptp.serving.vespa.package import from_zip

from . import expr as E
from .schema import Schema, parse

APP_PATH = "/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default"


@dataclass
class Runtime:
    max_ir_version: int = 10
    max_opset: dict[str, int] = field(default_factory=lambda: {"": 21, "ai.onnx.ml": 4})


class PackageError(ValueError):
    pass


@dataclass
class Active:
    generation: int
    schema: Schema
    files: dict[str, bytes]
    sessions: dict[str, ort.InferenceSession]
    parsed: dict[str, dict[str, E.Node]]  # profile -> function name -> parsed node
    phases: dict[str, tuple[E.Node | None, E.Node | None, int, list[str]]]


def _parse_tensor_literal(value: Any) -> Any:
    """Query inputs: numbers, dense [..] literals, mapped {{dim:key}:v,..} literals."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.startswith("["):
        return np.array(json.loads(s), dtype=np.float32)
    if s.startswith("{"):
        cells = {}
        for m in re.finditer(r"\{\s*\w+\s*:\s*([^}]+?)\s*\}\s*:\s*([-+0-9.eE]+)", s):
            cells[m.group(1)] = float(m.group(2))
        return cells
    return float(s)


class VespaEmulator:
    def __init__(self, runtime: Runtime | None = None, latency_ms: float = 0.0):
        self.runtime = runtime or Runtime()
        self.active: Active | None = None
        self.generation = 0
        self.docs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self._cols: dict[tuple[str | None, str | None], dict[str, Any]] = {}
        self.latency_ms = latency_ms
        self.deploy_log: list[dict[str, Any]] = []

    # ---- deploy ------------------------------------------------------------------------------------------------

    def _load(self, files: dict[str, bytes]) -> Active:
        if "services.xml" not in files:
            raise PackageError("services.xml is missing")
        sd = [p for p in files if p.startswith("schemas/") and p.endswith(".sd")]
        if not sd:
            raise PackageError("no schema in schemas/")
        try:
            schema = parse(files[sd[0]].decode())
        except Exception as e:  # pragma: no cover - parse errors are reported to the deployer
            raise PackageError(f"schema does not parse: {e}") from e
        sessions: dict[str, ort.InferenceSession] = {}
        parsed: dict[str, dict[str, E.Node]] = {}
        phases: dict[str, tuple[E.Node | None, E.Node | None, int, list[str]]] = {}
        for name in schema.profiles:
            prof = schema.resolved(name)
            fns: dict[str, E.Node] = {}
            for fname, src in prof.functions.items():
                try:
                    fns[fname] = E.parse(src)
                except E.ParseError as e:
                    raise PackageError(f"rank-profile {name}: function {fname}: {e}") from e
            for mname, model in prof.onnx.items():
                if model.file not in files:
                    raise PackageError(f"rank-profile {name}: onnx-model {mname}: {model.file} is not in the package")
                if mname not in sessions:
                    sessions[mname] = self._session(name, mname, files[model.file])
                sess = sessions[mname]
                names = {i.name: i for i in sess.get_inputs()}
                for onnx_input, source in model.inputs.items():
                    if onnx_input not in names:
                        raise PackageError(f"rank-profile {name}: model {mname} has no input {onnx_input!r}")
                    fn = source.removesuffix("()")
                    if fn not in fns:
                        raise PackageError(f"rank-profile {name}: input {onnx_input} refers to unknown function {fn}")
                    node = fns[fn]
                    shape = names[onnx_input].shape
                    if node.kind == "dense" and isinstance(shape[-1], int) and shape[-1] != node.value:
                        raise PackageError(
                            f"rank-profile {name}: model {mname} input {onnx_input} has {shape[-1]} features, "
                            f"function {fn} builds {node.value}"
                        )
                outs = {o.name for o in sess.get_outputs()}
                for onnx_output in model.outputs.values():
                    if onnx_output not in outs:
                        raise PackageError(f"rank-profile {name}: model {mname} has no output {onnx_output!r}")
            first = E.parse(prof.first_phase) if prof.first_phase else None
            second = E.parse(prof.second_phase) if prof.second_phase else None
            nodes: list[E.Node | None] = [first, second, *fns.values()]
            for expr_node in nodes:
                if expr_node is None:
                    continue
                refs = E.references(expr_node)
                for a in refs["attribute"] | refs["closeness"]:
                    if a not in schema.fields:
                        raise PackageError(f"rank-profile {name}: unknown attribute {a}")
                for r in refs["ref"]:
                    if r not in fns:
                        raise PackageError(f"rank-profile {name}: unknown function {r}")
                for q in refs["query"]:
                    if q not in prof.inputs:
                        raise PackageError(f"rank-profile {name}: query({q}) is not declared in inputs")
            for mf in prof.match_features:
                if mf not in fns:
                    raise PackageError(f"rank-profile {name}: match-feature {mf} is not a function")
            parsed[name] = fns
            phases[name] = (first, second, prof.rerank_count, prof.match_features)
        return Active(self.generation + 1, schema, files, sessions, parsed, phases)

    def _session(self, profile: str, mname: str, data: bytes) -> ort.InferenceSession:
        try:
            model = onnx.load_from_string(data)
        except Exception as e:
            raise PackageError(f"rank-profile {profile}: onnx-model {mname} is not a valid ONNX file: {e}") from e
        if model.ir_version > self.runtime.max_ir_version:
            raise PackageError(
                f"rank-profile {profile}: onnx-model {mname}: Unsupported model IR version: {model.ir_version}, "
                f"max supported IR version: {self.runtime.max_ir_version}"
            )
        for imp in model.opset_import:
            limit = self.runtime.max_opset.get(imp.domain)
            if limit is not None and imp.version > limit:
                raise PackageError(
                    f"rank-profile {profile}: onnx-model {mname}: opset {imp.version} of domain "
                    f"'{imp.domain or 'ai.onnx'}' is newer than this runtime supports ({limit})"
                )
        try:
            return ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise PackageError(f"rank-profile {profile}: onnx-model {mname} could not be loaded: {e}") from e

    def deploy(self, package_zip: bytes) -> tuple[int, dict[str, Any]]:
        with self.lock:
            try:
                files = from_zip(package_zip)
                active = self._load(files)
            except PackageError as e:
                self.deploy_log.append({"ok": False, "message": str(e)})
                return 400, {
                    "error-code": "INVALID_APPLICATION_PACKAGE",
                    "message": f"Invalid application package: {e}",
                }
            refeed = []
            if self.active:
                for f, t in self.active.schema.fields.items():
                    if f in active.schema.fields and active.schema.fields[f] != t:
                        refeed.append({"name": "field-type-change", "field": f})
                for f in set(self.active.schema.fields) - set(active.schema.fields):
                    for d in self.docs.values():
                        d.pop(f, None)
            self.generation = active.generation
            self.active = active
            self._cols.clear()
            self.deploy_log.append({"ok": True, "generation": self.generation})
            return 200, {
                "tenant": "default",
                "session-id": str(self.generation),
                "message": f"Session {self.generation} for tenant 'default' prepared and activated.",
                "configChangeActions": {"restart": [], "refeed": refeed, "reindex": []},
            }

    # ---- documents ---------------------------------------------------------------------------------------------

    def _check_fields(self, fields: dict[str, Any]) -> str | None:
        if not self.active:
            return "no application deployed"
        for f in fields:
            if f not in self.active.schema.fields:
                return f"No field '{f}' in the structure of type 'item'"
        return None

    def _coerce(self, name: str, value: Any) -> Any:
        assert self.active
        t = self.active.schema.fields[name]
        if t.startswith("tensor") and "{}" in t:
            if isinstance(value, dict) and "cells" in value:
                return {next(iter(c["address"].values())): float(c["value"]) for c in value["cells"]}
            return {str(k): float(v) for k, v in dict(value).items()}
        if t.startswith("tensor"):
            vals = value["values"] if isinstance(value, dict) else value
            arr = np.asarray(vals, dtype=np.float32)
            dim = int(re.search(r"\[(\d+)\]", t).group(1))  # type: ignore[union-attr]
            if arr.shape != (dim,):
                raise ValueError(f"tensor for {name} has shape {arr.shape}, field type is {t}")
            return arr
        if t.startswith("array"):
            return list(value)
        if t in ("float", "double"):
            return float(value)
        if t in ("int", "long"):
            return int(value)
        if t == "bool":
            return bool(value)
        return str(value)

    def _touch(self, doc: dict[str, Any]) -> None:
        m, c = doc.get("market"), doc.get("category")
        for key in [k for k in self._cols if k[0] in (None, m) and k[1] in (None, c)]:
            self._cols.pop(key, None)

    def put(self, doc_id: str, fields: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            err = self._check_fields(fields)
            if err:
                return 400, {"message": err}
            try:
                doc = {k: self._coerce(k, v) for k, v in fields.items()}
            except (ValueError, TypeError) as e:
                return 400, {"message": str(e)}
            old = self.docs.get(doc_id)
            if old:
                self._touch(old)
            self.docs[doc_id] = doc
            self._touch(doc)
            return 200, {"id": f"id:buyer:item::{doc_id}"}

    def update(self, doc_id: str, ops: dict[str, dict[str, Any]], create: bool) -> tuple[int, dict[str, Any]]:
        with self.lock:
            err = self._check_fields(ops)
            if err:
                return 400, {"message": err}
            doc = self.docs.get(doc_id)
            if doc is None:
                if not create:
                    return 404, {"message": "Document not found"}
                doc = self.docs[doc_id] = {}
            self._touch(doc)
            try:
                for f, op in ops.items():
                    if "assign" in op:
                        doc[f] = self._coerce(f, op["assign"])
                    elif "add" in op:
                        current = list(doc.get(f, []))
                        for v in op["add"]:
                            if v not in current:
                                current.append(v)
                        doc[f] = current
                    elif "increment" in op:
                        doc[f] = doc.get(f, 0) + op["increment"]
            except (ValueError, TypeError) as e:
                return 400, {"message": str(e)}
            self._touch(doc)
            return 200, {"id": f"id:buyer:item::{doc_id}"}

    def remove(self, doc_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            doc = self.docs.pop(doc_id, None)
            if doc:
                self._touch(doc)
            return 200, {}

    # ---- search ------------------------------------------------------------------------------------------------

    def _columns(self, market: str | None, category: str | None) -> dict[str, Any]:
        key = (market, category)
        if key in self._cols:
            return self._cols[key]
        assert self.active
        ids = [
            i
            for i, d in self.docs.items()
            if (market is None or d.get("market") == market) and (category is None or d.get("category") == category)
        ]
        docs = [self.docs[i] for i in ids]
        cols: dict[str, Any] = {"__ids": ids}
        for f, t in self.active.schema.fields.items():
            if t.startswith("tensor") and "{}" in t:
                cols[f] = E.Mapped(d.get(f, {}) for d in docs)
            elif t.startswith("tensor"):
                dim = int(re.search(r"\[(\d+)\]", t).group(1))  # type: ignore[union-attr]
                cols[f] = (
                    np.stack([d[f] if f in d else np.zeros(dim, np.float32) for d in docs])
                    if docs
                    else np.zeros((0, dim), np.float32)
                )
                cols["__has_" + f] = np.array([f in d for d in docs], dtype=bool)
            elif t.startswith("array"):
                cols[f] = [d.get(f, []) for d in docs]
            elif t == "string":
                cols[f] = [d.get(f, "") for d in docs]
            else:
                cols[f] = np.array([float(d.get(f, 0)) for d in docs], dtype=float)
        self._cols[key] = cols
        return cols

    def search(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        t0 = time.perf_counter()
        with self.lock:
            if not self.active:
                return 400, {"root": {"errors": [{"message": "no application deployed"}]}}
            yql = body.get("yql", "")
            ranking = body.get("ranking", "default")
            hits = int(body.get("hits", 10))
            m = re.match(r"select \* from item where (.+)$", yql.strip())
            if not m:
                return 400, {"root": {"errors": [{"message": f"cannot parse yql {yql!r}"}]}}
            clauses = [c.strip() for c in re.split(r"\s+and\s+", m.group(1))]
            market = category = None
            masks: list[Any] = []
            nearest: tuple[str, str, int] | None = None
            for c in clauses:
                if mm := re.match(r'(\w+) contains "([^"]*)"$', c):
                    f, v = mm.groups()
                    if f == "market":
                        market = v
                    elif f == "category":
                        category = v
                    else:
                        masks.append(("contains", f, v))
                elif mm := re.match(r"(\w+) = (true|false|-?[\d.]+)$", c):
                    masks.append(("eq", mm.group(1), mm.group(2)))
                elif mm := re.match(r"(\w+) (<|>|<=|>=) (-?[\d.]+)$", c):
                    masks.append((mm.group(2), mm.group(1), mm.group(3)))
                elif mm := re.match(r"\(\{targetHits:(\d+)\}nearestNeighbor\((\w+),\s*(\w+)\)\)$", c):
                    nearest = (mm.group(2), mm.group(3), int(mm.group(1)))
                elif c == "true":
                    continue
                else:
                    return 400, {"root": {"errors": [{"message": f"unsupported yql clause {c!r}"}]}}
            cols = self._columns(market, category)
            n = len(cols["__ids"])
            keep = np.ones(n, dtype=bool)
            for kind, f, v in masks:
                if f not in cols:
                    return 400, {"root": {"errors": [{"message": f"unknown field {f}"}]}}
                if kind == "contains":
                    keep &= np.array([v in x if isinstance(x, list) else x == v for x in cols[f]], dtype=bool)
                elif kind in ("<", ">", "<=", ">="):
                    col = np.asarray(cols[f], dtype=float)
                    keep &= {"<": col < float(v), ">": col > float(v), "<=": col <= float(v), ">=": col >= float(v)}[
                        kind
                    ]
                else:
                    target = {"true": 1.0, "false": 0.0}.get(v)
                    keep &= np.asarray(cols[f]) == (target if target is not None else float(v))
            if nearest:
                field_name = nearest[0]
                if field_name not in cols:
                    return 400, {"root": {"errors": [{"message": f"unknown field {field_name}"}]}}
                keep &= cols["__has_" + field_name]
            idx = np.nonzero(keep)[0]
            total = len(idx)
            if ranking == "unranked" or hits == 0:
                return 200, self._response([], total, t0)
            if ranking not in self.active.phases:
                return 400, {"root": {"errors": [{"message": f"rank profile '{ranking}' does not exist"}]}}
            sub = {
                k: (
                    E.Mapped(v[i] for i in idx)
                    if isinstance(v, E.Mapped)
                    else [v[i] for i in idx]
                    if isinstance(v, list)
                    else v[idx]
                )
                for k, v in cols.items()
            }
            queries = {}
            for k, v in body.items():
                if k.startswith("input.query(") and k.endswith(")"):
                    queries[k[len("input.query(") : -1]] = _parse_tensor_literal(v)
            first, second, rerank, match_features = self.active.phases[ranking]
            if nearest and nearest[1] not in queries:
                return 400, {"root": {"errors": [{"message": f"query tensor {nearest[1]} is missing"}]}}
            try:
                ctx = E.Context(len(idx), sub, queries, self.active.parsed[ranking], self._models(ranking))
                score = np.asarray(ctx.eval(first), dtype=float) if first is not None else np.zeros(len(idx))
                order = (
                    np.argsort(-score, kind="stable")[: nearest[2]] if nearest else np.argsort(-score, kind="stable")
                )
                if second is not None and len(order):
                    top = order[:rerank]
                    sub2 = {
                        k: (
                            E.Mapped(v[i] for i in top)
                            if isinstance(v, E.Mapped)
                            else [v[i] for i in top]
                            if isinstance(v, list)
                            else v[top]
                        )
                        for k, v in sub.items()
                    }
                    ctx2 = E.Context(len(top), sub2, queries, self.active.parsed[ranking], self._models(ranking))
                    s2 = np.asarray(ctx2.eval(second), dtype=float)
                    rer = top[np.argsort(-s2, kind="stable")]
                    score2 = dict(zip(top.tolist(), s2.tolist(), strict=True))
                    order = np.concatenate([rer, order[rerank:]])
                    final = [(int(i), score2.get(int(i), float(score[i]))) for i in order[:hits]]
                else:
                    final = [(int(i), float(score[i])) for i in order[:hits]]
                feats = {}
                if match_features and final:
                    chosen = np.array([i for i, _ in final])
                    sub3 = {
                        k: (
                            E.Mapped(v[i] for i in chosen)
                            if isinstance(v, E.Mapped)
                            else [v[i] for i in chosen]
                            if isinstance(v, list)
                            else v[chosen]
                        )
                        for k, v in sub.items()
                    }
                    ctx3 = E.Context(len(chosen), sub3, queries, self.active.parsed[ranking], self._models(ranking))
                    feats = {mf: np.asarray(ctx3.function(mf), dtype=float) for mf in match_features}
            except KeyError as e:
                return 400, {"root": {"errors": [{"message": f"ranking failed: {e}"}]}}
            children = []
            for pos, (i, rel) in enumerate(final):
                doc_ids = sub["__ids"]
                fields: dict[str, Any] = {"item_id": doc_ids[i]}
                if feats:
                    fields["matchfeatures"] = {k: _finite(v[pos]) for k, v in feats.items()}
                children.append({"id": f"id:buyer:item::{doc_ids[i]}", "relevance": _finite(rel), "fields": fields})
            return 200, self._response(children, total, t0)

    def _response(self, children: list[dict[str, Any]], total: int, t0: float) -> dict[str, Any]:
        if self.latency_ms:
            spent = (time.perf_counter() - t0) * 1000
            if spent < self.latency_ms:
                time.sleep((self.latency_ms - spent) / 1000)
        return {"root": {"id": "toplevel", "relevance": 1.0, "fields": {"totalCount": total}, "children": children}}

    def _models(self, profile: str) -> dict[str, Any]:
        assert self.active
        prof = self.active.schema.resolved(profile)
        out = {}
        for mname, model in prof.onnx.items():
            sess = self.active.sessions[mname]

            def run(ctx: E.Context, output: str | None, model=model, sess=sess) -> np.ndarray:
                feeds = {}
                for onnx_input, source in model.inputs.items():
                    feeds[onnx_input] = np.asarray(ctx.function(source.removesuffix("()")), dtype=np.float32)
                onnx_out = model.outputs.get(output or "", None) or sess.get_outputs()[0].name
                res = sess.run([onnx_out], feeds)[0]
                return np.asarray(res, dtype=float).reshape(ctx.n, -1)

            out[mname] = run
        return out

    # ---- HTTP --------------------------------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path == "/application/v2/tenant/default/prepareandactivate" and method == "POST":
            status, body = self.deploy(request.content)
            return httpx.Response(status, json=body)
        if path == "/application/v2/tenant/default/application/default" and method == "GET":
            return httpx.Response(200, json={"generation": self.generation})
        if path == APP_PATH + "/serviceconverge":
            return httpx.Response(
                200, json={"converged": True, "currentGeneration": self.generation, "wantedGeneration": self.generation}
            )
        if path == "/state/v1/health":
            # the container is up once an application is active; the config server always is
            up = self.active is not None or request.url.port in (19071, 29071)
            return httpx.Response(200, json={"status": {"code": "up" if up else "initializing"}})
        if path == "/search/" and method == "POST":
            status, body = self.search(json.loads(request.content or b"{}"))
            return httpx.Response(status, json=body)
        m = re.match(r"^/document/v1/buyer/item/docid/(.+)$", path)
        if m:
            doc_id = m.group(1)
            if method == "POST":
                status, body = self.put(doc_id, json.loads(request.content)["fields"])
            elif method == "PUT":
                create = request.url.params.get("create") == "true"
                status, body = self.update(doc_id, json.loads(request.content)["fields"], create)
            elif method == "DELETE":
                status, body = self.remove(doc_id)
            elif method == "GET":
                doc = self.docs.get(doc_id)
                if doc is None:
                    return httpx.Response(404, json={"message": "not found"})
                return httpx.Response(
                    200, json={"fields": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in doc.items()}}
                )
            else:
                return httpx.Response(405)
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"message": f"no route {method} {path}"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def _finite(x: float) -> float | str:
    return float(x) if math.isfinite(x) else "NaN"

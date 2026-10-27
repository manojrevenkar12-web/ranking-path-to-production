"""The registry in PostgreSQL (PTP_DATABASE_URL)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .. import clock

STATUSES = ("rejected", "candidate", "staging", "canary", "production", "archived", "rolled_back", "failed")
TABLES = [
    "quarantine",
    "monitor_runs",
    "relay_messages",
    "routing_log",
    "routing",
    "packages",
    "deployments",
    "gate_results",
    "stage_transitions",
    "model_versions",
    "pipeline_runs",
    "model_seq",
]


def url() -> str:
    u = os.environ.get("PTP_DATABASE_URL")
    if not u:
        raise RuntimeError("PTP_DATABASE_URL is not set")
    return u


class Registry:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or url()
        self._conn: psycopg.Connection[dict[str, Any]] | None = None

    def conn(self) -> psycopg.Connection[dict[str, Any]]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        c = self.conn()
        with c.transaction():
            yield c

    def bootstrap(self) -> None:
        sql = resources.files("ptp.registry").joinpath("schema.sql").read_text()
        self.conn().execute(sql)

    def reset(self) -> None:
        self.bootstrap()
        self.conn().execute("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")

    def q(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        cur = self.conn().execute(sql, args)
        return list(cur.fetchall()) if cur.description else []

    def one(self, sql: str, *args: Any) -> dict[str, Any] | None:
        rows = self.q(sql, *args)
        return rows[0] if rows else None

    # ---- pipeline runs ------------------------------------------------------------------------------------------

    def start_run(self, run_id: str, kind: str, engine: str, params: dict[str, Any]) -> None:
        self.conn().execute(
            "INSERT INTO pipeline_runs(run_id, kind, engine, started_at, state, params) VALUES (%s,%s,%s,%s,'RUNNING',%s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (run_id, kind, engine, clock.now(), Jsonb(params)),
        )

    def finish_run(
        self, run_id: str, state: str, outcome: str, version: str | None = None, error: str | None = None
    ) -> None:
        self.conn().execute(
            "UPDATE pipeline_runs SET state=%s, outcome=%s, version=COALESCE(%s, version), finished_at=%s, error=%s WHERE run_id=%s",
            (state, outcome, version, clock.now(), error, run_id),
        )

    def run(self, run_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM pipeline_runs WHERE run_id=%s", run_id)

    # ---- versions -----------------------------------------------------------------------------------------------

    def next_version(self, kind: str) -> str:
        prefix = {"ranker": "r", "tower": "t"}[kind]
        row = self.one(
            "INSERT INTO model_seq(kind, next) VALUES (%s, 2) ON CONFLICT (kind) DO UPDATE SET next = model_seq.next + 1 "
            "RETURNING next - 1 AS n",
            kind,
        )
        assert row
        return f"{prefix}{int(row['n']):04d}"

    def register(self, record: dict[str, Any]) -> None:
        cols = [
            "version", "kind", "created_at", "pipeline_run", "code_commit", "data_snapshot", "training", "features",
            "artifacts", "metrics", "dataset_profile", "model_card", "owner", "status",
        ]  # fmt: skip
        vals = [Jsonb(_jsonable(record[c])) if isinstance(record[c], (dict, list)) else record[c] for c in cols]
        self.conn().execute(
            f"INSERT INTO model_versions({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})", vals
        )
        self.conn().execute(
            "INSERT INTO stage_transitions(version, from_status, to_status, at, actor, reason) VALUES (%s, NULL, %s, %s, %s, %s)",
            (
                record["version"],
                record["status"],
                record["created_at"],
                "pipeline",
                f"registered by run {record['pipeline_run']}",
            ),
        )

    def version(self, version: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM model_versions WHERE version=%s", version)

    def versions(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            return self.q("SELECT * FROM model_versions WHERE kind=%s ORDER BY created_at, version", kind)
        return self.q("SELECT * FROM model_versions ORDER BY created_at, version")

    def transition(self, version: str, to: str, actor: str, reason: str) -> None:
        assert to in STATUSES, to
        with self.tx() as c:
            row = c.execute("SELECT status FROM model_versions WHERE version=%s FOR UPDATE", (version,)).fetchone()
            if row is None:
                raise KeyError(version)
            c.execute("UPDATE model_versions SET status=%s WHERE version=%s", (to, version))
            c.execute(
                "INSERT INTO stage_transitions(version, from_status, to_status, at, actor, reason) VALUES (%s,%s,%s,%s,%s,%s)",
                (version, row["status"], to, clock.now(), actor, reason),
            )

    def transitions(self, version: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM stage_transitions WHERE version=%s ORDER BY id", version)

    def production(self, kind: str) -> dict[str, Any] | None:
        return self.one(
            "SELECT * FROM model_versions WHERE kind=%s AND status='production' ORDER BY created_at DESC LIMIT 1", kind
        )

    # ---- gates --------------------------------------------------------------------------------------------------

    def record_gates(self, version: str | None, run_id: str | None, stage: str, results: list[Any]) -> None:
        now = clock.now()
        for r in results:
            self.conn().execute(
                "INSERT INTO gate_results(version, pipeline_run, stage, gate, passed, severity, blocking, waived_by, summary, evidence, at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    version,
                    run_id,
                    stage,
                    r.gate,
                    r.passed,
                    r.severity,
                    r.blocking,
                    r.waived_by,
                    r.summary,
                    Jsonb(_jsonable(r.evidence)),
                    now,
                ),
            )

    def gates(self, version: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        if version:
            return self.q(
                "SELECT * FROM gate_results WHERE version=%s OR pipeline_run=(SELECT pipeline_run FROM model_versions WHERE version=%s) ORDER BY id",
                version,
                version,
            )
        return self.q("SELECT * FROM gate_results WHERE pipeline_run=%s ORDER BY id", run_id)

    # ---- deployments, packages, routing --------------------------------------------------------------------------

    def open_deployment(self, version: str, trigger: str) -> tuple[int, bool]:
        """(id, created). An open deployment for the version is reused: a repeated dispatch never starts a second."""
        existing = self.one(
            "SELECT id FROM deployments WHERE version=%s AND state IN ('running','canary','waiting_embeddings')",
            version,
        )
        if existing:
            return int(existing["id"]), False
        row = self.one(
            "INSERT INTO deployments(version, trigger, started_at, state) VALUES (%s,%s,%s,'running') RETURNING id",
            version,
            trigger,
            clock.now(),
        )
        assert row
        return int(row["id"]), True

    def deployment(self, dep_id: int) -> dict[str, Any] | None:
        return self.one("SELECT * FROM deployments WHERE id=%s", dep_id)

    def deployments(self, version: str | None = None) -> list[dict[str, Any]]:
        if version:
            return self.q("SELECT * FROM deployments WHERE version=%s ORDER BY id", version)
        return self.q("SELECT * FROM deployments ORDER BY id")

    def step(self, dep_id: int, name: str, ok: bool, detail: str, data: dict[str, Any] | None = None) -> None:
        entry = {"step": name, "ok": ok, "detail": detail, "at": clock.now(), "data": _jsonable(data or {})}
        self.conn().execute(
            "UPDATE deployments SET steps = steps || %s::jsonb WHERE id=%s", (json.dumps([entry]), dep_id)
        )

    def set_deployment(self, dep_id: int, state: str, error: str | None = None, package_sha: str | None = None) -> None:
        finished = clock.now() if state in ("promoted", "rolled_back", "failed") else None
        self.conn().execute(
            "UPDATE deployments SET state=%s, error=COALESCE(%s, error), package_sha=COALESCE(%s, package_sha), "
            "finished_at=COALESCE(%s, finished_at) WHERE id=%s",
            (state, error, package_sha, finished, dep_id),
        )

    def open_deployments(self, state: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM deployments WHERE state=%s ORDER BY id", state)

    def save_package(self, env: str, generation: int, sha: str, profiles: list[str], zip_bytes: bytes) -> None:
        self.conn().execute(
            "INSERT INTO packages(env, generation, deployed_at, sha256, profiles, zip) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (env, generation) DO UPDATE SET deployed_at=EXCLUDED.deployed_at, sha256=EXCLUDED.sha256, "
            "profiles=EXCLUDED.profiles, zip=EXCLUDED.zip",
            (env, generation, clock.now(), sha, Jsonb(profiles), zip_bytes),
        )

    def active_package(self, env: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM packages WHERE env=%s ORDER BY deployed_at DESC, generation DESC LIMIT 1", env)

    def routing(self, surface: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM routing WHERE surface=%s", surface)

    def all_routing(self) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM routing ORDER BY surface")

    def set_routing(
        self, surface: str, production: str, canary: str | None, share: float, reason: str, previous: str | None = None
    ) -> None:
        now = clock.now()
        with self.tx() as c:
            old: dict[str, Any] | None = c.execute(
                "SELECT * FROM routing WHERE surface=%s FOR UPDATE", (surface,)
            ).fetchone()  # type: ignore[assignment]
            prev = previous if previous is not None else (old["previous"] if old else None)
            since = (old["canary_since"] if old and old["canary"] == canary else now) if canary else None
            c.execute(
                "INSERT INTO routing(surface, production, canary, share, canary_since, previous, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (surface) DO UPDATE SET production=EXCLUDED.production, canary=EXCLUDED.canary, share=EXCLUDED.share, "
                "canary_since=EXCLUDED.canary_since, previous=EXCLUDED.previous, updated_at=EXCLUDED.updated_at",
                (surface, production, canary, share, since, prev, now),
            )
            c.execute(
                "INSERT INTO routing_log(surface, production, canary, share, at, reason) VALUES (%s,%s,%s,%s,%s,%s)",
                (surface, production, canary, share, now, reason),
            )

    def routing_log(self) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM routing_log ORDER BY id")

    # ---- relay --------------------------------------------------------------------------------------------------

    def relay_seen(self, message_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM relay_messages WHERE message_id=%s", message_id)

    def relay_run_dispatched(self, run_id: str) -> bool:
        return (
            self.one("SELECT 1 AS x FROM relay_messages WHERE pipeline_run=%s AND action='dispatched'", run_id)
            is not None
        )

    def relay_record(self, message_id: str, run_id: str | None, event: dict[str, Any], action: str) -> None:
        now = clock.now()
        self.conn().execute(
            "INSERT INTO relay_messages(message_id, received_at, pipeline_run, event, action, dispatched_at) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (message_id) DO NOTHING",
            (message_id, now, run_id, Jsonb(event), action, now if action == "dispatched" else None),
        )

    def relay_messages(self) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM relay_messages ORDER BY received_at, message_id")

    # ---- quarantine ---------------------------------------------------------------------------------------------

    def quarantine(self, table: str, day: Any, market: str | None, reason: str, by: str) -> None:
        self.conn().execute(
            "INSERT INTO quarantine(table_name, day, market, reason, by_whom, at) VALUES (%s,%s,%s,%s,%s,%s)",
            (table, day, market, reason, by, clock.now()),
        )

    def quarantined(self, table: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM quarantine WHERE table_name=%s ORDER BY id", table)

    # ---- monitoring ---------------------------------------------------------------------------------------------

    def record_monitor(self, surface: str, version: str | None, healthy: bool, result: dict[str, Any]) -> None:
        self.conn().execute(
            "INSERT INTO monitor_runs(surface, version, at, healthy, result) VALUES (%s,%s,%s,%s,%s)",
            (surface, version, clock.now(), healthy, Jsonb(_jsonable(result))),
        )

    def monitors(self, surface: str | None = None) -> list[dict[str, Any]]:
        if surface:
            return self.q("SELECT * FROM monitor_runs WHERE surface=%s ORDER BY id", surface)
        return self.q("SELECT * FROM monitor_runs ORDER BY id")


def _jsonable(obj: Any) -> Any:
    return _finite(json.loads(json.dumps(obj, default=_default)))


def _finite(obj: Any) -> Any:
    """JSON has no Infinity or NaN; PostgreSQL refuses them. Keep them readable as strings."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:  # pragma: no cover
        pass
    return str(o)

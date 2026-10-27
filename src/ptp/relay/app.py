"""The relay: Pub/Sub push in, GitHub repository_dispatch out.

Pub/Sub delivers at least once, so the same message can arrive twice, and a pipeline run's event can be published
twice (a retried exit handler). The relay records each message id and each dispatched run in PostgreSQL and
dispatches a run once. It records a message only after GitHub accepted the dispatch: if GitHub fails, the relay
answers 503 and Pub/Sub redelivers. If the relay dies between GitHub's 204 and the record, the redelivery
dispatches again; the deploy workflow is idempotent per model version, so that costs a no-op run, never a second
deploy.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import jwt
from fastapi import FastAPI, Header, HTTPException, Request, Response

from .. import config, http
from ..events import PipelineEvent, decode_push
from ..registry.store import Registry

_jwks: dict[str, Any] = {"at": 0.0, "keys": {}}


def _keys(url: str) -> dict[str, Any]:
    if time.time() - _jwks["at"] > 300 or not _jwks["keys"]:
        with http.client(timeout=10) as c:
            r = c.get(url)
        r.raise_for_status()
        _jwks["keys"] = {k["kid"]: jwt.PyJWK(k) for k in r.json()["keys"]}
        _jwks["at"] = time.time()
    return _jwks["keys"]


def verify(token: str) -> dict[str, Any]:
    cfg = config.load("relay")["auth"]
    try:
        header = jwt.get_unverified_header(token)
        keys = _keys(cfg["jwks_url"])
        if header.get("kid") not in keys:
            _jwks["at"] = 0.0  # keys rotate: refetch once
            keys = _keys(cfg["jwks_url"])
        key = keys[header["kid"]]
        claims = jwt.decode(
            token,
            key=key.key,
            algorithms=["RS256"],
            audience=cfg["audience"],
            options={"require": ["exp", "iat", "iss", "aud"]},
        )
    except (jwt.PyJWTError, KeyError) as e:
        raise HTTPException(status_code=401, detail=f"invalid push token: {e}") from e
    if claims.get("iss") not in cfg["issuers"]:
        raise HTTPException(status_code=401, detail="wrong issuer")
    if claims.get("email") != cfg["service_account"] or not claims.get("email_verified"):
        raise HTTPException(status_code=403, detail="push token is not from the subscription's service account")
    return claims


def dispatch(event: PipelineEvent) -> int:
    t = config.load("targets")["github"]
    token = os.environ.get(config.load("relay")["dispatch"]["token_env"], "")
    payload = {
        "event_type": t["event_type"],
        "client_payload": {
            "version": event.version,
            "kind": event.kind,
            "run_id": event.run_id,
            "outcome": event.outcome,
            "artifact_key": (event.artifact or {}).get("key"),
            "sha256": (event.artifact or {}).get("sha256"),
            # extra keys for a dispatch target that needs them (the CI loop passes env and the bundle's run id)
            **json.loads(os.environ.get("PTP_DISPATCH_EXTRA", "{}")),
        },
    }
    with http.client(timeout=20) as c:
        r = c.post(
            f"{t['api'].rstrip('/')}/repos/{t['repository']}/dispatches",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
    return r.status_code


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="ml-pipeline-relay", docs_url=None, redoc_url=None)
    reg = registry or Registry()
    reg.bootstrap()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/pubsub/push", status_code=204)
    async def push(request: Request, authorization: str = Header(default="")) -> Response:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing push token")
        verify(authorization.removeprefix("Bearer "))
        try:
            envelope = json.loads(await request.body())
            message_id, event = decode_push(envelope)
        except (ValueError, KeyError) as e:
            # a message that can never be parsed must not be redelivered forever: acknowledge and record it
            reg.relay_record(f"bad-{hash(await request.body())}", None, {"error": str(e)}, "ignored")
            return Response(status_code=204)
        with reg.tx() as c:
            c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (event.run_id,))
            if reg.relay_seen(message_id):
                return Response(status_code=204)
            if event.outcome not in config.load("relay")["dispatch"]["outcomes"] or event.state != "SUCCEEDED":
                reg.relay_record(message_id, event.run_id, event.model_dump(), "ignored")
                return Response(status_code=204)
            if reg.relay_run_dispatched(event.run_id):
                reg.relay_record(message_id, event.run_id, event.model_dump(), "duplicate_run")
                return Response(status_code=204)
            status = dispatch(event)
            if status != 204:
                # not recorded: Pub/Sub will redeliver with backoff
                raise HTTPException(status_code=503, detail=f"GitHub answered {status}")
            reg.relay_record(message_id, event.run_id, event.model_dump(), "dispatched")
        return Response(status_code=204)

    return app

"""Pipeline events: every run ends with an exit handler that publishes its final state to a Pub/Sub topic, through
the Pub/Sub REST publish API. The relay receives them by push subscription."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from . import clock, config, http


class PipelineEvent(BaseModel):
    run_id: str
    pipeline: str  # ct-ranker, ct-tower, item-embeddings
    kind: str  # ranker, tower
    state: str  # SUCCEEDED, FAILED
    outcome: str  # candidate, blocked, no_data, error, embeddings_ready, embeddings_failed
    version: str | None = None
    artifact: dict[str, str] | None = None
    finished_at: int = Field(default_factory=clock.now)
    detail: dict[str, Any] = Field(default_factory=dict)


def _publish_url(topic: str) -> str:
    api = os.environ.get("PTP_PUBSUB_API", "https://pubsub.googleapis.com").rstrip("/")
    return f"{api}/v1/{topic}:publish"


def _token() -> str | None:
    tok = os.environ.get("PTP_PUBSUB_TOKEN")
    if tok:
        return tok
    if os.environ.get("PTP_PUBSUB_API"):
        return None  # an emulator
    try:  # on GCP: the workload's own credentials
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/pubsub"])
        creds.refresh(google.auth.transport.requests.Request())
        return str(creds.token)
    except Exception as e:  # pragma: no cover - only on GCP
        raise RuntimeError(f"no credentials to publish pipeline events: {e}") from e


def publish(event: PipelineEvent) -> str:
    cfg = config.load("pipeline")["events"]
    data = base64.b64encode(event.model_dump_json().encode()).decode()
    body = {
        "messages": [
            {
                "data": data,
                "attributes": {"run_id": event.run_id, "kind": event.kind, "outcome": event.outcome},
                "orderingKey": event.kind,
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with http.client() as c:
        r = c.post(_publish_url(cfg["topic"]), json=body, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"publish failed ({r.status_code}): {r.text[:200]}")
    return str(r.json()["messageIds"][0])


def decode_push(envelope: dict[str, Any]) -> tuple[str, PipelineEvent]:
    msg = envelope["message"]
    raw = base64.b64decode(msg["data"]).decode()
    return str(msg.get("messageId") or msg.get("message_id")), PipelineEvent.model_validate(json.loads(raw))

"""The relay: authenticated Pub/Sub push in, exactly one repository_dispatch per successful candidate run out."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ptp import config
from ptp import http as ptp_http
from ptp.events import PipelineEvent
from ptp.relay import app as relay_app
from ptpsim.emulators.github import GitHubEmulator
from ptpsim.emulators.pubsub import PubSubEmulator

pytestmark = pytest.mark.db


@pytest.fixture
def wired(registry: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    auth = config.load("relay")["auth"]
    ps = PubSubEmulator(
        push_url="http://relay.local/pubsub/push", audience=auth["audience"], service_account=auth["service_account"]
    )
    gh = GitHubEmulator(config.root(), config.load("targets")["github"]["repository"], "tok", lambda argv: 0)

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pubsub.local":
            return ps.handle(request)
        if request.url.host == "github.local":
            return gh.handle(request)
        return httpx.Response(502)

    ptp_http.install(httpx.MockTransport(route))
    monkeypatch.setenv("PTP_RELAY_JWKS_URL", "http://pubsub.local/oauth2/v3/certs")
    monkeypatch.setenv("PTP_GITHUB_API", "http://github.local")
    monkeypatch.setenv("PTP_GITHUB_TOKEN", "tok")
    relay_app._jwks.update(at=0.0, keys={})
    client = TestClient(relay_app.create_app(registry))

    class W:
        pass

    w = W()
    w.ps, w.gh, w.client, w.reg = ps, gh, client, registry
    yield w
    ptp_http.install(None)


def _envelope(event: PipelineEvent, message_id: str = "m1") -> dict[str, Any]:
    return {
        "message": {
            "data": base64.b64encode(event.model_dump_json().encode()).decode(),
            "messageId": message_id,
            "attributes": {},
        },
        "subscription": "s",
    }


def _event(run_id: str = "ct-ranker-1", outcome: str = "candidate", state: str = "SUCCEEDED") -> PipelineEvent:
    return PipelineEvent(
        run_id=run_id,
        pipeline="ct-ranker",
        kind="ranker",
        state=state,
        outcome=outcome,
        version="r0007",
        artifact={"key": "k", "sha256": "s"},
    )


def _push(w: Any, env: dict[str, Any], token: str | None = None) -> int:
    tok = token if token is not None else w.ps.token()
    return w.client.post(
        "/pubsub/push", content=json.dumps(env), headers={"Authorization": f"Bearer {tok}"}
    ).status_code


def test_a_candidate_is_dispatched_once_with_the_payload_the_workflow_reads(wired: Any) -> None:
    assert _push(wired, _envelope(_event())) == 204
    assert len(wired.gh.dispatches) == 1
    body = wired.gh.dispatches[0]
    assert body["event_type"] == "model-ready"
    assert body["client_payload"]["version"] == "r0007" and len(body["client_payload"]) <= 10
    assert [m["action"] for m in wired.reg.relay_messages()] == ["dispatched"]


def test_redelivery_of_the_same_message_and_a_second_message_for_the_same_run_do_not_dispatch_again(wired: Any) -> None:
    assert _push(wired, _envelope(_event(), "m1")) == 204
    assert _push(wired, _envelope(_event(), "m1")) == 204
    assert _push(wired, _envelope(_event(), "m2")) == 204
    assert len(wired.gh.dispatches) == 1
    assert sorted(m["action"] for m in wired.reg.relay_messages()) == ["dispatched", "duplicate_run"]


def test_blocked_and_failed_runs_are_recorded_not_dispatched(wired: Any) -> None:
    assert _push(wired, _envelope(_event("r1", outcome="blocked"), "a")) == 204
    assert _push(wired, _envelope(_event("r2", state="FAILED", outcome="error"), "b")) == 204
    assert wired.gh.dispatches == []
    assert {m["action"] for m in wired.reg.relay_messages()} == {"ignored"}


@pytest.mark.parametrize(
    "token_kwargs, status",
    [
        ({"email": "someone-else@example.iam.gserviceaccount.com"}, 403),
        ({"audience": "https://other.example/push"}, 401),
        ({"issuer": "https://evil.example"}, 401),
        ({"expires_in": -10}, 401),
    ],
)
def test_pushes_that_are_not_from_the_subscription_are_refused(
    wired: Any, token_kwargs: dict[str, Any], status: int
) -> None:
    assert _push(wired, _envelope(_event()), wired.ps.token(**token_kwargs)) == status
    assert wired.gh.dispatches == []


def test_a_token_signed_by_another_key_is_refused(wired: Any) -> None:
    other = PubSubEmulator(push_url="", audience=wired.ps.audience, service_account=wired.ps.service_account)
    assert _push(wired, _envelope(_event()), other.token()) == 401


def test_github_failing_makes_pubsub_retry_and_the_retry_dispatches(wired: Any) -> None:
    wired.gh.fail_next = 1
    assert _push(wired, _envelope(_event())) == 503
    assert wired.reg.relay_messages() == []
    assert _push(wired, _envelope(_event())) == 204
    assert len(wired.gh.dispatches) == 1


def test_a_message_nobody_can_parse_is_acknowledged_so_it_is_not_redelivered_forever(wired: Any) -> None:
    env = {"message": {"data": base64.b64encode(b"not json").decode(), "messageId": "x"}}
    assert _push(wired, env) == 204
    assert [m["action"] for m in wired.reg.relay_messages()] == ["ignored"]


def test_end_to_end_through_the_pubsub_rest_api(wired: Any) -> None:
    """events.publish -> Pub/Sub REST publish -> push to the relay -> dispatch."""
    from ptp import events

    wired.ps.sender = lambda url, env, headers: (
        wired.client.post("/pubsub/push", content=json.dumps(env), headers=headers).status_code
    )
    import os

    os.environ["PTP_PUBSUB_API"] = "http://pubsub.local"
    try:
        events.publish(_event("ct-ranker-9"))
        assert wired.ps.flush() == 1
    finally:
        os.environ.pop("PTP_PUBSUB_API")
    assert len(wired.gh.dispatches) == 1

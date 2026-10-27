"""The emulators as real HTTP servers, for CI: the Pub/Sub emulator (REST publish, JWKS, asynchronous push with
signed tokens) in front of the real relay, and a GitHub stand-in that records dispatches for the kind test."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from .emulators.pubsub import PubSubEmulator


def serve(handler_fn: Any, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _do(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            req = httpx.Request(
                self.command,
                f"http://{self.headers.get('host', 'localhost')}{self.path}",
                headers=dict(self.headers),
                content=body,
            )
            resp: httpx.Response = handler_fn(req)
            self.send_response(resp.status_code)
            data = resp.content
            self.send_header("content-type", resp.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = _do

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class GitHubStandIn:
    """Records repository_dispatch calls; GET /dispatches returns them (for assertions)."""

    def __init__(self, ps: PubSubEmulator):
        self.ps = ps
        self.dispatches: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/dispatches") and request.method == "POST":
            self.dispatches.append(json.loads(request.content))
            return httpx.Response(204)
        if request.url.path == "/dispatches":
            return httpx.Response(200, json=self.dispatches)
        if request.url.path == "/oauth2/v3/certs":
            return httpx.Response(200, json=self.ps.jwks())
        if request.url.path == "/token":
            return httpx.Response(200, json={"token": self.ps.token()})
        return httpx.Response(404)


def run_pubsub(port: int, push_url: str, audience: str, service_account: str) -> None:
    ps = PubSubEmulator(push_url=push_url, audience=audience, service_account=service_account)

    def send(url: str, envelope: dict[str, Any], headers: dict[str, str]) -> int:
        try:
            return httpx.post(url, json=envelope, headers=headers, timeout=30).status_code
        except httpx.HTTPError:
            return 503

    ps.sender = send
    serve(ps.handle, port)
    while True:  # push asynchronously, retrying what failed, as Pub/Sub does
        ps.flush()
        if ps._undelivered:
            time.sleep(2)
            ps.redeliver()
        time.sleep(0.5)


def run_github_stand_in(port: int, audience: str, service_account: str) -> None:
    ps = PubSubEmulator(push_url="", audience=audience, service_account=service_account)
    gh = GitHubStandIn(ps)
    serve(gh.handle, port)
    while True:
        time.sleep(3600)

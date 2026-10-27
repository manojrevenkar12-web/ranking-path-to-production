"""HTTP clients for everything the platform calls. A test or a simulated season can install a transport that routes
hosts to in-process emulators; production code never knows the difference."""

from __future__ import annotations

import httpx

_transport: httpx.BaseTransport | None = None


def install(transport: httpx.BaseTransport | None) -> None:
    global _transport
    _transport = transport


def client(timeout: float = 30.0) -> httpx.Client:
    if _transport is not None:
        return httpx.Client(transport=_transport, timeout=timeout)
    return httpx.Client(timeout=timeout)

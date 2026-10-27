"""A Pub/Sub emulator for the REST publish API and push subscriptions with OIDC authentication.

It signs push tokens with its own RSA key and serves the matching JWKS, so the relay verifies tokens exactly as it
verifies Google's. Delivery is at least once: a failed push is retried with backoff, and `duplicate_every` makes
it deliver some messages twice, as Pub/Sub can.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm


@dataclass
class Delivery:
    message_id: str
    attempt: int
    status: int


@dataclass
class PubSubEmulator:
    push_url: str
    audience: str
    service_account: str
    sender: Callable[[str, dict[str, Any], dict[str, str]], int] | None = None
    duplicate_every: int = 0
    max_attempts: int = 5
    kid: str = "emulator-1"
    deliveries: list[Delivery] = field(default_factory=list)
    published: list[dict[str, Any]] = field(default_factory=list)
    _count: int = 0
    _undelivered: list[dict[str, Any]] = field(default_factory=list)
    _queue: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwks(self) -> dict[str, Any]:
        jwk = json.loads(RSAAlgorithm.to_jwk(self.key.public_key()))
        jwk.update(kid=self.kid, alg="RS256", use="sig")
        return {"keys": [jwk]}

    def token(
        self,
        email: str | None = None,
        audience: str | None = None,
        issuer: str = "https://accounts.google.com",
        expires_in: int = 3600,
    ) -> str:
        now = int(time.time())
        claims = {
            "iss": issuer,
            "aud": audience or self.audience,
            "email": email or self.service_account,
            "email_verified": True,
            "iat": now,
            "exp": now + expires_in,
            "sub": "1234567890",
        }
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": self.kid})

    def envelope(self, msg: dict[str, Any]) -> dict[str, Any]:
        return {"message": msg, "subscription": "projects/vinted-buyer-ml-dev/subscriptions/ml-relay-push"}

    def push(self, msg: dict[str, Any]) -> bool:
        assert self.sender is not None
        for attempt in range(1, self.max_attempts + 1):
            status = self.sender(self.push_url, self.envelope(msg), {"Authorization": f"Bearer {self.token()}"})
            self.deliveries.append(Delivery(msg["messageId"], attempt, status))
            if 200 <= status < 300:
                return True
        self._undelivered.append(msg)
        return False

    def flush(self) -> int:
        """Deliver everything published so far (push is asynchronous to the publisher)."""
        n = 0
        while self._queue:
            self.push(self._queue.pop(0))
            n += 1
        return n

    def redeliver(self) -> None:
        pending, self._undelivered = self._undelivered, []
        for msg in pending:
            self.push(msg)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":publish") and request.method == "POST":
            body = json.loads(request.content)
            ids = []
            for m in body["messages"]:
                msg = {
                    "data": m["data"],
                    "attributes": m.get("attributes", {}),
                    "messageId": f"{10_000_000 + self._count:d}",
                    "publishTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.published.append(msg)
                ids.append(msg["messageId"])
                self._queue.append(msg)
                self._count += 1
                if self.duplicate_every and self._count % self.duplicate_every == 0:
                    self._queue.append(msg)  # at-least-once: the same message again
            return httpx.Response(200, json={"messageIds": ids})
        if request.url.path == "/oauth2/v3/certs":
            return httpx.Response(200, json=self.jwks())
        return httpx.Response(404)

    @staticmethod
    def decode(msg: dict[str, Any]) -> dict[str, Any]:
        return json.loads(base64.b64decode(msg["data"]))

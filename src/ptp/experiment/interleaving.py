"""Team-draft interleaving, for canaries.

Offline metrics on logged impressions can only re-order what the production model chose to show, so they see a
fraction of a new model's effect (docs/RESEARCH.md has the measurement). An A/B test sees all of it but needs far
more traffic than a canary gets. Interleaving shows each member one list merged from both models, credits every
interaction to the model that contributed that item, and asks which model members preferred: the same sessions
answer for both models, which is why it needs one to two orders of magnitude fewer sessions than an A/B test
(Chapelle et al. 2012; Radlinski et al. 2008).

The search and homepage services call `team_draft` for sessions the routing assigns to the canary; the canary
check calls `preference` on the session stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CANDIDATE, PRODUCTION = "C", "P"


def team_draft(
    production: list[str], candidate: list[str], k: int, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    """Merged list of up to k ids and the team ('C' or 'P') of each position."""
    merged: list[str] = []
    teams: list[str] = []
    seen: set[str] = set()
    picks = {CANDIDATE: 0, PRODUCTION: 0}
    lists = {CANDIDATE: candidate, PRODUCTION: production}
    pos = {CANDIDATE: 0, PRODUCTION: 0}
    while len(merged) < k:
        if picks[CANDIDATE] < picks[PRODUCTION] or (picks[CANDIDATE] == picks[PRODUCTION] and rng.random() < 0.5):
            team = CANDIDATE
        else:
            team = PRODUCTION
        other = PRODUCTION if team == CANDIDATE else CANDIDATE
        chosen = None
        for t in (team, other):
            lst = lists[t]
            while pos[t] < len(lst) and lst[pos[t]] in seen:
                pos[t] += 1
            if pos[t] < len(lst):
                chosen = (t, lst[pos[t]])
                break
        if chosen is None:
            break
        t, item = chosen
        merged.append(item)
        teams.append(t)
        seen.add(item)
        picks[t] += 1
    return merged, teams


def credit(
    teams: list[str], clicked: list[bool], favourited: list[bool], purchased: list[bool], weights: dict[str, float]
) -> tuple[float, float]:
    """(candidate credit, production credit) for one interleaved session."""
    c = p = 0.0
    for t, cl, fv, pu in zip(teams, clicked, favourited, purchased, strict=True):
        v = weights["click"] * cl + weights["favourite"] * fv + weights["purchase"] * pu
        if t == CANDIDATE:
            c += v
        else:
            p += v
    return c, p


@dataclass
class Preference:
    sessions: int
    wins: int
    losses: int
    ties: int

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_share(self) -> float:
        return self.wins / self.decided if self.decided else 0.5

    def interval(self, confidence: float) -> tuple[float, float]:
        """Wilson interval of the candidate's share of decided sessions."""
        n = self.decided
        if n == 0:
            return 0.0, 1.0
        z = _z(confidence)
        p = self.win_share
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

    def to_dict(self, confidence: float) -> dict[str, Any]:
        lo, hi = self.interval(confidence)
        return {
            "sessions": self.sessions,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_share": round(self.win_share, 4),
            "low": round(lo, 4),
            "high": round(hi, 4),
        }


def preference(rows: Any) -> Preference:
    """rows: sessions with credit_candidate and credit_production."""
    c = np.asarray(rows["credit_candidate"], dtype=float)
    p = np.asarray(rows["credit_production"], dtype=float)
    return Preference(len(c), int(np.sum(c > p)), int(np.sum(c < p)), int(np.sum(c == p)))


def _z(confidence: float) -> float:
    lo, hi = 0.0, 10.0
    two_sided = 1 - (1 - confidence) / 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < two_sided:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

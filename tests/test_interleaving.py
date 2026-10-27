from __future__ import annotations

import numpy as np
import pandas as pd

from ptp.experiment import interleaving as IL


def test_team_draft_merges_without_duplicates_and_balances_teams() -> None:
    rng = np.random.default_rng(0)
    prod = [f"p{i}" for i in range(20)]
    cand = [f"p{i}" for i in range(10, 30)]
    for _ in range(50):
        merged, teams = IL.team_draft(prod, cand, 20, rng)
        assert len(merged) == 20 and len(set(merged)) == 20
        assert abs(teams.count("C") - teams.count("P")) <= 1
        # each team's items keep the order its model gave them
        for team, lst in (("C", cand), ("P", prod)):
            mine = [m for m, t in zip(merged, teams, strict=True) if t == team]
            assert mine == sorted(mine, key=lst.index)


def test_team_draft_fills_from_the_other_list_when_one_runs_out() -> None:
    merged, teams = IL.team_draft(["a", "b", "c"], [], 3, np.random.default_rng(1))
    assert merged == ["a", "b", "c"] and set(teams) == {"P"}


def test_credit_goes_to_the_contributing_team() -> None:
    w = {"click": 1, "favourite": 2, "purchase": 3}
    c, p = IL.credit(["C", "P", "C"], [True, True, False], [True, False, False], [False, False, False], w)
    assert (c, p) == (3.0, 1.0)


def test_preference_detects_a_better_candidate_and_its_interval() -> None:
    rng = np.random.default_rng(2)
    rows = pd.DataFrame({"credit_candidate": rng.poisson(1.3, 600), "credit_production": rng.poisson(1.0, 600)})
    pref = IL.preference(rows)
    lo, hi = pref.interval(0.95)
    assert pref.win_share > 0.5 and lo > 0.5 and lo < pref.win_share < hi
    tie = IL.preference(pd.DataFrame({"credit_candidate": [1, 1], "credit_production": [1, 1]}))
    assert tie.decided == 0 and tie.interval(0.95) == (0.0, 1.0)

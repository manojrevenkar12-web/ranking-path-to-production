"""Why the offline gate uses unnormalised IPS-DCG.

Logs come from the production model: it ordered the results, and clicks are position-biased. On such logs, per-query
NDCG prefers the logging model even with inverse-propensity weights, because with a single click the weight
cancels out of the ratio. Unnormalised IPS-DCG recovers the model that is actually better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ptp.data import propensity
from ptp.gates import model as M


def _logs(
    seed: int = 0, queries: int = 4000, k: int = 20, eta: float = 0.75
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    better, logging = [], []
    for q in range(queries):
        rel = rng.normal(0, 1, k)
        log_score = rel + rng.normal(0, 1.5, k)  # the production model: noisy
        cand_score = rel + rng.normal(0, 0.5, k)  # the candidate: closer to the truth
        order = np.argsort(-log_score)
        bias = 1.0 / np.power(np.arange(1, k + 1), eta)
        p_click = bias * (1 / (1 + np.exp(-(1.5 * rel[order] - 1.5))))
        clicked = rng.random(k) < p_click
        for pos, i in enumerate(order):
            rows.append(
                {
                    "session_id": f"q{q}",
                    "market": "FR",
                    "position": pos,
                    "label": int(clicked[pos]),
                    "clicked": bool(clicked[pos]),
                }
            )
            better.append(cand_score[i])
            logging.append(log_score[i])
    return pd.DataFrame(rows), np.array(better), np.array(logging)


def _ndcg(frame: pd.DataFrame, scores: np.ndarray, w: np.ndarray) -> float:
    df = frame.assign(s=scores, w=w)
    disc = 1 / np.log2(np.arange(2, 12))
    out = []
    for _, g in df.groupby("session_id", sort=False):
        gain = (2.0 ** g["label"].to_numpy() - 1) * g["w"].to_numpy()
        if gain.max() <= 0:
            continue
        o = np.argsort(-g["s"].to_numpy())[:10]
        ideal = np.sort(gain)[::-1][:10]
        out.append(float((gain[o] * disc).sum() / (ideal * disc).sum()))
    return float(np.mean(out))


def test_normalised_ndcg_prefers_the_logging_model_even_with_ips() -> None:
    frame, better, logging = _logs()
    prop = {"per_position": (np.arange(1, 21) ** -0.75).tolist(), "eta": 0.75}
    w = propensity.weights(frame["position"].to_numpy(), prop, 20)
    assert _ndcg(frame, logging, w) > _ndcg(frame, better, w)


def test_ips_dcg_recovers_the_better_model() -> None:
    frame, better, logging = _logs()
    prop = {"per_position": (np.arange(1, 21) ** -0.75).tolist(), "eta": 0.75}
    a = M.dcg_per_query(frame, logging, 10, prop)["value"].to_numpy()
    b = M.dcg_per_query(frame, better, 10, prop)["value"].to_numpy()
    r = M.paired_bootstrap(a, b, 300, 0.95)
    assert r["low"] > 0, r


def test_propensity_is_measured_from_shuffled_sessions() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for q in range(3000):
        for pos in range(20):
            p = 0.4 * (pos + 1) ** -0.7
            rows.append({"session_id": f"q{q}", "position": pos, "clicked": rng.random() < p, "arm": "explore"})
    est = propensity.estimate(pd.DataFrame(rows), min_sessions=100)
    assert est["source"] == "explore"
    assert abs(est["eta"] - 0.7) < 0.08


def test_too_little_exploration_is_reported_not_guessed() -> None:
    frame = pd.DataFrame({"session_id": ["a"], "position": [0], "clicked": [True], "arm": ["explore"]})
    assert propensity.estimate(frame, min_sessions=300)["source"] == "none"
    w = propensity.weights(np.array([0, 5]), None, 20)
    assert w.tolist() == [1.0, 1.0]

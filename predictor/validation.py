"""
validation.py — out-of-sample temporal back-test of the goals model.

Strict time split (train < BACKTEST_TRAIN_END, test on the BACKTEST window) so
nothing about the future leaks into training. We report three standard 1X2
metrics on held-out matches and compare Poisson-only vs Poisson+Elo:

  * accuracy  — fraction where the most-probable outcome was correct
  * log-loss  — mean negative log-probability of the true outcome
  * RPS       — Ranked Probability Score, the football standard for ordered
                W/D/L forecasts (lower is better; ~0.19-0.21 is competitive)

This back-test uses only pre-2026 data, so it is unaffected by entered results —
it characterises the *engine*, not the live forecast (that is accuracy.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from predictor.goals_model import fit_goals_model, wdl_probs


def _rps(probs, outcome_index: int) -> float:
    """Ranked Probability Score for one ordered (W,D,L) forecast."""
    oh = [0, 0, 0]
    oh[outcome_index] = 1
    cp, co = np.cumsum(probs), np.cumsum(oh)
    return float(np.sum((cp - co) ** 2) / (len(probs) - 1))


def evaluate(model: dict, test: pd.DataFrame) -> tuple[float, float, float]:
    """Return ``(accuracy, log_loss, RPS)`` of ``model`` on ``test`` matches."""
    acc, ll, rps = [], [], []
    for r in test.itertuples(index=False):
        p = list(wdl_probs(model, r.home_team, r.away_team,
                           r.elo_h_pre, r.elo_a_pre, 0 if r.neutral_b else 1))
        if r.home_score > r.away_score:
            y = 0
        elif r.home_score == r.away_score:
            y = 1
        else:
            y = 2
        acc.append(int(np.argmax(p) == y))
        ll.append(-np.log(max(p[y], 1e-15)))
        rps.append(_rps(p, y))
    return float(np.mean(acc)), float(np.mean(ll)), float(np.mean(rps))


def backtest(matches_all: pd.DataFrame) -> pd.DataFrame:
    """Compare Poisson-only vs Poisson+Elo on the held-out window.

    ``matches_all`` must carry ``elo_h_pre`` / ``elo_a_pre`` columns.
    """
    tr = matches_all[(matches_all.date >= config.TRAIN_SINCE)
                     & (matches_all.date < config.BACKTEST_TRAIN_END)]
    te = matches_all[(matches_all.date >= config.BACKTEST_TEST_START)
                     & (matches_all.date <= config.BACKTEST_TEST_END)]
    rows = []
    for label, use_elo in [("Poisson only", False), ("Poisson + Elo", True)]:
        m = fit_goals_model(tr, use_elo=use_elo, ref_date=config.BACKTEST_TRAIN_END)
        a, l, r = evaluate(m, te)
        rows.append({"model": label, "accuracy": round(a, 3),
                     "log_loss": round(l, 3), "RPS": round(r, 4)})
    return pd.DataFrame(rows).set_index("model")

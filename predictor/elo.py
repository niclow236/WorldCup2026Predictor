"""
elo.py — World Football Elo ratings.

Standard World-Football-Elo update with a goal-difference multiplier. Because
the K-factor schedule (constants.k_factor) gives World-Cup matches the heaviest
weight, any 2026 results appended to the training table move ratings the most —
which is exactly how later-round predictions sharpen as the tournament unfolds.

Shootouts count as draws for rating purposes (we never see a "penalty win" as a
goal margin here; the knockout simulator handles shootouts separately).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import k_factor

INITIAL_RATING = 1500.0
HOME_ADVANTAGE_ELO = 100.0   # rating points added to the home side on non-neutral pitches


def run_elo(matches: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute final Elo ratings and the pre-match ratings of both sides.

    Parameters
    ----------
    matches : DataFrame
        Chronologically sorted matches with columns ``home_team, away_team,
        home_score, away_score, neutral_b, tournament``.

    Returns
    -------
    ratings : dict
        team -> final Elo rating.
    pre_h, pre_a : np.ndarray
        Per-row home/away pre-match ratings (used as a goals-model feature so
        the model sees each match's strength gap at the time it was played).
    """
    R: dict[str, float] = {}
    n = len(matches)
    pre_h = np.empty(n)
    pre_a = np.empty(n)
    ht, at = matches["home_team"].values, matches["away_team"].values
    hs, as_ = matches["home_score"].values, matches["away_score"].values
    neu, tour = matches["neutral_b"].values, matches["tournament"].values

    for i in range(n):
        rh = R.get(ht[i], INITIAL_RATING)
        ra = R.get(at[i], INITIAL_RATING)
        pre_h[i], pre_a[i] = rh, ra
        ha = 0.0 if neu[i] else HOME_ADVANTAGE_ELO
        # Expected score for the home side.
        we = 1.0 / (1.0 + 10.0 ** (-((rh + ha) - ra) / 400.0))
        gd = abs(hs[i] - as_[i])
        # Actual result (1 win / 0.5 draw / 0 loss for the home side).
        w = 1.0 if hs[i] > as_[i] else (0.0 if hs[i] < as_[i] else 0.5)
        # Goal-difference multiplier (World Football Elo convention).
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        change = k_factor(tour[i]) * g * (w - we)
        R[ht[i]] = rh + change
        R[at[i]] = ra - change

    return R, pre_h, pre_a

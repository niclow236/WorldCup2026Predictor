"""
goals_model.py — time-weighted Poisson goals model (+ optional Elo feature).

Model
-----
    log λ = μ + attack_team + defence_opp + β_h · home
                + β_e · (Elo_team − Elo_opp) / 100

Fit by a sparse ``PoissonRegressor`` with sample weights that decay with a
two-year half-life, on matches since ``config.TRAIN_SINCE`` — including any
entered 2026 results, so attack/defence strengths adapt to World-Cup form.

The fitted model is a plain dict (``att``, ``dfn``, ``hc``, ``elo_beta``,
``intercept``) so it is trivially serialisable and cheap to pass around. Helper
functions turn it into expected goals (``xg``), win/draw/loss probabilities
(``wdl_probs``), and a full Dixon-Coles-corrected scoreline grid
(``predict_scoreline``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

import config


def build_long(d: pd.DataFrame):
    """Reshape match rows into one row per (team, opponent) scoring event.

    Each match contributes two rows: home-attacking-away and away-attacking-home.
    Returns parallel arrays ``(team, opp, goals, home, elo_diff, dates)``.
    """
    n = len(d)
    ta = np.concatenate([d["home_team"].values, d["away_team"].values])
    td = np.concatenate([d["away_team"].values, d["home_team"].values])
    goals = np.concatenate([d["home_score"].values, d["away_score"].values]).astype(float)
    home = np.concatenate([np.ones(n), np.zeros(n)])
    elo_diff = np.concatenate([
        d["elo_h_pre"].values - d["elo_a_pre"].values,
        d["elo_a_pre"].values - d["elo_h_pre"].values,
    ]) / 100.0
    dates = np.concatenate([d["date"].values, d["date"].values])
    return ta, td, goals, home, elo_diff, dates


def fit_goals_model(d: pd.DataFrame, use_elo: bool | None = None,
                    half_life: float | None = None, ref_date=None) -> dict:
    """Fit the time-weighted Poisson goals model and return it as a dict.

    Parameters
    ----------
    d : DataFrame
        Training matches (already filtered to the desired window) carrying the
        ``elo_h_pre`` / ``elo_a_pre`` columns produced by :func:`elo.run_elo`.
    use_elo : bool
        Include the Elo-difference feature (defaults to config.USE_ELO_FEATURE).
    half_life : float
        Sample-weight half-life in days (defaults to config.HALF_LIFE_DAYS).
    ref_date : datetime-like or None
        "Now" for the recency weighting; defaults to the latest training date.
        Pinned explicitly during back-testing so weights are reproducible.
    """
    use_elo = config.USE_ELO_FEATURE if use_elo is None else use_elo
    half_life = config.HALF_LIFE_DAYS if half_life is None else half_life

    ta, td, goals, home, elo_diff, dates = build_long(d)
    ea = OneHotEncoder(handle_unknown="ignore")
    Xa = ea.fit_transform(ta.reshape(-1, 1))
    ed = OneHotEncoder(handle_unknown="ignore")
    Xd = ed.fit_transform(td.reshape(-1, 1))
    blocks = [Xa, Xd, sparse.csr_matrix(home.reshape(-1, 1))]
    if use_elo:
        blocks.append(sparse.csr_matrix(elo_diff.reshape(-1, 1)))
    X = sparse.hstack(blocks).tocsr()

    ref = np.datetime64(ref_date) if ref_date is not None else dates.max()
    w = 0.5 ** ((ref - dates).astype("timedelta64[D]").astype(float) / half_life)

    reg = PoissonRegressor(alpha=config.POISSON_ALPHA, max_iter=config.POISSON_MAX_ITER)
    reg.fit(X, goals, sample_weight=w)

    na = len(ea.categories_[0])
    nd = len(ed.categories_[0])
    c = reg.coef_
    return {
        "att": dict(zip(ea.categories_[0], c[:na])),
        "dfn": dict(zip(ed.categories_[0], c[na:na + nd])),
        "hc": c[na + nd],
        "elo_beta": c[na + nd + 1] if use_elo else 0.0,
        "intercept": reg.intercept_,
    }


def xg(model: dict, attacker: str, defender: str,
       elo_att: float, elo_def: float, home: float) -> float:
    """Expected goals for ``attacker`` vs ``defender`` under the goals model."""
    return float(np.exp(
        model["intercept"]
        + model["att"].get(attacker, 0.0)
        + model["dfn"].get(defender, 0.0)
        + model["hc"] * home
        + model["elo_beta"] * ((elo_att - elo_def) / 100.0)
    ))


def wdl_probs(model: dict, a: str, b: str, ea: float, eb: float,
              home: float, maxg: int = 10):
    """Return ``(P(home win), P(draw), P(away win))`` from two Poisson means."""
    la = xg(model, a, b, ea, eb, home)
    lb = xg(model, b, a, eb, ea, 0)
    pa = poisson.pmf(np.arange(maxg + 1), la)
    pb = poisson.pmf(np.arange(maxg + 1), lb)
    M = np.outer(pa, pb)
    s = M.sum()
    return np.tril(M, -1).sum() / s, np.trace(M) / s, np.triu(M, 1).sum() / s


# ---------------------------------------------------------------------------
# Dixon-Coles low-score correction + full scoreline grid
# ---------------------------------------------------------------------------
def dc_tau(i: int, j: int, la: float, lb: float, rho: float) -> float:
    """Dixon-Coles (1997) correction factor for the four low-scoring cells."""
    if i == 0 and j == 0:
        return 1 - la * lb * rho
    if i == 0 and j == 1:
        return 1 + la * rho
    if i == 1 and j == 0:
        return 1 + lb * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def predict_scoreline(model: dict, ratings: dict, a: str, b: str,
                      home_a: int = 0, home_b: int = 0,
                      rho: float | None = None, maxg: int = 10):
    """Most-likely scoreline + outcome probabilities for ``a`` vs ``b``.

    Returns ``((goals_a, goals_b), (pH, pD, pA), (la, lb), M)`` where ``M`` is
    the normalised, Dixon-Coles-corrected joint score-probability matrix.
    """
    rho = config.DIXON_COLES_RHO if rho is None else rho
    la = xg(model, a, b, ratings.get(a, 1500), ratings.get(b, 1500), home_a)
    lb = xg(model, b, a, ratings.get(b, 1500), ratings.get(a, 1500), home_b)
    pa = poisson.pmf(np.arange(maxg + 1), la)
    pb = poisson.pmf(np.arange(maxg + 1), lb)
    M = np.outer(pa, pb)
    for i in range(2):
        for j in range(2):
            M[i, j] *= dc_tau(i, j, la, lb, rho)
    M /= M.sum()
    pH, pD, pA = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    i, j = np.unravel_index(np.argmax(M), M.shape)
    return (int(i), int(j)), (pH, pD, pA), (la, lb), M

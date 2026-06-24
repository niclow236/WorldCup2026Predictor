"""
accuracy.py — score the PRE-TOURNAMENT forecast against ACTUAL 2026 results.

This is the honest, out-of-sample measure the question "how accurate is my model"
deserves: we take the model trained *without any 2026 data* (a genuine
pre-tournament forecast) and compare its predictions to the real results that
have since been played. No look-ahead — the forecast never saw these games.

Metrics (group-stage matches, clean win/draw/loss)
--------------------------------------------------
  * outcome accuracy   — did the most-probable outcome happen?
  * exact-score accuracy — did the most-likely scoreline happen?
  * Brier score        — multiclass (sum of squared prob errors over W/D/L)
  * RPS                — Ranked Probability Score (ordered W/D/L)
  * log-loss           — mean -log P(true outcome)

Goal-level (all played matches, regulation score)
--------------------------------------------------
  * goal MAE / RMSE    — predicted expected goals vs actual goals (per side)

Knockout
--------
  * winner accuracy    — did the projected stronger side actually advance?

A naive baseline (always predict the bookmaker-free prior: home/draw/away base
rates) is reported alongside so the metrics have a reference point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import HOSTS
from predictor.goals_model import predict_scoreline, xg

# Long-run base rates of international-match outcomes — a no-skill reference.
BASELINE_WDL = (0.45, 0.27, 0.28)


def _outcome_index(ga: int, gb: int) -> int:
    return 0 if ga > gb else (1 if ga == gb else 2)


def _rps(probs, y: int) -> float:
    oh = [0, 0, 0]
    oh[y] = 1
    cp, co = np.cumsum(probs), np.cumsum(oh)
    return float(np.sum((cp - co) ** 2) / (len(probs) - 1))


def _brier(probs, y: int) -> float:
    oh = np.zeros(3)
    oh[y] = 1
    return float(np.sum((np.asarray(probs) - oh) ** 2))


def score_forecast(model: dict, ratings: dict, group_actual: dict, ko_actual: dict):
    """Score a pre-tournament model against played results.

    Returns ``(per_match_df, summary)``. ``summary`` is a dict of aggregate
    metrics; ``per_match_df`` has one row per played match with the predicted vs
    actual detail behind those aggregates.
    """
    rows = []

    # --- group-stage matches: full probabilistic scoring -------------------
    for key, info in group_actual.items():
        a, b = tuple(info["by_team"].keys())
        ga, gb = info["by_team"][a], info["by_team"][b]
        (pi, pj), (pH, pD, pA), (la, lb), _ = predict_scoreline(
            model, ratings, a, b, home_a=int(a in HOSTS), home_b=int(b in HOSTS))
        y = _outcome_index(ga, gb)
        probs = [pH, pD, pA]
        rows.append({
            "stage": "group", "fixture": f"{a} v {b}",
            "pred_score": f"{pi}-{pj}", "actual_score": f"{ga}-{gb}",
            "pred_xg": f"{la:.2f}-{lb:.2f}",
            "pred_outcome": ["H", "D", "A"][int(np.argmax(probs))],
            "actual_outcome": ["H", "D", "A"][y],
            "outcome_hit": int(np.argmax(probs) == y),
            "exact_hit": int(pi == ga and pj == gb),
            "rps": _rps(probs, y), "brier": _brier(probs, y),
            "logloss": -np.log(max(probs[y], 1e-15)),
            "abs_err_a": abs(la - ga), "abs_err_b": abs(lb - gb),
        })

    # --- knockout matches: winner accuracy + goal errors -------------------
    for key, info in ko_actual.items():
        a, b = tuple(info["by_team"].keys())
        ga, gb = info["by_team"][a], info["by_team"][b]
        la = xg(model, a, b, ratings.get(a, 1500), ratings.get(b, 1500), 0)
        lb = xg(model, b, a, ratings.get(b, 1500), ratings.get(a, 1500), 0)
        actual_winner = a if ga > gb else (b if gb > ga else info.get("pen_winner"))
        pred_winner = a if la >= lb else b
        rows.append({
            "stage": "ko", "fixture": f"{a} v {b}",
            "pred_score": "-", "actual_score": f"{ga}-{gb}",
            "pred_xg": f"{la:.2f}-{lb:.2f}",
            "pred_outcome": pred_winner, "actual_outcome": actual_winner,
            "outcome_hit": int(pred_winner == actual_winner),
            "exact_hit": np.nan, "rps": np.nan, "brier": np.nan, "logloss": np.nan,
            "abs_err_a": abs(la - ga), "abs_err_b": abs(lb - gb),
        })

    df = pd.DataFrame(rows)
    summary: dict = {"n_group": int((df["stage"] == "group").sum()) if len(df) else 0,
                     "n_ko": int((df["stage"] == "ko").sum()) if len(df) else 0}
    if not len(df):
        summary["message"] = "No actual results entered yet — nothing to score."
        return df, summary

    grp = df[df["stage"] == "group"]
    if len(grp):
        summary["group_outcome_accuracy"] = round(grp["outcome_hit"].mean(), 3)
        summary["group_exact_accuracy"] = round(grp["exact_hit"].mean(), 3)
        summary["group_rps"] = round(grp["rps"].mean(), 4)
        summary["group_brier"] = round(grp["brier"].mean(), 4)
        summary["group_logloss"] = round(grp["logloss"].mean(), 4)
        # No-skill baseline (constant base-rate forecast) for the same matches.
        base_rps, base_ll = [], []
        for _, r in grp.iterrows():
            y = {"H": 0, "D": 1, "A": 2}[r["actual_outcome"]]
            base_rps.append(_rps(BASELINE_WDL, y))
            base_ll.append(-np.log(max(BASELINE_WDL[y], 1e-15)))
        summary["baseline_rps"] = round(float(np.mean(base_rps)), 4)
        summary["baseline_logloss"] = round(float(np.mean(base_ll)), 4)
        summary["rps_skill_vs_baseline"] = round(1 - summary["group_rps"] / summary["baseline_rps"], 3)

    ko = df[df["stage"] == "ko"]
    if len(ko):
        summary["ko_winner_accuracy"] = round(ko["outcome_hit"].mean(), 3)

    errs = pd.concat([df["abs_err_a"], df["abs_err_b"]])
    summary["goal_mae"] = round(float(errs.mean()), 3)
    summary["goal_rmse"] = round(float(np.sqrt((errs ** 2).mean())), 3)
    return df, summary

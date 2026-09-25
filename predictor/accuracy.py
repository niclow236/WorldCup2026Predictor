"""
accuracy.py — score a model's match predictions against ACTUAL 2026 results.

The headline use is the honest, out-of-sample question "how accurate is my
model?": take the model trained *without any 2026 data* (a genuine
pre-tournament forecast) and compare its predictions to the real results. No
look-ahead — the forecast never saw these games. ``evaluation.py`` reuses the
same scorer for the other model variants and for the replay of the live model.

Every match on one 90-minute W/D/L scale
----------------------------------------
Knockout ties that went to extra time or penalties count as 90-minute draws, so
all matches (group and knockout) are scored the same way:
  * outcome accuracy  — did the most-probable outcome happen?
  * exact-score       — did the most-likely scoreline happen? (skipped for
                        extra-time games, whose 90-minute score isn't recorded)
  * RPS               — Ranked Probability Score (ordered W/D/L)
  * Brier score       — multiclass (sum of squared prob errors over W/D/L)
  * log-loss          — mean -log P(true outcome)

Knockout advancement (who went through, incl. extra time and shootouts)
------------------------------------------------------------------------
  * P(advance) mirrors the simulator: 90 minutes -> extra time at a third of
    the scoring rate -> Elo-tilted shootout
  * accuracy, binary Brier and log-loss (a coin flip scores 0.25 / 0.693)

Goals
-----
  * MAE / RMSE of expected vs actual goals per side. Extra-time games are
    compared with 4/3 of the 90-minute expectation (the extra 30 minutes are
    played at the same per-minute rate in the simulator).

Summary keys: ``group_*`` covers the 72 group games (the long-standing headline
numbers), ``ko_*`` the knockout ties and ``all90_*`` every match on the
90-minute scale. A no-skill base-rate forecast is reported alongside so the
metrics have a reference point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import HOSTS
from predictor.goals_model import scoreline_grid, xg
from predictor.tournament import advance_prob, shootout_prob

# Long-run base rates of international-match outcomes — a no-skill reference.
BASELINE_WDL = (0.45, 0.27, 0.28)
OUTCOMES = ("H", "D", "A")
ET_GOAL_FACTOR = 4.0 / 3.0     # 120 minutes at the 90-minute per-minute rate


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


def _logloss(p: float) -> float:
    return float(-np.log(max(p, 1e-15)))


def actual_score_label(info: dict) -> str:
    """``"1-1 aet (pens 4-3)"``-style label for an entered result, in team order."""
    a, b = tuple(info["by_team"].keys())
    label = f"{info['by_team'][a]}-{info['by_team'][b]}"
    if info.get("aet"):
        label += " aet"
    if "pen_score" in info:
        label += f" (pens {info['pen_score'][a]}-{info['pen_score'][b]})"
    elif info.get("pen_winner"):
        label += f" ({info['pen_winner']} on pens)"
    return label


def score_matches(model: dict, ratings: dict, group_actual: dict, ko_actual: dict,
                  xg_fn=None) -> pd.DataFrame:
    """One row per played match: the model's prediction next to what happened.

    ``xg_fn(attacker, defender, home)`` supplies expected goals; by default the
    Poisson(+Elo) goals ``model`` with the given Elo ``ratings``. ``ratings`` also
    drive the shootout tilt, exactly as in the simulator.
    """
    if xg_fn is None:
        def xg_fn(att, dfn, home):
            return xg(model, att, dfn, ratings.get(att, 1500), ratings.get(dfn, 1500), home)

    rows = []
    for stage, played in (("group", group_actual), ("ko", ko_actual)):
        for info in played.values():
            a, b = tuple(info["by_team"].keys())
            ga, gb = info["by_team"][a], info["by_team"][b]
            # Hosts get home advantage in the group stage only (knockouts are neutral).
            home_a = int(stage == "group" and a in HOSTS)
            home_b = int(stage == "group" and b in HOSTS)
            la, lb = xg_fn(a, b, home_a), xg_fn(b, a, home_b)
            (pi, pj), probs, _ = scoreline_grid(la, lb)
            aet = bool(info.get("aet"))
            y = 1 if aet else _outcome_index(ga, gb)          # result after 90 minutes
            pred = int(np.argmax(probs))
            f = ET_GOAL_FACTOR if aet else 1.0
            row = {
                "date": info.get("date", ""), "stage": stage, "team_a": a, "team_b": b,
                "actual_score": actual_score_label(info), "pred_score": f"{pi}-{pj}",
                "xg_a": la, "xg_b": lb,
                "p_a": probs[0], "p_draw": probs[1], "p_b": probs[2],
                "pred_90": OUTCOMES[pred], "actual_90": OUTCOMES[y], "hit_90": int(pred == y),
                "exact_hit": np.nan if aet else int(pi == ga and pj == gb),
                "rps": _rps(probs, y), "brier": _brier(probs, y), "logloss": _logloss(probs[y]),
                "abs_err_a": abs(la * f - ga), "abs_err_b": abs(lb * f - gb),
            }
            if stage == "ko":
                p_adv = advance_prob(probs, la, lb,
                                     shootout_prob(ratings.get(a, 1500), ratings.get(b, 1500)))
                winner = a if ga > gb else (b if gb > ga else info["pen_winner"])
                pred_winner = a if p_adv >= 0.5 else b
                y_adv = int(winner == a)
                row.update({
                    "p_advance_a": p_adv, "pred_winner": pred_winner, "actual_winner": winner,
                    "advance_hit": int(pred_winner == winner),
                    "advance_brier": (p_adv - y_adv) ** 2,
                    "advance_logloss": _logloss(p_adv if y_adv else 1 - p_adv),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    """Aggregate metrics for a ``score_matches`` frame (see module docstring)."""
    summary: dict = {"n_group": int((df["stage"] == "group").sum()) if len(df) else 0,
                     "n_ko": int((df["stage"] == "ko").sum()) if len(df) else 0}
    if not len(df):
        summary["message"] = "No actual results entered yet — nothing to score."
        return summary

    def base(frame):
        ys = frame["actual_90"].map({"H": 0, "D": 1, "A": 2})
        return (float(np.mean([_rps(BASELINE_WDL, y) for y in ys])),
                float(np.mean([_logloss(BASELINE_WDL[y]) for y in ys])))

    grp = df[df["stage"] == "group"]
    if len(grp):
        summary["group_outcome_accuracy"] = round(grp["hit_90"].mean(), 3)
        summary["group_exact_accuracy"] = round(grp["exact_hit"].mean(), 3)
        summary["group_rps"] = round(grp["rps"].mean(), 4)
        summary["group_brier"] = round(grp["brier"].mean(), 4)
        summary["group_logloss"] = round(grp["logloss"].mean(), 4)
        base_rps, base_ll = base(grp)
        summary["baseline_rps"] = round(base_rps, 4)
        summary["baseline_logloss"] = round(base_ll, 4)
        summary["rps_skill_vs_baseline"] = round(1 - summary["group_rps"] / summary["baseline_rps"], 3)

    ko = df[df["stage"] == "ko"]
    if len(ko):
        summary["ko_winner_accuracy"] = round(ko["advance_hit"].mean(), 3)
        summary["ko_advance_brier"] = round(ko["advance_brier"].mean(), 4)
        summary["ko_advance_logloss"] = round(ko["advance_logloss"].mean(), 4)
        summary["ko_coinflip_brier"] = 0.25
        summary["ko_coinflip_logloss"] = round(float(np.log(2)), 4)

    summary["all90_outcome_accuracy"] = round(df["hit_90"].mean(), 3)
    summary["all90_rps"] = round(df["rps"].mean(), 4)
    summary["all90_baseline_rps"] = round(base(df)[0], 4)

    errs = pd.concat([df["abs_err_a"], df["abs_err_b"]])
    summary["goal_mae"] = round(float(errs.mean()), 3)
    summary["goal_rmse"] = round(float(np.sqrt((errs ** 2).mean())), 3)
    return summary


def score_forecast(model: dict, ratings: dict, group_actual: dict, ko_actual: dict, xg_fn=None):
    """Score a model against played results. Returns ``(per_match_df, summary)``."""
    df = score_matches(model, ratings, group_actual, ko_actual, xg_fn)
    return df, summarise(df)

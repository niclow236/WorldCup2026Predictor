"""
predictions.py — human-readable per-match and bracket forecasts.

Two products:
  * ``group_stage_predictions`` — a predicted scoreline for all 72 group games
    (entered results shown verbatim and flagged ✓; the rest use the most-likely
    Dixon-Coles-corrected scoreline).
  * ``project_bracket`` — the single *most-likely path* through the knockout
    bracket, built from the Monte-Carlo modal group finishers and third-place
    qualifiers. The probability tables from the Monte Carlo are the rigorous
    view; this is the "one bracket to print on a wall" companion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import GROUP_FIXTURES, GROUPS, HOSTS, KO_MATCH_ORDER, LATER, ROUND_OF, THIRD_PLACE_TABLE
from predictor.goals_model import predict_scoreline
from predictor.tournament import r32_pairs


def group_stage_predictions(model: dict, ratings: dict, group_actual: dict) -> pd.DataFrame:
    """Predicted (or actual) scoreline for every group-stage match."""
    rows = []
    for g, teams in GROUPS.items():
        for i, j in GROUP_FIXTURES:
            a, b = teams[i], teams[j]
            key = frozenset({a, b})
            if key in group_actual:
                ga, gb = group_actual[key]["by_team"][a], group_actual[key]["by_team"][b]
                res = f"{a} {ga}-{gb} {b}"
                tag = "✓ actual"
            else:
                (ga, gb), (pH, pD, pA), _, _ = predict_scoreline(
                    model, ratings, a, b, home_a=int(a in HOSTS), home_b=int(b in HOSTS))
                res = f"{a} {ga}-{gb} {b}"
                tag = f"pred (W/D/L {pH:.0%}/{pD:.0%}/{pA:.0%})"
            rows.append({"grp": g, "match": f"{a} v {b}", "result": res, "note": tag})
    return pd.DataFrame(rows)


def _projected_finishers(pos: dict, q3: dict):
    """Modal group winners / runners-up / thirds and the eight qualifying thirds."""
    proj_W, proj_R, proj_3 = {}, {}, {}
    for g in GROUPS:
        by1 = sorted(GROUPS[g], key=lambda t: pos[g][t][0], reverse=True)
        proj_W[g] = by1[0]
        by2 = sorted(GROUPS[g], key=lambda t: pos[g][t][1], reverse=True)
        proj_R[g] = next(t for t in by2 if t != proj_W[g])
        by3 = sorted(GROUPS[g], key=lambda t: pos[g][t][2], reverse=True)
        proj_3[g] = next(t for t in by3 if t not in (proj_W[g], proj_R[g]))
    qual_rate = sorted(GROUPS.keys(), key=lambda g: q3[proj_3[g]], reverse=True)
    proj_qual = frozenset(qual_rate[:8])
    T_proj = {slot: proj_3[src] for slot, src in THIRD_PLACE_TABLE[proj_qual].items()}
    return proj_W, proj_R, proj_3, T_proj


def project_bracket(model: dict, ratings: dict, simulator, pos: dict, q3: dict, ko_actual: dict):
    """Build the most-likely knockout bracket. Returns ``(df, champion, finalists)``."""

    def project_match(a: str, b: str):
        key = frozenset({a, b})
        if key in ko_actual:
            r = ko_actual[key]
            ga, gb = r["by_team"][a], r["by_team"][b]
            w = a if ga > gb else (b if gb > ga else r["pen_winner"])
            if ga != gb:
                sc = f"{ga}-{gb}"
            elif "pen_score" in r:
                ps = r["pen_score"]
                sc = f"{ga}-{gb} (pens {ps[w]}-{ps[b if w == a else a]})"
            else:
                sc = f"{ga}-{gb} (pens {w} wins)"
            return w, sc, "✓"
        (i, j), (pH, pD, pA), _, M = predict_scoreline(model, ratings, a, b)
        padv_a = pH + pD * simulator.pen_p(a, b)
        w = a if padv_a >= 0.5 else b
        padv = padv_a if w == a else (1 - padv_a)
        if i == j and padv <= 0.55:                 # genuinely even -> show a shootout
            pw_sc, pl_sc = simulator.pen_mode
            sc = f"{i}-{j} (pens {pw_sc}-{pl_sc})"
        else:                                        # representative decisive score
            sub = np.tril(M, -1) if w == a else np.triu(M, 1)
            ii, jj = np.unravel_index(np.argmax(sub), sub.shape)
            sc = f"{ii}-{jj}"
        return w, sc, ""

    proj_W, proj_R, proj_3, T_proj = _projected_finishers(pos, q3)
    pairs = r32_pairs(proj_W, proj_R, T_proj)
    win, brk = {}, []
    for no in KO_MATCH_ORDER:
        a, b = pairs[no] if no in pairs else (win[LATER[no][0]], win[LATER[no][1]])
        w, sc, tag = project_match(a, b)
        win[no] = w
        brk.append({"match": no, "round": ROUND_OF[no], "fixture": f"{a} v {b}",
                    "score": sc, "winner": w, "flag": tag})
    df = pd.DataFrame(brk)
    return df, win[104], [win[101], win[102]]

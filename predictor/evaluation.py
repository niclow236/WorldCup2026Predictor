"""
evaluation.py — the post-tournament verdict: what the models predicted vs what happened.

Runs once every result is in (all 104 matches) and answers, with no look-ahead:

1. **Is the data consistent?** ``reconstruct_bracket`` rebuilds the real
   tournament from the group results alone — FIFA 2026 tie-breakers, the
   best-third ranking, FIFA's 495-row slot table and the bracket wiring — and
   raises unless every entered knockout result fits exactly where it should.
2. **How good were the match predictions?** ``model_comparison`` scores every
   model on all 104 matches: a base-rate and an Elo-favourite baseline, the
   Poisson, Poisson+Elo and GB-hybrid models trained before kick-off, and the
   live Poisson+Elo model re-fitted before every match day (``live_replay``).
3. **How good were the tournament predictions?** ``score_stage_probabilities``
   scores the pre-tournament stage-reach probabilities (round of 32 through
   champion) against how far each team really went; ``group_comparison`` and
   ``bracket_comparison`` hold the projected group orders and bracket up against
   the real ones.
4. **How did the forecast evolve?** ``title_timeline`` re-runs the full Monte
   Carlo after every round, conditioned only on the results known by then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from constants import (ALL_TEAMS, GROUP_FIXTURES, GROUPS, KO_MATCH_ORDER, LATER, ROUND_OF,
                       THIRD_PLACE_FEEDERS, THIRD_PLACE_MATCH, THIRD_PLACE_TABLE)
from predictor.accuracy import BASELINE_WDL, _logloss, _rps, actual_score_label, score_matches, summarise
from predictor.data_io import build_matches_all
from predictor.elo import attach_elo
from predictor.goals_model import fit_goals_model
from predictor.tournament import r32_pairs

# Furthest stage reached, as an index (used to score stage-reach probabilities).
FINISH_LABELS = ["Group stage", "Round of 32", "Round of 16", "Quarter-final",
                 "Semi-final", "Runner-up", "Champion"]
# (forecast column, minimum finish index, label, number of teams that get there)
STAGES = [
    ("reach_R32_%", 1, "Reach round of 32", 32),
    ("reach_R16_%", 2, "Reach round of 16", 16),
    ("quarterfinal_%", 3, "Reach quarter-final", 8),
    ("semifinal_%", 4, "Reach semi-final", 4),
    ("final_%", 5, "Reach final", 2),
    ("champion_%", 6, "Win the World Cup", 1),
]
ROUND_FINISH = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "Final": 5}
TEAM_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}
LIVE_MODEL = "Poisson + Elo, live (re-fit daily)"   # model label of the live_replay() frame


def is_complete(group_actual: dict, ko_actual: dict) -> bool:
    """True once all 72 group games and all 32 knockout ties are entered."""
    return len(group_actual) == 72 and len(ko_actual) == 32


# ---------------------------------------------------------------------------
# 1 · The real tournament, rebuilt and validated from the results
# ---------------------------------------------------------------------------
def _table(teams, results) -> dict:
    """Points / goals / W-D-L for ``teams`` over the results played among them."""
    st = {t: {"pts": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} for t in teams}
    for a, b, ga, gb in results:
        if a not in st or b not in st:
            continue
        st[a]["gf"] += ga; st[a]["ga"] += gb; st[b]["gf"] += gb; st[b]["ga"] += ga
        if ga == gb:
            for t in (a, b):
                st[t]["pts"] += 1; st[t]["d"] += 1
        else:
            w, l_ = (a, b) if ga > gb else (b, a)
            st[w]["pts"] += 3; st[w]["w"] += 1; st[l_]["l"] += 1
    return st


def _blocks(teams, key) -> list[list[str]]:
    """``teams`` sorted by ``key`` (best first), grouped into runs of equal keys."""
    out: list[list[str]] = []
    for t in sorted(teams, key=key, reverse=True):
        if out and key(out[-1][0]) == key(t):
            out[-1].append(t)
        else:
            out.append([t])
    return out


def _rank_level(tied, results, overall) -> list[str]:
    """Order teams level on points by FIFA 2026 rules.

    Head-to-head points, goal difference and goals among the tied teams, reapplied
    to any subset still level; then overall goal difference and goals. Fair play
    and the FIFA ranking are not modelled — a tie that reaches them raises.
    """
    if len(tied) == 1:
        return list(tied)
    h2h = _table(tied, results)
    blocks = _blocks(tied, lambda t: (h2h[t]["pts"], h2h[t]["gf"] - h2h[t]["ga"], h2h[t]["gf"]))
    if len(blocks) > 1:
        return [t for blk in blocks for t in _rank_level(blk, results, overall)]
    blocks = _blocks(tied, lambda t: (overall[t]["gf"] - overall[t]["ga"], overall[t]["gf"]))
    if any(len(blk) > 1 for blk in blocks):
        raise ValueError(f"Cannot separate {tied} without fair-play data.")
    return [blk[0] for blk in blocks]


def group_tables(group_actual: dict) -> dict[str, pd.DataFrame]:
    """Final table of every group under the FIFA 2026 tie-breakers."""
    tables = {}
    for g, teams in GROUPS.items():
        results = []
        for i, j in GROUP_FIXTURES:
            a, b = teams[i], teams[j]
            rec = group_actual.get(frozenset((a, b)))
            if rec is None:
                raise ValueError(f"Group {g}: {a} v {b} has no entered result.")
            results.append((a, b, rec["by_team"][a], rec["by_team"][b]))
        st = _table(teams, results)
        order = [t for blk in _blocks(teams, lambda t: st[t]["pts"])
                 for t in _rank_level(blk, results, st)]
        tables[g] = pd.DataFrame([{
            "pos": p, "team": t, "W": st[t]["w"], "D": st[t]["d"], "L": st[t]["l"],
            "GF": st[t]["gf"], "GA": st[t]["ga"], "GD": st[t]["gf"] - st[t]["ga"], "Pts": st[t]["pts"],
        } for p, t in enumerate(order, 1)])
    return tables


def rank_thirds(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rank the twelve third-placed teams (points, goal difference, goals)."""
    thirds = [(g, tables[g].iloc[2]) for g in GROUPS]
    key = lambda x: (x[1]["Pts"], x[1]["GD"], x[1]["GF"])  # noqa: E731
    thirds.sort(key=key, reverse=True)
    if key(thirds[7]) == key(thirds[8]):
        raise ValueError("8th and 9th best third-placed teams are level — fair-play data needed.")
    return pd.DataFrame([{"rank": i, "group": g, "team": r["team"], "Pts": r["Pts"], "GD": r["GD"],
                          "GF": r["GF"], "qualified": i <= 8} for i, (g, r) in enumerate(thirds, 1)])


def reconstruct_bracket(group_actual: dict, ko_actual: dict):
    """Rebuild the real knockout bracket from the group results and validate it.

    The actual group tables and best-third ranking go through FIFA's slot table
    and ``r32_pairs`` to give the round-of-32 fixtures; winners (and, for the
    third-place play-off, semi-final losers) then feed forward. Every fixture
    must exist among the entered knockout results and every entered knockout
    result must be used, otherwise ``ValueError``. Returns
    ``(bracket_df, tables, thirds_df)``.
    """
    tables = group_tables(group_actual)
    thirds = rank_thirds(tables)
    W = {g: tables[g]["team"].iloc[0] for g in GROUPS}
    R = {g: tables[g]["team"].iloc[1] for g in GROUPS}
    qual = thirds[thirds["qualified"]]
    third_team = dict(zip(qual["group"], qual["team"]))
    T = {slot: third_team[src] for slot, src in THIRD_PLACE_TABLE[frozenset(qual["group"])].items()}
    pairs = r32_pairs(W, R, T)

    win, lose, rows, used = {}, {}, [], set()
    for no in KO_MATCH_ORDER:
        if no in pairs:
            a, b = pairs[no]
        elif no == THIRD_PLACE_MATCH:
            a, b = lose[THIRD_PLACE_FEEDERS[0]], lose[THIRD_PLACE_FEEDERS[1]]
        else:
            a, b = win[LATER[no][0]], win[LATER[no][1]]
        rec = ko_actual.get(frozenset((a, b)))
        if rec is None:
            raise ValueError(f"Match {no} ({ROUND_OF[no]}): the results imply {a} v {b}, "
                             "but no such knockout result is entered.")
        used.add(frozenset((a, b)))
        ga, gb = rec["by_team"][a], rec["by_team"][b]
        w = a if ga > gb else (b if gb > ga else rec["pen_winner"])
        win[no], lose[no] = w, (b if w == a else a)
        ordered = {**rec, "by_team": {a: ga, b: gb}}
        rows.append({"match": no, "round": ROUND_OF[no], "date": rec["date"], "team_a": a,
                     "team_b": b, "score": actual_score_label(ordered), "winner": w, "loser": lose[no]})
    extra = [" v ".join(sorted(k)) for k in ko_actual if k not in used]
    if extra:
        raise ValueError(f"Knockout results that do not fit the bracket: {extra}")
    return pd.DataFrame(rows), tables, thirds


def finish_index(bracket: pd.DataFrame) -> pd.Series:
    """Furthest stage each team reached, as an index into ``FINISH_LABELS``."""
    reached = {t: 0 for t in ALL_TEAMS}
    for r in bracket.itertuples(index=False):
        lvl = ROUND_FINISH.get(r.round)
        if lvl is None:                        # third-place play-off
            continue
        for t in (r.team_a, r.team_b):
            reached[t] = max(reached[t], lvl)
        if r.round == "Final":
            reached[r.winner] = 6
    return pd.Series(reached, name="finish")


# ---------------------------------------------------------------------------
# 2 · Match predictions: model comparison and the live replay
# ---------------------------------------------------------------------------
def live_replay(matches_hist: pd.DataFrame, group_actual: dict, ko_actual: dict) -> pd.DataFrame:
    """Score the *live* Poisson+Elo model: before each match day, re-fit Elo and
    the goals model on history plus every result known by then (exactly what the
    to-date pipeline did during the tournament) and predict that day's matches."""
    played = list(group_actual.values()) + list(ko_actual.values())
    frames = []
    for d in sorted({rec["date"] for rec in played}):
        def subset(results, keep):
            return {k: v for k, v in results.items() if keep(v["date"])}
        known_g = subset(group_actual, lambda x: x < d)
        known_k = subset(ko_actual, lambda x: x < d)
        matches, ratings = attach_elo(build_matches_all(matches_hist, known_g, known_k))
        model = fit_goals_model(matches[matches.date >= config.TRAIN_SINCE])
        frames.append(score_matches(model, ratings, subset(group_actual, lambda x: x == d),
                                    subset(ko_actual, lambda x: x == d)))
    return pd.concat(frames, ignore_index=True)


def _baseline_rows(ref: pd.DataFrame, ratings: dict) -> list[dict]:
    """No-skill base rate and 'higher Elo wins' reference rows."""
    grp, ko = ref[ref["stage"] == "group"], ref[ref["stage"] == "ko"]
    ys = ref["actual_90"].map({"H": 0, "D": 1, "A": 2})
    gys = ys[ref["stage"] == "group"]
    base = {
        "model": "Base rate (no skill)",
        "group_outcome_accuracy": float((gys == 0).mean()),       # always picks the 'home' side
        "group_rps": float(np.mean([_rps(BASELINE_WDL, y) for y in gys])),
        "group_logloss": float(np.mean([_logloss(BASELINE_WDL[y]) for y in gys])),
        "all90_outcome_accuracy": float((ys == 0).mean()),
        "all90_rps": float(np.mean([_rps(BASELINE_WDL, y) for y in ys])),
        "ko_winner_accuracy": 0.5, "ko_advance_brier": 0.25, "ko_advance_logloss": float(np.log(2)),
    }
    elo_a = ref["team_a"].map(lambda t: ratings.get(t, 1500))
    elo_b = ref["team_b"].map(lambda t: ratings.get(t, 1500))
    fav = np.where(elo_a >= elo_b, "H", "A")                     # never predicts a draw
    fav_ko = np.where(ko["team_a"].map(lambda t: ratings.get(t, 1500))
                      >= ko["team_b"].map(lambda t: ratings.get(t, 1500)), ko["team_a"], ko["team_b"])
    elo = {
        "model": "Elo favourite (higher rating wins)",
        "group_outcome_accuracy": float((fav[ref["stage"] == "group"] == grp["actual_90"]).mean()),
        "all90_outcome_accuracy": float((fav == ref["actual_90"]).mean()),
        "ko_winner_accuracy": float((fav_ko == ko["actual_winner"]).mean()),
    }
    return [base, elo]


COMPARISON_COLUMNS = ["group_outcome_accuracy", "group_rps", "group_logloss", "all90_outcome_accuracy",
                      "all90_rps", "ko_winner_accuracy", "ko_advance_brier", "ko_advance_logloss", "goal_mae"]
MATCH_KEY = ["stage", "team_a", "team_b"]


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = config.RNG_SEED):
    """Mean of per-match differences with a 95% bootstrap interval (resampling matches)."""
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def model_comparison(scored: dict[str, pd.DataFrame], ratings_pre: dict,
                     reference: str = "Poisson + Elo") -> pd.DataFrame:
    """One row per model: accuracy / RPS / log-loss on the same 104 matches.

    ``scored`` maps a model name to its ``score_matches`` frame; baselines use
    the pre-tournament Elo ``ratings_pre``. Each model's per-match RPS is also
    compared with the ``reference`` model's on the same matches: the mean
    difference and its 95% paired-bootstrap interval (an interval spanning 0
    means 104 matches cannot tell the two apart).
    """
    ref = scored[reference].set_index(MATCH_KEY)
    rows = _baseline_rows(scored[reference], ratings_pre)
    base_rps = np.array([_rps(BASELINE_WDL, {"H": 0, "D": 1, "A": 2}[o]) for o in ref["actual_90"]])
    per_match_rps = {rows[0]["model"]: base_rps}
    for name, df in scored.items():
        s = summarise(df)
        rows.append({"model": name, **{c: s.get(c, np.nan) for c in COMPARISON_COLUMNS}})
        per_match_rps[name] = df.set_index(MATCH_KEY).loc[ref.index, "rps"].to_numpy()
    for row in rows:
        if row["model"] in per_match_rps and row["model"] != reference:
            d, lo, hi = paired_bootstrap(per_match_rps[row["model"]] - ref["rps"].to_numpy())
            row.update({"rps_vs_ref": d, "rps_vs_ref_lo": lo, "rps_vs_ref_hi": hi})
    cols = ["model"] + COMPARISON_COLUMNS + ["rps_vs_ref", "rps_vs_ref_lo", "rps_vs_ref_hi"]
    return pd.DataFrame(rows).reindex(columns=cols)


def phase_breakdown(scored: dict[str, pd.DataFrame], cutoffs: list[tuple[str, str]]) -> pd.DataFrame:
    """Mean RPS per model within each phase: group matchdays 1-3, then knockouts.

    ``cutoffs`` are the first three ``checkpoints`` (the date each matchday ended).
    """
    def phase(r):
        if r.stage == "ko":
            return "Knockouts"
        return next(f"Matchday {k}" for k, (_, c) in enumerate(cutoffs[:3], 1) if r.date <= c)
    rows = []
    for name, df in scored.items():
        ph = [phase(r) for r in df.itertuples(index=False)]
        for label, grp in df.groupby(ph, sort=False):
            rows.append({"model": name, "phase": label, "matches": len(grp), "rps": grp["rps"].mean()})
    return pd.DataFrame(rows).pivot_table(index="model", columns="phase", values="rps", sort=False)


def match_table(pre: pd.DataFrame, live: pd.DataFrame | None, bracket: pd.DataFrame) -> pd.DataFrame:
    """All 104 matches: pre-tournament (and live) prediction next to the result."""
    rnd = {frozenset((r.team_a, r.team_b)): (r.round, r.match) for r in bracket.itertuples(index=False)}
    cols = ["p_a", "p_draw", "p_b", "pred_score", "hit_90", "p_advance_a", "pred_winner", "advance_hit"]
    out = pre[["date", "stage", "team_a", "team_b", "actual_score", "actual_90", "actual_winner"] + cols].copy()
    out = out.rename(columns={c: f"pre_{c}" for c in cols})
    if live is not None:
        lv = live[["stage", "team_a", "team_b"] + cols].rename(columns={c: f"live_{c}" for c in cols})
        out = out.merge(lv, on=["stage", "team_a", "team_b"], how="left")
    ko = out["stage"] == "ko"
    keys = [frozenset(p) for p in zip(out["team_a"], out["team_b"])]
    out.insert(1, "round", [rnd[k][0] if is_ko else f"Group {TEAM_GROUP[a]}"
                            for k, is_ko, a in zip(keys, ko, out["team_a"])])
    out.insert(2, "match", [rnd[k][1] if is_ko else np.nan for k, is_ko in zip(keys, ko)])
    return out.drop(columns="stage").sort_values(["date", "match"], na_position="first",
                                                 kind="stable").reset_index(drop=True)


def surprises(pre: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """The results the pre-tournament model found least likely (``n`` per stage).

    Group games: P(the actual 90-minute outcome). Knockout ties: P(the team
    that actually went through advancing) — who progressed is what mattered —
    listing only the ties the model called wrong (probability below 50%).
    """
    p_col = {"H": "p_a", "D": "p_draw", "A": "p_b"}
    rows = []
    for r in pre.itertuples(index=False):
        if r.stage == "ko":
            p = r.p_advance_a if r.actual_winner == r.team_a else 1 - r.p_advance_a
            what = f"{r.actual_winner} went through"
        else:
            p = getattr(r, p_col[r.actual_90])
            what = {"H": f"{r.team_a} win", "D": "draw", "A": f"{r.team_b} win"}[r.actual_90]
        rows.append({"stage": r.stage, "date": r.date, "fixture": f"{r.team_a} v {r.team_b}",
                     "result": r.actual_score, "what_happened": what, "model_probability": p})
    df = pd.DataFrame(rows).sort_values("model_probability", kind="stable")
    df = df[(df["stage"] == "group") | (df["model_probability"] < 0.5)]
    return (df.groupby("stage", sort=False).head(n)
            .sort_values(["stage", "model_probability"], kind="stable").reset_index(drop=True))


def calibration_bins(scored: pd.DataFrame, edges=(0, .1, .2, .3, .4, .5, .6, .7, 1.0)) -> pd.DataFrame:
    """Reliability table: pooled W/D/L probabilities vs how often those outcomes happened."""
    p = np.concatenate([scored["p_a"], scored["p_draw"], scored["p_b"]])
    y = np.concatenate([(scored["actual_90"] == o).astype(int) for o in ("H", "D", "A")])
    labels = [f"{100 * lo:.0f}–{100 * hi:.0f}%" for lo, hi in zip(edges[:-1], edges[1:])]
    bins = pd.cut(p, bins=list(edges), labels=labels, include_lowest=True)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin", observed=True)
    return pd.DataFrame({"forecasts": df.size(), "mean_forecast": df["p"].mean(),
                         "observed_frequency": df["y"].mean()}).reset_index().astype({"bin": str})


# ---------------------------------------------------------------------------
# 3 · Tournament predictions: stage reach, groups, bracket
# ---------------------------------------------------------------------------
def score_stage_probabilities(forecast: pd.DataFrame, finish: pd.Series) -> pd.DataFrame:
    """Brier score / log-loss of every stage-reach probability over all 48 teams,
    against a uniform no-skill forecast (each team equally likely), plus how many
    of the model's top-N candidates for a stage really got there."""
    fc = forecast.set_index("team")
    eps = 0.5 / config.N_SIMS            # never-simulated events are not literally impossible
    rows = []
    for col, lvl, label, n in STAGES:
        p = (fc[col].reindex(ALL_TEAMS) / 100).clip(eps, 1 - eps)
        y = (finish.reindex(ALL_TEAMS) >= lvl).astype(float)
        p0 = n / len(ALL_TEAMS)
        brier, brier0 = float(((p - y) ** 2).mean()), float(((p0 - y) ** 2).mean())
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        ll0 = float(-(y * np.log(p0) + (1 - y) * np.log(1 - p0)).mean())
        top = p.nlargest(n).index
        rows.append({"stage": label, "teams": n, "top_n_hits": int(y[top].sum()),
                     "mean_p_of_actual": float(p[y == 1].mean()), "brier": brier,
                     "uniform_brier": brier0, "brier_skill": 1 - brier / brier0,
                     "logloss": ll, "uniform_logloss": ll0})
    return pd.DataFrame(rows)


def team_comparison(forecast: pd.DataFrame, finish: pd.Series, tables: dict, pos: dict) -> pd.DataFrame:
    """Per team: pre-tournament stage probabilities vs how far it actually went.

    ``expected_rounds`` = sum of the six stage-reach probabilities (the number
    of stages the model expected the team to reach); ``over_under`` = stages
    actually reached minus that expectation.
    """
    fc = forecast.set_index("team")
    actual_pos = {r.team: r.pos for tbl in tables.values() for r in tbl.itertuples(index=False)}
    rows = []
    for t in ALL_TEAMS:
        g = TEAM_GROUP[t]
        p_pos = pos[g][t] / pos[g][t].sum()
        expected = sum(fc.loc[t, col] / 100 for col, _, _, _ in STAGES)
        rows.append({
            "team": t, "group": g, "elo_pre": int(fc.loc[t, "elo"]),
            **{col: fc.loc[t, col] for col, _, _, _ in STAGES},
            "p_group_winner_%": 100 * p_pos[0],
            "modal_group_pos": int(np.argmax(p_pos)) + 1, "actual_group_pos": actual_pos[t],
            "expected_rounds": expected, "actual_rounds": int(finish[t]),
            "over_under": int(finish[t]) - expected, "actual_finish": FINISH_LABELS[int(finish[t])],
        })
    return pd.DataFrame(rows).sort_values("champion_%", ascending=False, kind="stable").reset_index(drop=True)


def group_comparison(pos: dict, projected: tuple, tables: dict) -> pd.DataFrame:
    """Projected (modal) finishing order vs the real one, group by group."""
    proj_W, proj_R, proj_3, _ = projected
    rows = []
    for g, teams in GROUPS.items():
        first3 = [proj_W[g], proj_R[g], proj_3[g]]
        predicted = first3 + [t for t in teams if t not in first3]
        actual = tables[g]["team"].tolist()
        n = pos[g][actual[0]].sum()
        rows.append({
            "group": g, "predicted_order": " > ".join(predicted), "actual_order": " > ".join(actual),
            "winner_hit": predicted[0] == actual[0], "top2_hit": set(predicted[:2]) == set(actual[:2]),
            "exact_order_hit": predicted == actual,
            "positions_correct": sum(p == a for p, a in zip(predicted, actual)),
            "p_actual_winner_%": 100 * pos[g][actual[0]][0] / n,
        })
    return pd.DataFrame(rows)


def _round_sets(bracket: pd.DataFrame) -> dict[str, set]:
    sets = {rnd: set(bracket.loc[bracket["round"] == rnd, ["team_a", "team_b"]].values.ravel())
            for rnd in ("R32", "R16", "QF", "SF", "Final")}
    sets["Champion"] = set(bracket.loc[bracket["round"] == "Final", "winner"])
    return sets


def bracket_comparison(projected: pd.DataFrame, actual: pd.DataFrame):
    """Projected bracket vs the real one.

    Returns ``(per_match, per_round)``: each tie's projected vs actual fixture and
    winner, and for each round how many of the projected teams really got there.
    """
    m = projected[["match", "round", "team_a", "team_b", "winner"]].merge(
        actual[["match", "team_a", "team_b", "score", "winner"]], on="match", suffixes=("_proj", "_actual"))
    per_match = pd.DataFrame({
        "match": m["match"], "round": m["round"],
        "projected": m["team_a_proj"] + " v " + m["team_b_proj"],
        "projected_winner": m["winner_proj"],
        "actual": m["team_a_actual"] + " v " + m["team_b_actual"], "actual_score": m["score"],
        "actual_winner": m["winner_actual"],
        "fixture_hit": [{a, b} == {c, d} for a, b, c, d in
                        zip(m["team_a_proj"], m["team_b_proj"], m["team_a_actual"], m["team_b_actual"])],
        "winner_hit": m["winner_proj"] == m["winner_actual"],
    })
    ps, acts = _round_sets(projected), _round_sets(actual)
    labels = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
              "SF": "Semi-finals", "Final": "Final", "Champion": "Champion"}
    per_round = pd.DataFrame([{
        "stage": labels[k], "teams": len(acts[k]), "correct": len(ps[k] & acts[k]),
        "projected_but_missed": ", ".join(sorted(ps[k] - acts[k])),
        "unexpected_arrivals": ", ".join(sorted(acts[k] - ps[k])),
    } for k in labels])
    return per_match, per_round


# ---------------------------------------------------------------------------
# 4 · How the title odds evolved
# ---------------------------------------------------------------------------
def checkpoints(group_actual: dict, bracket: pd.DataFrame) -> list[tuple[str, str]]:
    """``(label, cutoff_date)`` after each group matchday and each knockout round."""
    by_team: dict[str, list[str]] = {t: [] for t in ALL_TEAMS}
    for rec in group_actual.values():
        for t in rec["by_team"]:
            by_team[t].append(rec["date"])
    cps = [(f"After matchday {k}" if k < 3 else "After group stage",
            max(sorted(ds)[k - 1] for ds in by_team.values())) for k in (1, 2, 3)]
    for rnd, label in (("R32", "After round of 32"), ("R16", "After round of 16"),
                       ("QF", "After quarter-finals"), ("SF", "After semi-finals")):
        cps.append((label, bracket.loc[bracket["round"] == rnd, "date"].max()))
    return cps


def title_timeline(pre_forecast: pd.DataFrame, group_actual: dict, ko_actual: dict,
                   bracket: pd.DataFrame, run_scenario) -> pd.DataFrame:
    """Champion / final / semi-final odds at every checkpoint (long format).

    ``run_scenario(group_known, ko_known, label) -> forecast`` re-runs the full
    pipeline model (Elo, goals model, GB hybrid, Monte Carlo) on the results
    known at that point. The pre-tournament forecast is reused as the first
    checkpoint; the last row is the actual outcome.
    """
    cols = ["champion_%", "final_%", "semifinal_%"]
    frames = [pre_forecast[["team"] + cols].assign(checkpoint="Pre-tournament", cutoff="")]
    for label, cutoff in checkpoints(group_actual, bracket):
        known_g = {k: v for k, v in group_actual.items() if v["date"] <= cutoff}
        known_k = {k: v for k, v in ko_actual.items() if v["date"] <= cutoff}
        fc = run_scenario(known_g, known_k, label)
        frames.append(fc[["team"] + cols].assign(checkpoint=label, cutoff=cutoff))
    final = bracket[bracket["round"] == "Final"].iloc[0]
    sf = bracket[bracket["round"] == "SF"]
    finalists, semis = {final.team_a, final.team_b}, set(sf.team_a) | set(sf.team_b)
    frames.append(pd.DataFrame({
        "team": ALL_TEAMS,
        "champion_%": [100.0 * (t == final.winner) for t in ALL_TEAMS],
        "final_%": [100.0 * (t in finalists) for t in ALL_TEAMS],
        "semifinal_%": [100.0 * (t in semis) for t in ALL_TEAMS],
    }).assign(checkpoint="Final result", cutoff=final.date))
    return pd.concat(frames, ignore_index=True)[["checkpoint", "cutoff", "team"] + cols]

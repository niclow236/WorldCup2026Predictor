"""
data_io.py — data acquisition, cleaning, and actual-result ingestion.

Responsibilities
----------------
1. Make sure the three historical CSVs exist locally (download on first run).
2. Clean the raw results into ``matches_hist`` — the historical training base
   (strictly pre-tournament, so 2026 scores can never leak in).
3. Parse ``actual_results_2026.csv`` (the real 2026 results) into the two dicts
   the simulator conditions on (``group_actual`` / ``ko_actual``).
4. Append entered 2026 results to the history to produce ``matches_all`` — the
   table the Elo and goals model train on (so World-Cup form feeds back in).

The actual-result dicts use ``frozenset({team_a, team_b})`` keys so a fixture
can be looked up regardless of which side is "home".
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
from constants import ALL_TEAMS, TOURNAMENT_START


# ---------------------------------------------------------------------------
# 1 · Acquisition
# ---------------------------------------------------------------------------
def _download_if_missing(path: str, filename: str) -> None:
    """Download ``filename`` from the configured base URL if it is not cached."""
    if os.path.exists(path):
        return
    if not config.ALLOW_DOWNLOAD:
        raise FileNotFoundError(
            f"{path} is missing and config.ALLOW_DOWNLOAD is False. "
            "Place the file manually or enable downloads."
        )
    url = f"{config.RESULTS_BASE_URL}/{filename}"
    print(f"  downloading {filename} …")
    df = pd.read_csv(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(results_raw, shootouts, former_names)`` as DataFrames.

    ``former_names`` is currently unused by the model but loaded for parity with
    the source dataset and possible future name-normalisation.
    """
    _download_if_missing(config.RESULTS_CSV, "results.csv")
    _download_if_missing(config.SHOOTOUTS_CSV, "shootouts.csv")
    _download_if_missing(config.FORMER_NAMES_CSV, "former_names.csv")
    results_raw = pd.read_csv(config.RESULTS_CSV)
    shootouts = pd.read_csv(config.SHOOTOUTS_CSV)
    former_names = pd.read_csv(config.FORMER_NAMES_CSV)
    return results_raw, shootouts, former_names


# ---------------------------------------------------------------------------
# 2 · Cleaning
# ---------------------------------------------------------------------------
def clean_results(results_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw results into the historical training base ``matches_hist``.

    Keeps only matches played *before* the tournament starts and drops unplayed
    fixtures (the dataset lists 2026 games with blank scores — they must not
    train the model), casts scores to int, normalises the neutral-venue flag,
    and sorts chronologically.

    The date cut-off is a leakage guard: a refreshed upstream ``results.csv``
    contains the real 2026 World Cup scores, which would otherwise leak into the
    "pre-tournament" model (and be double-counted in the to-date one). 2026
    results enter the pipeline only through ``actual_results_2026.csv``.
    """
    m = results_raw.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] < pd.Timestamp(TOURNAMENT_START)]
    m = m.dropna(subset=["home_score", "away_score"]).copy()
    m["home_score"] = m["home_score"].astype(int)
    m["away_score"] = m["away_score"].astype(int)
    m["neutral_b"] = m["neutral"].astype(str).str.upper().eq("TRUE")
    return m.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3 · Actual 2026 results
# ---------------------------------------------------------------------------
def _check_team(team: str) -> None:
    if team not in ALL_TEAMS:
        raise ValueError(
            f"'{team}' is not a 2026 team — check spelling against constants.GROUPS."
        )


def load_actuals(csv_path: str | None = None) -> tuple[dict, dict]:
    """Parse the actual-results CSV into ``(group_actual, ko_actual)`` dicts.

    CSV columns: ``date, stage, team_a, score_a, score_b, team_b, aet,
    pen_winner, pen_a, pen_b``. ``stage`` is ``group`` or ``ko``. Scores are
    final scores *including extra time* (never shootout kicks). For knockout
    rows, ``aet`` is 1 if the tie went to extra time (optional — a level score
    implies it); a level score also requires ``pen_winner``, with the shootout
    score in ``pen_a``/``pen_b`` optional. Each record keeps its ``date``.
    Returns empty dicts if the file is absent (that is the legitimate
    "pre-tournament" state).
    """
    path = csv_path or config.ACTUAL_RESULTS_CSV
    group_actual: dict = {}
    ko_actual: dict = {}
    if not os.path.exists(path):
        return group_actual, ko_actual

    df = pd.read_csv(path, dtype={"date": str})
    if "aet" not in df.columns:          # optional column (older files lack it)
        df["aet"] = np.nan
    for r in df.itertuples(index=False):
        a, b = str(r.team_a).strip(), str(r.team_b).strip()
        _check_team(a)
        _check_team(b)
        ga, gb = int(r.score_a), int(r.score_b)
        stage = str(r.stage).strip().lower()
        if stage not in ("group", "ko", "knockout"):
            raise ValueError(f"Unknown stage '{r.stage}' (expected 'group' or 'ko').")
        # Two teams meet at most once per stage (a knockout rematch of a group
        # fixture is legitimate, so the stages are checked separately).
        target = group_actual if stage == "group" else ko_actual
        key = frozenset({a, b})
        if key in target:
            raise ValueError(f"{a} v {b} is entered more than once in the {stage} stage.")
        date = str(r.date).strip()
        if stage == "group":
            group_actual[key] = {"by_team": {a: ga, b: gb}, "date": date}
            continue

        pen_winner = None if pd.isna(r.pen_winner) else str(r.pen_winner).strip()
        if ga == gb and not pen_winner:
            raise ValueError(
                f"{a} {ga}-{gb} {b} is level — a 'pen_winner' is required for knockout rows."
            )
        if pen_winner and (ga != gb or pen_winner not in (a, b)):
            raise ValueError(
                f"{a} {ga}-{gb} {b}: pen_winner '{pen_winner}' is only valid on a level score "
                "and must be one of the two teams."
            )
        aet = ga == gb or (not pd.isna(r.aet) and bool(int(r.aet)))
        rec = {"by_team": {a: ga, b: gb}, "pen_winner": pen_winner, "aet": aet, "date": date}
        pen_a = None if pd.isna(r.pen_a) else int(r.pen_a)
        pen_b = None if pd.isna(r.pen_b) else int(r.pen_b)
        if ga == gb and pen_a is not None and pen_b is not None:
            pw_sc = pen_a if pen_winner == a else pen_b
            pl_sc = pen_b if pen_winner == a else pen_a
            if pw_sc <= pl_sc:
                raise ValueError(
                    f"pen_winner '{pen_winner}' must have a higher shootout score than the loser."
                )
            rec["pen_score"] = {a: pen_a, b: pen_b}
        ko_actual[key] = rec
    return group_actual, ko_actual


def build_matches_all(matches_hist: pd.DataFrame, group_actual: dict, ko_actual: dict) -> pd.DataFrame:
    """Append entered 2026 results to ``matches_hist`` for strength estimation.

    Each entered result becomes a neutral ``FIFA World Cup`` row dated from the
    tournament start, so Elo and the goals model treat them as the heaviest-K,
    most-recent matches available.
    """
    rows = []
    d0 = pd.Timestamp(TOURNAMENT_START)
    entered = list(group_actual.items()) + list(ko_actual.items())
    for i, (_pair, info) in enumerate(entered):
        a, b = tuple(info["by_team"].keys())
        rows.append({
            "date": d0 + pd.Timedelta(days=i),
            "home_team": a, "away_team": b,
            "home_score": info["by_team"][a], "away_score": info["by_team"][b],
            "tournament": "FIFA World Cup", "city": "", "country": "",
            "neutral": "TRUE", "neutral_b": True,
        })
    if not rows:
        return matches_hist.copy()
    actual_df = pd.DataFrame(rows)
    return (pd.concat([matches_hist, actual_df], ignore_index=True)
            .sort_values("date").reset_index(drop=True))

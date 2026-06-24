"""
squad_values.py — optional Transfermarkt + gradient-boosting hybrid.

Downloads current squad market values, sums the top-23 players per nation, and
fits a ``HistGradientBoostingRegressor`` (Poisson loss) on engineered features
(Elo diff, rolling form, rest days, log squad-value diff, home flag). The fitted
model yields an expected-goals function ``xg_gb(a, b, home)`` that the tournament
layer can use *instead of* the linear Poisson model.

Everything here is best-effort: if the download or fit fails (e.g. offline, or
``config.USE_SQUAD_VALUE_GB`` is False) the caller falls back to Poisson+Elo.
Squad values are a *static* cross-sectional quality proxy — an approximation for
historical rows, reasonable because squad tier changes slowly and Elo already
captures historical form.
"""

from __future__ import annotations

import gzip
import urllib.request

import numpy as np
import pandas as pd

import config
from constants import ALL_TEAMS, TM_MAP


_SQUAD_VALUE_CACHE: tuple[dict[str, float], float, list[str]] | None = None


def _load_squad_values() -> tuple[dict[str, float], float, list[str]]:
    """Download Transfermarkt values; return ``(log_value_by_team, median, missing)``.

    Cached for the process: squad values depend only on the team list and
    SQUAD_SIZE, so both pipeline scenarios reuse a single download.
    """
    global _SQUAD_VALUE_CACHE
    if _SQUAD_VALUE_CACHE is not None:
        return _SQUAD_VALUE_CACHE
    req = urllib.request.Request(config.TRANSFERMARKT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        with gzip.open(resp, "rt", encoding="utf-8") as f:
            tm = pd.read_csv(f, usecols=["country_of_citizenship", "market_value_in_eur"])

    tm["nation"] = tm["country_of_citizenship"].map(lambda x: TM_MAP.get(x, x))
    tm = tm.dropna(subset=["market_value_in_eur", "nation"]).query("market_value_in_eur > 0")

    squad_val: dict[str, float] = {}
    missing: list[str] = []
    for t in ALL_TEAMS:
        top = tm[tm["nation"] == t].nlargest(config.SQUAD_SIZE, "market_value_in_eur")
        if len(top) >= 5:
            squad_val[t] = float(np.log1p(top["market_value_in_eur"].sum()))
        else:
            missing.append(t)
    median_sv = float(np.median(list(squad_val.values()))) if squad_val else 0.0
    for t in missing:
        squad_val[t] = median_sv
    _SQUAD_VALUE_CACHE = (squad_val, median_sv, missing)
    return _SQUAD_VALUE_CACHE


def _engineer_features(d: pd.DataFrame, ratings: dict, squad_val: dict, med_sv: float) -> pd.DataFrame:
    """Build the GB training frame: one row per (team, opponent) scoring event."""
    d = d.sort_values("date").reset_index(drop=True)
    feats = []
    last: dict = {}
    gfh: dict = {}
    gah: dict = {}
    for r in d.itertuples(index=False):
        for team, opp, gf, ga, ih in [
            (r.home_team, r.away_team, r.home_score, r.away_score, int(not r.neutral_b)),
            (r.away_team, r.home_team, r.away_score, r.home_score, 0),
        ]:
            rest = (r.date - last[team]).days if team in last else 180
            feats.append({
                "goals": float(gf),
                "elo_diff": (ratings.get(team, 1500) - ratings.get(opp, 1500)) / 100.0,
                "form_gf": float(np.mean(gfh[team][-5:])) if team in gfh else 1.2,
                "form_ga": float(np.mean(gah[team][-5:])) if team in gah else 1.2,
                "rest_days": float(min(rest, 365)),
                "squad_val_diff": squad_val.get(team, med_sv) - squad_val.get(opp, med_sv),
                "is_home": float(ih),
            })
        for team, gf, ga in [(r.home_team, r.home_score, r.away_score),
                             (r.away_team, r.away_score, r.home_score)]:
            gfh.setdefault(team, []).append(gf)
            gah.setdefault(team, []).append(ga)
            last[team] = r.date
    return pd.DataFrame(feats)


FEAT_COLS = ["elo_diff", "form_gf", "form_ga", "rest_days", "squad_val_diff", "is_home"]


def build_squad_value_model(matches_all: pd.DataFrame, ratings: dict):
    """Return ``(xg_gb, info)`` or ``(None, info)`` on any failure.

    ``xg_gb(a, b, home=0)`` gives expected goals for ``a`` against ``b``.
    ``info`` always carries an ``ok`` flag and a human-readable ``message``.
    """
    if not config.USE_SQUAD_VALUE_GB:
        return None, {"ok": False, "message": "USE_SQUAD_VALUE_GB is False — staying on Poisson+Elo."}

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_poisson_deviance

        squad_val, median_sv, missing = _load_squad_values()

        fd = _engineer_features(
            matches_all[matches_all.date >= config.TRAIN_SINCE], ratings, squad_val, median_sv)
        cut = int(len(fd) * 0.85)
        gb = HistGradientBoostingRegressor(
            loss="poisson", max_iter=config.GB_MAX_ITER, learning_rate=config.GB_LEARNING_RATE,
            max_depth=config.GB_MAX_DEPTH, min_samples_leaf=config.GB_MIN_SAMPLES_LEAF,
            random_state=config.RNG_SEED,
        ).fit(fd[FEAT_COLS].values[:cut], fd["goals"].values[:cut])
        dev = mean_poisson_deviance(
            fd["goals"].values[cut:], np.clip(gb.predict(fd[FEAT_COLS].values[cut:]), 1e-6, None))

        # Most-recent 5-match form snapshot per WC team for prediction time.
        recent_form: dict = {}
        for t in ALL_TEAMS:
            g = matches_all[(matches_all.home_team == t) | (matches_all.away_team == t)].tail(10)
            gf, ga = [], []
            for r in g.itertuples(index=False):
                if r.home_team == t:
                    gf.append(r.home_score); ga.append(r.away_score)
                else:
                    gf.append(r.away_score); ga.append(r.home_score)
            recent_form[t] = (gf[-5:] or [1.2], ga[-5:] or [1.2])

        def xg_gb(a: str, b: str, home: int = 0) -> float:
            feat = np.array([[
                (ratings.get(a, 1500) - ratings.get(b, 1500)) / 100.0,
                np.mean(recent_form[a][0]),
                np.mean(recent_form[a][1]),
                4.0,  # typical rest days during a tournament
                squad_val.get(a, median_sv) - squad_val.get(b, median_sv),
                float(home),
            ]])
            return float(np.clip(gb.predict(feat)[0], 0.05, 15.0))

        top5 = [(t, f"€{np.expm1(v) / 1e6:.0f}M")
                for t, v in sorted(squad_val.items(), key=lambda x: -x[1])[:5]]
        info = {
            "ok": True,
            "message": (f"GB hybrid active. Squad values: {48 - len(missing)}/48 found, "
                        f"{len(missing)} imputed. Holdout Poisson deviance {dev:.4f}."),
            "holdout_deviance": dev,
            "missing": missing,
            "top5_value": top5,
        }
        return xg_gb, info

    except Exception as e:  # noqa: BLE001 — best-effort; any failure => fallback
        return None, {"ok": False, "message": f"Squad-value hybrid skipped: {str(e)[:140]}"}

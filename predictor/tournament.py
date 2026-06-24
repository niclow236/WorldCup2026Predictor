"""
tournament.py — lambda matrices, group/knockout simulation, Monte Carlo.

The simulator is fully *conditioned* on entered results: where you have entered
a real group score it is used verbatim, and where you have entered a real
knockout winner that team advances. Everything still to come is sampled from the
goals model.

Design
------
``build_lambda_matrices`` pre-computes expected goals for every ordered team
pair once (neutral and home variants), so a single tournament simulation is just
Poisson draws and comparisons — ~0.3 ms each. ``TournamentSimulator`` bundles
those matrices with the realised-results dicts and exposes ``sim_tournament``
(one realisation) and ``monte_carlo`` (many).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from constants import (ALL_TEAMS, GROUP_FIXTURES, GROUPS, HOSTS, KO_MATCH_ORDER,
                       LATER, THIRD_PLACE_TABLE, WINNER_SLOT_ORDER)

INITIAL_RATING = 1500.0


def build_lambda_matrices(xg_fn) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """Pre-compute expected-goals matrices for all ordered team pairs.

    ``xg_fn(attacker, defender, home)`` -> expected goals. Returns
    ``(LAM_NEU, LAM_HOME, idx, all_teams)`` where ``idx`` maps team -> row index.
    """
    all_teams = list(ALL_TEAMS)
    idx = {t: i for i, t in enumerate(all_teams)}
    n = len(all_teams)
    lam_neu = np.zeros((n, n))
    lam_home = np.zeros((n, n))
    for a in all_teams:
        for b in all_teams:
            if a == b:
                continue
            lam_neu[idx[a], idx[b]] = xg_fn(a, b, 0)
            lam_home[idx[a], idx[b]] = xg_fn(a, b, 1)
    return lam_neu, lam_home, idx, all_teams


def r32_pairs(W: dict, R: dict, T: dict) -> dict[int, tuple[str, str]]:
    """Round-of-32 fixtures (Wikipedia match numbers) from group winners (W),
    runners-up (R) and third-place slot assignments (T)."""
    return {
        73: (R["A"], R["B"]), 74: (W["E"], T["E"]), 75: (W["F"], R["C"]), 76: (W["C"], R["F"]),
        77: (W["I"], T["I"]), 78: (R["E"], R["I"]), 79: (W["A"], T["A"]), 80: (W["L"], T["L"]),
        81: (W["D"], T["D"]), 82: (W["G"], T["G"]), 83: (R["K"], R["L"]), 84: (W["H"], R["J"]),
        85: (W["B"], T["B"]), 86: (W["J"], R["H"]), 87: (W["K"], T["K"]), 88: (R["D"], R["G"]),
    }


class TournamentSimulator:
    """Holds pre-computed lambdas + entered results and simulates tournaments."""

    def __init__(self, lam_neu, lam_home, idx, all_teams, ratings,
                 group_actual: dict, ko_actual: dict):
        self.lam_neu = lam_neu
        self.lam_home = lam_home
        self.idx = idx
        self.all_teams = all_teams
        self.n_teams = len(all_teams)
        self.elo_arr = np.array([ratings.get(t, INITIAL_RATING) for t in all_teams])
        self.group_actual = group_actual
        self.ko_actual = ko_actual
        self.pen_mode = self._compute_pen_mode()

    # -- shootouts ----------------------------------------------------------
    def pen_p(self, a: str, b: str) -> float:
        """Probability ``a`` wins a shootout — mild Elo tilt, near coin-flip."""
        return 1.0 / (1.0 + 10.0 ** (-(self.elo_arr[self.idx[a]] - self.elo_arr[self.idx[b]]) / 2000.0))

    @staticmethod
    def _compute_pen_mode(n: int = 4000, p: float = 0.76, seed: int = 7):
        """Modal (winner, loser) shootout score — used to display projected pens."""
        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(n):
            kw = rng.binomial(5, p)
            kl = rng.binomial(5, p)
            iters = 0
            while kw == kl:
                kw += int(rng.random() < p)
                kl += int(rng.random() < p)
                iters += 1
                if iters > 20:
                    kw += 1
                    break
            scores.append((kw, kl) if kw > kl else (kl, kw))
        return Counter(scores).most_common(1)[0][0]

    # -- single fixtures ----------------------------------------------------
    def sim_group(self, teams: list[str], rng):
        """Play a group's six games (entered scores used where present)."""
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        for i, j in GROUP_FIXTURES:
            a, b = teams[i], teams[j]
            key = frozenset({a, b})
            if key in self.group_actual:
                ga_, gb_ = self.group_actual[key]["by_team"][a], self.group_actual[key]["by_team"][b]
            else:
                la = self.lam_home[self.idx[a], self.idx[b]] if a in HOSTS else self.lam_neu[self.idx[a], self.idx[b]]
                lb = self.lam_home[self.idx[b], self.idx[a]] if b in HOSTS else self.lam_neu[self.idx[b], self.idx[a]]
                ga_, gb_ = rng.poisson(la), rng.poisson(lb)
            gf[a] += ga_; ga[a] += gb_; gf[b] += gb_; ga[b] += ga_
            if ga_ > gb_:
                pts[a] += 3
            elif ga_ < gb_:
                pts[b] += 3
            else:
                pts[a] += 1; pts[b] += 1
        # FIFA tie-breakers: points -> GD -> GF -> random (unmodelled fair-play/lots).
        order = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng.random()), reverse=True)
        return order, {t: {"pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in teams}

    def ko(self, a: str, b: str, rng) -> str:
        """Play a knockout match; return the winner (entered winner used if present)."""
        key = frozenset({a, b})
        if key in self.ko_actual:
            r = self.ko_actual[key]
            ga_, gb_ = r["by_team"][a], r["by_team"][b]
            if ga_ > gb_:
                return a
            if gb_ > ga_:
                return b
            return r["pen_winner"]
        # Regulation.
        ga_, gb_ = rng.poisson(self.lam_neu[self.idx[a], self.idx[b]]), rng.poisson(self.lam_neu[self.idx[b], self.idx[a]])
        if ga_ != gb_:
            return a if ga_ > gb_ else b
        # Extra time at ~1/3 scoring rate.
        ea_, eb_ = rng.poisson(self.lam_neu[self.idx[a], self.idx[b]] / 3.0), rng.poisson(self.lam_neu[self.idx[b], self.idx[a]] / 3.0)
        if ea_ != eb_:
            return a if ea_ > eb_ else b
        # Shootout.
        return a if rng.random() < self.pen_p(a, b) else b

    # -- third-place + full tournament -------------------------------------
    def resolve_thirds(self, group_order: dict, group_st: dict, rng):
        """Pick the eight best third-placed teams and their R32 slot mapping."""
        thirds = [(g, group_order[g][2], group_st[g][group_order[g][2]]) for g in GROUPS]
        thirds.sort(key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"], rng.random()), reverse=True)
        best8 = thirds[:8]
        qual = frozenset(g for g, _, _ in best8)
        third_team = {g: t for g, t, _ in best8}
        return qual, third_team, [g for g, _, _ in best8]

    def sim_tournament(self, rng) -> dict:
        """One full tournament realisation. Returns reached-stage team lists."""
        order, st = {}, {}
        for g, teams in GROUPS.items():
            o, s = self.sim_group(teams, rng)
            order[g] = o; st[g] = s
        W = {g: order[g][0] for g in GROUPS}
        R = {g: order[g][1] for g in GROUPS}
        qual, third_team, best8_groups = self.resolve_thirds(order, st, rng)
        T = {slot: third_team[src] for slot, src in THIRD_PLACE_TABLE[qual].items()}
        win = {}
        for no, (a, b) in r32_pairs(W, R, T).items():
            win[no] = self.ko(a, b, rng)
        for no in [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104]:
            f1, f2 = LATER[no]
            win[no] = self.ko(win[f1], win[f2], rng)
        return {
            "champion": win[104],
            "finalists": [win[101], win[102]],
            "semifinalists": [win[97], win[98], win[99], win[100]],
            "last16": [win[m] for m in range(89, 97)],
            "order": order,
            "best8_groups": best8_groups,
        }

    # -- Monte Carlo --------------------------------------------------------
    def monte_carlo(self, n_sims: int, seed: int):
        """Run ``n_sims`` tournaments; tally stage-reach and group-finish rates.

        Returns ``(forecast_df, pos, q3, n_sims)`` where ``forecast_df`` is sorted
        by champion %, ``pos[g][team]`` is P(finish 1st..4th), and ``q3[team]`` is
        P(qualify as a best third).
        """
        import pandas as pd

        rng = np.random.default_rng(seed)
        n = self.n_teams
        champ = np.zeros(n); fin = np.zeros(n); semi = np.zeros(n); r16 = np.zeros(n)
        pos = {g: {t: np.zeros(4) for t in GROUPS[g]} for g in GROUPS}
        q3 = {t: 0 for t in self.all_teams}
        for _ in range(n_sims):
            s = self.sim_tournament(rng)
            champ[self.idx[s["champion"]]] += 1
            for t in s["finalists"]:
                fin[self.idx[t]] += 1
            for t in s["semifinalists"]:
                semi[self.idx[t]] += 1
            for t in s["last16"]:
                r16[self.idx[t]] += 1
            for g in GROUPS:
                for rank, t in enumerate(s["order"][g]):
                    pos[g][t][rank] += 1
            for g in s["best8_groups"]:
                third = s["order"][g][2]
                q3[third] += 1
        forecast = pd.DataFrame({
            "team": self.all_teams,
            "elo": self.elo_arr.round(0).astype(int),
            "champion_%": 100 * champ / n_sims,
            "final_%": 100 * fin / n_sims,
            "semifinal_%": 100 * semi / n_sims,
            "reach_R16_%": 100 * r16 / n_sims,
        }).sort_values("champion_%", ascending=False).reset_index(drop=True)
        q3_pct = {t: 100 * q3[t] / n_sims for t in self.all_teams}
        return forecast, pos, q3_pct, n_sims

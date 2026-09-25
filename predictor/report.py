"""
report.py: render the final evaluation (``final_evaluation/report.md`` and its figures).

Every figure is drawn twice, for a light and a dark surface, and the report
embeds them with ``<picture>`` so GitHub serves whichever matches the reader's
theme. Colours come from one validated categorical palette (checked for
colour-vision deficiency and contrast in both modes); each chart also has its
numbers in a table right below it, so nothing is readable from colour alone.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "deemph": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "deemph": "#52514e",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}
DPI = 144
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT,
                         "svg.hashsalt": "wc2026", "path.simplify": True})
    return plt


def style_axes(ax, th, grid_axis="x"):
    """Recessive chrome: hairline solid grid on one axis, muted ticks, no box."""
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=th["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], length=0, labelsize=9)


def _hbar(ax, y, value, height, color, radius_px=4.0):
    """Horizontal bar from 0 to ``value``: square at the baseline, rounded data end."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    if value <= 0:
        return
    x_per_px = np.diff(ax.get_xlim())[0] / ax.bbox.width
    y_per_px = np.diff(ax.get_ylim())[0] / ax.bbox.height
    r = min(radius_px * x_per_px, value / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), value, height, linewidth=0, facecolor=color,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=abs(y_per_px / x_per_px)))
    ax.add_patch(Rectangle((0, y - height / 2), value - r, height, linewidth=0, facecolor=color))


def save_figure(fig, path, th):
    fig.savefig(path, dpi=DPI, facecolor=th["surface"], metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_title_odds(teams: pd.DataFrame, finishes: dict, th: dict, path: str, top: int = 16):
    """Pre-tournament title odds of the top teams, emphasising the final four."""
    plt = pyplot()
    d = teams.head(top).iloc[::-1].reset_index(drop=True)
    semis = d["actual_rounds"] >= 4
    fig, ax = plt.subplots(figsize=(8.2, 0.36 * top + 1.3), dpi=DPI)
    fig.patch.set_facecolor(th["surface"])
    style_axes(ax, th, "x")
    xmax = float(d["champion_%"].max()) * 1.18
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(d) - 0.4)
    fig.subplots_adjust(left=0.19, right=0.80, top=0.86, bottom=0.10)
    band_px = ax.bbox.height / len(d)
    h = min(0.62, 18.0 / band_px)                       # <= 18px thick, air between bars
    for i, r in d.iterrows():
        colour = th["series"][0] if semis[i] else th["deemph"]
        _hbar(ax, i, r["champion_%"], h, colour)
        ax.text(r["champion_%"] + xmax * 0.012, i, f"{r['champion_%']:.1f}%", va="center",
                ha="left", fontsize=9, color=th["ink"] if semis[i] else th["ink2"])
        ax.text(1.02, i, finishes[r["team"]], transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=9, color=th["ink"] if semis[i] else th["muted"],
                fontweight="bold" if r["actual_rounds"] == 6 else "normal")
    ax.set_yticks(range(len(d)), d["team"])
    ax.tick_params(axis="y", labelsize=9.5)
    for lbl, semi in zip(ax.get_yticklabels(), semis):
        lbl.set_color(th["ink"] if semi else th["ink2"])
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
    ax.text(1.02, len(d) - 0.1, "Actual finish", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=9, color=th["muted"])
    fig.text(0.02, 0.965, "Pre-tournament title odds vs where each team finished",
             fontsize=12.5, fontweight="bold", color=th["ink"], va="top")
    fig.text(0.02, 0.915, f"Top {top} of 48 teams by P(win the World Cup), forecast before a ball was kicked",
             fontsize=9.5, color=th["ink2"], va="top")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=th["series"][0], label="Reached the semi-finals"),
                       Patch(color=th["deemph"], label="Went out earlier")],
              loc="lower right", frameon=False, fontsize=9, labelcolor=th["ink2"],
              handlelength=1.0, handleheight=0.8)
    save_figure(fig, path, th)


def fig_title_timeline(timeline: pd.DataFrame, teams: list[str], finishes: dict, th: dict, path: str):
    """Small multiples: each final-four team's title odds after every round,
    highlighted against the other three in grey."""
    plt = pyplot()
    piv = timeline.pivot_table(index="checkpoint", columns="team", values="champion_%", sort=False)
    labels = list(piv.index)
    short = [s.replace("After ", "").replace("group stage", "Groups").replace("matchday ", "MD")
             .replace("round of ", "R").replace("quarter-finals", "QF").replace("semi-finals", "SF")
             .replace("Pre-tournament", "Pre").replace("Final result", "Result") for s in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), dpi=DPI, sharex=True, sharey=True)
    fig.patch.set_facecolor(th["surface"])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.09, hspace=0.34, wspace=0.08)
    for ax, team in zip(axes.ravel(), teams):
        style_axes(ax, th, "y")
        for other in teams:
            if other != team:
                ax.plot(x, piv[other], color=th["deemph"], linewidth=1.2, solid_capstyle="round")
        y = piv[team].to_numpy()
        ax.plot(x, y, color=th["series"][0], linewidth=2.2, solid_joinstyle="round",
                solid_capstyle="round", marker="o", markersize=5.5,
                markeredgecolor=th["surface"], markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{y[0]:.0f}%", (x[0], y[0]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=th["ink2"])
        peak = int(np.argmax(y[:-1]))                  # last forecast before the result
        if peak != 0:
            ax.annotate(f"{y[peak]:.0f}%", (x[peak], y[peak]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=th["ink"])
        ax.set_title(f"{team} ({finishes[team].lower()})", loc="left", fontsize=10.5, color=th["ink"],
                     fontweight="bold", pad=6)
        ax.set_ylim(-4, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
        ax.set_xticks(x, short)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.text(0.02, 0.965, "How the title odds of the final four evolved", fontsize=12.5,
             fontweight="bold", color=th["ink"], va="top")
    fig.text(0.02, 0.918, "P(win the World Cup) re-simulated after each round, using only the results known "
             "at that point (grey: the other three semi-finalists)", fontsize=9.5, color=th["ink2"], va="top")
    save_figure(fig, path, th)


def fig_calibration(cal: dict[str, pd.DataFrame], th: dict, path: str):
    """Reliability diagram: forecast probability vs observed frequency."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=DPI)
    fig.patch.set_facecolor(th["surface"])
    fig.subplots_adjust(left=0.13, right=0.96, top=0.83, bottom=0.12)
    style_axes(ax, th, "both")
    ax.plot([0, 1], [0, 1], color=th["axis"], linewidth=1.0, zorder=1)
    ax.text(0.97, 0.92, "perfect calibration", rotation=0, ha="right", va="bottom",
            fontsize=8.5, color=th["muted"])
    for k, (name, df) in enumerate(cal.items()):
        colour = th["series"][k]
        ax.plot(df["mean_forecast"], df["observed_frequency"], color=colour, linewidth=2.0,
                marker="o", markersize=6, markeredgecolor=th["surface"], markeredgewidth=1.5,
                solid_joinstyle="round", label=name, zorder=3 + k)
        # Direct label at the line's end (text in ink; the legend carries the colour key).
        ax.annotate(name, (df["mean_forecast"].iloc[-1], df["observed_frequency"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=th["ink2"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks]); ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Forecast probability of the outcome", fontsize=9.5, color=th["ink2"])
    ax.set_ylabel("How often it happened", fontsize=9.5, color=th["ink2"])
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=th["ink2"])
    fig.text(0.02, 0.965, "Were the match probabilities calibrated?", fontsize=12.5,
             fontweight="bold", color=th["ink"], va="top")
    fig.text(0.02, 0.915, "All 104 matches, 90-minute win / draw / loss probabilities pooled and binned",
             fontsize=9.5, color=th["ink2"], va="top")
    save_figure(fig, path, th)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def final_placings(bracket: pd.DataFrame) -> dict[str, str]:
    """Champion, runner-up, third and fourth, from the final and third-place play-off."""
    final = bracket[bracket["round"] == "Final"].iloc[0]
    third = bracket[bracket["round"] == "3rd"].iloc[0]
    return {final["winner"]: "Champion", final["loser"]: "Runner-up",
            third["winner"]: "Third place", third["loser"]: "Fourth place"}


def finish_labels(res: dict) -> dict[str, str]:
    """Every team's finish, with the semi-finalists split into their final placings."""
    labels = dict(zip(res["teams"]["team"], res["teams"]["actual_finish"]))
    labels.update(final_placings(res["bracket"]))
    return labels


def write_figures(res: dict, out_dir: str) -> None:
    """Render every figure in both themes (``<name>_light.png`` / ``<name>_dark.png``)."""
    finishes = finish_labels(res)
    final_four = list(final_placings(res["bracket"]))
    for mode, th in THEMES.items():
        fig_title_odds(res["teams"], finishes, th, os.path.join(out_dir, f"title_odds_{mode}.png"))
        fig_calibration(res["calibration"], th, os.path.join(out_dir, f"calibration_{mode}.png"))
        if res.get("timeline") is not None:
            fig_title_timeline(res["timeline"], final_four, finishes, th,
                               os.path.join(out_dir, f"title_timeline_{mode}.png"))


def _fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "n/a"
    return format(v, spec)


def _md_table(df: pd.DataFrame, fmt: dict | None = None, right: tuple = ()) -> str:
    """Render ``df`` as a GitHub markdown table.

    Numeric columns, and any pre-formatted ones named in ``right``, are
    right-aligned so their digits line up.
    """
    fmt = fmt or {}
    cols = list(df.columns)
    numeric = {c: c in right or (pd.api.types.is_numeric_dtype(df[c])
                                 and not pd.api.types.is_bool_dtype(df[c])) for c in cols}
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if numeric[c] else "---" for c in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append(_fmt(v, ".3f"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _picture(stem: str, alt: str) -> str:
    """Theme-aware image: GitHub serves the dark rendering to dark-mode readers."""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{stem}_dark.png">\n'
            f'  <img alt="{alt}" src="{stem}_light.png" width="760">\n</picture>')


def _pct(p: float, d: int = 1) -> str:
    """0.184 -> '18.4%'."""
    return _fmt(100 * p, f".{d}f") + "%" if not np.isnan(p) else "n/a"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _day(iso: str, year: bool = True) -> str:
    """'2026-07-19' -> '19 July 2026'."""
    from datetime import date
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%B}" + (f" {d.year}" if year else "")


def _score_from(label: str, first_is_a: bool) -> str:
    """Re-orient a ``"4-6"``-style score label so the named side comes first."""
    import re
    if first_is_a:
        return label
    label = re.sub(r"^(\d+)-(\d+)", r"\2-\1", label)
    return re.sub(r"pens (\d+)-(\d+)", r"pens \2-\1", label)


PROSE_FINISH = {"Champion": "first", "Runner-up": "second", "Third place": "third", "Fourth place": "fourth",
                "Semi-final": "out in the semi-finals", "Quarter-final": "out in the quarter-finals",
                "Round of 16": "out in the round of 16", "Round of 32": "out in the round of 32",
                "Group stage": "out in the group stage"}
YES, NO = "✓", "✗"


# ---------------------------------------------------------------------------
# final_evaluation/report.md
# ---------------------------------------------------------------------------
def write_final_report(res: dict, path: str) -> None:
    """Write the predicted-vs-actual report (markdown; figures and CSVs alongside)."""
    fc = res["pre_forecast"].reset_index(drop=True)
    teams, bracket, comp = res["teams"], res["bracket"], res["comparison"].set_index("model")
    acc, placings, finishes = res["acc_summary"], final_placings(bracket), finish_labels(res)
    champion = next(t for t, p in placings.items() if p == "Champion")
    final = bracket[bracket["round"] == "Final"].iloc[0]
    third = bracket[bracket["round"] == "3rd"].iloc[0]
    rank = {t: i + 1 for i, t in enumerate(fc["team"])}
    p_title = dict(zip(fc["team"], fc["champion_%"] / 100))
    per_match, per_round = res["bracket_match"], res["bracket_round"].set_index("stage")
    pm = per_match.set_index("match")
    ref, live_name = "Poisson + Elo", res.get("live_name")
    matches = res["matches"]
    ko_df = matches[~matches["round"].str.startswith("Group")]
    ko_miss = ko_df[ko_df["pre_advance_hit"] == 0]
    pens_miss = int(ko_miss["actual_score"].str.contains("pens").sum())
    grp_cmp = res["groups"]
    models = comp.drop(index=[m for m in comp.index if m.startswith(("Base rate", "Elo favourite"))])
    best = models["all90_rps"].idxmin()
    live_row = comp.loc[live_name] if live_name in comp.index else None
    L: list[str] = []

    # --- headline -----------------------------------------------------------
    final_score = _score_from(final["score"], final["winner"] == final["team_a"])
    L += ["# World Cup 2026: predicted vs actual", "",
          f"_The final evaluation of this forecasting pipeline: every forecast it made, scored against all 104 "
          f"matches of the 2026 FIFA World Cup ({_day(res['first_date'], year=False)} to "
          f"{_day(res['last_date'])}). Generated by `run_pipeline.py`; every number below is reproducible "
          f"from the repository._", "",
          f"**{champion} are world champions**, beating {final['loser']} {final_score.replace(' aet', '')}"
          f"{' after extra time' if 'aet' in final_score else ''} in the final. {third['winner']} beat "
          f"{third['loser']} {_score_from(third['score'], third['winner'] == third['team_a'])} for third place.", ""]

    sf_hit = int(per_round.loc["Semi-finals", "correct"])
    bits = ["all 4 semi-finalists" if sf_hit == 4 else f"{sf_hit} of 4 semi-finalists"]
    if pm.loc[101, "fixture_hit"] and pm.loc[102, "fixture_hit"]:
        bits.append("both semi-final pairings")
    if pm.loc[104, "fixture_hit"]:
        bits.append(f"the final ({final['team_a']} v {final['team_b']})")
    if pm.loc[104, "winner_hit"]:
        bits.append("the champion")
    if pm.loc[103, "fixture_hit"] and pm.loc[103, "winner_hit"]:
        bits.append(f"the third-place play-off ({third['winner']} over {third['loser']})")
    fav = fc.iloc[0]
    fav_line = (f"**{fav['team']}, {fav['champion_%']:.1f}%**, the eventual champions"
                if fav["team"] == champion else
                f"**{fav['team']}, {fav['champion_%']:.1f}%** ({champion} were {_ordinal(rank[champion])} "
                f"at {100 * p_title[champion]:.1f}%)")
    b = models.loc[best]
    best_line = f"**{best}** (RPS {b['all90_rps']:.4f})"
    if best != ref:
        best_line += (f", ahead of {ref} but within noise (95% CI of the difference "
                      f"{b['rps_vs_ref_lo']:+.4f} to {b['rps_vs_ref_hi']:+.4f})"
                      if b["rps_vs_ref_hi"] > 0 else f", clearly ahead of {ref}")
    L += ["## The verdict", "",
          "| Question | Answer |", "|---|---|",
          f"| Pre-tournament favourite | {fav_line} |",
          f"| Knockout bracket (the model's single most-likely path) | {_join(bits)} |",
          f"| 90-minute results called, all 104 matches | **{_pct(acc['all90_outcome_accuracy'])}** "
          f"(always backing the first-named team: "
          f"{_pct(comp.loc['Base rate (no skill)', 'all90_outcome_accuracy'])}) |",
          f"| Knockout winners called | **{int(round(acc['ko_winner_accuracy'] * acc['n_ko']))} of {acc['n_ko']}**"
          + (f"; {pens_miss} of the {len(ko_miss)} misses were penalty shoot-outs" if pens_miss else "") + " |",
          f"| Group winners · round-of-32 qualifiers called | **{int(grp_cmp['winner_hit'].sum())} of 12** · "
          f"**{int(per_round.loc['Round of 32', 'correct'])} of 32** |",
          f"| Probability quality: RPS over all 104 matches (lower is better) | **{acc['all90_rps']:.3f}** vs "
          f"{acc['all90_baseline_rps']:.3f} for a no-skill forecast |",
          f"| Best model on probability quality | {best_line} |"]
    if live_row is not None:
        noise = live_row["rps_vs_ref_lo"] < 0 < live_row["rps_vs_ref_hi"]
        L.append(f"| Did re-fitting on tournament results help? | **{'Yes' if live_row['rps_vs_ref'] < 0 else 'No'}**: "
                 f"the live model scored RPS {live_row['all90_rps']:.4f} vs {comp.loc[ref, 'all90_rps']:.4f} for the "
                 f"frozen pre-tournament model{' (within noise)' if noise else ''} |")
    L += ["", "Unless stated otherwise, \"the model\" is the pre-tournament forecast: trained only on matches "
          "played before kick-off and never updated. RPS is the ranked probability score, the standard accuracy "
          "measure for win/draw/loss forecasts.", ""]

    # --- 1 · favourites ------------------------------------------------------
    top = teams.head(16).copy()
    top.insert(0, "#", range(1, len(top) + 1))
    top["finish"] = top["team"].map(finishes)
    top3 = list(fc["team"][:3])
    s = (f"The model's three favourites were {_join(f'{t} ({100 * p_title[t]:.1f}%)' for t in top3)}; they "
         f"finished {_join(PROSE_FINISH[finishes[t]] for t in top3)}.")
    for t in [t for t in placings if rank[t] > 3]:
        s += f" {t} ({_ordinal(rank[t])} favourite, {100 * p_title[t]:.1f}%) finished {PROSE_FINISH[placings[t]]}."
    early = [t for t in fc["team"][:8] if teams.set_index("team").loc[t, "actual_rounds"] <= 2]
    if early:
        by_exit: dict[str, list[str]] = {}
        for t in early:
            by_exit.setdefault(finishes[t], []).append(f"{t} ({_ordinal(rank[t])})")
        s += (" Among the other top-eight favourites, "
              + _join(f"{_join(names)} went {PROSE_FINISH[fin]}" for fin, names in by_exit.items()) + ".")
    pct_cols = ["P(R16)", "P(QF)", "P(SF)", "P(final)", "P(title)"]
    L += ["## 1 · The favourites vs reality", "",
          _picture("title_odds", "Pre-tournament title odds of the top 16 teams, with each team's actual finish"),
          "", s, "",
          _md_table(top[["#", "team", "elo_pre", "reach_R16_%", "quarterfinal_%", "semifinal_%", "final_%",
                         "champion_%", "finish"]].rename(columns=dict(zip(
                             ["team", "elo_pre", "reach_R16_%", "quarterfinal_%", "semifinal_%", "final_%",
                              "champion_%", "finish"], ["Team", "Elo"] + pct_cols + ["Actual finish"]))),
                    {c: (lambda v: f"{v:.1f}%") for c in pct_cols}), ""]

    # --- 2 · stage by stage ---------------------------------------------------
    st = res["stage"]
    L += ["## 2 · Stage by stage", "",
          "Each stage-reach probability (48 teams × 6 stages) scored against how far every team really went. "
          "\"Uniform\" is the no-skill forecast that gives every team the same chance.", "",
          _md_table(pd.DataFrame({
              "Stage": st["stage"], "Teams": st["teams"],
              "Model's top-N that made it": [f"{h} of {n}" for h, n in zip(st["top_n_hits"], st["teams"])],
              "Avg. probability of those that did": st["mean_p_of_actual"].map(_pct),
              "Brier (model)": st["brier"].map(lambda v: f"{v:.4f}"),
              "Brier (uniform)": st["uniform_brier"].map(lambda v: f"{v:.4f}"),
              "Skill": st["brier_skill"].map(lambda v: _pct(v, 0)),
          }), right=("Model's top-N that made it", "Avg. probability of those that did", "Brier (model)",
                     "Brier (uniform)", "Skill")), "",
          f"Every stage beats the uniform forecast (Brier skill {_pct(st['brier_skill'].min(), 0)} to "
          f"{_pct(st['brier_skill'].max(), 0)}). The \"top-N\" column ranks teams by probability alone. The "
          f"bracket path in section 6 also respects who meets whom, which is why it finds {sf_hit} of 4 "
          f"semi-finalists where the probability ranking finds {int(st.iloc[3]['top_n_hits'])}.", ""]

    # --- 3 · model comparison -------------------------------------------------
    cmp_tbl = res["comparison"]

    def ci(r):
        if r["model"] == ref:
            return "reference"
        if np.isnan(r["rps_vs_ref"]):
            return "n/a"
        return f"{r['rps_vs_ref']:+.4f} ({r['rps_vs_ref_lo']:+.4f} to {r['rps_vs_ref_hi']:+.4f})"

    L += ["## 3 · Match predictions: which model was best?", "",
          "All models are scored on the same 104 matches. Group games use the 90-minute result; knockout ties are "
          "scored on their 90-minute result (a tie that went to extra time counts as a draw) and, separately, on "
          "who went through. The last column compares each model's per-match RPS with Poisson + Elo (negative = "
          "better), with a 95% paired-bootstrap interval: an interval that spans zero means 104 matches cannot "
          "separate the two.", "",
          _md_table(pd.DataFrame({
              "Model": cmp_tbl["model"],
              "Results called (90 min)": cmp_tbl["all90_outcome_accuracy"].map(_pct),
              "RPS": cmp_tbl["all90_rps"].map(lambda v: _fmt(v, ".4f")),
              "Group log-loss": cmp_tbl["group_logloss"].map(lambda v: _fmt(v, ".3f")),
              "Knockout winners": cmp_tbl["ko_winner_accuracy"].map(_pct),
              "Knockout Brier": cmp_tbl["ko_advance_brier"].map(lambda v: _fmt(v, ".3f")),
              "Goal MAE": cmp_tbl["goal_mae"].map(lambda v: _fmt(v, ".3f")),
              "RPS vs Poisson + Elo (95% CI)": cmp_tbl.apply(ci, axis=1),
          }), right=("Results called (90 min)", "RPS", "Group log-loss", "Knockout winners", "Knockout Brier",
                     "Goal MAE")), ""]
    if "Poisson (no Elo)" in comp.index:
        ne = comp.loc["Poisson (no Elo)"]
        L.append(f"- **The Elo feature earns its place.** Without it the Poisson model calls "
                 f"{_pct(ne['all90_outcome_accuracy'])} of results instead of {_pct(comp.loc[ref, 'all90_outcome_accuracy'])}, "
                 f"and its RPS is worse by {ne['rps_vs_ref']:.4f} (95% CI {ne['rps_vs_ref_lo']:+.4f} to "
                 f"{ne['rps_vs_ref_hi']:+.4f}).")
    if "GB squad-value hybrid" in comp.index:
        gb = comp.loc["GB squad-value hybrid"]
        L.append(f"- **The GB squad-value hybrid**, the engine behind the Monte-Carlo title odds, produced the best "
                 f"probabilities (RPS {gb['all90_rps']:.4f}, group log-loss {gb['group_logloss']:.3f}) while calling "
                 f"{_pct(gb['all90_outcome_accuracy'])} of results. Its edge over Poisson + Elo is suggestive rather "
                 f"than conclusive (95% CI {gb['rps_vs_ref_lo']:+.4f} to {gb['rps_vs_ref_hi']:+.4f}).")
    if live_row is not None:
        ph = res["phases"]
        worse = [c for c in ph.columns if ph.loc[live_name, c] > ph.loc[ref, c] + 0.002]
        where = [f"on {c.lower()}" if c.startswith("Matchday") else f"in the {c.lower()}" for c in worse]
        L.append(f"- **Re-fitting during the tournament did not help.** The live model (the pipeline's own to-date "
                 f"procedure, re-run before each of the {res['n_match_days']} match days on the results known by "
                 f"then) scored RPS {live_row['all90_rps']:.4f} against {comp.loc[ref, 'all90_rps']:.4f}. "
                 + (f"It kept pace early and fell behind {_join(where)} (table below). " if where else "")
                 + "World-Cup games carry the heaviest Elo weight, so a handful of results moved the ratings "
                 "further than they deserved; matchday 3 also brings dead rubbers and rotated squads.")
        L += ["", _md_table(ph.reset_index().rename(columns={"model": "Mean RPS by phase"}),
                            {c: (lambda v: _fmt(v, ".4f")) for c in ph.columns})]
    L.append("")

    # --- 4 · title odds over time ----------------------------------------------
    tl = res.get("timeline")
    if tl is not None:
        piv = tl.pivot_table(index="team", columns="checkpoint", values="champion_%", sort=False)
        cps = list(piv.columns)
        forecasts = [c for c in cps if c != "Final result"]
        leaders = piv[forecasts].idxmax()
        first_g = matches[matches["round"].str.startswith("Group")
                          & ((matches["team_a"] == champion) | (matches["team_b"] == champion))].iloc[0]
        opp = first_g["team_b"] if first_g["team_a"] == champion else first_g["team_a"]
        opening = _score_from(first_g["actual_score"], first_g["team_a"] == champion)
        c0, c1 = piv.loc[champion, cps[0]], piv.loc[champion, cps[1]]
        s = (f"The full model (GB hybrid, {res['n_sims']:,} simulated tournaments) was re-run after every round using "
             f"only the results known at that point. {champion} started "
             + (f"as favourites at {c0:.1f}%" if leaders[cps[0]] == champion else f"at {c0:.1f}%")
             + f"; after their opening {opening} against {opp} the model moved them to {c1:.1f}%")
        switch = [c for c in forecasts[1:] if leaders[c] != champion]
        if switch:
            s += f", and {leaders[switch[0]]} became favourites"
            last = leaders[forecasts[-1]]
            if last != champion:
                s += (f". {last} were still favourites going into the final ({piv.loc[last, forecasts[-1]]:.1f}% vs "
                      f"{piv.loc[champion, forecasts[-1]]:.1f}%)")
                if leaders[cps[0]] == champion:
                    s += ". In hindsight the pre-tournament call was the better one"
            else:
                s += f", before {champion} took the lead back"
        L += ["## 4 · How the title odds evolved", "",
              _picture("title_timeline", "Title odds of the final four after each round"), "", s + ".", ""]
        show = list(placings) + [t for t in fc["team"][:6] if t not in placings]
        L += [_md_table(piv.loc[show].reset_index().rename(columns={"team": "Team"}),
                        {c: (lambda v: f"{v:.1f}%") for c in cps}), ""]

    # --- 5 · group stage --------------------------------------------------------
    g = grp_cmp
    exact = g.loc[g["exact_order_hit"], "group"].tolist()
    s = (f"The model named **{int(g['winner_hit'].sum())} of 12 group winners**, the top two (in either order) in "
         f"{int(g['top2_hit'].sum())} groups and the exact finishing order in {len(exact)}"
         + (f" ({_join(exact)})" if exact else "")
         + f". It had {int(per_round.loc['Round of 32', 'correct'])} of the 32 round-of-32 qualifiers.")
    wrong = g[~g["winner_hit"]]
    if len(wrong):
        s += " Wrong group winners: " + "; ".join(
            f"Group {r['group']} went to {r['actual_order'].split(' > ')[0]} (the model had "
            f"{r['predicted_order'].split(' > ')[0]}, and gave {r['actual_order'].split(' > ')[0]} "
            f"{r['p_actual_winner_%']:.0f}%)" for _, r in wrong.iterrows()) + "."
    thirds = res["thirds"]
    L += ["## 5 · Group stage", "", s, "",
          f"Qualifying third-placed teams: {_join(thirds.loc[thirds['qualified'], 'team'])}.", "",
          _md_table(pd.DataFrame({
              "Group": g["group"], "Predicted order": g["predicted_order"], "Actual order": g["actual_order"],
              "Winner": g["winner_hit"].map({True: YES, False: NO}),
              "Positions right": g["positions_correct"].map(lambda v: f"{v} of 4"),
              "P(actual winner)": g["p_actual_winner_%"].map(lambda v: f"{v:.0f}%"),
          })), ""]

    # --- 6 · knockout bracket ----------------------------------------------------
    pr = res["bracket_round"]
    rounds = ["R32", "R16", "QF", "SF", "3rd", "Final"]
    agg = per_match.groupby("round", sort=False).agg(n=("match", "size"), fx=("fixture_hit", "sum"),
                                                        wn=("winner_hit", "sum")).reindex(rounds)
    L += ["## 6 · The knockout bracket, projected vs actual", "",
          "The pre-tournament model's single most-likely path through the bracket (modal group finishers, then "
          "the likelier side of each tie) against what happened. \"Correct\" counts the teams that reached each "
          "stage, whichever side of the draw they came through.", "",
          _md_table(pd.DataFrame({
              "Stage": pr["stage"], "Correct": [f"{c} of {n}" for c, n in zip(pr["correct"], pr["teams"])],
              "Projected but missed": pr["projected_but_missed"].replace("", "none"),
              "Unexpected arrivals": pr["unexpected_arrivals"].replace("", "none"),
          })), "",
          "Tie by tie (each match number compared with its projected counterpart):", "",
          _md_table(pd.DataFrame({
              "Round": agg.index, "Ties": agg["n"].astype(int),
              "Fixture right": [f"{int(a.fx)} of {int(a.n)}" for a in agg.itertuples()],
              "Winner right": [f"{int(a.wn)} of {int(a.n)}" for a in agg.itertuples()],
          }), right=("Fixture right", "Winner right")), "",
          "<details><summary>All 32 knockout ties</summary>", "",
          _md_table(pd.DataFrame({
              "Match": per_match["match"], "Round": per_match["round"], "Projected": per_match["projected"],
              "Projected winner": per_match["projected_winner"], "Actual": per_match["actual"],
              "Score": per_match["actual_score"], "Winner": per_match["actual_winner"],
              "Fixture": per_match["fixture_hit"].map({True: YES, False: NO}),
              "Winner right": per_match["winner_hit"].map({True: YES, False: NO}),
          })), "", "</details>", ""]

    # --- 7 · surprises -------------------------------------------------------------
    sp = res["surprises"]
    gs, ks = sp[sp["stage"] == "group"], sp[sp["stage"] == "ko"]
    L += ["## 7 · The biggest surprises", "",
          "The results the model found least likely. Group games: the probability it gave the actual 90-minute "
          "outcome. Knockout ties: the probability it gave the team that went through, for every tie it called "
          "wrong.", ""]
    for label, part in (("Group stage", gs), ("Knockouts", ks)):
        L += [f"**{label}**", "", _md_table(pd.DataFrame({
            "Date": part["date"], "Match": part["fixture"], "Result": part["result"],
            "What happened": part["what_happened"], "Model's probability": part["model_probability"].map(_pct),
        }), right=("Model's probability",)), ""]
    draws = int((gs["what_happened"] == "draw").sum())
    L += [f"{draws} of the {len(gs)} biggest group-stage shocks were draws against a clear favourite. In the "
          f"knockouts nothing was a real upset by the model's reckoning: the least likely team to go through still "
          f"had {_pct(ks['model_probability'].min(), 0)}, and {pens_miss} of its {len(ko_miss)} wrong calls were "
          f"settled on penalties.", ""]

    # --- 8 · calibration ----------------------------------------------------------
    cal = res["calibration"]
    first = next(iter(cal.values()))
    lo_f = first["bin"].str.extract(r"^(\d+)")[0].astype(float) / 100   # lower edge of each bin
    mid = first[(lo_f >= 0.5) & (lo_f < 0.7)]
    low = first[(lo_f >= 0.1) & (lo_f < 0.3)]

    def wmean(d, col):
        return float((d[col] * d["forecasts"]).sum() / d["forecasts"].sum())

    L += ["## 8 · Calibration", "", _picture("calibration", "Calibration of the match probabilities"), "",
          f"Draws were forecast almost exactly as often as they happened ({_pct(matches['pre_p_draw'].mean())} "
          f"forecast, {_pct((matches['actual_90'] == 'D').mean())} observed). The miss sits in the favourite/underdog "
          f"split: outcomes given 50–70% happened {_pct(wmean(mid, 'observed_frequency'), 0)} of the time (forecast "
          f"{_pct(wmean(mid, 'mean_forecast'), 0)}), and outcomes given 10–30% happened "
          f"{_pct(wmean(low, 'observed_frequency'), 0)} of the time (forecast {_pct(wmean(low, 'mean_forecast'), 0)}). "
          f"The model was too cautious about clear favourites. The top bin (over 70%) holds too few forecasts to "
          f"read much into.", ""]
    cal_tbl = None
    for name, df in cal.items():
        part = df.rename(columns={"forecasts": f"n ({name})", "mean_forecast": f"forecast ({name})",
                                  "observed_frequency": f"observed ({name})"})
        cal_tbl = part if cal_tbl is None else cal_tbl.merge(part, on="bin", how="outer", sort=False)
    fmt = {c: (lambda v: _pct(v, 0)) for c in cal_tbl.columns if c.startswith(("forecast", "observed"))}
    fmt.update({c: (lambda v: _fmt(v, ".0f")) for c in cal_tbl.columns if c.startswith("n (")})
    L += [_md_table(cal_tbl.rename(columns={"bin": "Forecast bin"}), fmt), ""]

    # --- 9 · over / under ---------------------------------------------------------
    ou = teams.sort_values("over_under", ascending=False, kind="stable")
    L += ["## 9 · Who beat the model, who fell short", "",
          "\"Expected stages\" adds up a team's six stage-reach probabilities (0 = out in the group, 6 = champion); "
          "the difference is how many rounds further, or shorter, it went than the model expected.", ""]
    for label, part in (("Beat the model", ou.head(8)), ("Fell short", ou.tail(8).iloc[::-1])):
        L += [f"**{label}**", "", _md_table(pd.DataFrame({
            "Team": part["team"], "Expected stages": part["expected_rounds"].map(lambda v: f"{v:.2f}"),
            "Reached": part["actual_rounds"], "Difference": part["over_under"].map(lambda v: f"{v:+.2f}"),
            "Finish": part["team"].map(finishes)}), right=("Expected stages", "Difference")), ""]

    # --- method ---------------------------------------------------------------------
    L += ["## Method and caveats", "",
          "- **Results.** All 104 scores, including extra time and shoot-outs, were cross-checked against "
          "Wikipedia's match reports, ESPN's scoreboards and the martj42 international-results dataset, with no "
          "discrepancies. Rebuilding the bracket from the group results alone (FIFA 2026 tie-breakers, the "
          "best-third ranking and FIFA's 495-row slot table) reproduces every knockout fixture; the pipeline "
          "re-checks this on every run.",
          f"- **Pre-tournament model.** Elo and a Poisson goals model trained on internationals up to "
          f"{res['history_end']} (the historical data is cut at kick-off, so no 2026 score can leak in); the GB "
          f"hybrid adds squad market values frozen as of {res['squad_as_of']}. {res['n_sims']:,} simulated "
          f"tournaments per forecast, seeded, so reruns reproduce every number exactly.",
          "- **Faithfulness to the forecasts made in June.** The per-match (Poisson + Elo) predictions are "
          "bit-for-bit the ones the pipeline produced during the tournament. The Monte-Carlo title odds also "
          "depend on the squad-value snapshot. The runs made during the tournament read a live snapshot that has "
          "since been rebuilt, so their odds can differ from the numbers here by a few tenths of a point. The one "
          "run whose output survives (11 July) had the same final four, final and champion.",
          "- **Knockout scoring.** A tie that went to extra time or penalties counts as a 90-minute draw. Who went "
          "through is scored separately, with the model's full advancement probability (extra time at the same "
          "scoring rate, then an Elo-tilted shoot-out). Goal errors for extra-time games use 4/3 of the 90-minute "
          "expectation.",
          "- **Sample size.** 104 matches is a small sample. Differences of a few thousandths of RPS between models "
          "are within noise, which is why the comparison reports bootstrap intervals.", "",
          "The tables behind this report are the CSV files in this folder."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

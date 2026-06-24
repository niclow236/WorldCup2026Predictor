"""
run_pipeline.py — end-to-end World Cup 2026 prediction pipeline.

Run:  python run_pipeline.py

It produces, in data/output/:

  Pre-tournament (NO 2026 data — a clean prior forecast)
    pretournament_group_predictions.csv
    pretournament_knockout_bracket.csv
    pretournament_probabilities.csv

  To-date (conditioned on data/input/actual_results_2026.csv)
    todate_group_predictions.csv
    todate_knockout_bracket.csv
    todate_probabilities.csv
    champion_probabilities.png

  Cross-cutting
    overall_winner.csv          — champion + podium contenders (to-date model)
    accuracy_report.csv         — pre-tournament forecast vs each actual result
    accuracy_summary.csv        — aggregate accuracy metrics
    backtest.csv                — out-of-sample engine validation
    summary.md                  — human-readable digest of everything above

The two scenarios share one code path (``build_scenario``); the only difference
is whether the actual-results dicts are populated. That is what makes "with vs
without 2026 data" a fair apples-to-apples comparison.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

import config
from constants import GROUPS
from predictor import accuracy, data_io, predictions, validation
from predictor.elo import run_elo
from predictor.goals_model import fit_goals_model, xg
from predictor.squad_values import build_squad_value_model
from predictor.tournament import TournamentSimulator, build_lambda_matrices


def _attach_elo(matches):
    """Run Elo over ``matches`` and attach per-row pre-match ratings."""
    ratings, pre_h, pre_a = run_elo(matches)
    matches = matches.copy()
    matches["elo_h_pre"] = pre_h
    matches["elo_a_pre"] = pre_a
    return matches, ratings


def build_scenario(matches_hist, group_actual, ko_actual, label):
    """Train Elo + goals model (+ optional GB) and build a simulator.

    Returns a dict bundle: matches_all (with elo), ratings, model, simulator,
    forecast, pos, q3, gb_info.
    """
    print(f"\n=== Scenario: {label} "
          f"({len(group_actual)} group, {len(ko_actual)} ko results) ===")
    matches_all = data_io.build_matches_all(matches_hist, group_actual, ko_actual)
    matches_all, ratings = _attach_elo(matches_all)

    model = fit_goals_model(matches_all[matches_all.date >= config.TRAIN_SINCE])

    # Choose the expected-goals function: GB hybrid if available, else Poisson+Elo.
    xg_gb, gb_info = build_squad_value_model(matches_all, ratings)
    print("  " + gb_info["message"])
    if xg_gb is not None:
        xg_fn = xg_gb
    else:
        def xg_fn(a, b, home):
            return xg(model, a, b, ratings.get(a, 1500), ratings.get(b, 1500), home)

    lam_neu, lam_home, idx, all_teams = build_lambda_matrices(xg_fn)
    sim = TournamentSimulator(lam_neu, lam_home, idx, all_teams, ratings, group_actual, ko_actual)

    print(f"  simulating {config.N_SIMS:,} tournaments …")
    forecast, pos, q3, _ = sim.monte_carlo(config.N_SIMS, config.RNG_SEED)

    return {
        "matches_all": matches_all, "ratings": ratings, "model": model,
        "simulator": sim, "forecast": forecast, "pos": pos, "q3": q3,
        "gb_info": gb_info,
    }


def _write(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(config.OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  wrote {name}")
    return path


def _write_chart(forecast: pd.DataFrame):
    if not config.WRITE_CHART:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = forecast.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(top["team"], top["champion_%"], color="#1f77b4")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xlabel("P(win 2026 World Cup) %")
        ax.margins(x=0.12)
        ax.set_title(f"Champion probabilities · {config.N_SIMS:,} simulations · to-date model")
        plt.tight_layout()
        path = os.path.join(config.OUTPUT_DIR, "champion_probabilities.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print("  wrote champion_probabilities.png")
    except Exception as e:  # noqa: BLE001
        print(f"  chart skipped: {str(e)[:80]}")


def main():
    # Windows consoles default to cp1252; force UTF-8 so arrows/emoji print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    config.ensure_dirs()
    print("Loading data …")
    results_raw, _shootouts, _former = data_io.load_raw()
    matches_hist = data_io.clean_results(results_raw)
    print(f"  historical matches: {len(matches_hist):,} "
          f"({matches_hist.date.min().date()} → {matches_hist.date.max().date()})")

    group_actual, ko_actual = data_io.load_actuals()

    # --- Scenario A: pre-tournament (no 2026 data) -------------------------
    pre = build_scenario(matches_hist, {}, {}, "pre-tournament (no 2026 data)")
    _write(predictions.group_stage_predictions(pre["model"], pre["ratings"], {}),
           "pretournament_group_predictions.csv")
    pre_bracket, pre_champ, pre_finalists = predictions.project_bracket(
        pre["model"], pre["ratings"], pre["simulator"], pre["pos"], pre["q3"], {})
    _write(pre_bracket, "pretournament_knockout_bracket.csv")
    _write(pre["forecast"].round(2), "pretournament_probabilities.csv")

    # --- Scenario B: to-date (conditioned on entered results) --------------
    todate = build_scenario(matches_hist, group_actual, ko_actual, "to-date (with 2026 data)")
    _write(predictions.group_stage_predictions(todate["model"], todate["ratings"], group_actual),
           "todate_group_predictions.csv")
    td_bracket, td_champ, td_finalists = predictions.project_bracket(
        todate["model"], todate["ratings"], todate["simulator"],
        todate["pos"], todate["q3"], ko_actual)
    _write(td_bracket, "todate_knockout_bracket.csv")
    _write(todate["forecast"].round(2), "todate_probabilities.csv")
    _write_chart(todate["forecast"])

    # --- Overall winner prediction (to-date model) -------------------------
    podium = todate["forecast"].head(5)[["team", "elo", "champion_%", "final_%", "semifinal_%"]].copy()
    _write(podium.round(2), "overall_winner.csv")

    # --- Accuracy: pre-tournament forecast vs actual results ---------------
    acc_df, acc_summary = accuracy.score_forecast(pre["model"], pre["ratings"], group_actual, ko_actual)
    if len(acc_df):
        _write(acc_df.round(4), "accuracy_report.csv")
    _write(pd.DataFrame([acc_summary]), "accuracy_summary.csv")

    # --- Back-test (engine validation) -------------------------------------
    bt = validation.backtest(pre["matches_all"])
    _write(bt.reset_index(), "backtest.csv")

    # --- Human-readable digest ---------------------------------------------
    _write_summary(pre, todate, pre_champ, td_champ, td_finalists,
                   acc_summary, bt, group_actual, ko_actual)

    print(f"\n🏆 To-date projected champion: {td_champ}  "
          f"(final: {td_finalists[0]} vs {td_finalists[1]})")
    print(f"   Pre-tournament projected champion: {pre_champ}")
    print(f"\nAll outputs in {config.OUTPUT_DIR}")


def _write_summary(pre, todate, pre_champ, td_champ, td_finalists,
                   acc_summary, bt, group_actual, ko_actual):
    lines = []
    lines.append(f"# World Cup 2026 — Prediction Pipeline Output\n")
    lines.append(f"_Generated {date.today().isoformat()} · {config.N_SIMS:,} simulations/scenario_\n")
    lines.append(f"- Entered results to date: **{len(group_actual)} group**, **{len(ko_actual)} knockout**")
    lines.append(f"- GB squad-value hybrid (to-date): {todate['gb_info']['message']}\n")

    lines.append("## 🏆 Overall winner prediction (to-date model)\n")
    top5 = todate["forecast"].head(5)
    lines.append("| Team | Elo | Champion % | Final % | Semi % |")
    lines.append("|---|---|---|---|---|")
    for _, r in top5.iterrows():
        lines.append(f"| {r['team']} | {int(r['elo'])} | {r['champion_%']:.1f} "
                     f"| {r['final_%']:.1f} | {r['semifinal_%']:.1f} |")
    lines.append(f"\n**Projected champion: {td_champ}** (final {td_finalists[0]} vs {td_finalists[1]})")
    lines.append(f"\nPre-tournament (no 2026 data) projected champion: **{pre_champ}** · "
                 f"top pick {pre['forecast'].iloc[0]['team']} "
                 f"({pre['forecast'].iloc[0]['champion_%']:.1f}%)\n")

    lines.append("## 🎯 Forecast accuracy (pre-tournament model vs actual results)\n")
    if acc_summary.get("n_group", 0) or acc_summary.get("n_ko", 0):
        for k, v in acc_summary.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- No actual results entered yet — nothing to score.")
    lines.append("")

    lines.append("## 🔬 Engine back-test (out-of-sample, 2018–2022)\n")
    lines.append("| model | accuracy | log_loss | RPS |")
    lines.append("|---|---|---|---|")
    for model_name, r in bt.iterrows():
        lines.append(f"| {model_name} | {r['accuracy']} | {r['log_loss']} | {r['RPS']} |")
    lines.append("\n_Lower RPS / log-loss is better; ~0.19–0.21 RPS is competitive._\n")

    path = os.path.join(config.OUTPUT_DIR, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  wrote summary.md")


if __name__ == "__main__":
    main()

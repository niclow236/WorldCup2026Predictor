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

  Cross-cutting
    champion_probabilities.png  — title odds, pre-tournament vs to-date
    overall_winner.csv          — champion + podium contenders (to-date model)
    accuracy_report.csv         — pre-tournament forecast vs each actual result
    accuracy_summary.csv        — aggregate accuracy metrics
    backtest.csv                — out-of-sample engine validation
    summary.md                  — human-readable digest of everything above

  Final evaluation (once all 104 results are entered), in final_evaluation/
    report.md                   — predicted vs actual: the tournament verdict
    *.csv, *.png                — the tables and figures behind the report

The two scenarios share one code path (``build_scenario``); the only difference
is whether the actual-results dicts are populated. That is what makes "with vs
without 2026 data" a fair apples-to-apples comparison.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

import config
from predictor import accuracy, data_io, evaluation, predictions, report, validation
from predictor.elo import attach_elo
from predictor.goals_model import fit_goals_model, xg
from predictor.squad_values import build_squad_value_model
from predictor.tournament import TournamentSimulator, build_lambda_matrices


def build_scenario(matches_hist, group_actual, ko_actual, label, verbose=True):
    """Train Elo + goals model (+ optional GB) and build a simulator.

    Returns a dict bundle: matches_all (with elo), ratings, model, xg_fn (the
    expected-goals function the simulator runs on), simulator, forecast, pos,
    q3, gb_info.
    """
    log = print if verbose else (lambda *_args, **_kw: None)
    log(f"\n=== Scenario: {label} "
        f"({len(group_actual)} group, {len(ko_actual)} ko results) ===")
    matches_all = data_io.build_matches_all(matches_hist, group_actual, ko_actual)
    matches_all, ratings = attach_elo(matches_all)

    model = fit_goals_model(matches_all[matches_all.date >= config.TRAIN_SINCE])

    # Choose the expected-goals function: GB hybrid if available, else Poisson+Elo.
    xg_gb, gb_info = build_squad_value_model(matches_all, ratings)
    log("  " + gb_info["message"])
    if xg_gb is not None:
        xg_fn = xg_gb
    else:
        def xg_fn(a, b, home):
            return xg(model, a, b, ratings.get(a, 1500), ratings.get(b, 1500), home)

    lam_neu, lam_home, idx, all_teams = build_lambda_matrices(xg_fn)
    sim = TournamentSimulator(lam_neu, lam_home, idx, all_teams, ratings, group_actual, ko_actual)

    log(f"  simulating {config.N_SIMS:,} tournaments …")
    forecast, pos, q3, _ = sim.monte_carlo(config.N_SIMS, config.RNG_SEED)

    return {
        "matches_all": matches_all, "ratings": ratings, "model": model, "xg_fn": xg_fn,
        "simulator": sim, "forecast": forecast, "pos": pos, "q3": q3,
        "gb_info": gb_info,
    }


def _write(df: pd.DataFrame, name: str, folder: str | None = None) -> str:
    path = os.path.join(folder or config.OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  wrote {os.path.relpath(path, config.OUTPUT_DIR)}")
    return path


def _write_chart(pre_fc: pd.DataFrame, todate_fc: pd.DataFrame, n_results: int):
    """Dumbbell of title odds per team: pre-tournament -> to-date."""
    if not config.WRITE_CHART:
        return
    try:
        th = report.THEMES["light"]
        plt = report.pyplot()
        d = pre_fc[["team", "champion_%"]].merge(todate_fc[["team", "champion_%"]], on="team",
                                                 suffixes=("_pre", "_now"))
        d["key"] = d[["champion_%_pre", "champion_%_now"]].max(axis=1)
        d = d.nlargest(15, "key").iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=report.DPI)
        fig.patch.set_facecolor(th["surface"])
        fig.subplots_adjust(left=0.17, right=0.95, top=0.85, bottom=0.14)
        report.style_axes(ax, th, "x")
        y = range(len(d))
        ax.hlines(y, d["champion_%_pre"], d["champion_%_now"], color=th["axis"], linewidth=2)
        for col, colour, label in (("champion_%_pre", th["series"][0], "Pre-tournament"),
                                   ("champion_%_now", th["series"][1], f"To date ({n_results} results)")):
            ax.scatter(d[col], y, s=52, color=colour, edgecolor=th["surface"], linewidth=1.5,
                       zorder=3, label=label, clip_on=False)
        ax.set_yticks(list(y), d["team"])
        ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
        ax.set_xlim(left=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False,
                  fontsize=9, labelcolor=th["ink2"])
        fig.text(0.02, 0.965, "Title odds: before the tournament vs to date", fontsize=12.5,
                 fontweight="bold", color=th["ink"], va="top")
        fig.text(0.02, 0.915, f"P(win the World Cup), {config.N_SIMS:,} simulated tournaments per forecast",
                 fontsize=9.5, color=th["ink2"], va="top")
        report.save_figure(fig, os.path.join(config.OUTPUT_DIR, "champion_probabilities.png"), th)
        print("  wrote champion_probabilities.png")
    except Exception as e:  # noqa: BLE001
        print(f"  chart skipped: {str(e)[:80]}")


def _final_evaluation(matches_hist, pre, pre_bracket, acc_df, acc_summary, group_actual, ko_actual):
    """Score every forecast against the finished tournament -> OUTPUT_DIR/final_evaluation/."""
    out = config.FINAL_DIR
    os.makedirs(out, exist_ok=True)
    print("\n=== Final evaluation: predicted vs actual ===")
    bracket, tables, thirds = evaluation.reconstruct_bracket(group_actual, ko_actual)
    print("  bracket rebuilt from the group results — all 32 knockout ties consistent")
    finish = evaluation.finish_index(bracket)

    # Match-level scoring of every model on the same 104 matches.
    pre_train = pre["matches_all"][pre["matches_all"].date >= config.TRAIN_SINCE]
    no_elo = fit_goals_model(pre_train, use_elo=False)
    scored = {"Poisson (no Elo)": accuracy.score_matches(no_elo, pre["ratings"], group_actual, ko_actual),
              "Poisson + Elo": acc_df}
    if pre["gb_info"]["ok"]:
        scored["GB squad-value hybrid"] = accuracy.score_matches(
            pre["model"], pre["ratings"], group_actual, ko_actual, xg_fn=pre["xg_fn"])
    live = timeline = None
    cutoffs = evaluation.checkpoints(group_actual, bracket)
    if config.RUN_LIVE_REPLAY:
        print("  replaying the live model before every match day …")
        live = evaluation.live_replay(matches_hist, group_actual, ko_actual)
        scored[evaluation.LIVE_MODEL] = live
        print(f"  re-simulating the title odds after each of {len(cutoffs)} rounds …")
        timeline = evaluation.title_timeline(
            pre["forecast"], group_actual, ko_actual, bracket,
            lambda g, k, label: build_scenario(matches_hist, g, k, label, verbose=False)["forecast"])

    comparison = evaluation.model_comparison(scored, pre["ratings"])
    bracket_match, bracket_round = evaluation.bracket_comparison(pre_bracket, bracket)
    calibration = {"pre-tournament": evaluation.calibration_bins(acc_df)}
    if live is not None:
        calibration["live"] = evaluation.calibration_bins(live)
    dates = sorted(rec["date"] for rec in list(group_actual.values()) + list(ko_actual.values()))
    res = {
        "bracket": bracket, "tables": tables, "thirds": thirds, "finish": finish,
        "pre_forecast": pre["forecast"], "acc_summary": acc_summary, "comparison": comparison,
        "phases": evaluation.phase_breakdown(scored, cutoffs),
        "stage": evaluation.score_stage_probabilities(pre["forecast"], finish),
        "teams": evaluation.team_comparison(pre["forecast"], finish, tables, pre["pos"]),
        "groups": evaluation.group_comparison(
            pre["pos"], predictions.projected_finishers(pre["pos"], pre["q3"]), tables),
        "bracket_match": bracket_match, "bracket_round": bracket_round,
        "surprises": evaluation.surprises(acc_df), "calibration": calibration,
        "matches": evaluation.match_table(acc_df, live, bracket), "timeline": timeline,
        "live_name": evaluation.LIVE_MODEL if live is not None else None,
        "n_match_days": len(set(dates)), "first_date": dates[0], "last_date": dates[-1],
        "n_sims": config.N_SIMS, "history_end": matches_hist.date.max().date().isoformat(),
        "squad_as_of": config.SQUAD_VALUE_AS_OF,
    }

    tables_long = pd.concat([t.assign(group=g) for g, t in tables.items()], ignore_index=True)
    cal_long = pd.concat([df.assign(model=name) for name, df in calibration.items()], ignore_index=True)
    for df, name in [
        (res["matches"], "match_predictions.csv"), (comparison, "model_comparison.csv"),
        (res["phases"].reset_index(), "phase_rps.csv"), (res["stage"], "stage_probabilities.csv"),
        (res["teams"], "team_comparison.csv"), (res["groups"], "group_comparison.csv"),
        (tables_long[["group"] + list(tables["A"].columns)], "group_tables.csv"),
        (bracket_match, "bracket_comparison.csv"), (bracket_round, "bracket_by_round.csv"),
        (res["surprises"], "surprises.csv"), (cal_long[["model"] + list(cal_long.columns[:-1])], "calibration.csv"),
    ] + ([(timeline, "title_timeline.csv")] if timeline is not None else []):
        _write(df.round(4), name, out)
    report.write_figures(res, out)
    report.write_final_report(res, os.path.join(out, "report.md"))
    print("  wrote final_evaluation/report.md (+ figures)")
    return res


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
    _write_chart(pre["forecast"], todate["forecast"], len(group_actual) + len(ko_actual))

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

    # --- Final evaluation (tournament complete) ----------------------------
    final = None
    if evaluation.is_complete(group_actual, ko_actual):
        final = _final_evaluation(matches_hist, pre, pre_bracket, acc_df, acc_summary,
                                  group_actual, ko_actual)
    else:
        print(f"\nFinal evaluation skipped: {len(group_actual)}/72 group and "
              f"{len(ko_actual)}/32 knockout results entered.")

    # --- Human-readable digest ---------------------------------------------
    _write_summary(pre, todate, pre_champ, td_champ, td_finalists,
                   acc_summary, bt, group_actual, ko_actual, final)

    print(f"\n🏆 To-date projected champion: {td_champ}  "
          f"(final: {td_finalists[0]} vs {td_finalists[1]})")
    print(f"   Pre-tournament projected champion: {pre_champ}")
    print(f"\nAll outputs in {config.OUTPUT_DIR}")


def _write_summary(pre, todate, pre_champ, td_champ, td_finalists,
                   acc_summary, bt, group_actual, ko_actual, final=None):
    dates = [rec["date"] for rec in list(group_actual.values()) + list(ko_actual.values())]
    through = f"results through {max(dates)}" if dates else "no results entered"
    lines = []
    lines.append("# World Cup 2026 — Prediction Pipeline Output\n")
    lines.append(f"_{through.capitalize()} · {config.N_SIMS:,} simulations/scenario_\n")
    lines.append(f"- Entered results to date: **{len(group_actual)} group**, **{len(ko_actual)} knockout**")
    lines.append(f"- GB squad-value hybrid (to-date): {todate['gb_info']['message']}\n")

    if final is not None:
        comp = final["comparison"].set_index("model")
        lines.append("## ✅ Final verdict (tournament complete)\n")
        lines.append(f"- Champion: **{td_champ}** — the pre-tournament model's favourite was "
                     f"**{pre['forecast'].iloc[0]['team']}** ({pre['forecast'].iloc[0]['champion_%']:.1f}%)")
        lines.append(f"- 90-minute results called: **{100 * acc_summary['all90_outcome_accuracy']:.1f}%** of 104; "
                     f"knockout winners: **{round(acc_summary['ko_winner_accuracy'] * acc_summary['n_ko'])}/"
                     f"{acc_summary['n_ko']}**")
        lines.append(f"- RPS over all 104 matches: **{acc_summary['all90_rps']:.4f}** "
                     f"(no-skill {acc_summary['all90_baseline_rps']:.4f}); best model: "
                     f"{comp['all90_rps'].idxmin()}")
        lines.append("- Full predicted-vs-actual report: [`final_evaluation/report.md`](final_evaluation/report.md)\n")

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

# World Cup 2026 — Prediction Pipeline Output

_Results through 2026-07-19 · 20,000 simulations/scenario_

- Entered results to date: **72 group**, **32 knockout**
- GB squad-value hybrid (to-date): GB hybrid active. Squad values (as of 2026-06-10): 48/48 found, 0 imputed. Holdout Poisson deviance 1.1634.

## ✅ Final verdict (tournament complete)

- Champion: **Spain** — the pre-tournament model's favourite was **Spain** (18.1%)
- 90-minute results called: **65.4%** of 104; knockout winners: **27/32**
- RPS over all 104 matches: **0.1598** (no-skill 0.2157); best model: GB squad-value hybrid
- Full predicted-vs-actual report: [`final_evaluation/report.md`](final_evaluation/report.md)

## 🏆 Overall winner prediction (to-date model)

| Team | Elo | Champion % | Final % | Semi % |
|---|---|---|---|---|
| Spain | 2323 | 100.0 | 100.0 | 100.0 |
| Argentina | 2245 | 0.0 | 100.0 | 100.0 |
| France | 2138 | 0.0 | 0.0 | 100.0 |
| England | 2190 | 0.0 | 0.0 | 100.0 |
| Switzerland | 1994 | 0.0 | 0.0 | 0.0 |

**Projected champion: Spain** (final Spain vs Argentina)

Pre-tournament (no 2026 data) projected champion: **Spain** · top pick Spain (18.1%)

## 🎯 Forecast accuracy (pre-tournament model vs actual results)

- **n_group**: 72
- **n_ko**: 32
- **group_outcome_accuracy**: 0.653
- **group_exact_accuracy**: 0.111
- **group_rps**: 0.1603
- **group_brier**: 0.5233
- **group_logloss**: 0.8765
- **baseline_rps**: 0.2073
- **baseline_logloss**: 1.0261
- **rps_skill_vs_baseline**: 0.227
- **ko_winner_accuracy**: 0.844
- **ko_advance_brier**: 0.1522
- **ko_advance_logloss**: 0.4768
- **ko_coinflip_brier**: 0.25
- **ko_coinflip_logloss**: 0.6931
- **all90_outcome_accuracy**: 0.654
- **all90_rps**: 0.1598
- **all90_baseline_rps**: 0.2157
- **goal_mae**: 0.928
- **goal_rmse**: 1.255

## 🔬 Engine back-test (out-of-sample, 2018–2022)

| model | accuracy | log_loss | RPS |
|---|---|---|---|
| Poisson only | 0.595 | 0.9 | 0.1783 |
| Poisson + Elo | 0.599 | 0.885 | 0.174 |

_Lower RPS / log-loss is better; ~0.19–0.21 RPS is competitive._

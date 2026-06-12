# FIFA World Cup 2026 Predictor

A self-contained, live-updating Jupyter notebook that forecasts every match of the 2026 FIFA World Cup — all 72 group games and the complete knockout bracket through the final on 19 July 2026.

Enter real results as they are played and re-run: team strengths re-train on the actual scores and the bracket locks in real winners, so every remaining prediction is conditioned on what actually happened.

## Features

- **Live updating** — enter any result in one cell (§4), re-run, and all downstream predictions refresh
- **Full tournament structure** — 48 teams, 12 groups (A–L), FIFA's 495-row third-place allocation table, exact R32 → R16 → QF → SF → Final bracket
- **In-house Elo** — World-Cup matches carry the heaviest K-factor (60), so entered results sharpen later-round predictions the most
- **Time-weighted Poisson goals model** — sparse `PoissonRegressor` with a 2-year half-life, Elo-difference feature, fit on ~49k internationals since 1872
- **Out-of-sample validation** — strict temporal split (train < 2018, test 2018–2022); reports accuracy, log-loss, and Ranked Probability Score
- **Monte Carlo simulation** — ~0.3 ms per full tournament run; outputs champion probabilities and stage-reach tables
- **Squad-value hybrid (§9)** — automatically augments the Poisson model with Transfermarkt squad market values via `HistGradientBoostingRegressor` before every simulation run

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/WorldCup2026Predictor.git
cd WorldCup2026Predictor

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib

# 4. Open the notebook
jupyter notebook world_cup_2026_predictor.ipynb
```

Run all cells top-to-bottom for the pre-tournament forecast. Data is downloaded automatically on first run and cached in `wc2026_data/` for offline use.

## Entering Real Results

Edit **cell §4** only. Group-stage games take a score:

```python
group_result("Spain", 3, 0, "Cape Verde")
```

Knockout games take the same format; add `pen_winner` when the score is level, and optionally `pen_a`/`pen_b` for the actual shootout score:

```python
ko_result("Brazil", 1, 1, "Croatia", pen_winner="Brazil", pen_a=4, pen_b=2)
```

Team names must match the spellings in §3 (e.g. `"South Korea"`, `"Ivory Coast"`, `"Czech Republic"`). After editing, select **Restart & Run All**.

## Data Sources

| Dataset | Source | Licence |
|---|---|---|
| International results (~49k matches, 1872–present) | [`martj42/international_results`](https://github.com/martj42/international_results) | CC0 |
| Squad market values | [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets) | CC0 |

Both datasets are downloaded automatically; no API key is required.

## Model Overview

```
Data  →  Elo ratings  →  Poisson goals model  →  GB hybrid (squad values)  →  Monte Carlo  →  Predicted scorelines + win probabilities
              ↑                    ↑                        ↑
        entered results     entered results         Transfermarkt data
```

The goals model is: `log λ = μ + attack_team + defence_opp + β_h·home + β_e·(Elo_team − Elo_opp)/100`, fit by sparse `PoissonRegressor` with sample weights decaying with a 2-year half-life on matches since 2008.

Pre-tournament champion probability benchmarks:

| Team | This model | Opta |
|---|---|---|
| Spain | ~22% | 16.1% |
| Argentina | ~15% | 10.4% |
| England / France | ~7% each | 11–13% |

The squad-value hybrid (§9) adds log squad value diff, rolling form, and rest days as features via `HistGradientBoostingRegressor` (Poisson loss), rewiring the lambda matrices before Monte Carlo runs. If the Transfermarkt download fails the notebook falls back silently to the Poisson+Elo model.

## Known Limitations

- Host advantage applied in group stage only; knockouts are treated as neutral
- Shootout outcomes near-random (mild Elo tilt only)
- Group fair-play / lots tie-breaks approximated by random draw
- The projected bracket in §11 is one modal path — use the §11c probability tables for rigorous estimates
- Squad values used as a static proxy across all historical training rows

## References

- Dixon & Coles (1997) — score correlation correction
- Lasek et al. (2013) — Elo for football
- Groll & Zeileis (2018–2026) — tournament forecasting benchmarks

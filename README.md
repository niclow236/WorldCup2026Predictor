# FIFA World Cup 2026 Predictor — Pipeline

A reproducible, command-line **forecasting pipeline** for the 2026 FIFA World Cup. It predicts a
scoreline for every match (all 72 group games and the full knockout bracket through the final), a
champion probability for all 48 teams, and — crucially — **scores its own pre-tournament forecast
against the real results as they come in.**

Run it once and it produces three views side by side:

1. **Pre-tournament forecast** — what the model predicted *before any 2026 result*, the honest prior.
2. **To-date forecast** — every remaining match re-predicted, conditioned on the real results entered
   so far (team strengths re-train on World-Cup form; played games are locked in).
3. **Overall winner prediction** + an **accuracy report** comparing (1) against what actually happened.

This replaces the original exploratory notebook (now archived in `notebooks/`).

---

## Repository layout

```
WorldCup2026Predictor/
├── config.py                 # runtime config: INPUT_DIR / OUTPUT_DIR + all tunables
├── constants.py              # tournament structure: groups, bracket wiring, FIFA 3rd-place table, K-factors
├── run_pipeline.py           # entry point — runs the whole pipeline
├── requirements.txt
├── README.md
├── code_desc.md              # in-depth, function-by-function description of the code
├── data/
│   ├── input/                # config.INPUT_DIR
│   │   ├── results.csv             # ~49k historical internationals (auto-downloaded)
│   │   ├── shootouts.csv
│   │   ├── former_names.csv
│   │   └── actual_results_2026.csv # ← YOU UPDATE THIS DAILY
│   └── output/               # config.OUTPUT_DIR (generated; git-ignored)
├── predictor/                # the engine
│   ├── data_io.py            # load/clean data; parse the daily actuals CSV
│   ├── elo.py                # World Football Elo
│   ├── goals_model.py        # time-weighted Poisson goals model (+ Elo feature)
│   ├── squad_values.py       # optional Transfermarkt + gradient-boosting hybrid
│   ├── tournament.py         # lambda matrices, group/knockout sim, Monte Carlo
│   ├── predictions.py        # per-match scorelines + projected bracket
│   ├── validation.py         # out-of-sample engine back-test
│   └── accuracy.py           # forecast-vs-actual scoring (predicted vs actual)
└── notebooks/                # the original exploratory notebook (archived)
```

`config.py` holds **choices** (paths, number of simulations, feature toggles). `constants.py` holds
**facts** about the tournament that never change. New tunables go in `config.py`.

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python run_pipeline.py
```

On first run the historical dataset is downloaded to `data/input/` and cached. The squad-value hybrid
also downloads a Transfermarkt snapshot; if that fails (offline), the pipeline falls back silently to
the Poisson+Elo model. Everything is written to `data/output/`.

---

## Daily update workflow

Each day, edit **one file** — `data/input/actual_results_2026.csv` — then re-run `python run_pipeline.py`.

```csv
date,stage,team_a,score_a,score_b,team_b,pen_winner,pen_a,pen_b
2026-06-24,group,Mexico,2,1,Czech Republic,,,
2026-07-05,ko,Brazil,1,1,Croatia,Brazil,4,2
```

- `stage` is `group` or `ko`.
- For **group** rows just give the score.
- For **knockout** rows that finished level, set `pen_winner` (required) and optionally the shootout
  score in `pen_a` / `pen_b`. Decisive knockouts leave those blank.
- Team names must match the spellings in `constants.GROUPS` (e.g. `South Korea`, `Czech Republic`,
  `Ivory Coast`, `Curaçao`, `DR Congo`).

> **Ask me to update it.** The intended workflow is that you paste the day's confirmed scorelines and I
> append them to this CSV (the source of truth stays a clean, diffable data file).

### Current state

Results are entered **through 23 June 2026** (all 48 matchday-1 and matchday-2 group games). The six
24 June games had not kicked off at generation time and are left blank.

---

## Outputs (`data/output/`)

| File | What it is |
|---|---|
| `pretournament_group_predictions.csv` | Predicted scoreline for all 72 group games, **no 2026 data** |
| `pretournament_knockout_bracket.csv`  | Most-likely knockout path, **no 2026 data** |
| `pretournament_probabilities.csv`     | Champion / final / semi / R16 probabilities for 48 teams, **no 2026 data** |
| `todate_group_predictions.csv`        | Group predictions conditioned on entered results (actuals flagged ✓) |
| `todate_knockout_bracket.csv`         | Knockout path conditioned on entered results |
| `todate_probabilities.csv`            | Champion/stage probabilities conditioned on entered results |
| `overall_winner.csv`                  | The headline champion pick + top-5 podium contenders |
| `champion_probabilities.png`          | Bar chart of title odds (to-date model) |
| `accuracy_report.csv`                 | Per-match predicted-vs-actual detail (pre-tournament model) |
| `accuracy_summary.csv`                | Aggregate accuracy metrics (see below) |
| `backtest.csv`                        | Out-of-sample engine validation (2018–2022) |
| `summary.md`                          | Human-readable digest tying it all together |

---

## How the model works

```
historical results ─┐
entered 2026 results ┴─► Elo ratings ─► Poisson goals model ─► GB squad-value hybrid ─► λ matrices
                                                                                          │
                                                            Monte Carlo (20k tournaments) ◄┘
                                                                                          │
                                          per-match scorelines · bracket · champion odds ◄┘
```

- **Elo** (`predictor/elo.py`) — World Football Elo with a goal-difference multiplier. World-Cup
  matches carry the heaviest K-factor (60), so entered 2026 results move ratings the most.
- **Poisson goals model** (`predictor/goals_model.py`) —
  `log λ = μ + attack_team + defence_opp + β_h·home + β_e·(Elo_team − Elo_opp)/100`, fit by a sparse
  `PoissonRegressor` with a two-year half-life on matches since 2008.
- **Squad-value hybrid** (`predictor/squad_values.py`) — adds log squad-value diff, rolling form, and
  rest days via a `HistGradientBoostingRegressor` (Poisson loss); rewires the expected-goals matrices
  when it loads successfully.
- **Monte Carlo** (`predictor/tournament.py`) — simulates the whole tournament 20,000× (~0.3 ms each),
  honouring entered scores in the group stage and entered winners in the knockouts. A Dixon-Coles
  low-score correction sharpens the displayed scorelines.

`code_desc.md` documents every function in detail.

---

## Measuring accuracy (predicted vs actual)

The right question is *"how good was the forecast the model made before it saw these games?"* —
`predictor/accuracy.py` answers it by scoring the **pre-tournament model** (trained on **zero** 2026
data) against the results you have actually entered. There is no look-ahead: the forecast never saw
those matches, so the numbers are a genuine out-of-sample report.

**Group-stage matches (clean win/draw/loss):**

| Metric | Meaning | Good = |
|---|---|---|
| `group_outcome_accuracy` | fraction where the most-probable result (W/D/L) happened | higher |
| `group_exact_accuracy`   | fraction where the most-likely *scoreline* happened | higher |
| `group_rps`              | Ranked Probability Score — the football standard for ordered W/D/L | lower (~0.19–0.21 competitive) |
| `group_brier`            | multiclass Brier score (squared probability error) | lower |
| `group_logloss`          | mean −log P(true outcome) | lower |
| `rps_skill_vs_baseline`  | RPS improvement over a no-skill base-rate forecast | higher (>0 = skilful) |

**Goal level (all played matches):** `goal_mae` / `goal_rmse` — predicted expected goals vs actual
goals per side.

**Knockouts:** `ko_winner_accuracy` — did the projected stronger side actually advance?

Each metric is reported next to a **no-skill baseline** (constant base-rate forecast) so the headline
number has a reference point. The engine's intrinsic quality is separately characterised by
`predictor/validation.py`, a strict train-before-2018 / test-2018–2022 back-test.

> **Latest run:** on the 48 played group games the pre-tournament model scored **60% outcome
> accuracy**, **RPS 0.167** (vs 0.197 baseline → **+15% skill**), goal MAE **0.95**. See
> `data/output/summary.md`.

---

## Improving the model: additional web data

The model currently learns from **results + Elo + squad market values**. Independent forecasters
(Opta, Groll/Zeileis) do better mainly by adding player- and market-level signals. Concrete,
free-ish data that would move the needle, roughly in order of expected value:

| Data | Source | Why it helps | Status |
|---|---|---|---|
| **Squad market values** | Transfermarkt (`dcaribou/transfermarkt-datasets`) | Cross-sectional team quality; best single non-results signal | ✅ implemented (GB hybrid) |
| **Bookmaker / betting-market odds** | The Odds API, Pinnacle, Betfair Exchange | Market-implied probabilities are a strong benchmark *and* a feature; also the fairest accuracy yardstick | ⏳ roadmap (see below) |
| **FIFA / Elo world ranking points** | FIFA.com, eloratings.net | An external strength prior that complements the in-house Elo, esp. for rarely-seen teams | ⏳ roadmap |
| **Player availability / injuries / suspensions** | physioroom, transfermarkt, team news | A missing key player (or an accumulated-cards suspension) materially shifts a single match | ⏳ roadmap |
| **Club-level xG / shot data** | FBref, Understat | Players' underlying attacking/defensive output is more stable than international goals (sparse data) | ⏳ roadmap |
| **Lineups / formation at kickoff** | official team sheets ~1h pre-match | Rotation in dead-rubber group games is a big, model-able effect | ⏳ roadmap |
| **Travel / rest / altitude / venue** | fixture list + venue metadata | Already partially captured (rest_days, host-home); altitude (Mexico City, Guadalajara) and travel load add signal | ◐ partial (rest + host) |
| **Weather (heat/humidity)** | Open-Meteo | Heat suppresses goals and favours deeper, fitter squads in the 2026 summer venues | ⏳ roadmap |

**The single most valuable next step is bookmaker odds**, for two reasons. (1) As a *feature*, the
market aggregates information the model can't see (team news, tactical matchups). (2) As an *accuracy
benchmark*, comparing our RPS/log-loss to the market's closing odds is the gold-standard test of skill
— "beating the closing line" is the only forecast claim that really counts. The clean way to add it:
fetch pre-match closing W/D/L odds, de-vig them to probabilities, store them per fixture, and extend
`accuracy.py` to report our metrics *and* the market's on the same matches.

---

## Configuration

Everything tunable lives in `config.py`:

- `INPUT_DIR` / `OUTPUT_DIR` and the individual file paths
- `N_SIMS` (Monte-Carlo count), `RNG_SEED` (reproducibility)
- `TRAIN_SINCE`, `HALF_LIFE_DAYS`, `USE_ELO_FEATURE` (goals model)
- `USE_SQUAD_VALUE_GB` and GB hyper-parameters (set `False` to stay on Poisson+Elo / run fully offline)
- `ALLOW_DOWNLOAD` (set `False` to forbid network access)
- `DIXON_COLES_RHO`, back-test windows, output formatting

---

## Known limitations

- Host advantage applied in the group stage only; knockouts treated as neutral.
- Shootouts near-random (mild Elo tilt only).
- Group fair-play / drawing-of-lots tie-breaks approximated by a random draw.
- The projected bracket is a single *modal* path — use the probability tables for the rigorous view.
- Squad values are a static proxy applied across all historical training rows.

## Data sources & references

| Dataset | Source | Licence |
|---|---|---|
| International results (~49k, 1872–present) | [`martj42/international_results`](https://github.com/martj42/international_results) | CC0 |
| Squad market values | [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets) | CC0 |

Methods: Dixon & Coles (1997); Lasek et al. (2013); Groll & Zeileis (2018–2026). Seeded
(`RNG_SEED = 42`) for full reproducibility.

# `code_desc.md` — in-depth description of the codebase

A function-by-function walkthrough of the World Cup 2026 prediction pipeline. Read top to bottom and
it follows the data flow: config → constants → data → Elo → goals model → squad-value hybrid →
simulation → predictions → validation → accuracy → orchestrator.

For each function: **what it takes, what it returns, how it works, and why it's done that way.**

---

## `config.py` — runtime configuration

No functions of consequence except `ensure_dirs()`. It is a flat namespace of tunables, deliberately
separated from `constants.py`:

- **`config.py` = choices** you might change between runs (paths, `N_SIMS`, feature toggles, training
  window, regularisation, Dixon-Coles ρ, back-test windows).
- **`constants.py` = facts** about the 2026 tournament that never change.

Key values: `INPUT_DIR` / `OUTPUT_DIR` (derived from `ROOT_DIR` so the repo is location-independent),
the individual CSV paths, `ALLOW_DOWNLOAD`, `TRAIN_SINCE`/`HALF_LIFE_DAYS`, `USE_ELO_FEATURE`,
`USE_SQUAD_VALUE_GB` + GB hyper-parameters, `N_SIMS`/`RNG_SEED`, `DIXON_COLES_RHO`, and the back-test
windows.

### `ensure_dirs()`
Creates `INPUT_DIR` and `OUTPUT_DIR` if absent (`os.makedirs(..., exist_ok=True)`). Called once at the
start of `run_pipeline.main()`.

---

## `constants.py` — tournament structure

Mostly data, plus two helpers.

- **`GROUPS`** — the 12 groups A–L, four teams each, spelled exactly as in the historical dataset so
  2026 squads join cleanly onto ~150 years of results. **`HOSTS`** — USA/Canada/Mexico (group-stage
  home advantage). **`ALL_TEAMS`** — flattened 48-team list; this ordering is the canonical index for
  every Monte-Carlo array. **`GROUP_FIXTURES`** — the six intra-group fixtures as index pairs.
- **`LATER`** — for each R16+ match, the two feeder match numbers whose winners meet. **`ROUND_OF`** —
  match number → round label. **`KO_MATCH_ORDER`** — the order to resolve knockout matches (feeders
  before consumers). **`WINNER_SLOT_ORDER`** — the eight R32 slots that receive third-placed teams.
- **`TM_MAP`** — Transfermarkt → dataset country-name fixes (e.g. `Korea, South` → `South Korea`).

### `k_factor(tournament) -> int`
Maps a competition string to its World-Football-Elo K-factor: `FIFA World Cup` = 60 (heaviest, so
entered results dominate), continental finals = 50, qualifiers / Nations League = 40, friendlies = 20,
else 30. Pure lookup.

### Module-level parse of `THIRD_PLACE_TABLE_RAW`
FIFA publishes, for each of the C(12,8) = **495** possible sets of groups that produce the eight best
third-placed teams, *which R32 slot each third goes to*. The raw block is one line per combination:
`<8 qualifying group letters> <8 slot assignments>`. The loop parses each into
`THIRD_PLACE_TABLE[frozenset(qual_letters)] = {slot_letter: source_group}` and asserts there are
exactly 495 rows (and 48 unique teams). Using a `frozenset` key means lookup is independent of letter
order.

---

## `predictor/data_io.py` — data acquisition, cleaning, ingestion

### `_download_if_missing(path, filename)`
If `path` doesn't exist, download `RESULTS_BASE_URL/filename` and cache it; raises if the file is
missing **and** `config.ALLOW_DOWNLOAD` is `False`. Keeps later runs fully offline once cached.

### `load_raw() -> (results_raw, shootouts, former_names)`
Ensures the three historical CSVs exist (downloading on first run) and returns them as DataFrames.
`former_names` is loaded for parity with the source dataset but is currently unused by the model.

### `clean_results(results_raw) -> matches_hist`
Produces the historical training base. Parses dates; **drops rows with missing scores** (the dataset
already lists 2026 fixtures with blank scores — they must never train the model); casts scores to int;
normalises the `neutral` flag into a boolean `neutral_b`; sorts chronologically and resets the index.

### `_check_team(team)`
Raises a clear `ValueError` if a team name isn't one of the 48 in `constants.GROUPS` — catches typos in
the daily actuals CSV early.

### `load_actuals(csv_path=None) -> (group_actual, ko_actual)`
Parses `data/input/actual_results_2026.csv` into the two dicts the simulator conditions on:

- `group_actual[frozenset({a, b})] = {"by_team": {a: ga, b: gb}}`
- `ko_actual[frozenset({a, b})] = {"by_team": {a, b → goals}, "pen_winner": team|None, "pen_score": {…}?}`

For knockout rows that finished level it **requires** a `pen_winner` and validates that, if a shootout
score is given, the winner's tally exceeds the loser's. Returns empty dicts if the file is absent —
that is the legitimate "pre-tournament" state, which is exactly how the pre-tournament scenario is run.
The `frozenset` keys make fixtures order-independent.

### `build_matches_all(matches_hist, group_actual, ko_actual) -> matches_all`
Appends every entered 2026 result to the history as a neutral `FIFA World Cup` row, dated from the
tournament start (one day apart, just to order them). Because World-Cup rows carry the heaviest Elo K
and are the most recent, this is what lets entered results *re-train* team strength. Returns
`matches_hist` unchanged when nothing is entered.

---

## `predictor/elo.py` — World Football Elo

### `run_elo(matches) -> (ratings, pre_h, pre_a)`
Single chronological pass computing each team's Elo. For every match it records the **pre-match**
ratings of both sides (`pre_h`, `pre_a`) — these become a goals-model feature, so the model sees the
strength gap *as it was* when each game was played (no leakage). Update details:

- Expected home score `we = 1/(1 + 10^(−((rh + ha) − ra)/400))`, with `ha = 100` on non-neutral
  pitches, `0` on neutral.
- Goal-difference multiplier `g`: 1.0 for a 0–1 margin, 1.5 for a 2-goal margin, `(11 + gd)/8` beyond —
  bigger wins move ratings more.
- `change = K · g · (result − we)`; the home side gains it and the away side loses it (zero-sum).

Shootouts are treated as draws here (margin 0); the knockout simulator handles shootouts separately.
Unrated teams start at `INITIAL_RATING = 1500`.

---

## `predictor/goals_model.py` — time-weighted Poisson goals model

### `build_long(d)`
Reshapes match rows into **one row per scoring event**: each match yields a home-attacks-away row and
an away-attacks-home row. Returns parallel arrays `(team, opp, goals, home, elo_diff, dates)` with
`elo_diff` scaled by /100. This "long" layout is what lets a single regression learn separate
attack and defence coefficients per team.

### `fit_goals_model(d, use_elo=None, half_life=None, ref_date=None) -> model(dict)`
Fits `log λ = μ + attack_team + defence_opp + β_h·home + β_e·elo_diff`:

- One-hot encodes attacker and defender (`OneHotEncoder(handle_unknown="ignore")`), stacks on the home
  flag and (optionally) the Elo-difference column into a sparse matrix.
- **Recency weighting**: each row's sample weight is `0.5 ** (age_in_days / half_life)` relative to
  `ref_date` (defaults to the latest training date). A two-year half-life means a match from 2 years
  ago counts half as much as today's. `ref_date` is pinned during back-testing for reproducibility.
- Fits a `PoissonRegressor` (L2 `alpha`), then unpacks coefficients into a plain dict
  (`att`, `dfn`, `hc`, `elo_beta`, `intercept`) — trivially serialisable and cheap to pass around.

### `xg(model, attacker, defender, elo_att, elo_def, home) -> float`
Evaluates the model to expected goals: `exp(intercept + att + dfn + hc·home + elo_beta·elo_diff/100)`.
Unknown teams contribute 0 (handled via `.get(..., 0.0)`).

### `wdl_probs(model, a, b, ea, eb, home, maxg=10) -> (pH, pD, pA)`
Builds two independent Poisson goal distributions (up to `maxg`), forms the outer-product score matrix,
and sums its lower triangle / diagonal / upper triangle for home-win / draw / away-win probabilities.
Used by the back-test.

### `dc_tau(i, j, la, lb, rho) -> float`
The Dixon-Coles (1997) correction factor for the four low-scoring cells (0-0, 0-1, 1-0, 1-1). Plain
Poisson independence slightly under-predicts draws and 1-0/0-1 games; `rho < 0` nudges mass toward
those cells. Returns 1.0 elsewhere.

### `predict_scoreline(model, ratings, a, b, home_a=0, home_b=0, rho=None, maxg=10)`
The display-grade match predictor. Computes both expected-goal means, builds the score matrix, applies
`dc_tau` to the four corner cells, renormalises, and returns
`((goals_a, goals_b), (pH, pD, pA), (la, lb), M)` — the most-likely scoreline, outcome probabilities,
the raw means, and the full normalised joint-probability matrix `M` (the bracket projector slices `M`
to find a *representative decisive* score for a projected winner).

---

## `predictor/squad_values.py` — optional Transfermarkt + GB hybrid

### `_load_squad_values() -> (log_value_by_team, median, missing)`  *(process-cached)*
Streams the gzipped Transfermarkt player CSV, maps citizenship to dataset spelling via `TM_MAP`, drops
zero/blank values, and for each WC team sums the top-`SQUAD_SIZE` (23) market values, storing
`log1p(sum)`. Teams with <5 players are imputed at the median. The result is cached at module level so
both pipeline scenarios reuse one download.

### `_engineer_features(d, ratings, squad_val, med_sv) -> DataFrame`
Builds the GB training frame, again one row per scoring event, with rolling state maintained in a
single pass: `elo_diff`, `form_gf`/`form_ga` (mean goals for/against over the last 5 games), `rest_days`
(days since the team's previous match, capped at 365; 180 if unseen), `squad_val_diff`, and `is_home`.
Form/rest are computed from *prior* matches only, so there's no leakage within the frame.

### `build_squad_value_model(matches_all, ratings) -> (xg_gb | None, info)`
Best-effort upgrade path. If `config.USE_SQUAD_VALUE_GB` is `False`, returns `(None, info)` immediately.
Otherwise it loads squad values, engineers features over the training window, fits a
`HistGradientBoostingRegressor(loss="poisson")` on the first 85% and reports holdout Poisson deviance on
the rest, builds a recent-form snapshot per WC team, and returns a closure:

- **`xg_gb(a, b, home=0) -> float`** — expected goals for `a` vs `b` from the GB model, using the teams'
  current Elo diff, recent form, a typical tournament rest of 4 days, squad-value diff, and the home
  flag, clipped to `[0.05, 15]`.

Any exception (offline, download failure, sklearn issue) is caught and returned as `(None, info)` with
a message, so the caller transparently falls back to Poisson+Elo. `info` always carries `ok` and
`message`.

---

## `predictor/tournament.py` — simulation engine

### `build_lambda_matrices(xg_fn) -> (LAM_NEU, LAM_HOME, idx, all_teams)`
Pre-computes expected goals for **every ordered team pair**, once, in both neutral and home variants,
using whatever `xg_fn(attacker, defender, home)` is supplied (the Poisson closure or `xg_gb`). This is
the key performance trick: after this, a full tournament simulation is just array indexing + Poisson
draws. `idx` maps team → matrix row.

### `r32_pairs(W, R, T) -> {match_no: (teamA, teamB)}`
Hard-codes the Round-of-32 wiring (Wikipedia match numbers 73–88) from group winners `W`, runners-up
`R`, and the third-place slot assignments `T`.

### class `TournamentSimulator`
Bundles the lambda matrices, the team index, an Elo array (for shootouts), and the entered-results
dicts. Constructed once per scenario.

- **`pen_p(a, b)`** — probability `a` wins a shootout, a deliberately mild Elo tilt
  (`/2000` denominator → near coin-flip), reflecting how close real shootouts are.
- **`_compute_pen_mode(...)`** *(static)* — simulates 4000 shootouts to find the single most common
  (winner, loser) score, used only to *display* a plausible shootout scoreline in projected brackets.
- **`sim_group(teams, rng) -> (order, stats)`** — plays a group's six fixtures. Uses the **entered
  score** where present, else Poisson draws (host gets the home λ). Applies FIFA tie-breakers
  points → goal difference → goals for → random (the random stands in for the unmodelled fair-play /
  drawing-of-lots steps). Returns the finishing order and per-team `{pts, gd, gf}`.
- **`ko(a, b, rng) -> winner`** — plays a knockout match. Returns the **entered winner** where present;
  otherwise regulation Poisson, then extra time at ⅓ the scoring rate, then a shootout via `pen_p`.
- **`resolve_thirds(order, stats, rng) -> (qual, third_team, best8_groups)`** — ranks the 12 third-placed
  teams by the same tie-breakers and takes the best 8; returns the `frozenset` of their groups (the key
  into `THIRD_PLACE_TABLE`) and the group→team map.
- **`sim_tournament(rng) -> dict`** — one full realisation: all groups, then resolve thirds, look up the
  R32 slot assignment, play R32 → Final via `r32_pairs` + `LATER`. Returns champion, finalists,
  semifinalists, last-16, the group orders, and the qualifying-third groups.
- **`monte_carlo(n_sims, seed) -> (forecast_df, pos, q3, n_sims)`** — runs `sim_tournament` `n_sims`
  times off one seeded RNG (reproducible), tallying how often each team becomes champion / reaches each
  stage and where it finishes its group. `forecast_df` is sorted by champion %; `pos[g][team]` is the
  P(finish 1st..4th) vector; `q3[team]` is P(qualify as a best third).

---

## `predictor/predictions.py` — human-readable forecasts

### `group_stage_predictions(model, ratings, group_actual) -> DataFrame`
One row per group match. Entered games show the actual score flagged `✓ actual`; the rest show the
most-likely Dixon-Coles scoreline plus the W/D/L split (hosts get home advantage via the `HOSTS` set).

### `_projected_finishers(pos, q3) -> (proj_W, proj_R, proj_3, T_proj)`
Turns the Monte-Carlo position frequencies into a single *modal* bracket input: each group's most-likely
winner, runner-up, and third; then the eight groups whose modal third qualifies most often, mapped
through `THIRD_PLACE_TABLE` into R32 slot assignments `T_proj`.

### `project_bracket(model, ratings, simulator, pos, q3, ko_actual) -> (df, champion, finalists)`
Walks the bracket in `KO_MATCH_ORDER`, resolving each tie via the inner `project_match(a, b)`:

- If the fixture was actually played (in `ko_actual`), show the real score / shootout and winner (`✓`).
- Otherwise pick the winner by `pH + pD·pen_p` (a draw is resolved toward the shootout-favoured side).
  If the modal score is a draw and the tie is genuinely even, show the modal shootout score; else slice
  the score matrix `M` to the winner's half and show a representative decisive scoreline.

Returns the bracket table plus the champion and the two finalists. This is the single "wall chart" path;
the probability tables are the rigorous companion.

---

## `predictor/validation.py` — out-of-sample engine back-test

### `_rps(probs, outcome_index)`
Ranked Probability Score for one ordered (W, D, L) forecast: mean squared error of the cumulative
distributions. The standard metric for ordered football outcomes.

### `evaluate(model, test) -> (accuracy, log_loss, RPS)`
For every test match, gets `wdl_probs`, scores accuracy (argmax correct?), log-loss (−log of the true
class probability), and RPS, then averages.

### `backtest(matches_all) -> DataFrame`
The strict temporal validation: train on `TRAIN_SINCE ≤ date < BACKTEST_TRAIN_END` (i.e. < 2018), test
on 2018–2022, for both Poisson-only and Poisson+Elo, with `ref_date` pinned to the train cutoff. This
characterises the **engine** (it ignores 2026 entirely), and confirms the Elo feature improves all three
metrics — typically RPS ≈ 0.174 with Elo vs 0.178 without.

---

## `predictor/accuracy.py` — forecast-vs-actual scoring

This is the answer to *"how accurate is my model?"* It scores the **pre-tournament** model (trained on
zero 2026 data) against the results actually entered — a true out-of-sample report with no look-ahead.

### `_outcome_index`, `_rps`, `_brier`
Small helpers: map a scoreline to 0/1/2 (home/draw/away); RPS for one ordered forecast; multiclass
Brier score (squared error of the 3-way probability vector vs the one-hot truth).

### `score_forecast(model, ratings, group_actual, ko_actual) -> (per_match_df, summary)`
Iterates the played matches:

- **Group games** get the full probabilistic treatment via `predict_scoreline` (hosts get home λ):
  predicted vs actual scoreline, expected goals, the predicted/actual outcome, and per-match
  `outcome_hit`, `exact_hit`, `rps`, `brier`, `logloss`, and per-side goal errors.
- **Knockout games** are scored on **winner accuracy** (predicted stronger side vs who actually
  advanced, shootouts included) plus goal errors — probabilistic W/D/L metrics don't cleanly apply once
  a draw can be decided on penalties.

The `summary` aggregates everything and, for the group matches, also computes the same RPS/log-loss for a
**no-skill base-rate forecast** (`BASELINE_WDL = 0.45/0.27/0.28`) and the resulting
`rps_skill_vs_baseline = 1 − model_rps/baseline_rps` (>0 means the model adds skill). `goal_mae` /
`goal_rmse` pool both teams across all played matches. Returns empty-but-valid output when nothing has
been entered yet.

---

## `run_pipeline.py` — orchestrator

### `_attach_elo(matches) -> (matches_with_elo, ratings)`
Runs `run_elo` and attaches the per-row `elo_h_pre` / `elo_a_pre` columns the goals model needs.

### `build_scenario(matches_hist, group_actual, ko_actual, label) -> bundle`
The shared code path for both scenarios — the *only* difference is whether the actuals dicts are
populated, which is what makes "with vs without 2026 data" a fair comparison. It builds `matches_all`,
runs Elo, fits the goals model, attempts the GB hybrid (choosing `xg_gb` if available, else a Poisson
closure as `xg_fn`), builds the lambda matrices and a `TournamentSimulator`, runs the Monte Carlo, and
returns everything as a dict bundle.

### `_write(df, name)` / `_write_chart(forecast)`
Thin output helpers: write a CSV to `OUTPUT_DIR`; render the champion-probability bar chart to PNG with
the non-interactive Agg backend (skipped gracefully if matplotlib is unavailable).

### `main()`
Forces UTF-8 stdout (Windows consoles default to cp1252 and choke on arrows/emoji), ensures the dirs,
loads and cleans data, then:

1. **Scenario A — pre-tournament** with empty actuals → writes `pretournament_*` outputs.
2. **Scenario B — to-date** with the loaded actuals → writes `todate_*` outputs and the chart.
3. **Overall winner** → top-5 podium from the to-date forecast.
4. **Accuracy** → scores Scenario A's model against the actuals → `accuracy_report.csv` /
   `accuracy_summary.csv`.
5. **Back-test** → engine validation on Scenario A's history → `backtest.csv`.
6. **`summary.md`** → a human-readable digest of the winner pick, accuracy, and back-test.

### `_write_summary(...)`
Renders `summary.md`: entered-result counts, GB status, the to-date podium table and champion, the
pre-tournament champion, the accuracy metrics, and the back-test table (rendered manually to avoid a
`tabulate` dependency).

---

## Performance & reproducibility notes

- A single tournament simulation is ~0.3 ms thanks to the pre-computed lambda matrices, so 20,000 sims
  per scenario run in a few seconds; the bulk of wall-clock time is the Transfermarkt download and the
  GB fit (run once, cached download reused across scenarios).
- Everything is seeded by `config.RNG_SEED` (Monte Carlo) and a fixed shootout-mode seed, so runs are
  bit-for-bit reproducible given the same input CSVs.
- The two scenarios are independent except for the shared cached squad-value download.

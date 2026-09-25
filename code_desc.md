# `code_desc.md`: in-depth description of the codebase

A function-by-function walkthrough of the World Cup 2026 prediction pipeline. Read top to bottom and
it follows the data flow: config → constants → data → Elo → goals model → squad-value hybrid →
simulation → predictions → validation → accuracy → final evaluation → report → orchestrator.

For each function: **what it takes, what it returns, how it works, and why it's done that way.**

---

## `config.py`: runtime configuration

No functions of consequence except `ensure_dirs()`. It is a flat namespace of tunables, deliberately
separated from `constants.py`:

- **`config.py` = choices** you might change between runs (paths, `N_SIMS`, feature toggles, training
  window, regularisation, Dixon-Coles ρ, back-test windows).
- **`constants.py` = facts** about the 2026 tournament that never change.

Key values: `INPUT_DIR` / `OUTPUT_DIR` / `FINAL_DIR` (derived from `ROOT_DIR` so the repo is
location-independent), the individual CSV paths (including `SQUAD_VALUES_CSV`, the frozen squad values),
`ALLOW_DOWNLOAD`, the two Transfermarkt URLs and `SQUAD_VALUE_AS_OF`, `TRAIN_SINCE`/`HALF_LIFE_DAYS`,
`USE_ELO_FEATURE`, `USE_SQUAD_VALUE_GB` + GB hyper-parameters, `N_SIMS`/`RNG_SEED`, `DIXON_COLES_RHO`,
the back-test windows and `RUN_LIVE_REPLAY` (the slow part of the final evaluation).

### `ensure_dirs()`
Creates `INPUT_DIR` and `OUTPUT_DIR` if absent (`os.makedirs(..., exist_ok=True)`). Called once at the
start of `run_pipeline.main()`.

---

## `constants.py`: tournament structure

Mostly data, plus two helpers.

- **`GROUPS`**: the 12 groups A–L, four teams each, spelled exactly as in the historical dataset so
  2026 squads join cleanly onto ~150 years of results. **`HOSTS`**: USA/Canada/Mexico (group-stage
  home advantage). **`ALL_TEAMS`**: flattened 48-team list; this ordering is the canonical index for
  every Monte-Carlo array. **`GROUP_FIXTURES`**: the six intra-group fixtures as index pairs.
- **`TOURNAMENT_START`**: the first match day. It is both the cut-off for the historical training
  data and the date from which entered results are stamped onto the timeline.
- **`LATER`**: for each R16+ match, the two feeder match numbers whose winners meet.
  **`THIRD_PLACE_MATCH` / `THIRD_PLACE_FEEDERS`**: match 103 is played by the *losers* of 101 and 102,
  so it is wired separately. **`ROUND_OF`**: match number → round label (`R32`…`Final`, `3rd`).
  **`KO_MATCH_ORDER`**: the order to resolve knockout matches (feeders before consumers, 73–104).
  **`WINNER_SLOT_ORDER`**: the eight R32 slots that receive third-placed teams.
- **`TM_MAP`**: Transfermarkt → dataset country-name fixes (e.g. `Korea, South` → `South Korea`).

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

## `predictor/data_io.py`: data acquisition, cleaning, ingestion

### `_download_if_missing(path, filename)`
If `path` doesn't exist, download `RESULTS_BASE_URL/filename` and cache it; raises if the file is
missing **and** `config.ALLOW_DOWNLOAD` is `False`. Keeps later runs fully offline once cached.

### `load_raw() -> (results_raw, shootouts, former_names)`
Ensures the three historical CSVs exist (downloading on first run) and returns them as DataFrames.
`former_names` is loaded for parity with the source dataset but is currently unused by the model.

### `clean_results(results_raw) -> matches_hist`
Produces the historical training base. Parses dates; **keeps only matches before `TOURNAMENT_START`**;
drops rows with missing scores (the dataset lists 2026 fixtures with blank scores; they must never
train the model); casts scores to int; normalises the `neutral` flag into a boolean `neutral_b`; sorts
chronologically and resets the index.

The date cut-off is a leakage guard: a refreshed upstream `results.csv` now contains the real 2026 World
Cup scores, which would otherwise leak into the "pre-tournament" model and be double-counted by the
to-date one. 2026 results enter the pipeline only through the actual-results CSV.

### `_check_team(team)`
Raises a clear `ValueError` if a team name isn't one of the 48 in `constants.GROUPS`, which catches typos in
the actual-results CSV early.

### `load_actuals(csv_path=None) -> (group_actual, ko_actual)`
Parses `data/input/actual_results_2026.csv` into the two dicts the simulator conditions on:

- `group_actual[frozenset({a, b})] = {"by_team": {a: ga, b: gb}, "date": …}`
- `ko_actual[frozenset({a, b})] = {"by_team": {…}, "pen_winner": team|None, "aet": bool, "date": …,
  "pen_score": {…}?}`

Scores include extra time (never shoot-out kicks). The optional `aet` column marks knockout ties that
went to extra time; a level score implies it. Validation: unknown teams, a fixture entered twice within
the same stage, a level knockout row without `pen_winner`, a `pen_winner` on a decisive score or naming a
third team, and a shoot-out score where the winner doesn't lead all raise `ValueError`. (A knockout
rematch of a group fixture is legitimate, so the stages are checked separately.) Returns empty dicts if
the file is absent (the legitimate "pre-tournament" state). The `frozenset` keys make fixtures
order-independent.

### `build_matches_all(matches_hist, group_actual, ko_actual) -> matches_all`
Appends every entered 2026 result to the history as a neutral `FIFA World Cup` row, dated from the
tournament start (one day apart, just to order them). Because World-Cup rows carry the heaviest Elo K
and are the most recent, this is what lets entered results *re-train* team strength. Returns
`matches_hist` unchanged when nothing is entered.

---

## `predictor/elo.py`: World Football Elo

### `run_elo(matches) -> (ratings, pre_h, pre_a)`
Single chronological pass computing each team's Elo. For every match it records the **pre-match**
ratings of both sides (`pre_h`, `pre_a`); these become a goals-model feature, so the model sees the
strength gap *as it was* when each game was played (no leakage). Update details:

- Expected home score `we = 1/(1 + 10^(−((rh + ha) − ra)/400))`, with `ha = 100` on non-neutral
  pitches, `0` on neutral.
- Goal-difference multiplier `g`: 1.0 for a 0–1 margin, 1.5 for a 2-goal margin, `(11 + gd)/8` beyond, so
  bigger wins move ratings more.
- `change = K · g · (result − we)`; the home side gains it and the away side loses it (zero-sum).

Shootouts are treated as draws here (margin 0); the knockout simulator handles shootouts separately.
Unrated teams start at `INITIAL_RATING = 1500`.

### `attach_elo(matches) -> (matches_with_elo, ratings)`
Runs `run_elo` and returns a copy of `matches` carrying the per-row `elo_h_pre` / `elo_a_pre` columns the
goals model trains on, plus the final ratings. Shared by `run_pipeline.build_scenario` and the live
replay.

---

## `predictor/goals_model.py`: time-weighted Poisson goals model

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
  (`att`, `dfn`, `hc`, `elo_beta`, `intercept`), trivially serialisable and cheap to pass around.

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

### `scoreline_grid(la, lb, rho=None, maxg=10) -> ((goals_a, goals_b), (pH, pD, pA), M)`
The model-agnostic core: turns two Poisson means into the Dixon-Coles-corrected joint score matrix `M`
(renormalised), the outcome probabilities and the modal scoreline. Because it takes λs rather than a
model, any expected-goals source (Poisson+Elo, the GB hybrid) can be scored the same way.

### `predict_scoreline(model, ratings, a, b, home_a=0, home_b=0, rho=None, maxg=10)`
The display-grade match predictor: computes both expected-goal means with `xg`, then calls
`scoreline_grid`. Returns `((goals_a, goals_b), (pH, pD, pA), (la, lb), M)`; the bracket projector
slices `M` to find a *representative decisive* score for a projected winner.

---

## `predictor/squad_values.py`: optional Transfermarkt + GB hybrid

### `_read_remote_csv(url, usecols)`
Streams a gzipped CSV over HTTP into a DataFrame (dates kept as ISO strings).

### `_compute_squad_values(as_of) -> DataFrame`
Downloads Transfermarkt's `players.csv` (citizenship) and `player_valuations.csv` (the dated history of
market values), takes each player's **latest valuation on or before `as_of`**, maps citizenship to the
dataset spelling via `TM_MAP`, and for each WC team sums the top-`SQUAD_SIZE` (23) values. Returns one
row per team: `team, squad_value_eur, n_players, as_of`.

### `_load_squad_values() -> (log_value_by_team, median, missing)`  *(process-cached)*
Reads the frozen values from `config.SQUAD_VALUES_CSV`; if the file is missing (or was built for a
different `SQUAD_VALUE_AS_OF`) it recomputes them with `_compute_squad_values` and rewrites the cache.
Stores `log1p(sum)` per team; teams with <5 valued players are imputed at the median. Freezing the values
matters twice over: the live Transfermarkt snapshot is rebuilt weekly, which made reruns drift, and
post-tournament price moves must never reach a "pre-tournament" forecast.

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

- **`xg_gb(a, b, home=0) -> float`**: expected goals for `a` vs `b` from the GB model, using the teams'
  current Elo diff, recent form, a typical tournament rest of 4 days, squad-value diff, and the home
  flag, clipped to `[0.05, 15]`.
- **`xg_gb.many(attackers, defenders, home)`**: the same for many pairs in one `predict` call; picked
  up by `build_lambda_matrices` (bit-identical numbers, ~100× faster).

Any exception (offline with no cache, download failure, sklearn issue) is caught and returned as
`(None, info)` with a message, so the caller transparently falls back to Poisson+Elo. `info` always
carries `ok` and `message`.

---

## `predictor/tournament.py`: simulation engine

### `build_lambda_matrices(xg_fn) -> (LAM_NEU, LAM_HOME, idx, all_teams)`
Pre-computes expected goals for **every ordered team pair**, once, in both neutral and home variants,
using whatever `xg_fn(attacker, defender, home)` is supplied (the Poisson closure or `xg_gb`); if
`xg_fn.many` exists all pairs go through it in one batch. This is the key performance trick: after this,
a full tournament simulation is just array indexing + Poisson draws. `idx` maps team → matrix row.

### `shootout_prob(elo_a, elo_b) -> float`
Probability `a` wins a shoot-out: a deliberately mild Elo tilt (`/2000` denominator → near coin-flip),
reflecting how close real shoot-outs are.

### `advance_prob(p90, la, lb, p_pens) -> float`
P(`a` goes through) from a knockout tie, mirroring `TournamentSimulator.ko`: the 90-minute probabilities,
then extra time as independent Poisson at a third of the 90-minute rates, then a shoot-out won with
`p_pens`. Used by the accuracy scoring of knockout ties.

### `r32_pairs(W, R, T) -> {match_no: (teamA, teamB)}`
Hard-codes the Round-of-32 wiring (Wikipedia match numbers 73–88) from group winners `W`, runners-up
`R`, and the third-place slot assignments `T`.

### class `TournamentSimulator`
Bundles the lambda matrices, the team index, an Elo array (for shootouts), and the entered-results
dicts. Constructed once per scenario.

- **`pen_p(a, b)`**: `shootout_prob` on the two teams' ratings.
- **`_compute_pen_mode(...)`** *(static)*: simulates 4000 shootouts to find the single most common
  (winner, loser) score, used only to *display* a plausible shootout scoreline in projected brackets.
- **`sim_group(teams, rng) -> (order, stats)`**: plays a group's six fixtures. Uses the **entered
  score** where present, else Poisson draws (host gets the home λ). Tie-breakers: points → goal
  difference → goals for → random (FIFA 2026 actually puts head-to-head first; the approximation is
  documented in the README and the random step stands in for fair play / lots). Returns the finishing
  order and per-team `{pts, gd, gf}`.
- **`ko(a, b, rng) -> winner`**: plays a knockout match. Returns the **entered winner** where present;
  otherwise regulation Poisson, then extra time at ⅓ the scoring rate, then a shootout via `pen_p`.
- **`resolve_thirds(order, stats, rng) -> (qual, third_team, best8_groups)`**: ranks the 12 third-placed
  teams by the same tie-breakers and takes the best 8; returns the `frozenset` of their groups (the key
  into `THIRD_PLACE_TABLE`) and the group→team map.
- **`sim_tournament(rng) -> dict`**: one full realisation: all groups, then resolve thirds, look up the
  R32 slot assignment, play R32 → Final in `KO_MATCH_ORDER` (the third-place play-off is skipped, since it
  decides no stage). Returns the champion, finalists (SF winners), semi-finalists (QF winners),
  quarter-finalists (R16 winners), last 16 (R32 winners), last 32 (all R32 participants), the group
  orders and the qualifying-third groups.
- **`monte_carlo(n_sims, seed) -> (forecast_df, pos, q3, n_sims)`**: runs `sim_tournament` `n_sims`
  times off one seeded RNG (reproducible), tallying how often each team reaches each stage and where it
  finishes its group. `forecast_df` has `champion_%`, `final_%`, `semifinal_%`, `quarterfinal_%`,
  `reach_R16_%`, `reach_R32_%` and is sorted by champion %; `pos[g][team]` counts finishes 1st..4th;
  `q3[team]` is P(qualify as a best third) in %.

---

## `predictor/predictions.py`: human-readable forecasts

### `group_stage_predictions(model, ratings, group_actual) -> DataFrame`
One row per group match. Entered games show the actual score flagged `actual`; the rest show the
most-likely Dixon-Coles scoreline plus the W/D/L split (hosts get home advantage via the `HOSTS` set).

### `projected_finishers(pos, q3) -> (proj_W, proj_R, proj_3, T_proj)`
Turns the Monte-Carlo position frequencies into a single *modal* bracket input: each group's most-likely
winner, runner-up, and third; then the eight groups whose modal third qualifies most often, mapped
through `THIRD_PLACE_TABLE` into R32 slot assignments `T_proj`. Also used by the final evaluation to
compare projected and actual group orders.

### `project_bracket(model, ratings, simulator, pos, q3, ko_actual) -> (df, champion, finalists)`
Walks the bracket in `KO_MATCH_ORDER` (all 32 ties, including the third-place play-off between the two
semi-final losers), resolving each tie via the inner `project_match(a, b)`:

- If the fixture was actually played (in `ko_actual`), show the real score / shootout and winner (`actual`).
- Otherwise pick the winner by `pH + pD·pen_p` (a draw is resolved toward the shootout-favoured side).
  If the modal score is a draw and the tie is genuinely even, show the modal shootout score; else slice
  the score matrix `M` to the winner's half and show a representative decisive scoreline.

Returns the bracket table (`match, round, fixture, team_a, team_b, score, winner, flag`) plus the
champion and the two finalists. This is the single "wall chart" path; the probability tables are the
rigorous companion.

---

## `predictor/validation.py`: out-of-sample engine back-test

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
metrics: typically RPS ≈ 0.174 with Elo vs 0.178 without.

---

## `predictor/accuracy.py`: forecast-vs-actual scoring of match predictions

This is the answer to *"how accurate is my model?"* at match level. Run on the **pre-tournament**
model (trained on zero 2026 data) it is a true out-of-sample report with no look-ahead; the final
evaluation reuses it for the other model variants and the live replay.

### `_outcome_index`, `_rps`, `_brier`, `_logloss`
Small helpers: map a scoreline to 0/1/2 (home/draw/away); RPS for one ordered forecast; multiclass
Brier score; clipped −log p.

### `actual_score_label(info) -> str`
Formats an entered result as `"1-1 aet (pens 4-3)"` in the record's team order.

### `score_matches(model, ratings, group_actual, ko_actual, xg_fn=None) -> DataFrame`
One row per played match. Expected goals come from `xg_fn(attacker, defender, home)` (by default the
Poisson(+Elo) `model` with `ratings`) and go through `scoreline_grid`. Every match is scored on its
**90-minute result** (a knockout tie that went to extra time counts as a draw): predicted vs actual
scoreline, W/D/L probabilities, `hit_90`, `exact_hit` (skipped for extra-time games, whose 90-minute
score isn't recorded), `rps`, `brier`, `logloss`, and per-side goal errors (extra-time games compared
with 4/3 of the 90-minute expectation). Knockout rows add `p_advance_a` from `advance_prob` (ratings
drive the shoot-out tilt, as in the simulator), the predicted and actual winner, `advance_hit`, and the
binary Brier / log-loss of the advancement forecast. Hosts get home advantage in the group stage only.

### `summarise(df) -> dict`
Aggregates a `score_matches` frame: `group_*` metrics on the 72 group games (with the no-skill base-rate
forecast `BASELINE_WDL = 0.45/0.27/0.28` and `rps_skill_vs_baseline`), `ko_*` metrics on the knockout
ties (with coin-flip references), `all90_*` metrics on every match, and `goal_mae` / `goal_rmse`.
Returns an empty-but-valid summary when nothing has been entered.

### `score_forecast(model, ratings, group_actual, ko_actual, xg_fn=None) -> (per_match_df, summary)`
`score_matches` + `summarise` in one call; what the pipeline writes to `accuracy_report.csv` /
`accuracy_summary.csv`.

---

## `predictor/evaluation.py`: the post-tournament verdict

Runs once all 104 results are entered (`is_complete`).

### Rebuilding the real tournament
- **`_table(teams, results)`**: points, goals and W/D/L over the results played among `teams`.
- **`_blocks(teams, key)`**: teams sorted by `key`, grouped into runs of equal keys.
- **`_rank_level(tied, results, overall)`**: orders teams level on points by the FIFA 2026 rules:
  head-to-head points, goal difference and goals among the tied teams, reapplied recursively to any
  still-level subset; then overall goal difference and goals. A tie that would need fair-play data
  raises.
- **`group_tables(group_actual)`**: the final table of every group (`pos, team, W, D, L, GF, GA, GD,
  Pts`); raises if a group fixture is missing.
- **`rank_thirds(tables)`**: ranks the twelve third-placed teams (points, goal difference, goals) and
  flags the best eight.
- **`reconstruct_bracket(group_actual, ko_actual) -> (bracket_df, tables, thirds_df)`**: runs the real
  tables through FIFA's slot table and `r32_pairs`, then feeds winners (and, for match 103, semi-final
  losers) forward. Every fixture must exist among the entered knockout results and every entered
  knockout result must be used, otherwise `ValueError`. This is the end-to-end integrity check on the
  data, and yields the real bracket with official match numbers.
- **`finish_index(bracket)`**: how far each team went (0 = group stage … 6 = champion).

### Match predictions
- **`live_replay(matches_hist, group_actual, ko_actual)`**: for each match day, re-fits Elo and the
  Poisson+Elo model on the history plus the results known before that day (the pipeline's own to-date
  procedure) and scores that day's matches. The concatenated frame is the live model's record.
- **`paired_bootstrap(diff)`**: mean of per-match differences with a 95% interval from resampling
  matches.
- **`model_comparison(scored, ratings_pre, reference="Poisson + Elo")`**: one row per model (plus a
  base-rate and a "higher Elo wins" baseline): accuracy, RPS, log-loss, knockout metrics and goal MAE,
  and each model's per-match RPS difference from the reference with its bootstrap interval.
- **`phase_breakdown(scored, cutoffs)`**: mean RPS per model on matchdays 1–3 and the knockouts.
- **`match_table(pre, live, bracket)`**: all 104 matches with the pre-tournament (and live)
  probabilities next to the result.
- **`surprises(pre, n)`**: the least likely group results, and the knockout ties the model called wrong.
- **`calibration_bins(scored)`**: pooled W/D/L probabilities binned against observed frequencies.

### Tournament predictions
- **`score_stage_probabilities(forecast, finish)`**: Brier score and log-loss of each stage-reach
  probability over all 48 teams against a uniform forecast, plus how many of the model's top-N candidates
  for each stage got there.
- **`team_comparison(forecast, finish, tables, pos)`**: per team: the six stage probabilities, the
  expected number of stages reached (their sum), the actual finish and the difference.
- **`group_comparison(pos, projected, tables)`**: projected vs actual finishing order per group.
- **`bracket_comparison(projected, actual) -> (per_match, per_round)`**: the projected bracket against
  the real one, tie by tie and stage by stage.
- **`checkpoints(group_actual, bracket)`**: the cut-off date of each group matchday and knockout round.
- **`title_timeline(pre_forecast, group_actual, ko_actual, bracket, run_scenario)`**: champion / final /
  semi-final odds at every checkpoint; `run_scenario` re-runs the full model (GB hybrid + Monte Carlo)
  on the results known by then.

---

## `predictor/report.py`: the final report and its figures

- **`THEMES`**: light and dark chart tokens (surface, ink, grid, a validated four-slot categorical
  palette). Every figure is rendered in both and embedded with `<picture>`, so GitHub shows the one that
  matches the reader's theme.
- **`pyplot()`, `style_axes()`, `save_figure()`**: matplotlib set-up with recessive chrome (hairline
  solid grid, muted ticks, no frame) and deterministic PNG output (no timestamp metadata).
- **`fig_title_odds`**: pre-tournament title odds of the top 16 as bars (square baseline, rounded data
  end), the final four emphasised, each team's actual finish alongside.
- **`fig_title_timeline`**: small multiples: each semi-finalist's title odds after every round,
  highlighted against the other three.
- **`fig_calibration`**: reliability diagram, pre-tournament vs live.
- **`write_figures(res, out_dir)`** / **`write_final_report(res, path)`**: render the figures and the
  markdown report from the evaluation results. Every sentence that depends on the outcome is computed
  from the data, and every chart has its numbers in a table beside it.

---

## `run_pipeline.py`: orchestrator

### `build_scenario(matches_hist, group_actual, ko_actual, label, verbose=True) -> bundle`
The shared code path for all scenarios: the *only* difference is which actual results are passed in,
which is what makes "with vs without 2026 data" a fair comparison. It builds `matches_all`, runs Elo,
fits the goals model, attempts the GB hybrid (choosing `xg_gb` if available, else a Poisson closure as
`xg_fn`), builds the lambda matrices and a `TournamentSimulator`, runs the Monte Carlo, and returns
everything (including `xg_fn`) as a dict bundle. The title-odds timeline calls it with `verbose=False`.

### `_write(df, name, folder=None)` / `_write_chart(pre_fc, todate_fc, n_results)`
Output helpers: write a CSV to `OUTPUT_DIR` (or a sub-folder); draw `champion_probabilities.png`, a
dumbbell of each team's title odds pre-tournament → to date.

### `_final_evaluation(...)`
Rebuilds and validates the bracket, scores every model (Poisson without Elo, Poisson + Elo, the GB
hybrid and, if `RUN_LIVE_REPLAY`, the live replay), re-simulates the title odds after each round,
writes the CSV tables, renders the figures and writes `final_evaluation/report.md`.

### `main()`
Forces UTF-8 stdout (Windows consoles default to cp1252 and choke on arrows/emoji), ensures the dirs,
loads and cleans data, then:

1. **Scenario A (pre-tournament)** with empty actuals → writes `pretournament_*` outputs.
2. **Scenario B (to-date)** with the loaded actuals → writes `todate_*` outputs and the chart.
3. **Overall winner** → top-5 podium from the to-date forecast.
4. **Accuracy** → scores Scenario A's model against the actuals → `accuracy_report.csv` /
   `accuracy_summary.csv`.
5. **Back-test** → engine validation on Scenario A's history → `backtest.csv`.
6. **Final evaluation** → once all 104 results are in, `final_evaluation/`.
7. **`summary.md`** → a human-readable digest of the winner pick, accuracy, back-test and (when complete)
   the final verdict.

### `_write_summary(...)`
Renders `summary.md`: entered-result counts, GB status, the final verdict when the tournament is
complete, the to-date podium table and champion, the pre-tournament champion, the accuracy metrics, and
the back-test table (rendered manually to avoid a `tabulate` dependency). The header states the date of
the latest result rather than the run date, so reruns reproduce the file exactly.

---

## Performance & reproducibility notes

- A single tournament simulation is ~0.3 ms thanks to the pre-computed lambda matrices, and the GB
  λ matrices are predicted in one batch, so a 20,000-simulation scenario takes about 15 seconds. A full
  run takes about four minutes, most of it the final evaluation's live replay (34 model fits) and seven
  extra scenarios for the title timeline; `RUN_LIVE_REPLAY = False` cuts it to about one minute.
- Every input is cached in `data/input/` (historical results to 6 June 2026, frozen squad values, the
  actual results), so runs work offline.
- Everything is seeded by `config.RNG_SEED` (Monte Carlo, GB model, bootstrap) and a fixed shootout-mode
  seed, so runs are bit-for-bit reproducible given the same input CSVs, which is why `data/output/` can
  be committed as the final record.

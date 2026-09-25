"""
config.py: central runtime configuration for the World Cup 2026 prediction pipeline.

Everything that a user might reasonably want to tune lives here:
  * filesystem layout (INPUT_DIR / OUTPUT_DIR and the individual file paths)
  * model hyper-parameters (training window, Elo K-factors live in constants.py)
  * Monte-Carlo settings (number of simulations, RNG seed)
  * feature toggles (Elo feature, squad-value GB hybrid, network downloads)

`constants.py` holds *structural* facts about the tournament that never change
(group composition, the bracket wiring, FIFA's third-place table). `config.py`
holds *choices* about how we run the model. Keep that distinction in mind when
deciding where a new value belongs.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# ROOT_DIR is the repository root (the folder this file sits in). All other
# paths are derived from it so the pipeline is location-independent.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# The two directories the whole pipeline reads from / writes to.
INPUT_DIR = os.path.join(ROOT_DIR, "data", "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "output")
# The post-tournament evaluation (report.md + its tables and figures).
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_evaluation")

# Individual input files.
RESULTS_CSV = os.path.join(INPUT_DIR, "results.csv")
SHOOTOUTS_CSV = os.path.join(INPUT_DIR, "shootouts.csv")
FORMER_NAMES_CSV = os.path.join(INPUT_DIR, "former_names.csv")

# The *actual* 2026 results: all 104 matches, verified against Wikipedia, ESPN
# and the martj42 dataset. Updated daily during the tournament; the pipeline
# re-reads it on every run.
ACTUAL_RESULTS_CSV = os.path.join(INPUT_DIR, "actual_results_2026.csv")

# Per-team squad market values frozen at SQUAD_VALUE_AS_OF (written on first
# use, then reused so every rerun sees exactly the same values).
SQUAD_VALUES_CSV = os.path.join(INPUT_DIR, "squad_values.csv")

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
# If an input CSV is missing locally, download it from these public sources.
# Set ALLOW_DOWNLOAD = False to force fully-offline operation (a missing file
# then raises instead of hitting the network).
ALLOW_DOWNLOAD = True
RESULTS_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
# Transfermarkt data (dcaribou/transfermarkt-datasets) for the optional GB
# hybrid: players.csv gives each player's citizenship, player_valuations.csv
# the dated history of market values.
TRANSFERMARKT_PLAYERS_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
TRANSFERMARKT_VALUATIONS_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/player_valuations.csv.gz")
# Squad values are taken as of the eve of the tournament: each player's latest
# valuation on or before this date. The live snapshot is rebuilt weekly, so
# reading "current" values made the pre-tournament forecast drift between runs
# (and would eventually let post-tournament price moves leak in).
SQUAD_VALUE_AS_OF = "2026-06-10"

# ---------------------------------------------------------------------------
# Goals model
# ---------------------------------------------------------------------------
# Only train the Poisson goals model on matches on/after this date, and decay
# each match's weight with the given half-life (days). 730 days = 2 years.
TRAIN_SINCE = "2008-01-01"
HALF_LIFE_DAYS = 730
POISSON_ALPHA = 1e-3          # L2 regularisation strength for PoissonRegressor
POISSON_MAX_ITER = 4000
USE_ELO_FEATURE = True        # include (Elo_team - Elo_opp)/100 in the goals model

# Dixon-Coles low-score correlation parameter used when turning two Poisson
# means into a scoreline distribution (negative => slight draw inflation).
DIXON_COLES_RHO = -0.13

# ---------------------------------------------------------------------------
# Squad-value gradient-boosting hybrid (optional, needs a network download)
# ---------------------------------------------------------------------------
USE_SQUAD_VALUE_GB = True     # if False, the pipeline stays on Poisson+Elo
SQUAD_SIZE = 23               # top-N players summed per nation
GB_MAX_ITER = 400
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 4
GB_MIN_SAMPLES_LEAF = 20

# ---------------------------------------------------------------------------
# Monte-Carlo simulation
# ---------------------------------------------------------------------------
N_SIMS = 20000                # tournaments simulated per scenario
RNG_SEED = 42                 # master seed -> fully reproducible runs

# ---------------------------------------------------------------------------
# Back-test (out-of-sample temporal validation, §7 in the old notebook)
# ---------------------------------------------------------------------------
BACKTEST_TRAIN_END = "2018-01-01"   # train strictly before this date
BACKTEST_TEST_START = "2018-01-01"  # test on this window (real WC 2018 + after)
BACKTEST_TEST_END = "2022-12-31"

# ---------------------------------------------------------------------------
# Final evaluation (runs once all 104 results are entered)
# ---------------------------------------------------------------------------
# Replay the live forecast: re-fit the model before every match day on the
# results known at that point, and re-run the full Monte Carlo after each round.
# This is what makes "live model vs pre-tournament model" measurable; it adds
# ~3 minutes to a run. Set False to skip it (the rest of the report still runs).
RUN_LIVE_REPLAY = True

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
# How many teams to show in the printed champion-probability table.
TOP_N_DISPLAY = 16
# Write a champion-probability bar chart PNG (requires matplotlib).
WRITE_CHART = True


def ensure_dirs() -> None:
    """Create the input/output directories if they do not yet exist."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

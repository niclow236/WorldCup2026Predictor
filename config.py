"""
config.py — central runtime configuration for the World Cup 2026 prediction pipeline.

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

# Individual input files.
RESULTS_CSV = os.path.join(INPUT_DIR, "results.csv")
SHOOTOUTS_CSV = os.path.join(INPUT_DIR, "shootouts.csv")
FORMER_NAMES_CSV = os.path.join(INPUT_DIR, "former_names.csv")

# The daily-updated file of *actual* 2026 results. Edit this each day; the
# pipeline re-reads it on every run. See data/input/actual_results_2026.csv.
ACTUAL_RESULTS_CSV = os.path.join(INPUT_DIR, "actual_results_2026.csv")

# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
# If an input CSV is missing locally, download it from these public sources.
# Set ALLOW_DOWNLOAD = False to force fully-offline operation (a missing file
# then raises instead of hitting the network).
ALLOW_DOWNLOAD = True
RESULTS_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
# Transfermarkt squad-value snapshot (used by the optional GB hybrid).
TRANSFERMARKT_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"

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

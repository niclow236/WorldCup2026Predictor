"""
predictor — the World Cup 2026 forecasting engine.

Module map (data flows top to bottom):

    data_io       load + clean historical results; parse the daily actuals CSV
    elo           World Football Elo ratings (re-computed incl. entered results)
    goals_model   time-weighted Poisson goals model (+ optional Elo feature)
    squad_values  optional Transfermarkt + gradient-boosting hybrid
    tournament    lambda matrices, group/knockout simulation, Monte Carlo
    predictions   per-match scorelines, projected standings + bracket
    validation    out-of-sample temporal back-test of the goals model
    accuracy      scores the pre-tournament forecast against actual 2026 results

`run_pipeline.py` at the repo root wires these together into the full pipeline.
"""

__all__ = [
    "data_io",
    "elo",
    "goals_model",
    "squad_values",
    "tournament",
    "predictions",
    "validation",
    "accuracy",
]

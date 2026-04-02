"""
results_logger.py
-----------------
Utilities for persisting Grassmann-IPCA + RFF backtest results across notebook
runs.  Each call to `append_results` adds the rows from `results_rff_df` (the
DataFrame produced by `make_results_df`) to a shared Parquet log file so that
experiments accumulate without needing to re-run earlier cells.

Typical usage in a notebook cell, right after `display(results_rff_df)`:

    from src.results_logger import append_results

    append_results(
        results_rff_df,
        num_stocks        = len(df.index.get_level_values("permno").unique()),
        window_len        = WINDOW_LEN,
        gamma             = GAMMA,
        num_iter_rff      = NUM_ITER_RFF,
        min_gradient_norm = 1e-4,   # omit or None if not used
        run_label         = "Run7-T24-z0_10",  # optional free-form tag
    )
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path to the shared log file — lives in cache/ next to the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = _REPO_ROOT / "cache" / "results_rff_log.parquet"


def append_results(
    results_rff_df: pd.DataFrame,
    *,
    num_stocks: int,
    window_len: int,
    gamma: float,
    num_iter_rff: int,
    min_gradient_norm: Optional[float] = None,
    run_label: str = "",
) -> pd.DataFrame:
    """Append *results_rff_df* (MultiIndex k/P/z) to the shared RFF results log.

    Parameters
    ----------
    results_rff_df:
        The DataFrame returned by ``make_results_df(aggregate(run_sweep(jobs)))``.
        Expected MultiIndex levels: ``(k, P, z)``; columns include ``avg_r2_oos``,
        ``avg_sharpe``, ``avg_subspace_stability``, etc.
    num_stocks:
        Size of the stock universe used in this run (e.g. 50 or 496).
    window_len:
        Rolling-window length in months (``WINDOW_LEN``).
    gamma:
        RFF bandwidth parameter (``GAMMA``).
    num_iter_rff:
        Number of independent RFF draws averaged per cell (``NUM_ITER_RFF``).
    min_gradient_norm:
        Convergence threshold passed to ``build_jobs``; ``None`` if not set.
    run_label:
        Optional free-form string to identify the experiment (e.g. ``"Run7-T24"``).

    Returns
    -------
    pd.DataFrame
        The full accumulated log after appending.
    """
    # Flatten the MultiIndex so every row is a plain record
    new_rows = results_rff_df.reset_index()

    # Attach run-level metadata
    new_rows["num_stocks"]        = num_stocks
    new_rows["window_len"]        = window_len
    new_rows["gamma"]             = gamma
    new_rows["num_iter_rff"]      = num_iter_rff
    new_rows["min_gradient_norm"] = min_gradient_norm
    new_rows["run_label"]         = run_label
    new_rows["run_timestamp"]     = datetime.datetime.utcnow().isoformat(timespec="seconds")

    # Load existing log (or start fresh)
    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_parquet(LOG_PATH, index=False)
    print(f"[results_logger] Appended {len(new_rows)} rows → {LOG_PATH}  "
          f"(total rows: {len(combined)})")
    return combined


def load_results() -> pd.DataFrame:
    """Load and return the full accumulated results log."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"No results log found at {LOG_PATH}. "
            "Run append_results() after at least one backtest."
        )
    return pd.read_parquet(LOG_PATH)

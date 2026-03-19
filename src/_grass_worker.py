"""
_grass_worker.py
================
All sweep logic that cannot live in a notebook cell:
  - _run_one        : single backtest job (must be importable by spawned processes)
  - _z_label        : canonical string key for a shrinkage value
  - build_rff_inputs: pre-computes all (n_feat, seed) RFF DataFrames
  - build_jobs      : builds the full (k, P, z, seed) Cartesian job list
  - run_sweep       : dispatches jobs in parallel and returns raw results
  - aggregate       : buckets raw results → results_by_factor_rff dict
  - make_results_df : converts aggregated dict → MultiIndex summary DataFrame

Usage from the notebook cell:
    from _grass_worker import run_sweep, make_results_df, Z_VALUES, _z_label
"""

import os
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest_GRASS_IPCA import run_ipca_grass_v2

# ---------------------------------------------------------------------------
# Thread-count caps — set before any numpy import in spawned workers
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ===========================================================================
# DEFAULT GRID  (override by passing kwargs to run_sweep)
# ===========================================================================
GAMMA            = 0.25
WINDOW_LEN       = 12
N_FEATURES_RFF   = [128, 1024, 4096, 16384]
NUM_FACTORS_LIST = [2, 4, 8, 12]
NUM_ITER_RFF     = 4

# Kelly (2024) log10(z) sweep.  0.0 = OLS baseline.
Z_VALUES = [0.0, 1e-2, 1.0]   # <- edit here to add/remove shrinkage levels

SCALAR_COLS = [
    "avg_r2_oos", "avg_sharpe",
    "avg_subspace_stability", "avg_max_principal_angle",
    "avg_mean_principal_angle", "avg_geodesic_accel",
    "avg_erank", "avg_erank_collapse", "avg_spectral_gap",
]


# ===========================================================================
# HELPERS
# ===========================================================================
def _z_label(z: float) -> str:
    """Stable string key for a shrinkage value (avoids float MultiIndex bugs)."""
    return f"z={z:.0e}"   # e.g. "z=0e+00", "z=1e-02", "z=1e+00"


# ===========================================================================
# WORKER  —  must be defined here (on-disk module) so spawned processes
#            can import it.  Never define this inside a notebook cell.
# ===========================================================================
def _run_one(job: dict) -> dict:
    """Single backtest run.  Receives a plain dict; returns tagged result dict."""
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"

    # Reconstruct the DataFrame from numpy arrays (avoids StringDtype pickle bug)
    df_trunc  = _payload_to_df(job["df_payload"])
    rff_cols  = job["df_payload"]["rff_cols"]

    res = run_ipca_grass_v2(
        df_trunc         = df_trunc,
        char_feats       = rff_cols,
        num_fact         = job["num_fact"],
        verbose          = job.get("verbose", False),
        window_len       = job["window_len"],
        max_nan          = 0.3,
        max_zero         = 0.3,
        min_non_nan_frac = 0.7,
        shrinkage        = job["shrinkage"],
    )
    res["_num_fact"]  = job["num_fact"]
    res["_n_feat"]    = job["n_feat"]
    res["_shrinkage"] = job["shrinkage"]
    res["_z_label"]   = job["z_label"]
    res["_seed"]      = job["seed"]
    return res


# ===========================================================================
# STEP 1 — PRE-COMPUTE RFF INPUTS
# ===========================================================================
def build_rff_inputs(
    X_chars,
    df_base,
    n_features_rff,
    num_iter_rff,
    gamma,
    RandomFourierFeatures,
):
    """
    Returns rff_inputs: {(n_feat, seed): (df_rff, rff_cols)}

    Parameters
    ----------
    X_chars               : (date x permno, m) DataFrame of imputed characteristics
    df_base               : original df with at least ["Price", "ret"] columns
    n_features_rff        : list of RFF feature counts to sweep
    num_iter_rff          : number of random seeds per feature count
    gamma                 : RBF bandwidth for RandomFourierFeatures
    RandomFourierFeatures : the RFF class (passed in to avoid import coupling)
    """
    rff_inputs = {}

    for n_feat, seed in itertools.product(n_features_rff, range(num_iter_rff)):
        rff      = RandomFourierFeatures(n_features=int(n_feat / 2), gamma=gamma)
        rff_data = rff.transform(X_chars, seed=seed)
        rff_cols = [f"rff_{j+1}" for j in range(rff_data.shape[1])]
        rff_df   = pd.DataFrame(rff_data, index=df_base.index, columns=rff_cols)
        df_rff   = (
            pd.concat([df_base[["Price", "ret"]], rff_df], axis=1)
            .dropna(subset=["ret"])
        )
        rff_inputs[(n_feat, seed)] = (df_rff, rff_cols)

    return rff_inputs


# ===========================================================================
# STEP 2 — BUILD JOB LIST
# ===========================================================================
def _df_to_payload(df, rff_cols):
    """
    Serialise a df_rff DataFrame as plain numpy arrays so joblib/loky can
    pickle it without hitting pandas StringDtype version mismatches between
    the main process and spawned workers.
    """
    all_cols = ["Price", "ret"] + list(rff_cols)
    return {
        "values"   : df[all_cols].to_numpy(dtype=float, na_value=float("nan")),
        "index"    : df.index,        # MultiIndex — picklable as-is
        "all_cols" : all_cols,
        "rff_cols" : list(rff_cols),
    }


def _payload_to_df(payload):
    """Reconstruct the DataFrame from the serialised payload inside a worker."""
    return pd.DataFrame(
        payload["values"],
        index   = payload["index"],
        columns = payload["all_cols"],
    )


def _job_cost(job: dict) -> float:
    """
    Heuristic wall-time proxy for a single job.

    Empirically the Grassmann CG optimiser scales roughly as:
        cost ~ k^1.5 * P * window_len * T_roll
    where T_roll = number of rolling windows = T - window_len.

    z (shrinkage) doesn't change cost — same linear solve either way.
    We don't know T_roll at job-build time so we use P * k^1.5 as a
    relative proxy; it's enough to correctly rank jobs by expected runtime.
    """
    return job["n_feat"] * (job["num_fact"] ** 1.5)


def build_jobs(
    rff_inputs,
    num_factors_list,
    n_features_rff,
    z_values,
    num_iter_rff,
    window_len,
):
    """
    Returns the full Cartesian (k, P, z, seed) job list, sorted longest-first.

    Longest-first ordering (LPT schedule) is the single most effective
    thing you can do for a heterogeneous job pool:
      - Slow jobs start immediately and run in parallel with each other.
      - Fast jobs fill the gaps at the end, minimising idle tail time.
      - Without it, one slow job can stall the pool after all others finish.

    DataFrames are pre-serialised to numpy arrays to avoid pandas pickle issues.
    """
    jobs = []
    for num_fact, n_feat, z, seed in itertools.product(
        num_factors_list, n_features_rff, z_values, range(num_iter_rff)
    ):
        df_rff, rff_cols = rff_inputs[(n_feat, seed)]
        jobs.append({
            "num_fact"   : num_fact,
            "n_feat"     : n_feat,
            "shrinkage"  : z,
            "z_label"    : _z_label(z),
            "seed"       : seed,
            "window_len" : window_len,
            "df_payload" : _df_to_payload(df_rff, rff_cols),  # numpy, not DataFrame
        })

    # LPT: sort descending by estimated cost so slow jobs start first
    jobs.sort(key=_job_cost, reverse=True)
    return jobs


def _optimal_workers(jobs: list[dict]) -> int:
    """
    Choose n_workers based on job count, CPU count, and peak memory.

    Rules:
      1. Never more than cpu_count - 1  (keep one core for OS / notebook)
      2. Never more than len(jobs)       (no point spawning idle workers)
      3. Memory guard: each worker holds one df_payload in RAM.
         Estimate peak RAM = n_workers * max_payload_bytes and cap so
         that we stay under 80% of available RAM.
      4. At least 1.
    """
    import math
    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available * 0.80
        max_payload = max(_job_cost(j) for j in jobs) * 8 * 10  # rough bytes heuristic
        mem_cap = max(1, int(avail_bytes / max_payload)) if max_payload > 0 else 9999
    except ImportError:
        mem_cap = 9999   # psutil not installed — skip memory guard

    n_cpu  = max(1, os.cpu_count() - 1)
    n_jobs = len(jobs)

    n_workers = min(n_cpu, n_jobs, mem_cap)
    return max(1, n_workers)


# ===========================================================================
# STEP 3 — PARALLEL EXECUTION
# ===========================================================================
def run_sweep(jobs, n_workers=None, verbose=False):
    """
    Dispatches all jobs in parallel via joblib/loky.

    Worker count selection (when n_workers=None):
      - Capped at cpu_count - 1
      - Capped at len(jobs)  (no idle workers)
      - Optionally capped by available RAM (requires psutil)

    Jobs should already be sorted longest-first (build_jobs does this).
    joblib pre_dispatch='all' sends all jobs to the queue immediately so
    workers always have the next job ready without waiting for the
    scheduler — important when job times vary widely.

    Parameters
    ----------
    verbose : bool, default False
        If True, print dispatch info and enable joblib progress messages.
    """
    if n_workers is None:
        n_workers = _optimal_workers(jobs)

    if verbose:
        n_jobs_total = len(jobs)
        cost_range   = (
            f"cost range {_job_cost(jobs[-1]):.0f}–{_job_cost(jobs[0]):.0f}"
            if jobs else ""
        )
        print(f"Dispatching {n_jobs_total} jobs across {n_workers} workers "
              f"({cost_range}, LPT order) ...")

    return Parallel(
        n_jobs       = n_workers,
        backend      = "loky",
        verbose      = 10 if verbose else 0,
        pre_dispatch = "all",   # all jobs queued immediately; workers never idle
    )(delayed(_run_one)(j) for j in jobs)


# ===========================================================================
# STEP 4 — AGGREGATE
# ===========================================================================
def aggregate(raw_results):
    """
    Buckets raw results by (k, P, z_label) and averages scalar metrics
    across seeds.  Returns results_by_factor_rff dict.
    """
    buckets = defaultdict(list)
    for r in raw_results:
        buckets[(r["_num_fact"], r["_n_feat"], r["_z_label"])].append(r)

    out = {}
    for (num_fact, n_feat, z), runs in buckets.items():
        out[(num_fact, n_feat, z)] = {
            # scalar averages
            "avg_r2_oos"              : np.mean([r["r2_oos"]                    for r in runs]),
            "avg_sharpe"              : np.mean([r["sharpe"]                    for r in runs]),
            "avg_subspace_stability"  : np.mean([r["mean_subspace_stability"]   for r in runs]),
            "avg_max_principal_angle" : np.mean([r["mean_max_principal_angle"]  for r in runs]),
            "avg_mean_principal_angle": np.mean([r["mean_mean_principal_angle"] for r in runs]),
            "avg_geodesic_accel"      : np.mean([r["mean_geodesic_accel"]       for r in runs]),
            "avg_erank"               : np.mean([r["mean_erank"]                for r in runs]),
            "avg_erank_collapse"      : np.mean([r["erank_collapse_frac"]       for r in runs]),
            "avg_spectral_gap"        : np.mean([r["mean_spectral_gap"]         for r in runs]),
            # per-seed time series
            "stability_dist_series"   : [r["stability_dist_series"]             for r in runs],
            "principal_angles_series" : [r["principal_angles_series"]           for r in runs],
            "geodesic_accel_series"   : [r["geodesic_accel_series"]             for r in runs],
            "erank_series"            : [r["erank_series"]                      for r in runs],
            "spectral_gap_series"     : [r["spectral_gap_series"]               for r in runs],
        }
    return out


# ===========================================================================
# STEP 5 — BUILD SUMMARY DATAFRAME
# ===========================================================================
def make_results_df(results_by_factor_rff):
    """
    Converts the aggregated dict to a (k, P, z)-MultiIndex DataFrame
    of scalar summary statistics.
    """
    scalar_rows = {
        k: {col: v for col, v in v_dict.items() if col in SCALAR_COLS}
        for k, v_dict in results_by_factor_rff.items()
    }
    df = pd.DataFrame(scalar_rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["k", "P", "z"])
    return df.sort_index()
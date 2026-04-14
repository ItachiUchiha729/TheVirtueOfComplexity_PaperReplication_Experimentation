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

  Market-cap-weighted variants (mktcap Sharpe):
  - build_rff_inputs_mktcap : same as build_rff_inputs but carries "mktcap" column
  - build_jobs_mktcap       : serialises mktcap in payload
  - _run_one_mktcap         : calls run_ipca_grass_mktcap
  - aggregate_mktcap        : like aggregate but also averages sharpe_mktcap
  - make_results_df_mktcap  : like make_results_df but adds avg_sharpe_mktcap col

Usage from the notebook cell:
    from _grass_worker import run_sweep, make_results_df, Z_VALUES, _z_label
    # mktcap variants:
    from _grass_worker import (build_rff_inputs_mktcap, build_jobs_mktcap,
                                run_sweep, aggregate_mktcap, make_results_df_mktcap)
"""

import os
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    # Package-context imports (e.g. "from src._grass_worker import ...")
    from .backtest_GRASS_IPCA import run_ipca_grass_v2
except ImportError:
    try:
        # Absolute package import fallback
        from src.backtest_GRASS_IPCA import run_ipca_grass_v2
    except ImportError:
        # Script-context fallback
        from backtest_GRASS_IPCA import run_ipca_grass_v2

try:
    # Package-context imports (e.g. "from src._grass_worker import ...")
    from .backtest_GRASS_IPCA_mktcap import run_ipca_grass_mktcap
except ImportError:
    try:
        # Absolute package import fallback
        from src.backtest_GRASS_IPCA_mktcap import run_ipca_grass_mktcap
    except ImportError:
        # Script-context fallback
        from backtest_GRASS_IPCA_mktcap import run_ipca_grass_mktcap

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
        df_trunc          = df_trunc,
        char_feats        = rff_cols,
        num_fact          = job["num_fact"],
        verbose           = job.get("verbose", False),
        window_len        = job["window_len"],
        max_nan           = 0.3,
        max_zero          = 0.3,
        min_non_nan_frac  = 0.7,
        shrinkage         = job["shrinkage"],
        min_gradient_norm = job.get("min_gradient_norm", 1e-6),
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
    min_gradient_norm: float = 1e-6,
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
    # Pre-serialise payloads once per unique (n_feat, seed) to avoid
    # redundant copies when multiple (k, z) jobs share the same data.
    payloads = {
        key: _df_to_payload(df_rff, rff_cols)
        for key, (df_rff, rff_cols) in rff_inputs.items()
    }

    jobs = []
    for num_fact, n_feat, z, seed in itertools.product(
        num_factors_list, n_features_rff, z_values, range(num_iter_rff)
    ):
        jobs.append({
            "num_fact"          : num_fact,
            "n_feat"            : n_feat,
            "shrinkage"         : z,
            "z_label"           : _z_label(z),
            "seed"              : seed,
            "window_len"        : window_len,
            "min_gradient_norm" : min_gradient_norm,
            "df_payload"        : payloads[(n_feat, seed)],
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


def run_sweep_mktcap(jobs, n_workers=None, verbose=False):
    """
    Identical to run_sweep but dispatches _run_one_mktcap instead of
    _run_one.  Use with job lists built by build_jobs_mktcap().
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
              f"({cost_range}, LPT order) [mktcap-weighted] ...")

    return Parallel(
        n_jobs       = n_workers,
        backend      = "loky",
        verbose      = 10 if verbose else 0,
        pre_dispatch = "all",
    )(delayed(_run_one_mktcap)(j) for j in jobs)


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


# ===========================================================================
# MKTCAP-WEIGHTED VARIANTS
# ===========================================================================
#
# These mirror the functions above but wire through run_ipca_grass_mktcap
# instead of run_ipca_grass_v2.  The only data-level change is that
# build_rff_inputs_mktcap requires df_base to contain a "mktcap" column
# (raw CRSP market equity: ABS(prc) * shrout / 1000, in millions).
#
# The mktcap column is carried through the serialised payload so spawned
# worker processes can access it without passing large objects over the
# pickling boundary more than once per (n_feat, seed) pair.
#
# Typical notebook usage:
#
#   # 0. Add mktcap to df (join CRSP msf query result)
#   df["mktcap"] = df_me_aligned["mktcap"]   # raw ABS(prc)*shrout/1000
#
#   # 1. Pre-compute RFF inputs (now carries mktcap)
#   rff_inputs = build_rff_inputs_mktcap(
#       X_chars=X_chars, df_base=df,
#       n_features_rff=N_FEATURES_RFF, num_iter_rff=NUM_ITER_RFF,
#       gamma=GAMMA, RandomFourierFeatures=RandomFourierFeatures,
#   )
#
#   # 2-4. Build jobs, run sweep, aggregate — run_sweep is unchanged
#   jobs        = build_jobs_mktcap(rff_inputs, ...)
#   raw_results = run_sweep(jobs)           # identical dispatcher
#   summary     = aggregate_mktcap(raw_results)
#   df_results  = make_results_df_mktcap(summary)
# ---------------------------------------------------------------------------

SCALAR_COLS_MKTCAP = SCALAR_COLS + ["avg_sharpe_mktcap"]


def build_rff_inputs_mktcap(
    X_chars,
    df_base,
    n_features_rff,
    num_iter_rff,
    gamma,
    RandomFourierFeatures,
):
    """
    Same as build_rff_inputs but includes the ``"mktcap"`` column so it is
    available inside the backtester for value-weighted portfolio construction.

    Parameters
    ----------
    (all identical to build_rff_inputs)
    df_base : pd.DataFrame
        Must contain ``"Price"``, ``"ret"``, **and** ``"mktcap"`` columns.
        ``mktcap`` should be raw CRSP market equity (ABS(prc)*shrout/1000,
        in millions) — **not** the cross-sectionally normalised "Size"
        characteristic.
    """
    rff_inputs = {}

    for n_feat, seed in itertools.product(n_features_rff, range(num_iter_rff)):
        rff      = RandomFourierFeatures(n_features=int(n_feat / 2), gamma=gamma)
        rff_data = rff.transform(X_chars, seed=seed)
        rff_cols = [f"rff_{j+1}" for j in range(rff_data.shape[1])]
        rff_df   = pd.DataFrame(rff_data, index=df_base.index, columns=rff_cols)
        df_rff   = (
            pd.concat([df_base[["Price", "ret", "mktcap"]], rff_df], axis=1)
            .dropna(subset=["ret"])
        )
        rff_inputs[(n_feat, seed)] = (df_rff, rff_cols)

    return rff_inputs


def _df_to_payload_mktcap(df, rff_cols):
    """
    Serialise a mktcap-augmented df_rff as plain numpy arrays.
    Identical to _df_to_payload but includes "mktcap" in the column list.
    """
    all_cols = ["Price", "ret", "mktcap"] + list(rff_cols)
    return {
        "values"   : df[all_cols].to_numpy(dtype=float, na_value=float("nan")),
        "index"    : df.index,
        "all_cols" : all_cols,
        "rff_cols" : list(rff_cols),
    }


def _run_one_mktcap(job: dict) -> dict:
    """
    Single mktcap-weighted backtest run.
    Drop-in replacement for _run_one that calls run_ipca_grass_mktcap.
    """
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"

    df_trunc  = _payload_to_df(job["df_payload"])   # _payload_to_df works unchanged
    rff_cols  = job["df_payload"]["rff_cols"]

    res = run_ipca_grass_mktcap(
        df_trunc          = df_trunc,
        char_feats        = rff_cols,
        num_fact          = job["num_fact"],
        verbose           = job.get("verbose", False),
        window_len        = job["window_len"],
        max_nan           = 0.3,
        max_zero          = 0.3,
        min_non_nan_frac  = 0.7,
        shrinkage         = job["shrinkage"],
        min_gradient_norm = job.get("min_gradient_norm", 1e-6),
    )
    res["_num_fact"]  = job["num_fact"]
    res["_n_feat"]    = job["n_feat"]
    res["_shrinkage"] = job["shrinkage"]
    res["_z_label"]   = job["z_label"]
    res["_seed"]      = job["seed"]
    return res


def build_jobs_mktcap(
    rff_inputs,
    num_factors_list,
    n_features_rff,
    z_values,
    num_iter_rff,
    window_len,
    min_gradient_norm: float = 1e-6,
):
    """
    Same as build_jobs but serialises payloads with _df_to_payload_mktcap
    (includes the mktcap column) and tags jobs so run_sweep dispatches
    _run_one_mktcap instead of _run_one.

    Pass the returned job list to the standard run_sweep — it will call
    ``job["worker_fn"](job)`` if present, falling back to _run_one.
    """
    payloads = {
        key: _df_to_payload_mktcap(df_rff, rff_cols)
        for key, (df_rff, rff_cols) in rff_inputs.items()
    }

    jobs = []
    for num_fact, n_feat, z, seed in itertools.product(
        num_factors_list, n_features_rff, z_values, range(num_iter_rff)
    ):
        jobs.append({
            "num_fact"          : num_fact,
            "n_feat"            : n_feat,
            "shrinkage"         : z,
            "z_label"           : _z_label(z),
            "seed"              : seed,
            "window_len"        : window_len,
            "min_gradient_norm" : min_gradient_norm,
            "df_payload"        : payloads[(n_feat, seed)],
        })

    jobs.sort(key=_job_cost, reverse=True)
    return jobs


def aggregate_mktcap(raw_results):
    """
    Like aggregate() but also averages ``sharpe_mktcap`` across seeds.
    Returns a dict with the same structure as aggregate(), with an extra
    ``avg_sharpe_mktcap`` key per (k, P, z) bucket.
    """
    buckets = defaultdict(list)
    for r in raw_results:
        buckets[(r["_num_fact"], r["_n_feat"], r["_z_label"])].append(r)

    out = {}
    for (num_fact, n_feat, z), runs in buckets.items():
        out[(num_fact, n_feat, z)] = {
            "avg_r2_oos"               : np.mean([r["r2_oos"]                    for r in runs]),
            "avg_sharpe"               : np.mean([r["sharpe_ew"]                 for r in runs]),
            "avg_sharpe_mktcap"        : np.mean([r["sharpe_mktcap"]             for r in runs]),
            "avg_subspace_stability"   : np.mean([r["mean_subspace_stability"]   for r in runs]),
            "avg_max_principal_angle"  : np.mean([r["mean_max_principal_angle"]  for r in runs]),
            "avg_mean_principal_angle" : np.mean([r["mean_mean_principal_angle"] for r in runs]),
            "avg_geodesic_accel"       : np.mean([r["mean_geodesic_accel"]       for r in runs]),
            "avg_erank"                : np.mean([r["mean_erank"]                for r in runs]),
            "avg_erank_collapse"       : np.mean([r["erank_collapse_frac"]       for r in runs]),
            "avg_spectral_gap"         : np.mean([r["mean_spectral_gap"]         for r in runs]),
            "stability_dist_series"    : [r["stability_dist_series"]             for r in runs],
            "principal_angles_series"  : [r["principal_angles_series"]           for r in runs],
            "geodesic_accel_series"    : [r["geodesic_accel_series"]             for r in runs],
            "erank_series"             : [r["erank_series"]                      for r in runs],
            "spectral_gap_series"      : [r["spectral_gap_series"]               for r in runs],
        }
    return out


def make_results_df_mktcap(results_by_factor_rff):
    """
    Like make_results_df() but includes the ``avg_sharpe_mktcap`` column.
    """
    scalar_rows = {
        k: {col: v for col, v in v_dict.items() if col in SCALAR_COLS_MKTCAP}
        for k, v_dict in results_by_factor_rff.items()
    }
    df = pd.DataFrame(scalar_rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["k", "P", "z"])
    return df.sort_index()
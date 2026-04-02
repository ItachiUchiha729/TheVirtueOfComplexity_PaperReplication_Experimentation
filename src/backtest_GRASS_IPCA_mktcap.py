"""
backtest_GRASS_IPCA_mktcap.py
=============================
Market-cap-weighted variant of the Grassmann-IPCA rolling backtest.

Identical to ``backtest_GRASS_IPCA.run_ipca_grass_v2`` in every respect
**except** the Sharpe-ratio calculation, which now uses value-weighted
(market-cap-weighted) portfolio returns rather than equal-weighted ones.

Methodology — replicating S&P-style value-weighting
----------------------------------------------------
At each OOS time step t the portfolio is constructed as follows:

  1.  **Weights** (beginning-of-period, i.e. *lagged* one month):
          w_i,t  =  mktcap_i,t-1  /  Σ_j mktcap_j,t-1

      Using *lagged* market cap is the academic standard (Fama-French 1992;
      CRSP Value-Weighted Index methodology).  It ensures the weighting
      information would have been available to an investor before period t
      begins and avoids look-ahead bias.

  2.  **Position sizing** (signal-proportional, mktcap-normalised):
          pos_i,t  =  ŷ_i,t  /  ( Σ_j w_j,t |ŷ_j,t| + ε )

      Dividing by the *mktcap-weighted* mean absolute signal keeps the
      position-normalisation consistent with the mktcap-weighted portfolio
      aggregation.  (Equal-weighted normalisation would create a subtle
      mismatch between sizing and aggregation.)

  3.  **Portfolio return**:
          port_ret_t  =  Σ_i w_i,t · pos_i,t · r_i,t

  4.  **Sharpe ratio** (annualised, assuming monthly returns):
          SR_mktcap  =  mean(port_ret) / std(port_ret, ddof=1) · √12

The function also returns the original equal-weighted Sharpe (``sharpe_ew``)
alongside the new ``sharpe_mktcap`` so the two can be compared directly.

WRDS / CRSP data note
---------------------
``df_trunc`` must contain a ``"mktcap"`` column (market equity in **millions**):

    mktcap  =  ABS(prc)  ×  shrout  /  1_000

where ``prc`` and ``shrout`` come from ``crsp.msf``.  Do **not** use the
``"Size"`` characteristic column — that is log(mktcap) cross-sectionally
z-scored (negative values are expected and normal) and carries no dollar-
magnitude information.

Example WRDS query (run once in notebook, then join to df)::

    sql = f\"\"\"
        SELECT msf.permno, msf.date,
               ABS(msf.prc) * msf.shrout / 1000.0 AS mktcap
        FROM crsp.msf
        WHERE msf.date >= '1993-12-01'
          AND msf.date <= '2025-12-31'
          AND msf.permno IN ({permnos_str})
    \"\"\"
    df_me = db.raw_sql(sql, date_cols=["date"])
    # Align date to first-of-month to match your panel index, then merge:
    df_me["date"] = df_me["date"].dt.to_period("M").dt.to_timestamp()
    df = df.join(
        df_me.set_index(["date", "permno"])["mktcap"],
        how="left",
    )

Dependencies
------------
numpy, GrassmannManifoldIPCAEstimator (from IPCA_Grass_estimator.py)
"""

import numpy as np

try:
    from IPCA_Grass_estimator import GrassmannManifoldIPCAEstimator
except ModuleNotFoundError:
    from src.IPCA_Grass_estimator import GrassmannManifoldIPCAEstimator

# Re-export the helpers so callers can import from one place.
try:
    from backtest_GRASS_IPCA import _principal_angles, _spectral_gap_ratio
except ModuleNotFoundError:
    from src.backtest_GRASS_IPCA import _principal_angles, _spectral_gap_ratio


# ===========================================================================
#  MAIN FUNCTION
# ===========================================================================

def run_ipca_grass_mktcap(
    df_trunc,
    char_feats,
    num_fact: int   = 2,
    window_len: int = 12,
    max_nan: float  = 0.3,
    max_zero: float = 0.3,
    min_non_nan_frac: float = 0.7,
    shrinkage: float = 0.0,
    min_gradient_norm: float = 1e-6,
    compute_seed_variance: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Market-cap-weighted rolling-window backtest for Grassmann-IPCA.

    Identical to ``run_ipca_grass_v2`` except:
      - ``df_trunc`` must contain a ``"mktcap"`` column (raw CRSP market
        equity: ABS(prc) × shrout / 1000, **not** the normalised "Size"
        characteristic).
      - The Sharpe ratio is computed from value-weighted portfolio returns
        (S&P-style: lagged mktcap weights, mktcap-normalised position sizing).
      - Both ``sharpe_mktcap`` *and* ``sharpe_ew`` (equal-weighted, for
        comparison) are returned.

    Parameters
    ----------
    df_trunc : pd.DataFrame
        Multi-index (date, permno) DataFrame containing:
          - ``"ret"``    : stock return (aligned/lagged)
          - ``"mktcap"`` : market equity in millions (ABS(prc)*shrout/1000)
          - *char_feats* : characteristic columns (e.g. RFF features)
    char_feats : list[str]
        Names of characteristic columns to use as instruments Z.
    (all other parameters identical to run_ipca_grass_v2)

    Returns
    -------
    dict — all keys from run_ipca_grass_v2, plus:
        sharpe_mktcap : float  — annualised value-weighted Sharpe ratio
        sharpe_ew     : float  — annualised equal-weighted Sharpe (identical
                                 to ``sharpe`` in run_ipca_grass_v2)
    Note: the ``"sharpe"`` key still exists and equals ``sharpe_ew`` so that
    the aggregate / make_results_df helpers work unchanged.
    """

    # ------------------------------------------------------------------
    # 1. BUILD PANELS
    # ------------------------------------------------------------------
    df_panel = df_trunc.copy().sort_index()

    rets_panel   = df_panel["ret"].unstack("permno").sort_index()
    mktcap_panel = df_panel["mktcap"].unstack("permno").sort_index()

    X_list = []
    for col in char_feats:
        x = df_panel[col].unstack("permno").sort_index()
        X_list.append(x)

    # ------------------------------------------------------------------
    # 2. ALIGN ASSETS ACROSS ALL PANELS (returns + chars + mktcap)
    # ------------------------------------------------------------------
    common_assets = rets_panel.columns.intersection(mktcap_panel.columns)
    for x in X_list:
        common_assets = common_assets.intersection(x.columns)

    rets_panel   = rets_panel[common_assets]
    mktcap_panel = mktcap_panel[common_assets]
    X_list       = [x[common_assets] for x in X_list]

    # ------------------------------------------------------------------
    # 3. DROP ASSETS WITH TOO MANY MISSING / ZERO VALUES
    #    (based on returns and chars; mktcap missingness handled at
    #    prediction time via cross-sectional mean imputation)
    # ------------------------------------------------------------------
    zero_frac_rets = (rets_panel == 0).mean()
    nan_frac_rets  = rets_panel.isna().mean()

    good_assets = (nan_frac_rets <= max_nan) & (zero_frac_rets <= max_zero)
    for x in X_list:
        good_assets &= (x.isna().mean()  <= max_nan)
        good_assets &= ((x == 0).mean() <= max_zero)

    rets_panel   = rets_panel.loc[:, good_assets]
    mktcap_panel = mktcap_panel.loc[:, good_assets]
    X_list       = [x.loc[:, good_assets] for x in X_list]

    # ------------------------------------------------------------------
    # 4. DROP EARLY ROWS WITH HEAVY NaNs (burn-in)
    # ------------------------------------------------------------------
    min_non_nan = int(min_non_nan_frac * rets_panel.shape[1])

    valid_time = rets_panel.notna().sum(axis=1) >= min_non_nan
    for x in X_list:
        valid_time &= (x.notna().sum(axis=1) >= min_non_nan)

    rets_panel   = rets_panel.loc[valid_time]
    mktcap_panel = mktcap_panel.loc[valid_time]
    X_list       = [x.loc[valid_time] for x in X_list]

    # ------------------------------------------------------------------
    # 5. LAG MKTCAP BY ONE PERIOD (beginning-of-period weights)
    #    S&P / Fama-French standard: weight returns at t using mktcap
    #    observed at t-1.  This avoids look-ahead bias and matches the
    #    convention used in CRSP value-weighted index construction.
    # ------------------------------------------------------------------
    mktcap_panel = mktcap_panel.shift(1)   # row t now holds mktcap_{t-1}
    # Row 0 becomes all-NaN after the shift — it will be handled by the
    # same `valid` mask that filters the OOS predictions.

    # ------------------------------------------------------------------
    # 6. IMPUTE REMAINING NaNs WITH CROSS-SECTIONAL MEAN
    # ------------------------------------------------------------------
    rets_panel = rets_panel.apply(lambda s: s.fillna(s.mean()), axis=1)
    X_list     = [x.apply(lambda s: s.fillna(s.mean()), axis=1)
                  for x in X_list]
    # Mktcap: impute with cross-sectional median (robust to outliers).
    # Negative / zero values are replaced with the median so they don't
    # create negative weights.
    mktcap_panel = mktcap_panel.apply(
        lambda s: s.where(s > 0).fillna(s[s > 0].median()), axis=1
    )
    mktcap_panel = mktcap_panel.fillna(0.0)   # any remaining NaN → 0 weight

    # ------------------------------------------------------------------
    # 7. STANDARDISE CHARACTERISTICS CROSS-SECTIONALLY
    # ------------------------------------------------------------------
    X_list = [
        (x.sub(x.mean(axis=1), axis=0))
         .div(x.std(axis=1).replace(0, np.nan), axis=0)
        for x in X_list
    ]
    X_list = [x.fillna(0.0) for x in X_list]

    # ------------------------------------------------------------------
    # 8. STACK INTO NUMPY TENSORS
    # ------------------------------------------------------------------
    Z          = np.stack([x.to_numpy(dtype=float) for x in X_list], axis=2)
    rets_stock = rets_panel.to_numpy(dtype=float)
    mktcap_arr = mktcap_panel.to_numpy(dtype=float)   # (T, N), lagged

    T, N   = rets_stock.shape
    m_dim  = Z.shape[2]
    preds_stock = np.full_like(rets_stock, np.nan, dtype=float)

    d_proj_random_baseline = float(
        np.sqrt(2 * num_fact * (m_dim - num_fact) / (m_dim + 1))
    )

    # ------------------------------------------------------------------
    # 9. ACCUMULATOR INITIALISATION
    # ------------------------------------------------------------------
    W_prev                  = None
    stability_dists         = []
    erank_series            = []
    principal_angles_series = []
    spectral_gap_series     = []
    d_proj_norm_series      = []
    d_proj_seeds_series     = []

    # ------------------------------------------------------------------
    # 10. ROLLING WINDOW LOOP  (unchanged from run_ipca_grass_v2)
    # ------------------------------------------------------------------
    for t in range(window_len, T):

        rets_tr = rets_stock[t - window_len:t]
        Z_tr    = Z[t - window_len:t]

        est = GrassmannManifoldIPCAEstimator(
            num_assets  = N,
            num_fact    = num_fact,
            num_charact = Z.shape[2],
            win_len     = rets_tr.shape[0],
            shrinkage   = shrinkage,
        )
        W_t, f_tr, _ = est.fit(
            data              = [rets_tr, Z_tr],
            optimizer         = "ConjugateGradient",
            max_iterations    = 200,
            min_gradient_norm = min_gradient_norm,
            verbosity         = 0,
            log_verbosity     = 0,
            initial_point     = W_prev if W_prev is not None else None,
        )

        # Metric 1 + 2: projection distance and principal angles
        if W_prev is not None:
            angles = _principal_angles(W_t, W_prev)
            principal_angles_series.append(angles)
            d_proj = float(np.sqrt(2.0 * np.sum(np.sin(angles) ** 2)))
            stability_dists.append(d_proj)
            d_proj_norm = (
                d_proj / d_proj_random_baseline
                if d_proj_random_baseline > 0 else np.nan
            )
            d_proj_norm_series.append(float(d_proj_norm))

        # Metric 7: between-seed d_proj
        if compute_seed_variance:
            est_rand = GrassmannManifoldIPCAEstimator(
                num_assets  = N,
                num_fact    = num_fact,
                num_charact = m_dim,
                win_len     = rets_tr.shape[0],
                shrinkage   = shrinkage,
            )
            W_t_rand, _, _ = est_rand.fit(
                data              = [rets_tr, Z_tr],
                optimizer         = "ConjugateGradient",
                max_iterations    = 200,
                min_gradient_norm = min_gradient_norm,
                verbosity         = 0,
                log_verbosity     = 0,
                initial_point     = None,
            )
            M_seeds = W_t.T @ W_t_rand
            d_seeds = float(np.sqrt(max(0.0, 2.0 * (num_fact - np.sum(M_seeds ** 2)))))
            d_proj_seeds_series.append(d_seeds)

        W_prev = W_t

        # Metric 4 + 5: effective rank and spectral gap
        Sigma_f = (f_tr.T @ f_tr) / f_tr.shape[0]
        eigvals  = np.linalg.eigvalsh(Sigma_f)
        eigvals  = np.maximum(eigvals, 0.0)
        total = eigvals.sum()
        if total > 0:
            p     = eigvals / total
            p     = p[p > 0]
            erank = np.exp(-np.sum(p * np.log(p)))
        else:
            erank = 1.0
        erank_series.append(erank)
        spectral_gap_series.append(_spectral_gap_ratio(eigvals, num_fact))

        # Forecast
        f_next         = f_tr.mean(axis=0)
        Lambda_t       = Z[t] @ W_t
        preds_stock[t] = Lambda_t @ f_next

    # ------------------------------------------------------------------
    # 11. PERFORMANCE EVALUATION
    # ------------------------------------------------------------------
    valid  = ~np.isnan(preds_stock).any(axis=1)
    y_true = rets_stock[valid]          # (T_oos, N)
    y_pred = preds_stock[valid]         # (T_oos, N)
    mc_oos = mktcap_arr[valid]          # (T_oos, N) — already lagged

    # ── MSE / R² (equal-weighted, standard) ───────────────────────────
    mse_oos = np.mean((y_true - y_pred) ** 2)
    r2_oos  = 1.0 - (
        np.sum((y_true - y_pred) ** 2) /
        np.sum((y_true - y_true.mean()) ** 2)
    )

    # ── Equal-weighted Sharpe (same formula as run_ipca_grass_v2) ─────
    pos_ew   = y_pred / (np.mean(np.abs(y_pred), axis=1, keepdims=True) + 1e-12)
    port_ew  = np.mean(pos_ew * y_true, axis=1)
    sharpe_ew = (
        np.mean(port_ew) /
        (np.std(port_ew, ddof=1) + 1e-12) *
        np.sqrt(12)
    )

    # ── Market-cap-weighted Sharpe (S&P / Fama-French style) ──────────
    #
    # Step 1: portfolio weights  w_i,t = mktcap_i,t-1 / Σ mktcap_j,t-1
    #   (mktcap_arr was already lagged by 1 row above)
    mc_clipped = np.maximum(mc_oos, 0.0)           # no negative weights
    mc_sum     = mc_clipped.sum(axis=1, keepdims=True)
    w_mktcap   = mc_clipped / (mc_sum + 1e-12)     # (T_oos, N), rows sum to 1
    #
    # Step 2: position sizing — proportional to signal, normalised by
    #   the mktcap-weighted mean absolute signal (consistent with step 3)
    signal_scale = np.sum(w_mktcap * np.abs(y_pred), axis=1, keepdims=True)
    pos_mktcap   = y_pred / (signal_scale + 1e-12)  # (T_oos, N)
    #
    # Step 3: portfolio return — mktcap-weighted sum
    port_mktcap  = np.sum(w_mktcap * pos_mktcap * y_true, axis=1)  # (T_oos,)
    #
    # Step 4: annualised Sharpe (monthly data → × √12)
    sharpe_mktcap = (
        np.mean(port_mktcap) /
        (np.std(port_mktcap, ddof=1) + 1e-12) *
        np.sqrt(12)
    )

    # ------------------------------------------------------------------
    # 12. AGGREGATE GEOMETRIC DIAGNOSTICS  (identical to v2)
    # ------------------------------------------------------------------
    mean_stability = (
        float(np.mean(stability_dists)) if stability_dists else np.nan
    )

    if principal_angles_series:
        max_angles  = [a[0] for a in principal_angles_series]
        mean_angles = [float(np.mean(a)) for a in principal_angles_series]
        mean_max_principal_angle  = float(np.mean(max_angles))
        mean_mean_principal_angle = float(np.mean(mean_angles))
    else:
        mean_max_principal_angle  = np.nan
        mean_mean_principal_angle = np.nan

    if len(stability_dists) >= 2:
        accel_series        = list(np.diff(stability_dists))
        mean_geodesic_accel = float(np.mean(np.abs(accel_series)))
    else:
        accel_series        = []
        mean_geodesic_accel = np.nan

    mean_erank = float(np.mean(erank_series)) if erank_series else np.nan
    erank_collapse_frac = (
        float(np.mean(np.array(erank_series) < 0.5 * num_fact))
        if erank_series else np.nan
    )
    mean_spectral_gap = (
        float(np.mean(spectral_gap_series)) if spectral_gap_series else np.nan
    )
    mean_d_proj_norm = (
        float(np.mean(d_proj_norm_series)) if d_proj_norm_series else np.nan
    )
    mean_d_proj_seeds = (
        float(np.mean(d_proj_seeds_series)) if d_proj_seeds_series else np.nan
    )
    d_proj_seeds_flat_frac = (
        float(np.mean(
            np.array(d_proj_seeds_series) > 0.5 * np.sqrt(2 * num_fact)
        )) if d_proj_seeds_series else np.nan
    )

    # ------------------------------------------------------------------
    # 13. PRINT SUMMARY
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print("  Grassmann-IPCA Backtest Results  [mktcap-weighted Sharpe]")
        print("=" * 65)

        print("\n── Performance ──────────────────────────────────────────────")
        print(f"  OOS MSE                              : {mse_oos:.6e}")
        print(f"  OOS R²                               : {r2_oos:.4f}")
        print(f"  Sharpe (equal-weighted)              : {sharpe_ew:.3f}")
        print(f"  Sharpe (mktcap-weighted, S&P-style)  : {sharpe_mktcap:.3f}")

        print("\n── Metric 1 : Projection distance d_proj ────────────────────")
        print(f"  Mean d_proj                          : {mean_stability:.4f}"
              "  (lower = more stable)")

        print("\n── Metric 2 : Principal angles ──────────────────────────────")
        print(f"  Mean max angle   (rad)               : {mean_max_principal_angle:.4f}")
        print(f"  Mean mean angle  (rad)               : {mean_mean_principal_angle:.4f}")

        print("\n── Metric 3 : Geodesic acceleration ─────────────────────────")
        print(f"  Mean |Δd_proj|                       : {mean_geodesic_accel:.4f}")

        print("\n── Metric 4 : Effective rank erank(Σ̂_f) ────────────────────")
        print(f"  Mean erank                           : {mean_erank:.4f}"
              f"  (max = {num_fact})")
        print(f"  erank collapse fraction              : {erank_collapse_frac:.4f}")

        print("\n── Metric 5 : Spectral gap ratio ────────────────────────────")
        print(f"  Mean gap_ratio                       : {mean_spectral_gap:.4f}")

        print("\n── Metric 6 : Normalised d_proj ─────────────────────────────")
        print(f"  Random baseline                      : {d_proj_random_baseline:.4f}")
        print(f"  Mean d_proj_norm                     : {mean_d_proj_norm:.4f}")

        if compute_seed_variance:
            print("\n── Metric 7 : Between-seed d_proj ───────────────────────────")
            print(f"  Mean d_proj (warm vs random init)    : {mean_d_proj_seeds:.4f}"
                  f"  (max = {np.sqrt(2*num_fact):.4f})")
            print(f"  Flat-Hessian fraction                : {d_proj_seeds_flat_frac:.4f}")

        print("=" * 65)

    # ------------------------------------------------------------------
    # 14. RETURN RESULTS
    # ------------------------------------------------------------------
    return {
        # ── Performance ───────────────────────────────────────────────
        "r2_oos"         : r2_oos,
        "sharpe"         : sharpe_ew,       # kept for backward compat with aggregate()
        "sharpe_ew"      : sharpe_ew,
        "sharpe_mktcap"  : sharpe_mktcap,

        # ── Metric 1 ──────────────────────────────────────────────────
        "mean_subspace_stability" : mean_stability,
        "stability_dist_series"   : stability_dists,

        # ── Metric 2 ──────────────────────────────────────────────────
        "mean_max_principal_angle"  : mean_max_principal_angle,
        "mean_mean_principal_angle" : mean_mean_principal_angle,
        "principal_angles_series"   : principal_angles_series,

        # ── Metric 3 ──────────────────────────────────────────────────
        "mean_geodesic_accel"   : mean_geodesic_accel,
        "geodesic_accel_series" : accel_series,

        # ── Metric 4 ──────────────────────────────────────────────────
        "mean_erank"          : mean_erank,
        "erank_series"        : erank_series,
        "erank_collapse_frac" : erank_collapse_frac,

        # ── Metric 5 ──────────────────────────────────────────────────
        "mean_spectral_gap"    : mean_spectral_gap,
        "spectral_gap_series"  : spectral_gap_series,

        # ── Metric 6 ──────────────────────────────────────────────────
        "mean_d_proj_norm"        : mean_d_proj_norm,
        "d_proj_norm_series"      : d_proj_norm_series,
        "d_proj_random_baseline"  : d_proj_random_baseline,

        # ── Metric 7 ──────────────────────────────────────────────────
        "mean_d_proj_seeds"       : mean_d_proj_seeds,
        "d_proj_seeds_series"     : d_proj_seeds_series,
        "d_proj_seeds_flat_frac"  : d_proj_seeds_flat_frac,
    }

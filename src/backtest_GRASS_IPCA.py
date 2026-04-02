"""
backtest_IPCA_Grass.py
======================
Rolling-window out-of-sample backtest for the Grassmann-IPCA model,
with a suite of geometric diagnostics that operationalise the
"Virtue of Complexity" framework (Deep, Lesniewski, Missaoui & Pakala, 2026).

Overview
--------
At each time step t the estimator:
  1. Fits W_t ∈ Gr(m, k) on the trailing `window_len` observations.
  2. Computes predicted returns as  r̂_t = (Z_t W_t) f̄,
     where f̄ is the mean in-window factor return.
  3. Records a battery of geometric metrics (see "Geometric Diagnostics"
     section below) to distinguish *virtuous* from *illusory* complexity.

Geometric Diagnostics
---------------------
The paper proves (Theorem 6.1) that illusory complexity manifests as:
  (a) Flat Hessian curvature on Gr(m, k),
  (b) Subspace instability across resamples, and
  (c) Effective-rank collapse of the fitted factor covariance.

The metrics here measure all three manifestations:

  METRIC 1 — Projection distance (d_proj)  [existing]
      ‖P_t − P_{t-1}‖_F,  P = W Wᵀ
      Low → stable subspace; high → factor space rotating each window.

  METRIC 2 — Principal (canonical) angles  [NEW — Metric 1 from suggestions]
      θ₁ ≥ … ≥ θ_k are the angles between V_t and V_{t-1}.
      They strictly generalise d_proj (which equals 2 Σ sin²(θᵢ) / 2).
      Tracking the *max* angle reveals which factor direction is drifting;
      tracking the *mean* gives an average-case stability number.

  METRIC 3 — Geodesic acceleration  [NEW — Metric 2 from suggestions]
      Δd_t = d_proj_t − d_proj_{t-1}
      A model with virtuous complexity should have *decreasing* d_proj as
      more data arrives.  Persistent large |Δd| flags the optimizer
      wandering on the flat curvature directions proved in Theorem 6.1.

  METRIC 4 — Effective rank erank(Σ̂_f)  [existing]
      erank = exp(H(p)),  p_i = λ_i / tr(Σ̂_f)
      erank ≪ k → only a few factors active; rest fitting noise.
      erank → k → all k factors genuinely identified.

  METRIC 5 — Spectral gap ratio  [NEW — Metric 4 from suggestions]
      gap_ratio = (λ_k − λ_{k+1}) / λ_1
      Directly measures the signal/noise separation predicted by the
      block-spectral structure of Theorem 6.1 (Eq. 26-27).
      Large gap → clean identification; near zero → last factor sits on
      the noise floor and the subspace is underdetermined.

  METRIC 6 — Normalised d_proj  [NEW]
      d_proj_norm = d_proj / d_proj_random_baseline
      where d_proj_random_baseline = sqrt(2k(m-k)/(m+1)) is the expected
      distance between two Haar-uniform random points on Gr(m,k).
      Removes the ambient-dimension effect so d_proj is comparable across
      different P values.
      0 → perfectly stable, 1 → as unstable as two independent random draws.

  METRIC 7 — Between-seed d_proj  [NEW — direct Theorem 6.1(i) test]
      Fit a second solution W_t_rand from a fresh random W0 and compute
      d_proj between warm-start solution W_t and W_t_rand.
      Directly tests Theorem 6.1(i): flat Hessian → O(1) rotations →
      large d_proj_seeds. Curved Hessian → both inits converge → small
      d_proj_seeds. Independent of rolling-window design.

Dependencies
------------
  numpy, GrassmannManifoldIPCAEstimator (from IPCA_Grass_estimator.py)

Usage
-----
  Call `run_ipca_grass_v2(df_trunc, char_feats, ...)` and inspect the
  returned dict for all metrics and their per-window time series.
"""

import numpy as np

# ---------------------------------------------------------------------------
# The GrassmannManifoldIPCAEstimator must be importable from the same package.
# Adjust the import path to match your project layout.
# ---------------------------------------------------------------------------
try:
    from IPCA_Grass_estimator import GrassmannManifoldIPCAEstimator
except ModuleNotFoundError:
    from src.IPCA_Grass_estimator import GrassmannManifoldIPCAEstimator


# ===========================================================================
#  HELPER FUNCTIONS
# ===========================================================================

def _principal_angles(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    """
    Compute the principal (canonical) angles between two subspaces
    represented by matrices with orthonormal columns.

    The principal angles θ₁ ≥ θ₂ ≥ … ≥ θ_k ∈ [0, π/2] are defined via
    the singular values σᵢ of W_aᵀ W_b:

        cos(θᵢ) = σᵢ

    They satisfy the identity:

        ‖P_a − P_b‖²_F = 2 Σᵢ sin²(θᵢ)

    so they strictly generalise the projection-distance metric.

    Parameters
    ----------
    W_a, W_b : ndarray of shape (m, k)
        Orthonormal bases for two k-dimensional subspaces of R^m.

    Returns
    -------
    angles : ndarray of shape (k,)
        Principal angles in *radians*, sorted descending (largest first).
        angle ≈ 0   → directions are (nearly) identical.
        angle ≈ π/2 → directions are orthogonal (completely unrelated).
    """
    # Gram matrix between the two bases: k × k
    M = W_a.T @ W_b

    # Singular values are cosines of the principal angles
    sv = np.linalg.svd(M, compute_uv=False)

    # Numerical noise can push sv slightly outside [-1, 1]
    sv = np.clip(sv, -1.0, 1.0)

    angles = np.arccos(sv)          # convert to radians
    return np.sort(angles)[::-1]    # largest angle first


def _spectral_gap_ratio(eigvals: np.ndarray, num_fact: int) -> float:
    """
    Compute the normalised spectral gap between the k-th and (k+1)-th
    eigenvalues of the fitted factor covariance Σ̂_f.

    From Theorem 6.1 (Deep et al., 2026), the true Σ*_f has block-spectral
    structure: the top k₀ eigenvalues are the signal eigenvalues
    λ₁, …, λ_{k₀}, while the remaining k − k₀ eigenvalues equal the noise
    floor σ².  Consequently:

        gap = λ_k − λ_{k+1}  (here λ are Σ̂_f eigenvalues, sorted desc.)

    is *large* when all k factors are signal, and *near zero* when the last
    factor is sitting on the noise floor.

    We normalise by the leading eigenvalue to get a dimensionless ratio in
    [0, 1]:

        gap_ratio = gap / λ_1

    Parameters
    ----------
    eigvals : ndarray
        Eigenvalues of Σ̂_f (any order; function will sort them).
    num_fact : int
        Nominal number of factors k.

    Returns
    -------
    gap_ratio : float
        ∈ [0, 1].  1.0 means perfect signal/noise separation;
        0.0 means the k-th factor is indistinguishable from noise.
    """
    # Sort eigenvalues descending
    lam = np.sort(eigvals)[::-1]

    leading = lam[0] if lam[0] > 0 else 1.0  # avoid division by zero

    if len(lam) <= num_fact:
        # Σ̂_f is k × k, so there is no (k+1)-th eigenvalue.
        # Return the ratio of the k-th eigenvalue to the leading one as a
        # proxy: this still collapses toward zero when the last factor is weak.
        gap = lam[-1]
    else:
        gap = lam[num_fact - 1] - lam[num_fact]

    return float(np.clip(gap / leading, 0.0, 1.0))


# ===========================================================================
#  MAIN BACKTEST FUNCTION
# ===========================================================================

def run_ipca_grass_v2(
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
    Rolling-window out-of-sample backtest for Grassmann-IPCA with a full
    suite of geometric complexity diagnostics.

    Data preparation
    ----------------
    The function builds balanced panels from a multi-index DataFrame,
    drops assets with too many missing / zero values, burns in until
    a minimum coverage threshold is met, then imputes and standardises
    the remaining NaNs.

    Rolling estimation
    ------------------
    For each time step t > window_len:
      • Fit GrassmannManifoldIPCAEstimator on the trailing `window_len`
        rows to obtain W_t ∈ Gr(m, k) and in-window factor series f̂.
      • Predict r̂_t = (Z_t W_t) mean(f̂).

    Geometric diagnostics
    ---------------------
    At each window the following are computed and accumulated:

      d_proj         — projection distance ‖P_t − P_{t-1}‖_F
      principal_angles  — k canonical angles between V_t and V_{t-1}
      geodesic_accel — first difference of d_proj series (acceleration)
      erank          — effective rank of Σ̂_f = f̂ᵀf̂ / T
      spectral_gap   — normalised λ_k / λ_1 gap (signal/noise separation)

    Performance metrics
    -------------------
      OOS R²   — 1 − Σ(r − r̂)² / Σ(r − r̄)²
      Sharpe   — annualised market-timing Sharpe ratio

    Parameters
    ----------
    df_trunc : pd.DataFrame
        Multi-index (date, permno) DataFrame containing:
          - 'ret'        : stock return (already lagged / aligned)
          - *char_feats* : characteristic columns
    char_feats : list[str]
        Names of characteristic columns to use as instruments Z.
    num_fact : int, optional
        Number of latent factors k.  Default 2.
    window_len : int, optional
        Rolling estimation window in periods.  Default 12.
    max_nan : float, optional
        Maximum fraction of NaN returns (or char values) allowed per
        asset; assets above this threshold are dropped.  Default 0.3.
    max_zero : float, optional
        Maximum fraction of exact-zero returns allowed per asset.
        Default 0.3.
    min_non_nan_frac : float, optional
        Minimum fraction of assets with non-NaN values required for a
        time period to be kept (burn-in filter).  Default 0.7.

    Returns
    -------
    results : dict with keys:

        Performance
        ~~~~~~~~~~~
        r2_oos               : float  — OOS R²
        sharpe               : float  — annualised Sharpe ratio

        Subspace stability (Metric 1 / original)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_subspace_stability : float
            Mean d_proj = mean ‖P_t − P_{t-1}‖_F across windows.
        stability_dist_series   : list[float]
            Per-window d_proj values.

        Principal angles (Metric 2 — NEW)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_max_principal_angle  : float
            Mean of the *largest* principal angle per window (radians).
            This is the worst-case direction drift.
        mean_mean_principal_angle : float
            Mean of the *average* principal angle per window (radians).
        principal_angles_series   : list[ndarray]
            Per-window arrays of shape (k,) holding all k angles.

        Geodesic acceleration (Metric 3 — NEW)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_geodesic_accel       : float
            Mean |Δd_proj| across consecutive windows.  Near-zero means
            the subspace is converging; large means persistent wandering.
        geodesic_accel_series     : list[float]
            Per-window Δd_proj values (length = len(stability_dists) − 1).

        Effective rank (Metric 4 / original)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_erank             : float  — mean erank(Σ̂_f)
        erank_series           : list[float]
        erank_collapse_frac    : float  — fraction of windows erank < k/2

        Spectral gap (Metric 5 — NEW)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_spectral_gap      : float  — mean gap_ratio across windows
        spectral_gap_series    : list[float]
            Per-window gap_ratio values ∈ [0, 1].

        Normalised d_proj (Metric 6 — NEW)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_d_proj_norm       : float
            Mean of d_proj / d_proj_random_baseline across windows.
            Baseline = sqrt(2k(m-k)/(m+1)), the expected d_proj between
            two Haar-random points on Gr(m,k). Values near 1 indicate
            the subspace moves as much as random; near 0 indicates
            genuine stability independent of ambient dimension.
        d_proj_norm_series     : list[float]
            Per-window normalised d_proj values.
        d_proj_random_baseline : float
            The Haar-random expected d_proj = sqrt(2k(m-k)/(m+1)).

        Between-seed d_proj (Metric 7 — NEW)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_d_proj_seeds      : float
            Mean d_proj between warm-start solution and a second
            independent random-init solution, computed per window.
            Large → flat Hessian (different inits land far apart).
            Small → curved Hessian (both inits find the same basin).
            This is the direct empirical test of Theorem 6.1(i).
        d_proj_seeds_series    : list[float]
            Per-window between-seed d_proj values.
            Only populated when compute_seed_variance=True.
    """

    # ------------------------------------------------------------------
    # 1. BUILD PANELS
    # ------------------------------------------------------------------
    df_panel = df_trunc.copy().sort_index()

    # Returns panel: shape (T, N)
    rets_panel = df_panel["ret"].unstack("permno").sort_index()

    # Characteristic panels: list of (T, N) DataFrames, one per feature.
    # Returns are already aligned/shifted when df_trunc was constructed,
    # so no additional lag is applied here.
    X_list = []
    for col in char_feats:
        x = df_panel[col].unstack("permno").sort_index()
        X_list.append(x)

    # ------------------------------------------------------------------
    # 2. ALIGN ASSETS ACROSS ALL PANELS
    # ------------------------------------------------------------------
    # Only keep assets present in *every* panel (returns + all chars).
    common_assets = rets_panel.columns
    for x in X_list:
        common_assets = common_assets.intersection(x.columns)

    rets_panel = rets_panel[common_assets]
    X_list     = [x[common_assets] for x in X_list]

    # ------------------------------------------------------------------
    # 3. DROP ASSETS WITH TOO MANY MISSING / ZERO VALUES
    # ------------------------------------------------------------------
    zero_frac_rets = (rets_panel == 0).mean()
    nan_frac_rets  = rets_panel.isna().mean()

    good_assets = (nan_frac_rets <= max_nan) & (zero_frac_rets <= max_zero)
    for x in X_list:
        good_assets &= (x.isna().mean()  <= max_nan)
        good_assets &= ((x == 0).mean() <= max_zero)

    rets_panel = rets_panel.loc[:, good_assets]
    X_list     = [x.loc[:, good_assets] for x in X_list]

    # ------------------------------------------------------------------
    # 4. DROP EARLY ROWS WITH HEAVY NaNs (burn-in)
    # ------------------------------------------------------------------
    # Require at least `min_non_nan_frac` fraction of assets to have
    # non-NaN values in both returns and every characteristic.
    min_non_nan = int(min_non_nan_frac * rets_panel.shape[1])

    valid_time = rets_panel.notna().sum(axis=1) >= min_non_nan
    for x in X_list:
        valid_time &= (x.notna().sum(axis=1) >= min_non_nan)

    rets_panel = rets_panel.loc[valid_time]
    X_list     = [x.loc[valid_time] for x in X_list]

    # ------------------------------------------------------------------
    # 5. IMPUTE REMAINING NaNs WITH CROSS-SECTIONAL MEAN
    # ------------------------------------------------------------------
    rets_panel = rets_panel.apply(lambda s: s.fillna(s.mean()), axis=1)
    X_list     = [x.apply(lambda s: s.fillna(s.mean()), axis=1)
                  for x in X_list]

    # ------------------------------------------------------------------
    # 6. STANDARDISE CHARACTERISTICS CROSS-SECTIONALLY
    # ------------------------------------------------------------------
    # For each time period, demean and scale each characteristic to unit
    # cross-sectional standard deviation.  This puts all instruments on
    # a comparable scale and stabilises the Grassmann optimisation.
    X_list = [
        (x.sub(x.mean(axis=1), axis=0))
         .div(x.std(axis=1).replace(0, np.nan), axis=0)
        for x in X_list
    ]
    X_list = [x.fillna(0.0) for x in X_list]

    # ------------------------------------------------------------------
    # 7. STACK INTO NUMPY TENSORS
    # ------------------------------------------------------------------
    # Z : (T, N, m)  — characteristic tensor
    # rets_stock : (T, N)  — return matrix
    Z          = np.stack([x.to_numpy(dtype=float) for x in X_list], axis=2)
    rets_stock = rets_panel.to_numpy(dtype=float)

    T, N = rets_stock.shape
    m_dim = Z.shape[2]   # ambient dimension of Gr(m,k)
    preds_stock = np.full_like(rets_stock, np.nan, dtype=float)

    # Haar-random baseline for normalised d_proj (Metric 6).
    # Two points drawn from the uniform (Haar) measure on Gr(m,k) have:
    #   E[d_proj^2] = 2k(m-k)/(m+1)
    # This is derived from E[sin^2(theta_i)] = (m-k)/(m+1) for each
    # principal angle under the Haar measure.
    d_proj_random_baseline = float(
        np.sqrt(2 * num_fact * (m_dim - num_fact) / (m_dim + 1))
    )

    # ------------------------------------------------------------------
    # 8. ACCUMULATOR INITIALISATION
    # ------------------------------------------------------------------

    # ── Original metrics (from run_ipca_grass_v1) ─────────────────────
    W_prev          = None   # W from previous window (for stability)
    stability_dists = []     # d_proj = ‖P_t − P_{t-1}‖_F per window
    erank_series    = []     # erank(Σ̂_f) per window

    # ── NEW Metric 2: Principal angles ────────────────────────────────
    # At each window we store the full array of k principal angles.
    # From these we derive per-window max and mean angles as summaries.
    principal_angles_series   = []   # list of ndarray shape (k,)

    # ── NEW Metric 3: Geodesic acceleration ───────────────────────────
    # Computed post-loop as the first difference of stability_dists.
    # (No per-window accumulator needed; derived from stability_dists.)

    # ── NEW Metric 5: Spectral gap ────────────────────────────────────
    spectral_gap_series = []   # gap_ratio ∈ [0, 1] per window


    # ── NEW Metric 6: Normalised d_proj ──────────────────────────
    d_proj_norm_series  = []

    # ── NEW Metric 7: Between-seed d_proj ────────────────────────
    d_proj_seeds_series = []
    # ------------------------------------------------------------------
    # 9. ROLLING WINDOW LOOP
    # ------------------------------------------------------------------
    for t in range(window_len, T):

        # ── Extract training window ────────────────────────────────────
        rets_tr = rets_stock[t - window_len:t]   # (window_len, N)
        Z_tr    = Z[t - window_len:t]            # (window_len, N, m)

        # ── Instantiate and fit the Grassmann-IPCA estimator ──────────
        est = GrassmannManifoldIPCAEstimator(
            num_assets  = N,
            num_fact    = num_fact,
            num_charact = Z.shape[2],
            win_len     = rets_tr.shape[0],
            shrinkage   = shrinkage,
        )

        # W_t, f_tr, _ = est.fit(
        #     data           = [rets_tr, Z_tr],
        #     optimizer      = "ConjugateGradient",
        #     max_iterations = 50, # reduce for faster backtest; increase for more accurate convergence diagnostics
        #     verbosity      = 0,
        #     log_verbosity  = 0,
        # )
        W_t, f_tr, _ = est.fit(
            data              = [rets_tr, Z_tr],
            optimizer         = "ConjugateGradient",
            max_iterations    = 200,
            min_gradient_norm = min_gradient_norm,
            verbosity         = 0,
            log_verbosity     = 0,
            initial_point     = W_prev if W_prev is not None else None, # warm-start from previous window's solution to compare d_proj better
        )
        # W_t  : (m, k) orthonormal — a point on Gr(m, k)
        # f_tr : (window_len, k) — in-window factor returns

        # ==============================================================
        # METRIC 1 (original): Projection distance d_proj
        # --------------------------------------------------------------
        # The projection onto the k-dim subspace V_t is P_t = W_t W_tᵀ.
        # d_proj = ‖P_t − P_{t-1}‖_F measures how far the subspace has
        # rotated since the last window.  Because W has orthonormal
        # columns (Stiefel representative), P = W Wᵀ directly.
        #
        # Interpretation:
        #   Low  → stable factor space  → virtuous complexity signal
        #   High → rotating noise dirs  → illusory complexity signal
        # ==============================================================
        if W_prev is not None:
            # ==========================================================
            # METRIC 2: Principal angles between V_t and V_{t-1}
            # ----------------------------------------------------------
            # Computed FIRST because d_proj is derived from angles:
            #   d_proj² = 2 Σᵢ sin²(θᵢ)
            # This avoids forming two (P × P) projection matrices,
            # reducing O(P²k) to O(Pk² + k³).  For P=4096, k=24 this
            # is ~170× fewer FLOPs.
            # ==========================================================
            angles = _principal_angles(W_t, W_prev)   # (k,) descending, O(Pk² + k³)
            principal_angles_series.append(angles)

            # METRIC 1: Projection distance — derived from principal angles
            d_proj = float(np.sqrt(2.0 * np.sum(np.sin(angles) ** 2)))
            stability_dists.append(d_proj)

            # ==============================================================
            # METRIC 6 (NEW): Normalised d_proj
            # --------------------------------------------------------------
            # d_proj_norm = d_proj / d_proj_random_baseline
            # Baseline = sqrt(2k(m-k)/(m+1)) is the expected d_proj between
            # two Haar-uniform random points on Gr(m,k).
            #
            # Removing the ambient-dimension effect makes d_proj comparable
            # across different P (RFF feature) values. Without this, d_proj
            # grows toward sqrt(2k) purely because m=P is large, even when
            # the warm-started solution is genuinely stable.
            #
            # Interpretation:
            #   near 0 → genuine subspace stability (warm-start barely moved)
            #   near 1 → as unstable as independent random draws
            #   >> 1   → numerical instability
            # ==============================================================
            d_proj_norm = d_proj / d_proj_random_baseline if d_proj_random_baseline > 0 else np.nan
            d_proj_norm_series.append(float(d_proj_norm))

        # ==============================================================
        # METRIC 7 (NEW): Between-seed d_proj
        # --------------------------------------------------------------
        # Fit a second solution from a fresh random initialisation
        # (no warm-start) and compute d_proj between this random solution
        # and the warm-started solution W_t.
        #
        # This directly operationalises Theorem 6.1(i):
        #   Flat Hessian (k > k0, z=0): the loss landscape has no curvature
        #   along noise directions, so a random init lands in a completely
        #   different basin → large d_proj_seeds (near sqrt(2k)).
        #
        #   Curved Hessian (z=10, k~k0): regularisation creates a well-
        #   defined basin; random and warm inits both find it → small
        #   d_proj_seeds.
        #
        # Unlike Metric 1, this test is INDEPENDENT of the rolling-window
        # design: each window is self-contained and the comparison is
        # between two solutions on the same data with different starts.
        # ==============================================================
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
                initial_point     = None,   # fresh random init — intentional
            )
            # d_proj between warm-start and random-init via k×k Gram
            # matrix — avoids forming two (P × P) projections.
            # Identity: ‖P_a − P_b‖²_F = 2(k − ‖W_aᵀ W_b‖²_F)
            M_seeds = W_t.T @ W_t_rand        # (k, k)  O(Pk²)
            d_seeds = float(np.sqrt(max(0.0, 2.0 * (num_fact - np.sum(M_seeds ** 2)))))
            d_proj_seeds_series.append(d_seeds)

        W_prev = W_t

        # ==============================================================
        # METRIC 4 (original): Effective rank of Σ̂_f
        # --------------------------------------------------------------
        # Σ̂_f = (1/T) f̂ᵀ f̂   is the sample factor covariance.
        # erank(Σ̂_f) = exp(H(p)),  where p_i = λ_i / tr(Σ̂_f).
        #
        # By Theorem 6.1, when k > k₀ (true factors), Σ*_f has k − k₀
        # eigenvalues equal to the noise floor σ².  Those weak components
        # contribute little to erank, so erank collapses toward k₀.
        #
        # Interpretation:
        #   erank → k   → all k factors genuinely identified (virtuous)
        #   erank ≪ k   → effective dimension collapsed (illusory)
        # ==============================================================
        Sigma_f = (f_tr.T @ f_tr) / f_tr.shape[0]   # (k, k)
        eigvals  = np.linalg.eigvalsh(Sigma_f)        # ascending
        eigvals  = np.maximum(eigvals, 0.0)            # clip numerical noise

        total = eigvals.sum()
        if total > 0:
            p     = eigvals / total
            p     = p[p > 0]                           # avoid log(0)
            erank = np.exp(-np.sum(p * np.log(p)))     # exp(Shannon entropy)
        else:
            erank = 1.0

        erank_series.append(erank)

        # ==============================================================
        # METRIC 5 (NEW): Spectral gap ratio
        # --------------------------------------------------------------
        # gap_ratio = (λ_k − λ_{k+1}) / λ_1
        # where λᵢ are the eigenvalues of Σ̂_f in descending order.
        #
        # From Theorem 6.1, the true Σ*_f has block-spectral structure:
        # the top k₀ eigenvalues are signal eigenvalues λ₁, …, λ_{k₀};
        # the rest equal the noise floor σ².  The gap at position k
        # therefore measures the signal/noise separation directly.
        #
        # When Σ̂_f is k × k (as here), there is no explicit (k+1)-th
        # eigenvalue, so we use the k-th eigenvalue itself as the gap
        # proxy — it collapses toward σ² when the last factor is noise.
        #
        # Interpretation:
        #   gap_ratio → 1   → clean signal/noise separation (virtuous)
        #   gap_ratio → 0   → last factor sitting on noise floor
        # ==============================================================
        gap_ratio = _spectral_gap_ratio(eigvals, num_fact)
        spectral_gap_series.append(gap_ratio)

        # ── Forecast next-period return ────────────────────────────────
        # Simple predictor: project characteristics through the fitted
        # loading matrix W_t, then scale by the mean in-window factor.
        f_next       = f_tr.mean(axis=0)     # (k,)  historical mean factor
        Lambda_t     = Z[t] @ W_t            # (N, k) loadings at time t
        preds_stock[t] = Lambda_t @ f_next   # (N,)  predicted returns

    # ------------------------------------------------------------------
    # 10. PERFORMANCE EVALUATION
    # ------------------------------------------------------------------
    valid  = ~np.isnan(preds_stock).any(axis=1)
    y_true = rets_stock[valid]
    y_pred = preds_stock[valid]

    mse_oos = np.mean((y_true - y_pred) ** 2)
    r2_oos  = 1.0 - (
        np.sum((y_true - y_pred) ** 2) /
        np.sum((y_true - y_true.mean()) ** 2)
    )

    # Market-timing Sharpe (annualised assuming monthly data → × √12).
    # Position size is proportional to the predicted return, rescaled by
    # the mean absolute predicted return to avoid size effects.
    pos      = y_pred / (np.mean(np.abs(y_pred), axis=1, keepdims=True) + 1e-12)
    port_ret = np.mean(pos * y_true, axis=1)
    sharpe   = (
        np.mean(port_ret) /
        (np.std(port_ret, ddof=1) + 1e-12) *
        np.sqrt(12)
    )

    # ------------------------------------------------------------------
    # 11. AGGREGATE GEOMETRIC DIAGNOSTICS
    # ------------------------------------------------------------------

    # ── Metric 1 aggregate: mean projection distance ───────────────────
    mean_stability = (
        float(np.mean(stability_dists)) if stability_dists else np.nan
    )

    # ── Metric 2 aggregate: principal angle summaries ─────────────────
    if principal_angles_series:
        # Max angle per window (worst-case direction drift)
        max_angles  = [a[0] for a in principal_angles_series]
        # Mean angle per window (average-case drift)
        mean_angles = [float(np.mean(a)) for a in principal_angles_series]

        mean_max_principal_angle  = float(np.mean(max_angles))
        mean_mean_principal_angle = float(np.mean(mean_angles))
    else:
        mean_max_principal_angle  = np.nan
        mean_mean_principal_angle = np.nan

    # ── Metric 3 aggregate: geodesic acceleration ─────────────────────
    # The acceleration is the first-difference of the d_proj series.
    # A converging model has acceleration → 0 (or negative on average).
    if len(stability_dists) >= 2:
        accel_series = list(np.diff(stability_dists))   # Δd_t
        mean_geodesic_accel = float(np.mean(np.abs(accel_series)))
    else:
        accel_series        = []
        mean_geodesic_accel = np.nan

    # ── Metric 4 aggregate: effective rank ────────────────────────────
    mean_erank = float(np.mean(erank_series)) if erank_series else np.nan

    # Fraction of windows where erank fell below half the nominal k.
    # Flags windows where the optimizer collapsed into the noise eigenspace.
    erank_collapse_frac = (
        float(np.mean(np.array(erank_series) < 0.5 * num_fact))
        if erank_series else np.nan
    )

    # ── Metric 5 aggregate: spectral gap ──────────────────────────────
    mean_spectral_gap = (
        float(np.mean(spectral_gap_series)) if spectral_gap_series else np.nan
    )

    # ── Metric 6 aggregate: normalised d_proj ─────────────────────────
    mean_d_proj_norm = (
        float(np.mean(d_proj_norm_series)) if d_proj_norm_series else np.nan
    )

    # ── Metric 7 aggregate: between-seed d_proj ───────────────────────
    mean_d_proj_seeds = (
        float(np.mean(d_proj_seeds_series)) if d_proj_seeds_series else np.nan
    )
    # Fraction of windows where seeds landed far apart (> 50% of max).
    # High fraction → Hessian is flat in many windows → illusory regime.
    d_proj_seeds_flat_frac = (
        float(np.mean(
            np.array(d_proj_seeds_series) > 0.5 * np.sqrt(2 * num_fact)
        )) if d_proj_seeds_series else np.nan
    )

    # ------------------------------------------------------------------
    # 12. PRINT SUMMARY
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 65)
        print("  Grassmann-IPCA Backtest Results")
        print("=" * 65)

        print("\n── Performance ──────────────────────────────────────────────")
        print(f"  OOS MSE                              : {mse_oos:.6e}")
        print(f"  OOS R²                               : {r2_oos:.4f}")
        print(f"  Market-timing Sharpe (annualised)    : {sharpe:.3f}")

        print("\n── Metric 1 : Projection distance d_proj ────────────────────")
        print(f"  Mean d_proj                          : {mean_stability:.4f}"
              "  (lower = more stable)")

        print("\n── Metric 2 : Principal angles (NEW) ────────────────────────")
        print(f"  Mean max angle   (rad)               : {mean_max_principal_angle:.4f}"
              "  (worst-case dir drift; 0 = stable)")
        print(f"  Mean mean angle  (rad)               : {mean_mean_principal_angle:.4f}"
              "  (avg dir drift; 0 = stable)")

        print("\n── Metric 3 : Geodesic acceleration (NEW) ───────────────────")
        print(f"  Mean |Δd_proj|                       : {mean_geodesic_accel:.4f}"
              "  (near 0 = converging subspace)")

        print("\n── Metric 4 : Effective rank erank(Σ̂_f) ────────────────────")
        print(f"  Mean erank                           : {mean_erank:.4f}"
              f"  (max = {num_fact})")
        print(f"  erank collapse fraction              : {erank_collapse_frac:.4f}"
              "  (fraction of windows erank < k/2)")

        print("\n── Metric 5 : Spectral gap ratio (NEW) ──────────────────────")
        print(f"  Mean gap_ratio                       : {mean_spectral_gap:.4f}"
              "  (1 = clean signal/noise sep; 0 = last factor on noise floor)")

        print("\n── Metric 6 : Normalised d_proj (NEW) ───────────────────────")
        print(f"  Random baseline                      : {d_proj_random_baseline:.4f}"
              f"  (expected d_proj for 2 Haar-random pts on Gr({m_dim},{num_fact}))")
        print(f"  Mean d_proj_norm                     : {mean_d_proj_norm:.4f}"
              "  (0=stable, 1=as unstable as random)")

        if compute_seed_variance:
            print("\n── Metric 7 : Between-seed d_proj (NEW) ─────────────────────")
            print(f"  Mean d_proj (warm vs random init)    : {mean_d_proj_seeds:.4f}"
                  f"  (max = {np.sqrt(2*num_fact):.4f})")
            print(f"  Flat-Hessian fraction                : {d_proj_seeds_flat_frac:.4f}"
                  "  (frac windows where seeds landed > 50pct of max apart)")

        print("=" * 65)

    # ------------------------------------------------------------------
    # 13. RETURN FULL RESULTS DICT
    # ------------------------------------------------------------------
    return {
        # ── Performance ───────────────────────────────────────────────
        "r2_oos"  : r2_oos,
        "sharpe"  : sharpe,

        # ── Metric 1: Projection distance (original) ──────────────────
        "mean_subspace_stability" : mean_stability,
        "stability_dist_series"   : stability_dists,

        # ── Metric 2: Principal angles (NEW) ──────────────────────────
        "mean_max_principal_angle"  : mean_max_principal_angle,
        "mean_mean_principal_angle" : mean_mean_principal_angle,
        "principal_angles_series"   : principal_angles_series,

        # ── Metric 3: Geodesic acceleration (NEW) ─────────────────────
        "mean_geodesic_accel"   : mean_geodesic_accel,
        "geodesic_accel_series" : accel_series,

        # ── Metric 4: Effective rank (original) ───────────────────────
        "mean_erank"          : mean_erank,
        "erank_series"        : erank_series,
        "erank_collapse_frac" : erank_collapse_frac,

        # ── Metric 5: Spectral gap ratio (NEW) ────────────────────────
        "mean_spectral_gap"    : mean_spectral_gap,
        "spectral_gap_series"  : spectral_gap_series,

        # ── Metric 6: Normalised d_proj (NEW) ─────────────────────────
        # d_proj / sqrt(2k(m-k)/(m+1)) removes ambient-dimension effect.
        # Enables valid cross-P comparisons of subspace stability.
        "mean_d_proj_norm"        : mean_d_proj_norm,
        "d_proj_norm_series"      : d_proj_norm_series,
        "d_proj_random_baseline"  : d_proj_random_baseline,

        # ── Metric 7: Between-seed d_proj (NEW) ───────────────────────
        # d_proj between warm-start and fresh-random solution per window.
        # Direct test of Theorem 6.1(i): flat Hessian → large d_proj_seeds.
        "mean_d_proj_seeds"       : mean_d_proj_seeds,
        "d_proj_seeds_series"     : d_proj_seeds_series,
        "d_proj_seeds_flat_frac"  : d_proj_seeds_flat_frac,
    }
"""
generate_synthetic_panel.py
===========================
Synthetic panel generator for validating geometric diagnostics against
known-true-k₀ factor structure.  Produces a DataFrame compatible with
the `run_ipca_grass_v2` rolling backtest pipeline.

Design Philosophy
-----------------
This generator serves as a SANITY CHECK, not a proof.  The goal is to verify
that under the population assumptions of Theorem 6.1, the geometric diagnostics
(spectral gap, erank, d_proj, anisotropy) correctly identify the true signal
rank and reproduce the predicted curvature-chain effects.  The fact that the
real-data findings (Section 9) match these controlled predictions — despite
the many ways real returns violate the stylized assumptions — provides
evidence that the geometric framework captures genuine structure rather than
artifacts of model misspecification.

Key Design Choices
------------------
1. EIGENVALUE STRUCTURE (two-parameter decay model):
   λ_j = snr_base × decay^(j-1) + σ²

   Instead of hand-picking k₀ eigenvalues to produce a specific spectral gap
   shape, we use a single geometric decay with two free parameters (snr_base
   and eigenvalue_decay).  This is honest: we control the signal-to-noise
   ratio and the decay rate, but the resulting spectral gap cliff, erank
   profile, and ridge sensitivity all emerge endogenously rather than being
   engineered.

2. SUBSPACE DRIFT (calibrated from real data):
   W*(t) follows a slow random walk on Gr(m, k₀).  Rather than picking
   w_drift_scale to produce visually appealing d_proj plots, we calibrate
   it from the real data's observed d_proj at a reference configuration
   (k=12, z=50, P=1024 → d_proj ≈ 0.6).  This ensures the synthetic
   subspace dynamics live in the same regime as the empirical ones.

3. STANDARD / UNCONTROVERSIAL CHOICES:
   - AR(1) factor returns (standard in IPCA simulation literature)
   - Toeplitz characteristic correlation (captures "nearby characteristics
     are more similar" without imposing specific factor structure)
   - Cross-sectional heteroskedasticity via log-normal idiosyncratic vol
   - No missingness, heavy tails, or intercept (these are nuisances that
     complicate interpretation without testing new theory)

4. WHAT THIS DOES NOT TEST:
   The real-data finding that peak Sharpe sits PAST the spectral gap cliff
   ("productive illusion" at k=20-24) likely reflects genuine economic
   complexity — nonlinearity, time-varying factor counts, structural breaks
   — that a clean k₀-factor linear model cannot produce.  If the synthetic
   experiment does not reproduce this, that is itself informative: it
   suggests the productive illusion is not an artifact of the diagnostics
   but a real feature of returns data.

Data Generating Process
-----------------------
    r_{i,t} = (Z_{i,t} @ W*(t)) @ f_t + ε_{i,t}

where:
    Z_{i,t} ∈ R^m        ~ N(0, Σ_z)          cross-sectionally correlated characteristics
    W*(t)   ∈ Gr(m, k₀)                        slow random walk on Grassmann manifold
    f_t     ∈ R^{k₀}     ~ AR(1)               factor returns with structured covariance
    ε_{i,t} ∈ R           ~ N(0, σ_i²)         heteroskedastic idiosyncratic noise

The population return covariance conditional on Z_t has eigenvalues
{λ_1, ..., λ_{k₀}, σ², ..., σ²}, producing the block spectral structure
required by Theorem 6.1.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _grassmann_step(W, delta_scale, rng):
    """
    Take a small random step on Gr(m, k) starting from W (m×k, orthonormal cols).
    Uses QR retraction of W + delta_scale × horizontal_tangent.

    The tangent vector is projected to the horizontal space (orthogonal to the
    current subspace) before being added.  This ensures the perturbation only
    rotates the subspace rather than scaling or deforming it.
    """
    m, k = W.shape
    Z = rng.normal(size=(m, k)) * delta_scale
    Z = Z - W @ (W.T @ Z)          # project to horizontal space
    W_new, _ = np.linalg.qr(W + Z)
    return W_new[:, :k]


def _calibrate_drift_scale(k0, window_len, target_dproj):
    """
    Back out w_drift_scale from a target d_proj value.

    Rough model: over a window of length T, the subspace accumulates
    T independent Grassmann steps.  Each step of size δ contributes
    roughly δ to each of the k₀ principal angles, so:

        d_proj ≈ sqrt(2 × k₀ × T) × δ

    Inverting: δ = target_dproj / sqrt(2 × k₀ × T)
    """
    return target_dproj / np.sqrt(2 * k0 * window_len)


# ──────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_panel(
    T: int = 360,               # total months (e.g. 30 years)
    N: int = 50,                # number of assets
    m: int = 30,                # number of raw characteristics
    k0: int = 12,               # TRUE number of signal factors
    seed: int = 42,

    # ── Signal structure (two-parameter decay model) ─────────────
    # λ_j = snr_base × decay^(j-1) + σ²
    # This controls the eigenvalue profile without hand-picking each one.
    snr_base: float = 5.0,          # λ_1 / σ² excess ratio (how far above noise)
    eigenvalue_decay: float = 0.82, # geometric decay rate per factor
    sigma_eps: float = 1.0,         # idiosyncratic noise std (σ)

    # ── Factor dynamics ──────────────────────────────────────────
    factor_ar1: float = 0.5,    # AR(1) persistence of factor returns
    factor_vol: float = 0.3,    # innovation vol of factor returns

    # ── Subspace dynamics ────────────────────────────────────────
    # If w_drift_scale is None, it is calibrated from target_dproj.
    # target_dproj is the d_proj observed at (k=k₀, z=50, P=1024)
    # in the real-data backtest (≈ 0.6 from mapping_report_v3).
    w_drift_scale: float | None = None,
    target_dproj: float = 0.6,      # calibration target from real data
    calibration_window: int = 24,   # window length used in calibration

    # ── Characteristic structure ─────────────────────────────────
    z_corr: float = 0.3,        # Toeplitz correlation for characteristics
    z_scale: float = 1.0,       # overall characteristic scale

    # ── Cross-sectional heteroskedasticity ───────────────────────
    hetero_strength: float = 0.3,   # log-normal spread of σ_i
):
    """
    Generate a synthetic IPCA panel with known true factor rank k₀.

    Parameters
    ----------
    T : int
        Number of time periods (months).
    N : int
        Number of assets.
    m : int
        Number of raw characteristics.  Must be ≥ k₀.
    k0 : int
        True number of signal factors.
    seed : int
        Random seed for reproducibility.
    snr_base : float
        Signal-to-noise ratio for the strongest factor.
        λ_1 = snr_base + σ².  Default 5.0 gives λ_1 = 6.0 at σ²=1.
    eigenvalue_decay : float
        Geometric decay rate for signal eigenvalues.
        λ_j = snr_base × decay^(j-1) + σ².
        With defaults (5.0, 0.82, k₀=12):
          λ_1 = 6.00, λ_6 = 2.87, λ_12 = 1.60
        The last eigenvalue sits 60% above the noise floor — enough to be
        detectable but marginal enough to create an ambiguous boundary.
    sigma_eps : float
        Idiosyncratic noise std.  The noise floor is σ² = sigma_eps².
    factor_ar1 : float
        AR(1) persistence for factor returns.
    factor_vol : float
        Innovation volatility for factor returns.
    w_drift_scale : float or None
        Per-period Grassmann step size for the loading matrix W*(t).
        If None, calibrated automatically from target_dproj.
    target_dproj : float
        Target d_proj for calibrating w_drift_scale.
        Default 0.6 comes from real-data backtest at (k=12, z=50, P=1024).
    z_corr : float
        Toeplitz correlation parameter for characteristics.
    z_scale : float
        Overall scale of characteristics.
    hetero_strength : float
        Strength of cross-sectional heteroskedasticity in [0, ~2].
        0 = homoskedastic.  0.3 = mild (realistic).

    Returns
    -------
    df : pd.DataFrame
        MultiIndex (date, permno) with columns:
        - 'ret' : stock return
        - 'Price' : dummy (always 1.0, required by backtest pipeline)
        - char_1, ..., char_m : raw characteristics
    truth : dict
        Ground-truth information:
        - true_k0, signal_eigenvalues, sigma_eps, W_star_series, etc.
    char_cols : list[str]
        Names of characteristic columns.
    """
    rng = np.random.default_rng(seed)
    assert m >= k0, f"Need m >= k0 ({m} < {k0})"

    # ── Signal eigenvalue structure (two-parameter decay) ────────
    # λ_j = snr_base × decay^(j-1) + σ²
    sigma_sq = sigma_eps ** 2
    signal_eigenvalues = np.array([
        snr_base * (eigenvalue_decay ** j) + sigma_sq
        for j in range(k0)
    ])

    # Factor return covariance structure: diag(sqrt(λ - σ²))
    # The excess eigenvalues above noise drive the signal strength
    excess = np.maximum(signal_eigenvalues - sigma_sq, 1e-8)
    L_f = np.diag(np.sqrt(excess))  # (k0, k0)

    # ── Subspace drift calibration ───────────────────────────────
    if w_drift_scale is None:
        w_drift_scale = _calibrate_drift_scale(
            k0, calibration_window, target_dproj
        )

    # ── Characteristic covariance (Toeplitz) ─────────────────────
    idx = np.arange(m)
    Sigma_z = z_corr ** np.abs(idx[:, None] - idx[None, :])
    Lz = np.linalg.cholesky(Sigma_z + 1e-10 * np.eye(m))

    # ── True loading matrix W*(t): random walk on Gr(m, k₀) ─────
    A0 = rng.normal(size=(m, k0))
    W_star_0, _ = np.linalg.qr(A0)
    W_star_0 = W_star_0[:, :k0]

    W_star_series = [W_star_0]
    W_t = W_star_0.copy()
    for t in range(1, T):
        W_t = _grassmann_step(W_t, w_drift_scale, rng)
        W_star_series.append(W_t.copy())

    # ── Cross-sectional heteroskedasticity ───────────────────────
    u = rng.normal(size=N)
    sigma_i = sigma_eps * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 0.1, np.percentile(sigma_i, 99))

    # ── Generate panel ───────────────────────────────────────────
    dates = pd.date_range("1994-01-01", periods=T, freq="MS")
    permnos = np.arange(10001, 10001 + N)
    char_cols = [f"char_{j+1}" for j in range(m)]

    rows = []
    f_prev = np.zeros(k0)
    all_f = np.zeros((T, k0))

    for t in range(T):
        W_t = W_star_series[t]  # (m, k0)

        # Factor returns: AR(1) with structured covariance
        innov = L_f @ rng.normal(size=k0) * factor_vol
        f_t = factor_ar1 * f_prev + innov
        f_prev = f_t
        all_f[t] = f_t

        # Characteristics: Z_{i,t} ~ N(0, Sigma_z)
        Z_t = (rng.normal(size=(N, m)) @ Lz.T) * z_scale  # (N, m)

        # Loadings and signal
        Lambda_t = Z_t @ W_t          # (N, k0)
        signal = Lambda_t @ f_t       # (N,)

        # Idiosyncratic noise
        eps = rng.normal(size=N) * sigma_i

        # Returns
        ret_t = signal + eps

        for i in range(N):
            row = {
                "date": dates[t],
                "permno": permnos[i],
                "ret": ret_t[i],
                "Price": 1.0,
            }
            for j in range(m):
                row[char_cols[j]] = Z_t[i, j]
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "permno"]).sort_index()

    truth = {
        "true_k0": k0,
        "signal_eigenvalues": signal_eigenvalues,
        "snr_base": snr_base,
        "eigenvalue_decay": eigenvalue_decay,
        "sigma_eps": sigma_eps,
        "w_drift_scale": w_drift_scale,
        "W_star_series": W_star_series,
        "factor_returns": all_f,
        "sigma_i": sigma_i,
        "params": {
            "T": T, "N": N, "m": m, "k0": k0,
            "snr_base": snr_base,
            "eigenvalue_decay": eigenvalue_decay,
            "factor_ar1": factor_ar1, "factor_vol": factor_vol,
            "w_drift_scale": w_drift_scale,
            "target_dproj": target_dproj,
            "z_corr": z_corr, "z_scale": z_scale,
            "sigma_eps": sigma_eps, "hetero_strength": hetero_strength,
        },
    }

    return df, truth, char_cols
"""
generate_synthetic_panel.py
===========================
Synthetic panel generator for validating geometric diagnostics against
known-true-k₀ factor structure.  Produces a DataFrame compatible with
the `run_ipca_grass_v2` rolling backtest pipeline.

Add this function to IPCA_Grass_estimator.py (or import from here).

Design principles
-----------------
1. Known true signal rank k₀ with well-separated eigenvalues.
2. Time-varying factor loadings via slow Grassmann random walk of W*(t).
3. Controlled SNR so the spectral gap cliff at k₀ is visible.
4. Output is a (date, permno)-MultiIndex DataFrame identical in format
   to the real-data panel used in experimentation_IPCA_rff.ipynb.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm


def _random_skew(n, k, rng):
    """Random skew-symmetric matrix for Grassmann tangent perturbation."""
    A = rng.normal(size=(n, k))
    # Project to horizontal space: (I - WW^T) A  — but we want a general
    # tangent direction, so just use a fresh random matrix and let the
    # caller handle the retraction.
    return A


def _grassmann_step(W, delta_scale, rng):
    """
    Take a small random step on Gr(m, k) starting from W (m×k, orthonormal cols).
    Uses QR retraction of W + delta_scale * random_tangent.
    """
    m, k = W.shape
    # Random tangent: project out the current subspace
    Z = rng.normal(size=(m, k)) * delta_scale
    Z = Z - W @ (W.T @ Z)  # horizontal lift
    W_new, _ = np.linalg.qr(W + Z)
    return W_new[:, :k]


def generate_synthetic_panel(
    T: int = 360,           # total months (e.g. 30 years)
    N: int = 50,            # number of assets
    m: int = 20,            # number of raw characteristics
    k0: int = 5,            # TRUE number of signal factors
    seed: int = 42,

    # Signal structure
    signal_eigenvalues: np.ndarray | None = None,
    # If None, defaults to geometrically decaying: [5, 3, 2, 1.5, 1] (length k0)
    sigma_eps: float = 1.0,   # idiosyncratic noise std

    # Factor dynamics
    factor_ar1: float = 0.5,   # AR(1) persistence of factor returns
    factor_vol: float = 0.3,   # innovation vol of factor returns

    # Subspace dynamics (how fast W* rotates)
    w_drift_scale: float = 0.02,  # per-period Grassmann step size
    # 0 = static W*, 0.02 = slow drift, 0.1 = fast drift

    # Characteristic structure
    z_corr: float = 0.3,    # Toeplitz correlation for characteristics
    z_scale: float = 1.0,

    # Cross-sectional heteroskedasticity
    hetero_strength: float = 0.3,
):
    """
    Generate a synthetic IPCA panel with known true factor rank k₀.

    The data generating process is:
        r_{i,t} = β_{i,t}' f_t + ε_{i,t}
        β_{i,t} = Z_{i,t} @ Γ_t'     (Γ_t on Grassmann, k₀ × m)

    where Γ_t follows a slow random walk on Gr(m, k₀).

    The signal eigenvalue structure of Σ_f is controlled so that:
    - The top k₀ eigenvalues are well above the noise floor σ²
    - This produces a clear spectral gap cliff when k > k₀

    Returns
    -------
    df : pd.DataFrame
        MultiIndex (date, permno) with columns:
        - 'ret' : stock return
        - 'Price' : dummy (always 1.0)
        - char_1, char_2, ..., char_m : raw characteristics
    truth : dict
        Contains W_star_series, true_k0, signal_eigenvalues, etc.
    char_cols : list[str]
        Names of characteristic columns.
    """
    rng = np.random.default_rng(seed)

    # ── Signal eigenvalue structure ──────────────────────────────────
    if signal_eigenvalues is None:
        # Geometrically decaying: ensures clear separation from noise
        signal_eigenvalues = np.array([5.0 * (0.7 ** i) for i in range(k0)])
    else:
        signal_eigenvalues = np.asarray(signal_eigenvalues)
        assert len(signal_eigenvalues) == k0

    # Factor return covariance: Σ_f = diag(signal_eigenvalues)
    # The factor returns f_t ~ AR(1) with this variance structure
    L_f = np.diag(np.sqrt(signal_eigenvalues))  # (k0, k0)

    # ── Characteristic covariance (Toeplitz) ─────────────────────────
    idx = np.arange(m)
    Sigma_z = z_corr ** np.abs(idx[:, None] - idx[None, :])
    Lz = np.linalg.cholesky(Sigma_z + 1e-10 * np.eye(m))

    # ── True loading matrix W*(t): slow random walk on Gr(m, k₀) ────
    A0 = rng.normal(size=(m, k0))
    W_star_0, _ = np.linalg.qr(A0)
    W_star_0 = W_star_0[:, :k0]

    W_star_series = [W_star_0]
    W_t = W_star_0.copy()
    for t in range(1, T):
        W_t = _grassmann_step(W_t, w_drift_scale, rng)
        W_star_series.append(W_t.copy())

    # ── Cross-sectional heteroskedasticity ───────────────────────────
    u = rng.normal(size=N)
    sigma_i = sigma_eps * np.exp(hetero_strength * u)
    sigma_i = np.clip(sigma_i, 0.1, np.percentile(sigma_i, 99))

    # ── Generate panel ───────────────────────────────────────────────
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
        "sigma_eps": sigma_eps,
        "W_star_series": W_star_series,
        "factor_returns": all_f,
        "sigma_i": sigma_i,
        "params": {
            "T": T, "N": N, "m": m, "k0": k0,
            "factor_ar1": factor_ar1, "factor_vol": factor_vol,
            "w_drift_scale": w_drift_scale,
            "z_corr": z_corr, "z_scale": z_scale,
            "sigma_eps": sigma_eps, "hetero_strength": hetero_strength,
        },
    }

    return df, truth, char_cols

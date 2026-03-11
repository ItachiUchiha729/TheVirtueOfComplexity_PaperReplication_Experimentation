"""
Backtesting framework for a simple IPCA-style model on factor/predictor features.

Interpretation (time-series setting):
- You already have a feature matrix X_t (your "factors" / RFF features / predictors).
- In each rolling window, we estimate K latent factors via PCA on X_train
  (low-rank factor structure), then regress y_train on these factors.
- Forecast is produced from the test observation's projected latent factors.

We also store geometry metrics per window:
- spectrum of X'X, effective rank, stable rank, explained variance
- coefficient concentration (HHI), beta spread, L2 norm, L1/L2 ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score

from src.config import ANNUALIZATION_FACTOR, DEFAULT_TRAIN_WINDOW


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > 0 else 0.0


def effective_rank_from_eigs(eigs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective rank = exp(H), where H is Shannon entropy of normalized eigenvalues.
    Uses nonnegative eigenvalues.
    """
    eigs = np.asarray(eigs, dtype=float)
    eigs = np.clip(eigs, 0.0, None)
    s = eigs.sum()
    if s <= eps:
        return 0.0
    p = eigs / s
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def stable_rank_from_svals(svals: np.ndarray, eps: float = 1e-12) -> float:
    """
    Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s_i^2) / max(s_i^2)
    Here we apply to X_train (singular values).
    """
    svals = np.asarray(svals, dtype=float)
    if svals.size == 0:
        return 0.0
    denom = float(np.max(svals) ** 2)
    if denom <= eps:
        return 0.0
    return float(np.sum(svals**2) / denom)


def herfindahl_index(w: np.ndarray, eps: float = 1e-12) -> float:
    """
    HHI on absolute weights: sum_i (|w_i| / sum|w|)^2.
    Higher => more concentrated.
    """
    w = np.asarray(w, dtype=float)
    a = np.abs(w)
    s = a.sum()
    if s <= eps:
        return 0.0
    p = a / s
    return float(np.sum(p**2))


@dataclass
class IPCAConfig:
    n_components: int = 5              # K latent factors
    use_ridge_on_factors: bool = True  # ridge on factor regression
    ridge_lambda: float = 1.0          # ridge strength on factor regression (not scaled by T)
    standardize_features: bool = True  # z-score X in each window before PCA


class BacktestIPCA:
    """
    Rolling-window backtesting for PCA-based latent factor regression (IPCA-style on factor set).

    Inputs:
      - features: shape (n_samples, n_features) => your factor/predictor matrix X_t
      - returns:  shape (n_samples,)          => target return y_t aligned with X_t

    Output results per t:
      - forecast
      - timing_return = forecast * realized_return (same as your ridge BT)
      - plus geometry metrics for X_train and implied betas in original feature space
    """

    def __init__(
        self,
        train_window: int = DEFAULT_TRAIN_WINDOW,
        cfg: IPCAConfig = IPCAConfig(),
        dtype: Type[np.floating[Any]] = np.float32,
        random_state: int = 0,
    ) -> None:
        self.train_window = train_window
        self.cfg = cfg
        self.dtype = dtype
        self.random_state = random_state

        self.n_features: Optional[int] = None
        self.complexity_ratio: Optional[float] = None
        self.backtest_results: Optional[pd.DataFrame] = None
        self.prediction: Optional[pd.Series] = None
        self.performance_metrics: Optional[Dict[str, float]] = None

    def predict(
        self,
        features: np.ndarray | pd.DataFrame,
        returns: np.ndarray | pd.Series,
    ) -> "BacktestIPCA":

        # Convert to arrays + index
        if isinstance(features, pd.DataFrame):
            X = features.values
            x_index = features.index
            feature_names = list(features.columns)
        else:
            X = np.asarray(features)
            x_index = None
            feature_names = None

        if isinstance(returns, pd.Series):
            y = returns.values
            y_index = returns.index
        else:
            y = np.asarray(returns)
            y_index = x_index

        n_samples, self.n_features = X.shape
        self.complexity_ratio = self.n_features / self.train_window

        if y.shape[0] != n_samples:
            raise ValueError(f"features has {n_samples} rows but returns has {y.shape[0]}")

        K = int(self.cfg.n_components)
        if K <= 0:
            raise ValueError("n_components must be >= 1")

        results = []

        for t in range(self.train_window, n_samples):
            X_train = X[t - self.train_window : t].astype(self.dtype, copy=False)
            y_train = y[t - self.train_window : t].astype(self.dtype, copy=False)

            X_test = X[t : t + 1].astype(self.dtype, copy=False)
            y_test = float(y[t])

            # Standardize X per window (common for PCA / factor extraction)
            if self.cfg.standardize_features:
                mu = X_train.mean(axis=0, keepdims=True)
                sd = X_train.std(axis=0, keepdims=True)
                sd = np.where(sd > 0, sd, 1.0)
                Xtr = (X_train - mu) / sd
                Xte = (X_test - mu) / sd
            else:
                mu = None
                sd = None
                Xtr = X_train
                Xte = X_test

            # PCA factor extraction on training window
            # Use full_svd for stability on small T
            pca = PCA(
                n_components=min(K, Xtr.shape[1], Xtr.shape[0]),
                svd_solver="full",
                random_state=self.random_state,
            )
            F_train = pca.fit_transform(Xtr)  # (T, K)
            F_test = pca.transform(Xte)       # (1, K)

            # Regression on latent factors
            if self.cfg.use_ridge_on_factors:
                # Here ridge_lambda is on factor regression directly; no paper-scaling needed
                reg = Ridge(alpha=float(self.cfg.ridge_lambda), fit_intercept=False, solver="svd")
            else:
                reg = LinearRegression(fit_intercept=False)

            reg.fit(F_train, y_train)
            forecast = reg.predict(F_test).reshape(-1)[0].item()
            b = np.asarray(reg.coef_).reshape(-1)  # keep for w computation below

            timing_return = forecast * y_test

            # Map factor regression back to implied weights in original feature space:
            # y_hat = (X_std @ V_K) b = X_std @ (V_K b) => w_std = V_K b
            # If standardized, implied w in original scale: w = w_std / sd
            V = pca.components_.T  # (P, K) : columns are loading directions
            w_std = V @ b          # (P,)
            if self.cfg.standardize_features:
                w = (w_std / sd.reshape(-1)).astype(float)
            else:
                w = w_std.astype(float)

            # Geometry diagnostics on X_train
            # PCA gives explained_variance_ which are eigenvalues of covariance of Xtr
            eigs = np.asarray(pca.explained_variance_, dtype=float)  # length K
            evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
            svals = np.asarray(pca.singular_values_, dtype=float)

            eff_rank = effective_rank_from_eigs(eigs)
            stab_rank = stable_rank_from_svals(svals)

            beta_norm = float(np.sqrt(np.sum(w**2)))
            beta_l1 = float(np.sum(np.abs(w)))
            beta_spread = float(_safe_std(w))
            beta_hhi = herfindahl_index(w)

            # “Effective complexity” style scalars (all heuristic but useful)
            # - L1/L2 ratio (proxy for concentration; lower => concentrated)
            l1_l2 = float(beta_l1 / (beta_norm + 1e-12))
            # - fraction variance captured by top component(s)
            top1_evr = float(evr[0]) if evr.size > 0 else 0.0
            topk_evr = float(np.sum(evr)) if evr.size > 0 else 0.0

            obs_index = y_index[t] if y_index is not None else t

            row = {
                "index": obs_index,
                "forecast": forecast,
                "timing_return": timing_return,
                "market_return": y_test,

                # Model/geometry (time-varying)
                "beta_norm": beta_norm,
                "beta_l1": beta_l1,
                "beta_l1_l2": l1_l2,
                "beta_hhi": beta_hhi,
                "beta_spread": beta_spread,

                "eff_rank_cov": eff_rank,
                "stable_rank_X": stab_rank,
                "top1_evr": top1_evr,
                "topk_evr": topk_evr,
                "n_components_used": int(eigs.size),
            }

            # Optional: store a few spectrum points for plotting (keep it light)
            for i in range(min(5, eigs.size)):
                row[f"eig_{i+1}"] = float(eigs[i])
                row[f"evr_{i+1}"] = float(evr[i])

            results.append(row)

        self.backtest_results = pd.DataFrame(results).set_index("index")
        self.prediction = self.backtest_results["forecast"]
        return self

    def calc_performance(self, annualization_factor: int = ANNUALIZATION_FACTOR) -> Dict[str, float]:
        if self.backtest_results is None:
            raise RuntimeError("Must call predict() before calc_performance()")

        data = self.backtest_results.dropna()

        # CAPM alpha/beta vs market_return like your ridge BT
        market_model = LinearRegression().fit(
            data[["market_return"]],
            data["timing_return"],
        )
        strategy_beta = float(market_model.coef_[0])
        strategy_alpha = float(market_model.intercept_)

        sqrt_factor = np.sqrt(annualization_factor)

        timing_mean = float(data["timing_return"].mean() * annualization_factor)
        timing_std = float(data["timing_return"].std() * sqrt_factor)

        market_mean = float(data["market_return"].mean() * annualization_factor)
        market_std = float(data["market_return"].std() * sqrt_factor)

        actual_direction = data["market_return"] > 0
        predicted_direction = data["forecast"] > 0

        self.performance_metrics = {
            # Performance
            "Market Sharpe Ratio": (market_mean / market_std) if market_std > 0 else np.nan,
            "Expected Return": timing_mean,
            "Volatility": timing_std,
            "R2": r2_score(data["market_return"], data["forecast"]),
            "SR": (timing_mean / timing_std) if timing_std > 0 else np.nan,
            "IR": ((timing_mean - market_mean * strategy_beta) / timing_std) if timing_std > 0 else np.nan,
            "Alpha": strategy_alpha,
            "Precision": precision_score(actual_direction, predicted_direction),
            "Recall": recall_score(actual_direction, predicted_direction),
            "Accuracy": accuracy_score(actual_direction, predicted_direction),

            # Geometry summaries (window-averaged)
            "beta_norm_mean": float(data["beta_norm"].mean()),
            "beta_hhi_mean": float(data["beta_hhi"].mean()),
            "beta_spread_mean": float(data["beta_spread"].mean()),
            "eff_rank_cov_mean": float(data["eff_rank_cov"].mean()),
            "stable_rank_X_mean": float(data["stable_rank_X"].mean()),
            "top1_evr_mean": float(data["top1_evr"].mean()),
            "topk_evr_mean": float(data["topk_evr"].mean()),
        }
        return self.performance_metrics
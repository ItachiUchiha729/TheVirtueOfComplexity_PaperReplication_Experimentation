from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd


@dataclass
class TrueIPCAConfig:
    n_factors: int = 5                 # K
    n_als_iters: int = 10              # ALS iterations per window
    ridge_gamma: float = 1e-6          # L2 on Gamma update
    ridge_factor: float = 1e-6         # L2 on factor (f_t) cross-sec solve
    standardize_chars: bool = True     # z-score characteristics within window
    factor_forecast: Literal["last", "ar1"] = "ar1"
    portfolio: Literal["ls_rank"] = "ls_rank"
    ls_frac: float = 0.2              # top/bottom quantile for long-short


def _ridge_solve(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Solve (A^T A + lam I) x = A^T b."""
    k = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ b)


def _fit_ar1(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit AR(1) per factor: f_t = a + phi f_{t-1} + e_t
    Returns (a, phi) arrays length K.
    """
    # F shape (T, K)
    Ft = F[1:, :]
    Fm1 = F[:-1, :]
    # add intercept
    X = np.column_stack([np.ones(Fm1.shape[0]), Fm1])  # (T-1, 1+K) but we fit per factor
    a = np.zeros(F.shape[1])
    phi = np.zeros(F.shape[1])
    for k in range(F.shape[1]):
        y = Ft[:, k]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]  # [a, phi]
        a[k] = beta[0]
        phi[k] = beta[1]
    return a, phi


def _make_ls_weights(scores: pd.Series, frac: float) -> pd.Series:
    """Long top frac, short bottom frac, equal-weight within buckets, dollar-neutral."""
    n = len(scores)
    if n < 5:
        # too few names, return zeros
        return pd.Series(0.0, index=scores.index)

    scores = scores.dropna()
    n = len(scores)
    if n == 0:
        return pd.Series(0.0, index=scores.index)

    q = max(1, int(np.floor(frac * n)))
    ranked = scores.sort_values()
    short_idx = ranked.index[:q]
    long_idx = ranked.index[-q:]

    w = pd.Series(0.0, index=scores.index)
    w.loc[long_idx] = 1.0 / q
    w.loc[short_idx] = -1.0 / q
    return w


class BacktestTrueIPCA:
    """
    True IPCA rolling backtest (panel setting).

    Model:
      r_{i,t} = x_{i,t}^T Gamma f_t + e_{i,t}

    ALS:
      - Given Gamma: for each t, solve f_t by ridge regression across assets.
      - Given {f_t}: solve Gamma by ridge regression across all (i,t).

    OOS forecast:
      - Forecast f_t on test date using last factors or AR(1) on training factors.
      - Predict rhat_{i,t} = x_{i,t}^T Gamma fhat_t.
      - Form LS portfolio on rhat and realize next return using actual r_{i,t}.
    """

    def __init__(self, train_window: int, cfg: TrueIPCAConfig = TrueIPCAConfig()):
        self.train_window = int(train_window)
        self.cfg = cfg

        self.backtest_results: Optional[pd.DataFrame] = None
        self.gamma_: Optional[np.ndarray] = None  # (P, K)

    def _als_fit(self, X: np.ndarray, y: np.ndarray, T: int, N: int, P: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (T, N, P)
        y: (T, N)
        Returns Gamma (P,K), F (T,K)
        """
        rng = np.random.default_rng(0)
        Gamma = rng.normal(scale=0.01, size=(P, K))

        for _ in range(self.cfg.n_als_iters):
            # Step 1: given Gamma, solve f_t for each t
            F = np.zeros((T, K))
            for t in range(T):
                Xt = X[t]  # (N,P)
                Zt = Xt @ Gamma  # (N,K)
                F[t] = _ridge_solve(Zt, y[t], self.cfg.ridge_factor)

            # Step 2: given F, solve Gamma
            # Stack across (t,i): y_{t,i} ≈ (x_{t,i}^T Gamma) f_t = (f_t ⊗ x_{t,i})^T vec(Gamma)
            # Build design matrix A with rows: kron(f_t, x_{t,i}) of length (K*P)
            A = np.zeros((T * N, K * P))
            b = y.reshape(T * N)
            row = 0
            for t in range(T):
                ft = F[t]  # (K,)
                # kron(ft, x) gives (K*P,)
                A[row:row + N, :] = np.kron(ft, X[t])  # (N, K*P)
                row += N

            vecG = _ridge_solve(A, b, self.cfg.ridge_gamma)  # (K*P,)
            Gamma = vecG.reshape(K, P).T  # back to (P,K)

        return Gamma, F

    def predict(self, features: pd.DataFrame, returns: pd.Series) -> "BacktestTrueIPCA":
        if not isinstance(features.index, pd.MultiIndex) or not isinstance(returns.index, pd.MultiIndex):
            raise ValueError("features and returns must have MultiIndex (date, asset).")

        if features.index.names[0] is None:
            # not required, but helps readability
            pass

        # Align
        common_index = features.index.intersection(returns.index)
        Xdf = features.loc[common_index].copy()
        yser = returns.loc[common_index].copy()

        # Dates
        dates = Xdf.index.get_level_values(0).unique()
        dates = dates.sort_values()

        P = Xdf.shape[1]
        K = int(self.cfg.n_factors)

        results = []

        for t_idx in range(self.train_window, len(dates)):
            test_date = dates[t_idx]
            train_dates = dates[t_idx - self.train_window:t_idx]

            X_train_df = Xdf.loc[train_dates]
            y_train_ser = yser.loc[train_dates]

            # panel -> (T, N, P)
            # We need a consistent asset set per date; simplest is inner-join assets across train+test
            train_assets = X_train_df.index.get_level_values(1)
            assets_common = train_assets.unique()

            # Build tensors date-by-date
            X_list = []
            y_list = []
            for d in train_dates:
                Xd = X_train_df.xs(d, level=0).reindex(assets_common)
                yd = y_train_ser.xs(d, level=0).reindex(assets_common)
                X_list.append(Xd.values.astype(float))
                y_list.append(yd.values.astype(float))

            X_train = np.stack(X_list, axis=0)  # (T,N,P)
            y_train = np.stack(y_list, axis=0)  # (T,N)

            # Standardize characteristics within window (optional)
            if self.cfg.standardize_chars:
                mu = X_train.reshape(-1, P).mean(axis=0, keepdims=True)
                sd = X_train.reshape(-1, P).std(axis=0, keepdims=True)
                sd = np.where(sd > 0, sd, 1.0)
                X_train = (X_train - mu) / sd
            else:
                mu = None
                sd = None

            Tn, N, Pn = X_train.shape
            assert Pn == P

            # Fit true IPCA on training window
            Gamma, F = self._als_fit(X_train, y_train, T=Tn, N=N, P=P, K=K)
            self.gamma_ = Gamma

            # Forecast factors for test date
            if self.cfg.factor_forecast == "last":
                f_hat = F[-1].copy()
            else:  # "ar1"
                if F.shape[0] < 3:
                    f_hat = F[-1].copy()
                else:
                    a, phi = _fit_ar1(F)
                    f_hat = a + phi * F[-1]

            # Build test X (same assets_common intersected with test date assets)
            X_test_df = Xdf.xs(test_date, level=0).reindex(assets_common)
            y_test_df = yser.xs(test_date, level=0).reindex(assets_common)

            X_test = X_test_df.values.astype(float)
            if self.cfg.standardize_chars:
                X_test = (X_test - mu) / sd

            # Predict cross-sectional expected returns
            scores = X_test @ Gamma @ f_hat  # (N,)
            scores_ser = pd.Series(scores, index=assets_common)

            # Portfolio (simple long-short by predicted score)
            if self.cfg.portfolio == "ls_rank":
                w = _make_ls_weights(scores_ser, frac=self.cfg.ls_frac)
            else:
                raise ValueError("Unknown portfolio type")

            realized = pd.Series(y_test_df.values.astype(float), index=assets_common)

            port_ret = float((w * realized).sum())

            results.append(
                {
                    "date": test_date,
                    "portfolio_return": port_ret,
                    "factor_forecast": self.cfg.factor_forecast,
                    "K": K,
                    "train_window": self.train_window,
                }
            )

        self.backtest_results = pd.DataFrame(results).set_index("date")
        return self

    def calc_performance(self, annualization_factor: int = 252) -> Dict[str, float]:
        if self.backtest_results is None or self.backtest_results.empty:
            raise RuntimeError("Run predict() first.")

        r = self.backtest_results["portfolio_return"].astype(float)
        mean_ann = float(r.mean() * annualization_factor)
        vol_ann = float(r.std() * np.sqrt(annualization_factor))
        sr = np.nan if vol_ann == 0 else mean_ann / vol_ann

        return {
            "Expected Return": mean_ann,
            "Volatility": vol_ann,
            "SR": float(sr) if np.isfinite(sr) else np.nan,
        }
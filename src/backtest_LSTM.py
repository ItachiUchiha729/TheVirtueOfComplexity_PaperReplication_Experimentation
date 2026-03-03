# backtest_lstm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score

import torch
import torch.nn as nn

from src.config import ANNUALIZATION_FACTOR, DEFAULT_TRAIN_WINDOW

@dataclass(frozen=True)
class _ArrayView:
    '''Small helper to consistently extract .values and .index from numpy/pandas inputs.'''
    values : np.ndarray
    index : Optional[pd.Index]

def _as_array_view(x : Union[pd.DataFrame, pd.Series, np.ndarray])-> _ArrayView:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return _ArrayView(values = x.values, index = x.index)
    return _ArrayView(values = x, index = None)

#LSTM bactester class
class BacktestLSTM(nn.Module):
    """
    Rolling window LSTM backtester that refits the LSTM from scratc at each step

    - Uses the same rolling training window lenght 'T' and produces  1-step forecasts.
    - Stores the same outputs in 'backtest_results' : coefficient_norm, forecast, timing_return, market_return
    - Keeps complexity_ratio = n_features /T, has another param_complexity_ratio = n_params / T(this signifies how deep the model is)
    - Sequence modeling : forecast at time t uses the last 'seq_len' feature vectors at time t.
     Within each rolling training window, we create supervised samples using sliding sequences, this updates 
     cell state and hidden state of the LSTM, this is how the LSTM learns the temporal dependencies in the data.
        """
    
    def __init__(self,
                 d_in: int,
                 hidden_size: int = 32,
                 num_layers:int = 1,
                 seq_len: int = 12,
                 dropout: float = 0.0,
                 T: int = DEFAULT_TRAIN_WINDOW,
                 epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3):
        super().__init__()

        if d_in < 1:
            raise ValueError("Input dimension must be at least 1.")
        if hidden_size < 1:
            raise ValueError("Hidden size must be at least 1.")
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        if seq_len < 1:
            raise ValueError("Sequence length must be at least 1.")
        if T < 2:
            raise ValueError("Training window length T must be at least 2.")
        if seq_len > T:
            raise ValueError("Sequence length seq_len cannot be greater than training window length T.")
        
        #in pytorch LSTM, dropput is only applied between layers and only when num_layers>1

        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size = d_in, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True, 
                            dropout = lstm_dropout) #(batch, seq, features)
        
        self.head = nn.Linear(hidden_size, 1) #output is a single value for regression
        self.seq_len = int(seq_len)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)

        #Rolling estimation window lenght
        self.train_window = int(T)

        #Complexity ratios
        self.n_features = int(d_in)
        self.complexity_ratio = self.n_features / self.train_window

        #Complexity ratios to compare the depth of neural nets
        self.n_params = int(sum(p.numel() for p in self.parameters() if p.requires_grad)) #numel is the number of elements in the tensor, this gives the total number of trainable parameters in the model
        self.param_complexity_ratio = self.n_params / self.train_window

        #Outputs 
        self.backtested_results:Optional[pd.DataFrame] = None #Optional is type hinting that this variable can be either a pd.DataFrame or None.
        self.predictions: Optional[np.Series] = None
        self.performance: Optional[float] = None
        self.performance_metrics : Optional[dict] = None

    def forward(self, x_seq : torch.Tensor) -> torch.Tensor:
        """x_seq is of shape (batch_size, seq_len, n_features),
        Returns: (batch, 1)"""

        out, _ = self.lstm(x_seq)
        last = out[:, -1, :] #get the output of the last time step, this is of shape (batch, hidden_size)

        return self.head(last) # nn.Linear will apply a linear transformation to the last output, this will give us the final prediction of shape (batch, 1)
    
    def _reset_parameters(self) -> None:
        """
        IMPORTANT : we refit from scrach at each rolling step
        Calling reset_arams() on each module returns weights to a fresh random init"""

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _build_supervised_from_window(
            self, 
            window_X: np.ndarray,
            window_y: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray  ]:
        """ 
        Create supervised samples inside ONE rolling training window.
        
        We want the model to learn : (X__t-seq_len+1, ..., X_t) -> y_t
        
        If window length is T, we can form (T - seq_len + 1) samples,"""

        T = window_X.shape[0]
        S = self.seq_len
        n = T - S + 1 #number of samples we can form

        if n < 1:
            raise ValueError(f"Sequence length seq_len={S} is too long for the training window length T={T}. Cannot form any samples.")
        
        X_seq = np.empty((n, S, window_X.shape[1]), dtype = np.float32) #shape (n_samples, seq_len, n_features)
        y_tgt = np.empty((n, ), dtype = np.float32) #shape (n_samples, )

        # sammple i corresponds to window index (i + S -1) as the precition taret time

        for i in range(n):
            X_seq[i] = window_X[i : i + S]
            y_tgt[i] = window_y[i + S - 1]

        return X_seq, y_tgt
        
    def fit(self, 
            X_seq: np.ndarray, 
            y : np.ndarray, *, 
            epochs: Optional[int] = None,
            batch_size : Optional[int] = None,
            lr: Optional[float] = None) -> "BacktestLSTM":
        """Fit the LSTM model on the given supervised samples (X_seq, y).
        X_seq is of shape (n_samples, seq_len, n_features)
        y is of shape (n_samples, )"""

        epochs = self.epochs if epochs is None else int(epochs)
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        lr = self.lr if lr is None else float(lr)
        device = self._device()
        self.to(device)

        X_tensor = torch.tensor(X_seq, dtype = torch.float32, device = device) #shape (n_samples, seq_len, n_features   
        y_tensor = torch.tensor(y, dtype = torch.float32, device = device).unsqueeze(1) #shape (n_samples, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

        opt = torch.optim.Adam(self.parameters(), lr = lr)
        loss_fn = nn.MSELoss() 
        
        for _ in range(epochs):
            self.train()
            for bx, by in loader:
                opt.zero_grad()
                pred = self(bx) #equivalent to self.forward(bx), this will give us the predictions of shape (batch_size, 1)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()

        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                y : Optional[Union[pd.DataFrame, np.ndarray]] = None) -> np.ndarray:
        #Optinal and Union are type hinting tools, Optional means the argument can be of the specified type or None, Union means the argument can be of any of the specified types.
        """ Two modes of prediction :
        1. Inference mode : predict(X) -> np.ndarray
        - Returns predictions aligned to rows of X.
        -First (seq_len - 1) predictions are Nan(insufficent history to form a sequence), then we have valid predictions for the rest of the rows.
        
        2)Backtest mode: predict(fetures, returns) -> self
        -Rolling refit from scratch each step.
        Stores bactest_results and prediction, compatible with calc_performance()"""

        X_view = _as_array_view(X)
        Xv = np.asarray(X_view.values)

        if Xv.ndim != 2:
            raise ValueError(f"Input features X must be 2-dimensional, got {Xv.ndim} dimensions.")
        
        if y is None:
            #infernece mode, no rolling dir
            n_samples, d_in = Xv.shape
            if d_in != self.n_features:
                raise ValueError(f"X has {d_in} features but model was initailsed with d_in = {self.n_features}")
        
            preds = np.full((n_samples, ), np.nan, dtype = float)

            if n_samples < self.seq_len:
                return preds
            
            device = self._device()
            self.to(device)
            self.eval()

            # Build all sequences ending at each time t >= seq_len-1
            X_seq = np.empty((n_samples - self.seq_len + 1, self.seq_len, d_in), dtype=np.float32)
            for i in range(n_samples - self.seq_len + 1):
                X_seq[i] = Xv[i : i + self.seq_len]

            X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = self(X_tensor).detach().cpu().view(-1).numpy()

            preds[self.seq_len - 1 :] = out
            return preds
        # --- backtest mode (rolling) ---
        y_view = _as_array_view(y)
        yv = np.asarray(y_view.values)

        if yv.ndim != 1:
            raise ValueError("returns must be 1D with shape (n_samples,)")

        n_samples, d_in = Xv.shape
        if len(yv) != n_samples:
            raise ValueError(f"features has {n_samples} samples but returns has {len(yv)}")

        if n_samples <= self.train_window:
            raise ValueError(
                f"Need n_samples > train_window; got n_samples={n_samples}, train_window={self.train_window}"
            )
        if self.seq_len > self.train_window:
            raise ValueError("seq_len must be <= train_window for rolling backtest.")

        # Keep the same definition as the repo for plotting VoC curves
        self.n_features = int(d_in)
        self.complexity_ratio = self.n_features / self.train_window
        self.n_params = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.param_complexity_ratio = self.n_params / self.train_window

        index = y_view.index if y_view.index is not None else X_view.index

        results = []
        prediction_indices = range(self.train_window, n_samples)

        for t in prediction_indices:
            # Window used for estimation: [t-T, t)
            wX = Xv[t - self.train_window : t].astype(np.float32)
            wy = yv[t - self.train_window : t].astype(np.float32)

            # Build supervised dataset from this window (sliding sequences)
            X_seq, y_seq = self._build_supervised_from_window(wX, wy)

            # Refit from scratch to match backtest.py methodology
            self._reset_parameters()
            self.fit(X_seq, y_seq)

            # Forecast at time t uses features ending at t: [t-seq_len+1, ..., t]
            test_seq = Xv[t - self.seq_len + 1 : t + 1].astype(np.float32)
            test_seq = test_seq.reshape(1, self.seq_len, d_in)

            device = self._device()
            self.to(device)
            self.eval()
            with torch.no_grad():
                forecast = float(self(torch.tensor(test_seq, dtype=torch.float32, device=device)).item())

            # Parameter norm analogue to kβ̂k (but for NN weights)
            coefficient_norm = float(
                np.sqrt(
                    sum(float(torch.sum(p.detach() ** 2).item()) for p in self.parameters() if p.requires_grad)
                )
            )

            realized = float(yv[t])
            timing_return = forecast * realized
            obs_index = index[t] if index is not None else t

            results.append(
                {
                    "index": obs_index,
                    "coefficient_norm": coefficient_norm,
                    "forecast": forecast,
                    "timing_return": timing_return,
                    "market_return": realized,
                }
            )

        self.backtest_results = pd.DataFrame(results).set_index("index")
        self.prediction = self.backtest_results["forecast"]
        return self
    
    def calc_performance(self, annualization_factor: int = ANNUALIZATION_FACTOR) -> dict:
        """
        Same metrics + naming as backtest.py / your BacktestNN (so plots/tables stay consistent).
        """
        if self.backtest_results is None:
            raise RuntimeError("Must call predict(features, returns) before calc_performance()")

        data = self.backtest_results.dropna()
        if data.empty:
            raise RuntimeError("No valid rows in backtest results after dropping NaNs.")

        market_model = LinearRegression().fit(data[["market_return"]], data["timing_return"])
        strategy_beta = float(market_model.coef_[0])
        strategy_alpha = float(market_model.intercept_)

        sqrt_factor = float(np.sqrt(annualization_factor))

        timing_mean = float(data["timing_return"].mean() * annualization_factor)
        timing_std = float(data["timing_return"].std() * sqrt_factor)

        market_mean = float(data["market_return"].mean() * annualization_factor)
        market_std = float(data["market_return"].std() * sqrt_factor)

        actual_direction = data["market_return"] > 0
        predicted_direction = data["forecast"] > 0

        market_sharpe = np.nan if market_std == 0 else market_mean / market_std
        strategy_sharpe = np.nan if timing_std == 0 else timing_mean / timing_std
        information_ratio = (
            np.nan
            if timing_std == 0
            else (timing_mean - market_mean * strategy_beta) / timing_std
        )

        self.performance_metrics = {
            "beta_norm_mean": float(data["coefficient_norm"].mean()),
            "Market Sharpe Ratio": float(market_sharpe) if np.isfinite(market_sharpe) else np.nan,
            "Expected Return": timing_mean,
            "Volatility": timing_std,
            "R2": float(r2_score(data["market_return"], data["forecast"])),
            "SR": float(strategy_sharpe) if np.isfinite(strategy_sharpe) else np.nan,
            "IR": float(information_ratio) if np.isfinite(information_ratio) else np.nan,
            "Alpha": strategy_alpha,
            "Precision": float(precision_score(actual_direction, predicted_direction, zero_division=0)),
            "Recall": float(recall_score(actual_direction, predicted_direction, zero_division=0)),
            "Accuracy": float(accuracy_score(actual_direction, predicted_direction)),
            # Extra fields (won't break existing code if ignored)
            "feature_complexity_ratio": float(self.complexity_ratio),
            "param_complexity_ratio": float(self.param_complexity_ratio),
            "n_params": float(self.n_params),
        }

        self.performance = self.performance_metrics["R2"]
        return self.performance_metrics

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> dict:
        """
        Convenience wrapper (mirrors BacktestNN.evaluate()).
        """
        self.predict(X, y)
        return self.calc_performance()

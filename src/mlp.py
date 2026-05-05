"""
CrossSectionalMLP — shared-weight per-asset predictor for nonlinear Grassmannian analysis.

Architecture:  f_θ : R^{m0} → R  (tanh MLP, L hidden layers of width K)
Cross-section: F_θ(Z_t)_i = f_θ(Z_{i,t})  →  r̂_t ∈ R^N

Ridge penalty on the output layer only, matching the companion paper's
regularisation strategy for RFF-IPCA.

Reference: Deep, Lesniewski, Missaoui, Pakala (2026) Section 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossSectionalMLP(nn.Module):
    """
    Parameters
    ----------
    m0 : int   per-asset characteristic dimension  (73 in the companion dataset)
    K  : int   hidden-layer width                  (complexity axis: 16/32/64/128)
    L  : int   number of hidden layers             (fixed at 2 per paper Section 2.1)
    """

    def __init__(self, m0: int, K: int, L: int = 2) -> None:
        super().__init__()
        if L < 1:
            raise ValueError("L must be >= 1")

        layers: list[nn.Module] = [nn.Linear(m0, K), nn.Tanh()]
        for _ in range(L - 1):
            layers += [nn.Linear(K, K), nn.Tanh()]
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(K, 1)   # ridge applied to this layer only

        self.m0 = m0
        self.K = K
        self.L = L

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        Z : Tensor (N, m0)  — cross-section of characteristics at time t

        Returns
        -------
        r_hat : Tensor (N,)  — predicted returns
        """
        return self.output(self.hidden(Z)).squeeze(-1)

    def forward_single(self, z: torch.Tensor) -> torch.Tensor:
        """
        Single-asset forward pass for use with torch.func.vmap.

        Parameters
        ----------
        z : Tensor (m0,)

        Returns
        -------
        r_hat_i : scalar Tensor
        """
        return self.output(self.hidden(z.unsqueeze(0))).squeeze()


def ridge_loss(
    model: CrossSectionalMLP,
    Z: torch.Tensor,
    r: torch.Tensor,
    z: float,
) -> torch.Tensor:
    """
    MSE + ridge on the output layer weights.

    L(θ) = (1/N) ||r - F_θ(Z)||² + z * ||W^{out}||²_F
    """
    r_hat = model(Z)
    mse = ((r - r_hat) ** 2).mean()
    ridge = z * (model.output.weight ** 2).sum()
    return mse + ridge


def train_mlp(
    model: CrossSectionalMLP,
    Z_train: torch.Tensor,
    r_train: torch.Tensor,
    z: float = 10.0,
    lr: float = 1e-3,
    max_epochs: int = 200,
    tol: float = 1e-5,
    patience: int = 10,
) -> list[float]:
    """
    Train model in-place via Adam with early stopping.

    Parameters
    ----------
    Z_train : (T*N, m0) or (T, N, m0) — will be flattened to (T*N, m0)
    r_train : (T*N,)   or (T, N)      — will be flattened to (T*N,)
    z       : ridge coefficient on output weights
    patience: stop if loss doesn't improve by > tol for this many epochs

    Returns
    -------
    losses : list of per-epoch training loss (float)
    """
    Z = Z_train.reshape(-1, model.m0)
    r = r_train.reshape(-1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    best_loss = float("inf")
    wait = 0

    model.train()
    for _ in range(max_epochs):
        opt.zero_grad()
        loss = ridge_loss(model, Z, r, z)
        loss.backward()
        opt.step()
        val = loss.item()
        losses.append(val)

        if best_loss - val > tol:
            best_loss = val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.eval()
    return losses

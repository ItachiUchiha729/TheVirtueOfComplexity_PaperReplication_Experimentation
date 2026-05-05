"""
Stacked-gradient matrix and Grassmannian diagnostics for nonlinear predictors.

The core object is G_θ(Z_t) ∈ Mat_{N, m0}(R), whose i-th row is ∇f_θ(Z_{i,t}).
This is NOT the full Jacobian J_Z F_θ (which degenerates to R^N for pooled architectures).

Reference: Deep, Lesniewski, Missaoui, Pakala (2026)
    Definition 3.1  — stacked-gradient matrix G_θ(Z_t)
    Definition 4.1  — local predictive subspace V_θ(Z_t) = Im(G_θ(Z_t))
    Eq (13)-(14)    — effective rank via Shannon entropy
    Eq (16)         — smoothed Gram matrix M_t  (N×N, return-space)
    Definition 8.1  — Grassmann velocity
    Section 8.1     — principal-angle anisotropy ρ_{θ,t} = θ̄ / θ_max
    Eq (7)          — geometric alpha  α^geo_{t+1} = (I - P_{V_t}) r_{t+1}
    Section 8.3     — predictive alignment η_{t+1} = ||P_{V_t} r||² / ||r||²
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mlp import CrossSectionalMLP


# ---------------------------------------------------------------------------
# Stacked-gradient matrix  G_θ(Z_t)
# ---------------------------------------------------------------------------

def stacked_gradient_matrix(
    model: nn.Module,
    Z_t: torch.Tensor,
) -> np.ndarray:
    """
    Compute G_θ(Z_t) ∈ R^{N × m0} where row i = ∇f_θ(Z_{i,t}).

    Uses torch.func.vmap + jacrev for a vectorised forward-mode Jacobian —
    equivalent to N separate backward passes but ~N× faster.

    Parameters
    ----------
    model : CrossSectionalMLP (or any nn.Module with a forward_single method)
    Z_t   : Tensor (N, m0)  — cross-section of characteristics at time t

    Returns
    -------
    G : ndarray (N, m0)
    """
    try:
        from torch.func import vmap, grad
    except ImportError:
        from functorch import vmap, grad   # older torch fallback

    model.eval()

    # We need a stateless function for vmap; extract params/buffers
    try:
        from torch.func import functional_call
    except ImportError:
        from functorch import functional_call

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def f_single(z: torch.Tensor) -> torch.Tensor:
        """scalar output for one asset"""
        return functional_call(model, (params, buffers), (z.unsqueeze(0),)).squeeze()

    grad_fn = grad(f_single)           # R^{m0} → R^{m0}
    G = vmap(grad_fn)(Z_t)             # (N, m0)
    return G.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Subspace extraction from G_θ
# ---------------------------------------------------------------------------

def grassmann_subspace(
    G: np.ndarray,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract the rank-k predictive subspace from the stacked-gradient matrix.

    Thin SVD:  G = U Σ R^T  with U ∈ R^{N×m0}
    Subspace:  V_t^(k) = span(U[:, :k]) ∈ Gr(k, N)   (paper Eq. 15)
    Effective rank via Shannon entropy of normalised squared singular values
    (paper Eq. 13-14).

    Parameters
    ----------
    G : ndarray (N, m0)
    k : int | None   if None, use full rank min(N, m0)

    Returns
    -------
    U_k   : ndarray (N, k)    orthonormal basis for V_t^(k)
    sigma : ndarray (m0,)     singular values (descending)
    erank : float             effective rank ∈ [1, m0]
    """
    U, sigma, _ = np.linalg.svd(G, full_matrices=False)   # U: (N, r), r=min(N,m0)

    # Effective rank (paper Eq. 13-14)
    s2 = sigma ** 2
    total = s2.sum()
    if total > 1e-12:
        p = s2 / total
        p = p[p > 1e-15]
        erank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        erank = 1.0

    if k is None:
        k = len(sigma)
    k = min(k, U.shape[1])

    return U[:, :k].copy(), sigma, erank


# ---------------------------------------------------------------------------
# Grassmannian distance metrics  (reuse companion paper helpers)
# ---------------------------------------------------------------------------

def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Principal angles between span(U1) and span(U2).

    θ_i = arccos(σ_i(U1^T U2)),  sorted descending.

    Parameters
    ----------
    U1, U2 : ndarray (N, k)   orthonormal bases for two k-dim subspaces

    Returns
    -------
    angles : ndarray (k,)   in radians, largest first
    """
    M = U1.T @ U2
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    angles = np.arccos(sv)
    return np.sort(angles)[::-1].copy()


def projection_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    ||P1 - P2||_F  where Pi = Ui Ui^T.

    Relation to principal angles:  d² = 2 Σ sin²(θ_i/2)
    (paper Section 4.1 in companion, Definition 4.2 here).
    """
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    return float(np.linalg.norm(P1 - P2, "fro"))


def grassmann_velocity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Geodesic distance d_G(V1, V2) = sqrt(Σ θ_i²).

    Discrete analogue of Definition 8.1 in the paper.
    """
    angles = principal_angles(U1, U2)
    return float(np.sqrt(np.sum(angles ** 2)))


def anisotropy_ratio(angles: np.ndarray) -> float:
    """
    ρ_{θ,t} = θ̄ / θ_max  (paper Section 8.1 / companion Finding 4).

    Low ρ → concentrated motion (one dominant direction, signal-driven).
    High ρ → diffuse rotation (all directions comparable, noise-driven).
    """
    theta_max = float(angles[0])
    if theta_max < 1e-12:
        return 1.0
    theta_bar = float(np.mean(angles))
    return theta_bar / theta_max


def spectral_gap(sigma: np.ndarray, k: int) -> float:
    """
    Normalised gap between the k-th and (k+1)-th singular values of G.

    gap = (σ_k − σ_{k+1}) / σ_1  ∈ [0, 1]
    """
    if len(sigma) <= k or sigma[0] < 1e-12:
        return 0.0
    if len(sigma) <= k:
        gap = sigma[k - 1]
    else:
        gap = sigma[k - 1] - sigma[k]
    return float(np.clip(gap / sigma[0], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Predictive alignment and geometric alpha  (paper Sections 6, 8.3)
# ---------------------------------------------------------------------------

def predictive_alignment(U_k: np.ndarray, r_next: np.ndarray) -> float:
    """
    η_{t+1} = ||P_{V_t} r_{t+1}||² / ||r_{t+1}||²  ∈ [0, 1]

    Fraction of cross-sectional return energy in the model's local
    predictive subspace (paper Section 8.3, using η to avoid confusion
    with geometric alpha ρ_{θ,t}).

    Parameters
    ----------
    U_k    : ndarray (N, k)  orthonormal basis for V_t^(k)
    r_next : ndarray (N,)    realised next-period cross-sectional returns
    """
    proj = U_k @ (U_k.T @ r_next)         # P_{V_t} r_{t+1}
    r_norm2 = float(np.dot(r_next, r_next))
    if r_norm2 < 1e-12:
        return 0.0
    return float(np.dot(proj, proj) / r_norm2)


def geometric_alpha(U_k: np.ndarray, r_next: np.ndarray) -> np.ndarray:
    """
    α^geo_{t+1} = (I - P_{V_t}) r_{t+1}  ∈ R^N  (paper Eq. 7)

    Component of realised return orthogonal to the predictive subspace.
    Zero iff r_{t+1} ∈ V_t^(k) (local no-residual-alpha condition).
    """
    proj = U_k @ (U_k.T @ r_next)
    return r_next - proj


# ---------------------------------------------------------------------------
# Smoothed subspace  M_t  (paper Eq. 16 / Section 7.2)
# ---------------------------------------------------------------------------

def smoothed_subspace(
    G_list: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Smoothed rank-k predictive subspace from a rolling window of G matrices.

    M_t = (1/w) Σ_{s=t-w+1}^{t} G_s G_s^T  ∈  Mat_{N,N}(R)  (paper Eq. 16)

    The N×N (return-space) Gram matrix aggregates local sensitivity over the
    window. Top-k eigenvectors form the smoothed subspace V̂_t^(k).

    Parameters
    ----------
    G_list : list of w ndarrays, each (N, m0)
    k      : effective factor dimension to retain

    Returns
    -------
    U_k   : ndarray (N, k)   orthonormal basis for V̂_t^(k)
    lam   : ndarray (N,)     eigenvalues of M_t (descending)
    erank : float            effective rank of spectrum
    """
    w = len(G_list)
    N = G_list[0].shape[0]
    M = np.zeros((N, N))
    for G in G_list:
        M += G @ G.T
    M /= w

    lam_all, U_all = np.linalg.eigh(M)            # ascending order
    idx = np.argsort(lam_all)[::-1]               # descending
    lam = lam_all[idx]
    U_all = U_all[:, idx]

    # Effective rank
    l_pos = lam[lam > 1e-12]
    total = l_pos.sum()
    if total > 0:
        p = l_pos / total
        erank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        erank = 1.0

    k = min(k, U_all.shape[1])
    return U_all[:, :k].copy(), lam, erank

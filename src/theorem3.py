"""Theorem 3: Greedy benchmark coverage and dimension bounds.

Implements:
- The greedy benchmark selection algorithm with the (1 - 1/e) Nemhauser
  guarantee (Theorem 3 in proofs/theorem3_proof.tex).
- ``coverage_function`` over arbitrary subsets.
- ``dimension_bounds`` (n_signal <= D <= n - 1) from the dimension-bounds
  theorem.
- ``uncovered_directions`` characterising the blind spot.

The coverage function is the fraction of total benchmark variance captured
by the linear span of the selected benchmark loading vectors:

    f(T) = tr(P_T @ Sigma) / tr(Sigma)

where P_T is the orthogonal projector onto span{l_j : j in T}, and
l_j = sqrt(lambda_j) v_j is the loading vector of benchmark j (when Sigma is
the correlation matrix; we accept any PSD matrix Sigma).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .theorem1 import mp_threshold
from .utils import eigendecomp


# ----------------------------------------------------------------------------
# Dimension bounds
# ----------------------------------------------------------------------------
@dataclass
class DimensionBounds:
    n_signal: int   # eigenvalues above MP threshold
    upper: int      # n - 1
    lambda_plus: float


def dimension_bounds(S: np.ndarray) -> DimensionBounds:
    """n_signal eigenvalues above MP edge <= D <= n - 1."""
    S = np.asarray(S, dtype=float)
    n, k = S.shape
    C = np.corrcoef(S, rowvar=False)
    w, _ = eigendecomp(C)
    thr = mp_threshold(n, k)
    n_sig = int((w > thr).sum())
    return DimensionBounds(n_signal=n_sig, upper=max(n - 1, 0), lambda_plus=float(thr))


# ----------------------------------------------------------------------------
# Coverage function
# ----------------------------------------------------------------------------
def coverage_function(Sigma: np.ndarray, subset: Sequence[int]) -> float:
    """f(T) = tr(P_T Sigma) / tr(Sigma).

    P_T is the orthogonal projector onto the column subspace of Sigma's
    columns indexed by T. For a correlation matrix this equals the fraction
    of total variance captured by the selected benchmarks.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    k = Sigma.shape[0]
    total = float(np.trace(Sigma))
    if total <= 0:
        return 0.0
    if not len(subset):
        return 0.0
    idx = np.asarray(list(subset), dtype=int)
    A = Sigma[:, idx]                       # k x |T|
    # Orthonormal basis of the column span via QR (rank-revealing).
    Q, _ = np.linalg.qr(A)
    PSigma = Q @ (Q.T @ Sigma)              # P_T @ Sigma
    return float(np.trace(PSigma) / total)


# ----------------------------------------------------------------------------
# Greedy algorithm
# ----------------------------------------------------------------------------
@dataclass
class GreedyResult:
    order: list[int]                  # benchmark indices in selection order
    marginal_gains: list[float]       # f gain at each step
    cumulative_coverage: list[float]  # f after each step
    redundancies: list[int]           # benchmarks with marginal gain < eta_redundant
    minimum_subset: list[int]         # smallest prefix achieving target coverage
    target_coverage: float


def greedy_select(
    Sigma: np.ndarray,
    target: float = 0.9,
    eta_redundant: float = 0.01,
) -> GreedyResult:
    """Greedy benchmark selection (Theorem 3, Nemhauser et al 1978).

    At each step pick the benchmark whose addition gives the largest marginal
    gain in coverage. Stop only after every benchmark is ordered (so the
    return contains the full ranking, not just the prefix).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    k = Sigma.shape[0]
    remaining = list(range(k))
    selected: list[int] = []
    gains: list[float] = []
    cov: list[float] = []
    prev = 0.0
    while remaining:
        best_j, best_g = remaining[0], -np.inf
        for j in remaining:
            f_new = coverage_function(Sigma, selected + [j])
            g = f_new - prev
            if g > best_g:
                best_g, best_j, best_f = g, j, f_new
        selected.append(best_j)
        remaining.remove(best_j)
        gains.append(float(best_g))
        cov.append(float(best_f))
        prev = best_f

    # Locate the minimum prefix achieving the target.
    min_prefix: list[int] = []
    for i, c in enumerate(cov):
        if c >= target:
            min_prefix = selected[: i + 1]
            break
    if not min_prefix:
        min_prefix = selected[:]  # target unachievable; return full set

    redundancies = [selected[i] for i, g in enumerate(gains) if g < eta_redundant]

    return GreedyResult(
        order=selected,
        marginal_gains=gains,
        cumulative_coverage=cov,
        redundancies=redundancies,
        minimum_subset=min_prefix,
        target_coverage=target,
    )


# ----------------------------------------------------------------------------
# Blind spot characterisation
# ----------------------------------------------------------------------------
@dataclass
class BlindSpot:
    uncovered_eigenvectors: np.ndarray   # k x r matrix
    uncovered_eigenvalues: np.ndarray    # length r
    uncovered_fraction: float            # 1 - f(selected)
    selected_subset: list[int]


def uncovered_directions(
    Sigma: np.ndarray,
    selected_subset: Sequence[int],
    rel_tol: float = 1e-6,
) -> BlindSpot:
    """Identify principal directions of Sigma orthogonal to the selected span.

    Returns the eigenvectors v_i with non-negligible eigenvalues whose
    projection onto span{Sigma[:, j] : j in selected_subset} is below tol.
    These are the "blind directions" of the selected benchmark suite.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    k = Sigma.shape[0]
    eigvals, eigvecs = eigendecomp(Sigma)

    if not len(selected_subset):
        return BlindSpot(
            uncovered_eigenvectors=eigvecs,
            uncovered_eigenvalues=eigvals,
            uncovered_fraction=1.0,
            selected_subset=[],
        )

    sel = np.asarray(list(selected_subset), dtype=int)
    A = Sigma[:, sel]
    Q, _ = np.linalg.qr(A)

    # Projection norm of each eigenvector onto Q's column space
    proj = Q.T @ eigvecs
    energy = (proj ** 2).sum(axis=0)   # in [0, 1]

    keep = (energy < (1.0 - rel_tol)) & (eigvals > rel_tol * eigvals.max())
    return BlindSpot(
        uncovered_eigenvectors=eigvecs[:, keep],
        uncovered_eigenvalues=eigvals[keep] * (1.0 - energy[keep]),
        uncovered_fraction=1.0 - coverage_function(Sigma, list(sel)),
        selected_subset=list(sel),
    )

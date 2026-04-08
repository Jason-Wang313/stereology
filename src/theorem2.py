"""Theorem 2: Lipschitz indistinguishability bound.

For convex capability profiles K, L contained in a ball of radius R,
m benchmark "width" measurements with tolerance epsilon imply

    delta_H(K, L) <= epsilon + pi * R / m

(Theorem 2 in proofs/theorem2_proof.tex). The minimum benchmark count for
a target Hausdorff distance delta is

    m >= pi * R / (delta - epsilon).

The module also provides:
- ``capability_radius_estimate``: estimate R from the score matrix.
- ``ranking_unreliability``: Corollary 2.1 (top-1 reliability bound).
- ``rank_reversal_susceptible``: Corollary 2.2 (d_eff < n - 1 implies
  rank reversal is possible).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Theorem 2 core
# ----------------------------------------------------------------------------
def lipschitz_bound(epsilon: float, R: float, m: int) -> float:
    """delta_H upper bound: epsilon + pi * R / m."""
    if m <= 0:
        raise ValueError("m must be positive")
    return float(epsilon + np.pi * R / m)


def minimum_benchmarks(delta: float, epsilon: float, R: float) -> float:
    """Smallest m guaranteeing delta_H <= delta."""
    gap = delta - epsilon
    if gap <= 0:
        return float("inf")
    return float(np.ceil(np.pi * R / gap))


def capability_radius_estimate(S: np.ndarray) -> float:
    """Estimate the capability radius R from the score matrix.

    The score matrix is centered, then R is the largest centred row norm
    in standardised units. This bounds all observed profiles inside a ball
    of radius R.
    """
    S = np.asarray(S, dtype=float)
    mu = S.mean(axis=0)
    sigma = S.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    Z = (S - mu) / sigma
    return float(np.linalg.norm(Z, axis=1).max())


# ----------------------------------------------------------------------------
# Corollary 2.1: Ranking reliability
# ----------------------------------------------------------------------------
@dataclass
class RankingReliability:
    n_models: int
    delta_min: float
    indistinguishability_radius: float
    pair_violation_rate: float
    upper_bound_on_top1_correct: float
    pairs_within_radius: int


def aggregate_score(S: np.ndarray) -> np.ndarray:
    """Mean across benchmarks (the simplest aggregator)."""
    return S.mean(axis=1)


def ranking_unreliability(
    S: np.ndarray,
    R: float | None = None,
    epsilon: float = 0.0,
) -> RankingReliability:
    """Corollary 2.1 from Theorem 2.

    Counts the fraction of model pairs whose aggregate-score gap is below
    the indistinguishability radius delta_0 = pi R / m.
    """
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    if R is None:
        R = capability_radius_estimate(S)
    delta_0 = np.pi * R / m

    agg = aggregate_score(S)
    diffs = np.abs(agg[:, None] - agg[None, :])
    iu = np.triu_indices(n, k=1)
    pair_diffs = diffs[iu]
    n_pairs = pair_diffs.size
    n_close = int((pair_diffs < delta_0).sum())
    rate = n_close / max(n_pairs, 1)
    upper_bound = max(0.0, 1.0 - rate)

    return RankingReliability(
        n_models=n,
        delta_min=float(pair_diffs.min()) if pair_diffs.size else 0.0,
        indistinguishability_radius=float(delta_0),
        pair_violation_rate=float(rate),
        upper_bound_on_top1_correct=float(upper_bound),
        pairs_within_radius=n_close,
    )


# ----------------------------------------------------------------------------
# Corollary 2.2: Rank reversal susceptibility
# ----------------------------------------------------------------------------
def rank_reversal_susceptible(d_eff: float, n_models: int) -> bool:
    """True iff d_eff < n - 1 (rank reversal can occur, Cor 2.2)."""
    return d_eff < (n_models - 1)


def reversal_dimension_gap(d_eff: float, n_models: int) -> float:
    """How many ``missing'' independent directions for n models to be orderable."""
    return float(max(0.0, (n_models - 1) - d_eff))

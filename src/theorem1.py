"""Theorem 1: Effective Dimensionality.

Implements the participation ratio, the Marchenko-Pastur correction,
the variance-capture upper bound, and bootstrap confidence intervals.
See ``proofs/theorem1_proof.tex`` for the formal statement.

The participation ratio is defined as
    d_eff = (sum lambda_i)^2 / sum lambda_i^2,
where {lambda_i} are the eigenvalues of the benchmark correlation matrix.
For a correlation matrix sum lambda_i = k (the number of benchmarks), so
d_eff = k^2 / sum lambda_i^2 and the bound 1 <= d_eff <= k holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils import correlation_matrix, eigendecomp


# ----------------------------------------------------------------------------
# Core formulas
# ----------------------------------------------------------------------------
def participation_ratio(eigvals: np.ndarray) -> float:
    """d_eff = (sum lambda)^2 / sum lambda^2."""
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return 0.0
    return float(eigvals.sum() ** 2 / np.square(eigvals).sum())


def mp_threshold(n: int, k: int) -> float:
    """Upper edge of the Marchenko-Pastur bulk: lambda_+ = (1 + sqrt(k/n))^2.

    Eigenvalues above this threshold are signal under the null of independent
    standardised benchmarks; everything below is indistinguishable from
    finite-sample noise.
    """
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    gamma = k / n
    return float((1.0 + np.sqrt(gamma)) ** 2)


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    """Per-component fraction of total variance, sorted descending."""
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return np.zeros_like(eigvals)
    return eigvals / total


def variance_capture_bound(d_eff: float, D_total: int) -> float:
    """Theorem 1(a): captured fraction <= d_eff / D under isotropic capability."""
    if D_total <= 0:
        raise ValueError("D_total must be positive")
    return float(min(1.0, d_eff / D_total))


# ----------------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------------
def bootstrap_d_eff(
    S: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    use_mp: bool = False,
) -> tuple[float, float, float, np.ndarray]:
    """Bootstrap confidence interval for d_eff over models.

    Parameters
    ----------
    S : (n_models, k_benchmarks) score matrix
    n_boot : number of resamples (with replacement, over rows)
    seed : RNG seed
    use_mp : if True, restrict eigenvalues to those above the MP threshold

    Returns
    -------
    point estimate, 2.5th percentile, 97.5th percentile, full sample.
    """
    rng = np.random.default_rng(seed)
    n, k = S.shape
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Sb = S[idx]
        # Drop columns with zero variance in this resample
        std = Sb.std(axis=0)
        keep = std > 1e-12
        if keep.sum() < 2:
            samples[b] = np.nan
            continue
        Cb = correlation_matrix(Sb[:, keep])
        wb, _ = eigendecomp(Cb)
        if use_mp:
            thr = mp_threshold(n, int(keep.sum()))
            wb = wb[wb > thr]
            if wb.size == 0:
                samples[b] = 0.0
                continue
        samples[b] = participation_ratio(wb)

    samples = samples[~np.isnan(samples)]
    point_C = correlation_matrix(S)
    w0, _ = eigendecomp(point_C)
    if use_mp:
        thr = mp_threshold(n, k)
        w0 = w0[w0 > thr]
    point = participation_ratio(w0)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi), samples


# ----------------------------------------------------------------------------
# High-level API
# ----------------------------------------------------------------------------
@dataclass
class DimensionalityResult:
    """Container for the full Theorem 1 analysis on a score matrix."""
    n_models: int
    k_benchmarks: int
    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    d_eff: float
    d_eff_mp: float
    mp_threshold: float
    n_signal_eigs: int
    bootstrap_ci: tuple[float, float] = (np.nan, np.nan)             # CI on d_eff
    bootstrap_ci_mp: tuple[float, float] = (np.nan, np.nan)          # CI on d_eff_mp
    bootstrap_samples: Optional[np.ndarray] = field(default=None, repr=False)
    bootstrap_samples_mp: Optional[np.ndarray] = field(default=None, repr=False)

    def variance_capture_bound(self, D_total: int) -> float:
        return variance_capture_bound(self.d_eff, D_total)


def analyze_dimensionality(
    S: np.ndarray,
    bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> DimensionalityResult:
    """Run the full Theorem 1 analysis on a score matrix.

    Parameters
    ----------
    S : (n_models, k_benchmarks)
    bootstrap : whether to compute a bootstrap CI for d_eff
    n_boot : bootstrap iterations
    seed : RNG seed
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("score matrix must be 2-D")
    n, k = S.shape

    C = correlation_matrix(S)
    eigvals, _ = eigendecomp(C)
    eigvals = np.clip(eigvals, 0.0, None)

    thr = mp_threshold(n, k)
    signal = eigvals[eigvals > thr]

    d_eff_full = participation_ratio(eigvals)
    d_eff_mp = participation_ratio(signal) if signal.size else 0.0

    ci = (np.nan, np.nan)
    ci_mp = (np.nan, np.nan)
    boot_samples: Optional[np.ndarray] = None
    boot_samples_mp: Optional[np.ndarray] = None
    if bootstrap and n >= 8:
        _, lo, hi, boot_samples = bootstrap_d_eff(
            S, n_boot=n_boot, seed=seed, use_mp=False
        )
        ci = (lo, hi)
        _, lo_mp, hi_mp, boot_samples_mp = bootstrap_d_eff(
            S, n_boot=n_boot, seed=seed + 1, use_mp=True
        )
        ci_mp = (lo_mp, hi_mp)

    return DimensionalityResult(
        n_models=n,
        k_benchmarks=k,
        eigenvalues=eigvals,
        explained_variance=explained_variance_ratio(eigvals),
        d_eff=d_eff_full,
        d_eff_mp=d_eff_mp,
        mp_threshold=thr,
        n_signal_eigs=int(signal.size),
        bootstrap_ci=ci,
        bootstrap_ci_mp=ci_mp,
        bootstrap_samples=boot_samples,
        bootstrap_samples_mp=boot_samples_mp,
    )

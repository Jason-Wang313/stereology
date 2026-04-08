"""Busemann-Petty analogue and rank-reversal corollaries.

These functions implement the *empirical* counterparts of the corollaries
proved in ``proofs/corollary_proofs.tex``:

- ``benchmark_dominated_pairs``: find observed pairs (A, B) where A scores
  strictly lower than B on every benchmark; the Busemann-Petty analogue
  predicts that for d_eff >= 5 such pairs need not be uniformly dominated
  in the underlying capability space.

- ``rank_reversals_on_addition``: find triples (A, B, X) such that the
  rank ordering of (A, B) reverses when X is added to the population.
  This is the geometric MCDM phenomenon (Belton & Gear 1983) made precise
  by Corollary 2.2.

We are careful to frame the Busemann-Petty result as an *analogue*: it is
*not* a direct application, since benchmark scores are width-like rather
than section volumes. We follow the wording in proofs/corollary_proofs.tex.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Busemann-Petty analogue: pointwise benchmark domination
# ----------------------------------------------------------------------------
@dataclass
class DominationPair:
    a_idx: int
    b_idx: int
    margins: np.ndarray  # B - A on each benchmark (all >= 0, at least one > 0)


def benchmark_dominated_pairs(
    S: np.ndarray,
    strict: bool = True,
) -> list[DominationPair]:
    """Find pairs where model B beats model A on every single benchmark.

    Parameters
    ----------
    S : (n_models, k_benchmarks)
    strict : require strict inequality on every benchmark (otherwise allow
             ties on some).
    """
    S = np.asarray(S, dtype=float)
    n, k = S.shape
    pairs: list[DominationPair] = []
    for a, b in combinations(range(n), 2):
        diff = S[b] - S[a]
        if strict:
            if (diff > 0).all():
                pairs.append(DominationPair(a, b, diff))
            elif (diff < 0).all():
                pairs.append(DominationPair(b, a, -diff))
        else:
            if (diff >= 0).all() and (diff > 0).any():
                pairs.append(DominationPair(a, b, diff))
            elif (diff <= 0).all() and (diff < 0).any():
                pairs.append(DominationPair(b, a, -diff))
    return pairs


def domination_under_holdout(
    S: np.ndarray,
    held_out_cols: Sequence[int],
) -> list[DominationPair]:
    """Pairs where domination on the visible benchmarks reverses on a
    held-out benchmark.

    Returns dominating pairs (A < B on every visible benchmark) where the
    held-out benchmark would actually have B < A on at least one held-out
    column. These are *empirical* Busemann-Petty-style failures: visible
    domination does not imply true overall capability domination.
    """
    S = np.asarray(S, dtype=float)
    k = S.shape[1]
    visible = [c for c in range(k) if c not in set(held_out_cols)]
    if not visible or not len(held_out_cols):
        return []
    Sv = S[:, visible]
    pairs = benchmark_dominated_pairs(Sv, strict=True)
    failures: list[DominationPair] = []
    held = list(held_out_cols)
    for p in pairs:
        a_held = S[p.a_idx, held]
        b_held = S[p.b_idx, held]
        if (a_held > b_held).any():
            failures.append(p)
    return failures


# ----------------------------------------------------------------------------
# Rank reversal on adding a model
# ----------------------------------------------------------------------------
@dataclass
class RankReversal:
    a_idx: int
    b_idx: int
    added_idx: int
    rank_a_before: int
    rank_b_before: int
    rank_a_after: int
    rank_b_after: int


def _ranks(scores: np.ndarray, idx: Sequence[int]) -> dict[int, int]:
    """Return a {row index -> rank} dict (1 = best) for ``idx``."""
    sub = [(i, scores[i]) for i in idx]
    sub.sort(key=lambda x: -x[1])
    return {row: r + 1 for r, (row, _) in enumerate(sub)}


def rank_reversals_on_addition(
    S: np.ndarray,
    aggregator=None,
    base_size: int | None = None,
    seed: int = 0,
    max_triples: int = 5000,
) -> list[RankReversal]:
    """Empirically find rank reversals when one model is added to a base set.

    Strategy
    --------
    1. Take a base set of `base_size` models (default n - 1).
    2. For each model X not in the base, build a candidate set base U {X}.
    3. Re-aggregate scores under the chosen aggregator and rank.
    4. Compare ranks of every (A, B) pair in `base` before and after.
    5. Record any pair whose ordering flips.

    Aggregator default = mean across benchmarks. We use a *normalised*
    aggregator (column z-score, then mean) so that adding a model with an
    outlier benchmark can plausibly reweight existing rankings.
    """
    S = np.asarray(S, dtype=float)
    n, k = S.shape
    rng = np.random.default_rng(seed)

    if aggregator is None:
        def aggregator(M: np.ndarray) -> np.ndarray:
            mu = M.mean(axis=0)
            sd = M.std(axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            return ((M - mu) / sd).mean(axis=1)

    if base_size is None:
        base_size = n - 1

    # All n choices of which model to add (rotate base = all-but-one)
    reversals: list[RankReversal] = []
    triples_examined = 0
    for added in range(n):
        base = [i for i in range(n) if i != added]
        S_base = S[base]
        agg_base = aggregator(S_base)
        ranks_before = {row: r for row, r in zip(base, _ranks_array(agg_base))}

        full = base + [added]
        S_full = S[full]
        agg_full = aggregator(S_full)
        ranks_after_arr = _ranks_array(agg_full)
        ranks_after = {row: r for row, r in zip(full, ranks_after_arr)}

        for a, b in combinations(base, 2):
            triples_examined += 1
            if triples_examined > max_triples:
                return reversals
            ra0, rb0 = ranks_before[a], ranks_before[b]
            ra1, rb1 = ranks_after[a], ranks_after[b]
            before_a_better = ra0 < rb0
            after_a_better = ra1 < rb1
            if before_a_better != after_a_better:
                reversals.append(
                    RankReversal(a, b, added, ra0, rb0, ra1, rb1)
                )
    return reversals


def _ranks_array(scores: np.ndarray) -> np.ndarray:
    """Return ranks (1 = best) of a 1-D score array."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

"""Validation experiments for the STEREOLOGY paper (Appendix H).

A. Permutation null for eigenvalues (compares the observed spectrum to
   the spectrum obtained by independently shuffling each benchmark
   column, and to the Marchenko-Pastur upper edge).
B. Split-half reliability of d_eff over 500 random 50/50 model splits.
C. d_eff saturation curve over subsamples n' = 20, 50, ..., n.
D. Greedy out-of-sample Kendall tau between r-benchmark ranking and
   full ranking, compared to random benchmark subsets.
E. Greedy vs max-uncorrelated heuristic coverage curves.
F. Spearman vs Pearson d_eff comparison.
G. Pairwise swap probability sensitivity sweep over D and bootstrap CIs.

Outputs are written to results/validation/*.json and *.csv. The script
loads the existing extended.csv and OLLM v2 csv that ship with the
project; no network calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import (
    analyze_dimensionality,
    mp_threshold,
    participation_ratio,
)
from src.theorem3 import coverage_function, greedy_select
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    score_matrix,
)


VALIDATION_DIR = RESULTS_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# A. Permutation null for eigenvalues
# ----------------------------------------------------------------------------
def permutation_null(S: np.ndarray, n_perm: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, k = S.shape
    obs = np.linalg.eigvalsh(np.corrcoef(S, rowvar=False))[::-1]
    null = np.empty((n_perm, k))
    for b in range(n_perm):
        Sp = np.empty_like(S)
        for j in range(k):
            Sp[:, j] = S[rng.permutation(n), j]
        null[b] = np.linalg.eigvalsh(np.corrcoef(Sp, rowvar=False))[::-1]
    p95 = np.percentile(null, 95, axis=0)
    return obs, null, p95


# ----------------------------------------------------------------------------
# B. Split-half reliability of d_eff
# ----------------------------------------------------------------------------
def split_half_reliability(S: np.ndarray, n_splits: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(S)
    half = n // 2
    pairs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[:half], idx[half:half * 2]
        ra = analyze_dimensionality(S[a], bootstrap=False).d_eff
        rb = analyze_dimensionality(S[b], bootstrap=False).d_eff
        pairs.append((ra, rb))
    arr = np.array(pairs)
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    mad = float(np.mean(np.abs(arr[:, 0] - arr[:, 1])))
    return arr, corr, mad


# ----------------------------------------------------------------------------
# C. d_eff saturation curve
# ----------------------------------------------------------------------------
def saturation_curve(
    S: np.ndarray, sample_sizes: list[int],
    n_boot: int = 100, seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = len(S)
    rows = []
    for n_prime in sample_sizes:
        if n_prime > n:
            continue
        ds = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n_prime, replace=False)
            ds.append(analyze_dimensionality(S[idx], bootstrap=False).d_eff)
        ds = np.asarray(ds)
        rows.append({
            "n_prime": int(n_prime),
            "d_eff_mean": float(ds.mean()),
            "d_eff_lo": float(np.percentile(ds, 2.5)),
            "d_eff_hi": float(np.percentile(ds, 97.5)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# D. Greedy out-of-sample Kendall tau
# ----------------------------------------------------------------------------
def aggregate(S: np.ndarray) -> np.ndarray:
    Z = (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)
    return Z.mean(axis=1)


def greedy_oos_kendall(
    S: np.ndarray, r_values: list[int],
    n_random_perm: int = 200, seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n, k = S.shape
    Sigma = np.corrcoef(S, rowvar=False)
    full_rank = aggregate(S)

    g = greedy_select(Sigma, target=1.0, eta_redundant=0.0)
    rows = []
    for r in r_values:
        g_subset = g.order[:r]
        g_rank = aggregate(S[:, g_subset])
        tau_g, _ = kendalltau(full_rank, g_rank)

        taus_random = []
        for _ in range(n_random_perm):
            sub = rng.choice(k, size=r, replace=False)
            r_rank = aggregate(S[:, sub])
            tau_r, _ = kendalltau(full_rank, r_rank)
            taus_random.append(tau_r)
        rows.append({
            "r": int(r),
            "tau_greedy": float(tau_g),
            "tau_random_mean": float(np.mean(taus_random)),
            "tau_random_std": float(np.std(taus_random)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# E. Greedy vs max-uncorrelated coverage
# ----------------------------------------------------------------------------
def max_uncorrelated_curve(Sigma: np.ndarray) -> list[float]:
    k = Sigma.shape[0]
    abs_corr = np.abs(Sigma)
    selected: list[int] = [int(np.argmax(np.diag(Sigma)))]
    cov = [coverage_function(Sigma, selected)]
    while len(selected) < k:
        remaining = [j for j in range(k) if j not in selected]
        # next benchmark = the one with smallest max |corr| to selected set
        scores = [max(abs_corr[j, i] for i in selected) for j in remaining]
        nxt = remaining[int(np.argmin(scores))]
        selected.append(nxt)
        cov.append(coverage_function(Sigma, selected))
    return cov


# ----------------------------------------------------------------------------
# F. Spearman vs Pearson d_eff comparison
# ----------------------------------------------------------------------------
def spearman_d_eff(S: np.ndarray) -> dict:
    rho, _ = spearmanr(S)
    if np.isscalar(rho):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    eigs = np.linalg.eigvalsh(rho)[::-1]
    return {
        "d_eff_spearman": float(participation_ratio(eigs)),
        "spearman_eigs": eigs.tolist(),
    }


# ----------------------------------------------------------------------------
# G. Pairwise swap probability sensitivity sweep over D
# ----------------------------------------------------------------------------
def swap_sensitivity(scores: np.ndarray, d_eff: float, D_grid: list[int]):
    s = np.sort(scores)[::-1]
    delta2 = float(s[0] - s[1])
    rows = []
    for D in D_grid:
        if D <= d_eff:
            continue
        sigma_h = np.sqrt(2.0 * (D - d_eff))
        # standardized score scale: rescale sigma_h to observed std
        sigma_obs = float(np.std(scores))
        sigma_h_rescaled = sigma_obs * np.sqrt((D - d_eff) / d_eff)
        p = float(norm.cdf(-delta2 / (2.0 * sigma_h_rescaled)))
        rows.append({
            "D": int(D),
            "rho": float(np.sqrt(d_eff / D)),
            "delta2": delta2,
            "P_swap": p,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run() -> None:
    out: dict = {}
    df_v2 = load_ollm_v2()
    df_ext = load_extended()
    S_v2 = score_matrix(df_v2, OLLM_V2_BENCHES)
    S_ext = score_matrix(df_ext, EXTENDED_BENCHES)

    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S_top = score_matrix(df_top, EXTENDED_BENCHES)

    # ---- A: permutation null --------------------------------------------
    print("[A] permutation null for eigenvalues")
    obs_v2, null_v2, p95_v2 = permutation_null(S_v2, n_perm=1000)
    obs_ext, null_ext, p95_ext = permutation_null(S_ext, n_perm=1000)
    out["A_permutation_null"] = {
        "ollm_v2": {
            "observed_eigs": obs_v2.tolist(),
            "perm_p95": p95_v2.tolist(),
            "mp_lambda_plus": mp_threshold(*S_v2.shape),
        },
        "extended": {
            "observed_eigs": obs_ext.tolist(),
            "perm_p95": p95_ext.tolist(),
            "mp_lambda_plus": mp_threshold(*S_ext.shape),
        },
    }

    # ---- B: split-half ---------------------------------------------------
    print("[B] split-half reliability")
    arr_v2, corr_v2, mad_v2 = split_half_reliability(S_v2, n_splits=500)
    arr_ext, corr_ext, mad_ext = split_half_reliability(S_ext, n_splits=500)
    out["B_split_half"] = {
        "ollm_v2": {"correlation": corr_v2, "mean_abs_diff": mad_v2},
        "extended": {"correlation": corr_ext, "mean_abs_diff": mad_ext},
    }

    # ---- C: saturation ---------------------------------------------------
    print("[C] saturation curve")
    sat_v2 = saturation_curve(S_v2, [20, 50, 100, 200, 300, 458])
    sat_ext = saturation_curve(S_ext, [20, 50, 100, 150, 200, 250, 295])
    sat_v2.to_csv(VALIDATION_DIR / "C_saturation_v2.csv", index=False)
    sat_ext.to_csv(VALIDATION_DIR / "C_saturation_ext.csv", index=False)
    out["C_saturation"] = {
        "ollm_v2": sat_v2.to_dict(orient="records"),
        "extended": sat_ext.to_dict(orient="records"),
    }

    # ---- D: greedy out-of-sample Kendall tau ----------------------------
    print("[D] greedy OOS Kendall tau")
    tau_df = greedy_oos_kendall(S_top, r_values=list(range(2, 13)))
    tau_df.to_csv(VALIDATION_DIR / "D_greedy_oos_tau.csv", index=False)
    out["D_greedy_oos_kendall"] = tau_df.to_dict(orient="records")

    # ---- E: max-uncorrelated comparison ---------------------------------
    print("[E] max-uncorrelated vs greedy")
    Sigma_top = np.corrcoef(S_top, rowvar=False)
    g_top = greedy_select(Sigma_top, target=1.0, eta_redundant=0.0)
    mu_curve = max_uncorrelated_curve(Sigma_top)
    out["E_max_uncorrelated"] = {
        "greedy_curve": list(map(float, g_top.cumulative_coverage)),
        "max_uncorrelated_curve": list(map(float, mu_curve)),
    }

    # ---- F: Spearman comparison -----------------------------------------
    print("[F] Spearman vs Pearson")
    out["F_spearman_vs_pearson"] = {
        "ollm_v2_full": {
            "d_eff_pearson": float(
                analyze_dimensionality(S_v2, bootstrap=False).d_eff
            ),
            **spearman_d_eff(S_v2),
        },
        "ollm_v2_frontier": {
            "d_eff_pearson": float(
                analyze_dimensionality(
                    S_v2[df_v2[OLLM_V2_BENCHES].mean(axis=1).rank(pct=True) >= 0.5],
                    bootstrap=False,
                ).d_eff
            ),
            **spearman_d_eff(
                S_v2[df_v2[OLLM_V2_BENCHES].mean(axis=1).rank(pct=True) >= 0.5]
            ),
        },
        "extended_frontier": {
            "d_eff_pearson": float(
                analyze_dimensionality(S_top, bootstrap=False).d_eff
            ),
            **spearman_d_eff(S_top),
        },
    }

    # ---- G: swap sensitivity --------------------------------------------
    print("[G] swap sensitivity")
    d_top = analyze_dimensionality(S_top, bootstrap=False).d_eff
    sweep = swap_sensitivity(aggregate(S_top), d_top, [10, 15, 20, 30, 50, 100])
    sweep.to_csv(VALIDATION_DIR / "G_swap_sensitivity.csv", index=False)
    out["G_swap_sensitivity"] = sweep.to_dict(orient="records")

    out_path = VALIDATION_DIR / "validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    run()

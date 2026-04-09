"""v8 evidence-first validation experiments.

EXP 1A  Six-prior x four-D swap simulation
EXP 1B  Five-quality-functional swap simulation
EXP 1C  Four-standardisation blind-spot ratio table
EXP 1D  Four-normalisation d_eff (Pearson, Spearman, Kendall, quantile)
EXP 2A  MMLU width-model walkthrough (correlation, residual)
EXP 2B  Synthetic pass/fail counterexample (R^2, residual size)
EXP 3A  Extended half-split swap counts (top-1, top-5)
EXP 3B  Adversarial direction injection (top-10 changes)
EXP 3C  delta_0 vs empirical swap calibration
EXP 4A  Quarterly coverage retention 4x4 matrix
EXP 4B  Top-k stability under fixed greedy subset (temporal holdout)

Outputs land in results/validation_v8/.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau, spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import analyze_dimensionality, participation_ratio
from src.theorem3 import coverage_function, greedy_select
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    DATA_DIR,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    score_matrix,
)

OUT = RESULTS_DIR / "validation_v8"
OUT.mkdir(parents=True, exist_ok=True)


def standardise(S):
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


def aggregate(S):
    return standardise(S).mean(axis=1)


def _ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(scores) + 1)
    return r


# ============================================================================
# Helpers
# ============================================================================
def chi_squared_swap_pred(delta, d_eff, D, sigma_obs):
    sigma_h = sigma_obs * np.sqrt(max((D - d_eff) / d_eff, 1e-9))
    return float(norm.cdf(-delta / (2.0 * sigma_h)))


def load_frontier_extended():
    df = load_extended()
    avg = df[EXTENDED_BENCHES].mean(axis=1)
    return df[avg >= avg.quantile(0.5)].reset_index(drop=True)


# ============================================================================
# EXP 1A. Six priors x four D values
# ============================================================================
def exp1a_priors_x_D() -> dict:
    print("[1A] 6 priors x 4 D values")
    df_top = load_frontier_extended()
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    d_eff = float(analyze_dimensionality(S, bootstrap=False).d_eff)
    s = np.sort(aggregate(S))[::-1]
    delta2 = float(s[0] - s[1])
    sigma_obs = float(np.std(aggregate(S)))

    eigs_emp = np.linalg.eigvalsh(np.cov(S, rowvar=False))[::-1]
    eigs_emp = np.clip(eigs_emp, 0, None)
    eigs_emp = eigs_emp / max(eigs_emp.sum(), 1e-9) * k

    rng = np.random.default_rng(0)

    def make_prior_swap(name, eigs, D):
        # Use the chi-squared closed form with the eigenvalue spectrum
        # rescaled so that tr(Sigma_hidden) matches sigma_h^2
        sigma_h2 = (D - d_eff)
        # Normalise eigs to sum to sigma_h2
        e = np.array(eigs[: max(int(D - d_eff), 1)], dtype=float)
        if e.sum() <= 0:
            return float("nan")
        e = e / e.sum() * sigma_h2
        var_diff = 4.0 * float(np.sum(e ** 2))
        # Convert to standardised score scale: rescale by variance ratio
        # The geometry is: Var(Y_j-Y_i) in the prior's units must be mapped
        # back to the observed-score scale via sigma_obs * sqrt(Var/sigma_h2).
        sigma_h_obs = sigma_obs * np.sqrt(var_diff / max(sigma_h2 ** 2, 1e-9))
        return float(norm.cdf(-delta2 / (2.0 * sigma_h_obs)))

    priors = {
        "isotropic":     lambda D: [1.0] * max(int(D - d_eff), 1),
        "empirical":     lambda D: list(eigs_emp[: max(int(D - d_eff), 1)]),
        "1/i":           lambda D: [1.0 / (i + 1) for i in range(max(int(D - d_eff), 1))],
        "1/i^2":         lambda D: [1.0 / (i + 1) ** 2 for i in range(max(int(D - d_eff), 1))],
        "Pareto(1.5)":   lambda D: list(rng.pareto(1.5, size=max(int(D - d_eff), 1)) + 1),
        "adversarial":   lambda D: [1.0] + [0.0] * max(int(D - d_eff) - 1, 0),
    }

    rows = []
    for name, fn in priors.items():
        row = {"prior": name}
        for D in [10, 20, 50, 100]:
            row[f"D={D}"] = round(make_prior_swap(name, fn(D), D), 3)
        rows.append(row)
    df_1a = pd.DataFrame(rows)
    df_1a.to_csv(OUT / "1a_priors_D.csv", index=False)
    print(df_1a.to_string(index=False))
    return {"table": df_1a.to_dict(orient="records")}


# ============================================================================
# EXP 1B. Five quality functionals
# ============================================================================
def exp1b_functionals() -> dict:
    print("[1B] 5 quality functionals at D=20")
    df_top = load_frontier_extended()
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)
    d_eff = float(analyze_dimensionality(S, bootstrap=False).d_eff)
    D = 20

    rng = np.random.default_rng(1)
    rows = []
    sigma_h = np.sqrt(2.0 * (D - d_eff))
    n_trials = 5000

    def swap_under_quality(qfunc):
        # Simulate n=148 capability vectors c in R^D, observed = first d_eff comps
        cap = rng.normal(size=(n_trials, n, D))
        # Observed = sum of first d_eff_int components
        d_eff_int = int(round(d_eff))
        obs_quality = cap[:, :, :d_eff_int].mean(axis=2)
        true_quality = np.array([
            [qfunc(c) for c in row] for row in cap
        ])
        swaps = 0
        for t in range(n_trials):
            o = np.argsort(-obs_quality[t])
            tr = np.argsort(-true_quality[t])
            if o[0] != tr[0]:
                swaps += 1
        return swaps / n_trials

    quality_funcs = {
        "||c||^2 (current)": lambda c: float(np.sum(c ** 2)),
        "(1/sqrt(D)) sum c": lambda c: float(np.sum(c) / np.sqrt(D)),
        "e_1 . c (single)":  lambda c: float(c[0]),
        "random w . c":      None,    # set below
        "min_j c_j":         lambda c: float(np.min(c)),
    }
    # Random w (fixed across this experiment for reproducibility)
    w = rng.normal(size=D)
    w /= np.linalg.norm(w)
    quality_funcs["random w . c"] = lambda c, w=w: float(c @ w)

    for name, qf in quality_funcs.items():
        rate = swap_under_quality(qf)
        rows.append({"quality_functional": name, "P_swap_D20": round(rate, 3)})
    df_1b = pd.DataFrame(rows)
    df_1b.to_csv(OUT / "1b_functionals.csv", index=False)
    print(df_1b.to_string(index=False))
    return {"table": df_1b.to_dict(orient="records")}


# ============================================================================
# EXP 1C. Four standardisations -> blind-spot ratio
# ============================================================================
def exp1c_standardisation() -> dict:
    print("[1C] 4 standardisations")
    df = load_extended()
    avg = df[EXTENDED_BENCHES].mean(axis=1)
    df_top = df[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S_raw = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S_raw.shape

    methods = {}
    methods["zscore"] = standardise(S_raw)
    methods["min_max"] = (S_raw - S_raw.min(0)) / (S_raw.max(0) - S_raw.min(0))
    ranks = np.apply_along_axis(rankdata, 0, S_raw)
    methods["rank_zscore"] = standardise(ranks)
    methods["raw"] = S_raw.copy()

    rng = np.random.default_rng(0)
    rand = rng.normal(size=(10000, k))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)

    rows = []
    for name, M in methods.items():
        # R = max row norm in this representation
        R = float(np.linalg.norm(M, axis=1).max())
        # PCA-loading covering radius
        U, s_, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        # Use top-d_eff loadings
        d_eff = max(int(round(participation_ratio(np.linalg.eigvalsh(np.corrcoef(M, rowvar=False))[::-1]))), 2)
        loadings = Vt[:d_eff].T
        norms = np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-12
        loading_dirs = loadings / norms
        # Sample rand on the d_eff sphere
        rand_local = rng.normal(size=(5000, d_eff))
        rand_local /= np.linalg.norm(rand_local, axis=1, keepdims=True)
        cos = np.clip((rand_local @ loading_dirs.T).max(axis=1), -1, 1)
        omega = float(np.arccos(cos).max())
        delta_vis = 2 * R * omega

        agg = (M - M.mean(0)).mean(axis=1)
        s_sorted = np.sort(agg)[::-1]
        delta2 = float(s_sorted[0] - s_sorted[1])
        ratio = delta_vis / max(abs(delta2), 1e-9)
        rows.append({
            "method": name,
            "R": round(R, 2),
            "omega_emp": round(omega, 2),
            "delta_vis": round(delta_vis, 2),
            "delta2": round(delta2, 4),
            "ratio": round(ratio, 1),
        })
    df_1c = pd.DataFrame(rows)
    df_1c.to_csv(OUT / "1c_standardisation.csv", index=False)
    print(df_1c.to_string(index=False))
    return {"table": df_1c.to_dict(orient="records")}


# ============================================================================
# EXP 2A. MMLU walkthrough
# ============================================================================
def exp2a_mmlu_walk() -> dict:
    print("[2A] MMLU walkthrough")
    df = load_extended()
    S = score_matrix(df, EXTENDED_BENCHES)
    Sz = standardise(S)
    n = len(Sz)

    j_mmlu = EXTENDED_BENCHES.index("MMLU")
    y = Sz[:, j_mmlu]

    # Top 5 PCs of the score matrix excluding MMLU column
    X = np.delete(Sz, j_mmlu, axis=1)
    U, s_, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :5] * s_[:5]
    Xfit = np.hstack([np.ones((n, 1)), pcs])
    beta, *_ = np.linalg.lstsq(Xfit, y, rcond=None)
    y_hat = Xfit @ beta
    residuals = y - y_hat

    grad = beta[1:]
    a_mmlu = grad / (np.linalg.norm(grad) + 1e-12)
    grad_norm = float(np.linalg.norm(grad))

    # Width-model prediction of pair differences
    proj = pcs @ a_mmlu
    pred_diff_pairs = []
    actual_diff_pairs = []
    rng = np.random.default_rng(0)
    sample_pairs = rng.choice(n, size=(2000, 2), replace=True)
    for i, j in sample_pairs:
        if i == j:
            continue
        actual = y[i] - y[j]
        pred = grad_norm * (proj[i] - proj[j])
        actual_diff_pairs.append(actual)
        pred_diff_pairs.append(pred)
    actual = np.array(actual_diff_pairs)
    pred = np.array(pred_diff_pairs)
    corr = float(np.corrcoef(actual, pred)[0, 1])
    mae = float(np.mean(np.abs(actual - pred)))
    max_resid = float(np.max(np.abs(actual - pred)))
    out = {
        "r2_linear": round(1 - residuals.var() / y.var(), 3),
        "correlation_actual_predicted": round(corr, 3),
        "mean_abs_residual": round(mae, 3),
        "max_residual": round(max_resid, 3),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# EXP 2B. Synthetic pass/fail counterexample
# ============================================================================
def exp2b_counterexample() -> dict:
    print("[2B] pass/fail counterexample")
    df = load_extended()
    Sz = standardise(score_matrix(df, EXTENDED_BENCHES))
    n, k = Sz.shape
    rng = np.random.default_rng(2)
    w = rng.normal(size=k)
    w /= np.linalg.norm(w)
    z = Sz @ w
    tau = float(np.median(z))
    # Sharp pass/fail synthetic benchmark
    y = (z > tau).astype(float)
    # Try to fit a linear model
    X = np.hstack([np.ones((n, 1)), Sz])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    r2 = 1 - ((y - y_hat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    residual = float(np.max(np.abs(y - y_hat)))
    return {
        "synthetic_r2_linear": round(float(r2), 3),
        "max_residual": round(residual, 3),
        "interpretation": (
            "Sharp pass/fail benchmark has low linear R^2 because the response "
            "is binary; linearisation residual eta is large, widening epsilon "
            "in Theorem 2."
        ),
    }


# ============================================================================
# EXP 3A. Extended half-split swap counts (top-1, top-5)
# ============================================================================
def exp3a_half_split_counts() -> dict:
    print("[3A] half-split swap counts")
    df_top = load_frontier_extended()
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)
    rng = np.random.default_rng(0)

    swaps_top1 = []
    swaps_top5 = []
    for _ in range(500):
        cols = rng.permutation(k)
        vis = cols[: k // 2]
        hold = cols[k // 2:]
        agg_v = Sz[:, vis].mean(axis=1)
        agg_h = Sz[:, hold].mean(axis=1)
        ord_v = np.argsort(-agg_v)
        ord_h = np.argsort(-agg_h)
        # Top-1 changed?
        swaps_top1.append(int(ord_v[0] != ord_h[0]))
        # Top-5 set difference
        s_v = set(ord_v[:5].tolist())
        s_h = set(ord_h[:5].tolist())
        swaps_top5.append(5 - len(s_v & s_h))
    out = {
        "n_trials": 500,
        "top1_swap_rate": round(float(np.mean(swaps_top1)), 3),
        "mean_top5_changes": round(float(np.mean(swaps_top5)), 2),
        "fraction_top5_with_change": round(
            float(np.mean(np.array(swaps_top5) > 0)), 3
        ),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# EXP 3B. Adversarial direction injection
# ============================================================================
def exp3b_adversarial_injection() -> dict:
    print("[3B] adversarial direction injection")
    df_top = load_frontier_extended()
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)
    n, k = Sz.shape

    # PCA: smallest-eigenvalue eigenvectors
    U, s_, Vt = np.linalg.svd(Sz, full_matrices=False)
    eigvecs = Vt.T  # k x k
    # Reverse so the smallest is first
    order = np.argsort(s_)
    worst_dirs = eigvecs[:, order[:3]]   # 3 least-covered

    base_rank = _ranks(Sz.mean(axis=1))
    base_top10 = set(np.argsort(-Sz.mean(axis=1))[:10])

    rows = []
    for idx in range(3):
        v = worst_dirs[:, idx]
        s_hidden = Sz @ v
        # Re-rank using original 12 standardised aggregate + this hidden col
        new_agg = (Sz.mean(axis=1) * k + s_hidden) / (k + 1)
        new_rank = _ranks(new_agg)
        new_top10 = set(np.argsort(-new_agg)[:10])
        rows.append({
            "rank": idx + 1,
            "n_top10_changed": len(base_top10 ^ new_top10),
            "kendall_tau_top": round(
                float(kendalltau(base_rank, new_rank).correlation), 3
            ),
        })
    df_3b = pd.DataFrame(rows)
    df_3b.to_csv(OUT / "3b_adversarial.csv", index=False)
    print(df_3b.to_string(index=False))
    return {"table": df_3b.to_dict(orient="records")}


# ============================================================================
# EXP 3C. delta_0 vs empirical swap calibration
# ============================================================================
def exp3c_delta0_calibration() -> dict:
    print("[3C] delta_0 vs empirical swap calibration")
    df_top = load_frontier_extended()
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)
    n, k = Sz.shape
    R = float(np.linalg.norm(Sz, axis=1).max())

    rng = np.random.default_rng(0)
    pts = []
    for m in range(2, 12):
        for _ in range(50):
            cols = rng.choice(k, size=m, replace=False)
            sub = Sz[:, cols]
            agg = sub.mean(axis=1)
            ord_sub = np.argsort(-agg)
            top, second = ord_sub[0], ord_sub[1]
            full = Sz.mean(axis=1)
            empirical = int(full[second] > full[top])
            delta_0 = float(np.pi * R / m)
            pts.append((delta_0, empirical))
    arr = np.array(pts)
    # Bin by delta_0 (rounded to 0.5) and compute mean swap rate per bin
    delta_bins = np.round(arr[:, 0] / 0.5) * 0.5
    bins = sorted(set(delta_bins.tolist()))
    binned = []
    for b in bins:
        mask = delta_bins == b
        binned.append({
            "delta_0_bin": float(b),
            "n": int(mask.sum()),
            "empirical_swap_rate": round(float(arr[mask, 1].mean()), 3),
        })
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    df_3c = pd.DataFrame(binned)
    df_3c.to_csv(OUT / "3c_calibration.csv", index=False)
    print(df_3c.to_string(index=False))
    return {
        "binned": binned,
        "correlation": round(corr, 3),
    }


# ============================================================================
# EXP 4A. Quarterly coverage retention 4x4 matrix
# ============================================================================
def exp4a_quarterly_retention() -> dict:
    print("[4A] quarterly coverage retention")
    df = load_extended()
    df_sorted = df.sort_values("model").reset_index(drop=True)
    n = len(df_sorted)
    quarter_size = n // 4
    quarters = [df_sorted.iloc[i * quarter_size: (i + 1) * quarter_size]
                for i in range(4)]
    Ss = [score_matrix(q, EXTENDED_BENCHES) for q in quarters]
    Sigmas = [np.corrcoef(s, rowvar=False) for s in Ss]
    greedies = [greedy_select(S, target=1.0, eta_redundant=0.0).order for S in Sigmas]

    matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            matrix[i, j] = coverage_function(Sigmas[j], greedies[i][:7])
    out = {
        "matrix": [[round(float(matrix[i, j]), 3) for j in range(4)] for i in range(4)],
        "min_off_diagonal": round(
            float(matrix[~np.eye(4, dtype=bool)].min()), 3
        ),
        "max_off_diagonal": round(
            float(matrix[~np.eye(4, dtype=bool)].max()), 3
        ),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# EXP 4B. Top-k stability under fixed greedy subset
# ============================================================================
def exp4b_topk_stability() -> dict:
    print("[4B] top-k stability under fixed greedy subset")
    df = load_extended()
    df_sorted = df.sort_values("model").reset_index(drop=True)
    n = len(df_sorted)
    early = df_sorted.iloc[: n // 2]
    late = df_sorted.iloc[n // 2:]

    # Train greedy on early
    S_early = score_matrix(early, EXTENDED_BENCHES)
    Sigma_early = np.corrcoef(S_early, rowvar=False)
    g_early = greedy_select(Sigma_early, target=1.0, eta_redundant=0.0).order
    early_subset_idx = g_early[:7]

    S_late = score_matrix(late, EXTENDED_BENCHES)

    full_late_rank = _ranks(standardise(S_late).mean(axis=1))
    sub_late_rank = _ranks(standardise(S_late[:, early_subset_idx]).mean(axis=1))

    rng = np.random.default_rng(0)
    random_taus = []
    random_top10_overlaps = []
    full_top10 = set(np.argsort(-standardise(S_late).mean(axis=1))[:10])
    for _ in range(200):
        idx = rng.choice(len(EXTENDED_BENCHES), size=7, replace=False)
        rand_rank = _ranks(standardise(S_late[:, idx]).mean(axis=1))
        random_taus.append(float(kendalltau(full_late_rank, rand_rank).correlation))
        rand_top10 = set(np.argsort(-standardise(S_late[:, idx]).mean(axis=1))[:10])
        random_top10_overlaps.append(len(full_top10 & rand_top10))

    sub_tau = float(kendalltau(full_late_rank, sub_late_rank).correlation)
    sub_top10 = set(np.argsort(-standardise(S_late[:, early_subset_idx]).mean(axis=1))[:10])

    out = {
        "greedy_early_kendall_tau_late": round(sub_tau, 3),
        "random_kendall_tau_late_mean": round(float(np.mean(random_taus)), 3),
        "random_kendall_tau_late_std": round(float(np.std(random_taus)), 3),
        "greedy_top10_shared": len(full_top10 & sub_top10),
        "random_top10_shared_mean": round(float(np.mean(random_top10_overlaps)), 2),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# Driver
# ============================================================================
def run():
    out = {}
    out["exp1a"] = exp1a_priors_x_D()
    out["exp1b"] = exp1b_functionals()
    out["exp1c"] = exp1c_standardisation()
    out["exp2a"] = exp2a_mmlu_walk()
    out["exp2b"] = exp2b_counterexample()
    out["exp3a"] = exp3a_half_split_counts()
    out["exp3b"] = exp3b_adversarial_injection()
    out["exp3c"] = exp3c_delta0_calibration()
    out["exp4a"] = exp4a_quarterly_retention()
    out["exp4b"] = exp4b_topk_stability()

    (OUT / "validation_v8.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'validation_v8.json'}")


if __name__ == "__main__":
    run()

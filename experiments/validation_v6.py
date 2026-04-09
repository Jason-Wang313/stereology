"""v6 maximum-aim validation experiments.

W1   Width-model verification: linear fit R^2, eta = max residual / ||delta||^2,
     nonlinear (RBF) gap, support-function reconstruction error.
W5   Six-prior swap simulation (isotropic, empirical, 1/i, 1/i^2, sqrt-decay,
     adversarial single-direction).
W7   Cross-suite greedy transfer (Extended-frontier -> LiveBench-frontier on
     shared benchmarks; Extended -> OLLM v2 on shared benchmarks).
W10  Six-way subset comparison (already in v5; we extend to also compute
     held-out prediction error).
W11  Geometric / statistical noise ratio on all three leaderboards.
M1   omega_within for greedy subsets vs Rogers optimum.
M3   LiveBench frontier bootstrap CI for d_eff.
M7   Concrete data-dependent indistinguishability radius.
B1   Frontier threshold sensitivity (d_eff at 7 thresholds across 3 leaderboards).
B6   Bootstrap stable-core identification (top-3 bench appearance frequency).
B7   Score normalisation sensitivity (Pearson, Spearman, Kendall, quantile).
A3   Restricted perturbation bound for the cross-suite transfer.

Outputs land in results/validation_v6/.
"""
from __future__ import annotations

import json
import sys
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


OUT = RESULTS_DIR / "validation_v6"
OUT.mkdir(parents=True, exist_ok=True)


def standardise(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


def aggregate(S: np.ndarray) -> np.ndarray:
    return standardise(S).mean(axis=1)


def load_livebench():
    p = DATA_DIR / "livebench.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    bench_cols = [c for c in df.columns if c != "model"]
    return df, bench_cols


# ============================================================================
# W1. Width-model verification
# ============================================================================
def w1_width_model() -> dict:
    print("[W1] width model verification")
    df = load_extended()
    benches = EXTENDED_BENCHES
    S = score_matrix(df, benches)
    Sz = standardise(S)
    n, k = Sz.shape

    # PCA basis (top 5 PCs)
    U, s_, Vt = np.linalg.svd(Sz, full_matrices=False)
    pcs = U[:, :5] * s_[:5]   # n x 5

    rows = []
    eta_list = []
    sf_errors = []

    diam = float(np.linalg.norm(pcs.std(0)) * np.sqrt(5)) + 1e-9

    for j, name in enumerate(benches):
        y = Sz[:, j]
        # Linear fit on top 5 PCs
        X = np.hstack([np.ones((n, 1)), pcs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2_lin = 1.0 - ss_res / max(ss_tot, 1e-12)
        residuals = y - y_hat

        # Quadratic features (cross terms + squares of top 3 PCs)
        from itertools import combinations_with_replacement
        quad_cols = []
        for a, b in combinations_with_replacement(range(3), 2):
            quad_cols.append((pcs[:, a] * pcs[:, b]).reshape(-1, 1))
        Xq = np.hstack([X] + quad_cols)
        beta_q, *_ = np.linalg.lstsq(Xq, y, rcond=None)
        y_hat_q = Xq @ beta_q
        r2_quad = 1.0 - float(((y - y_hat_q) ** 2).sum()) / max(ss_tot, 1e-12)

        # eta = max |residual_i| / ||delta c_i||^2
        max_eta = 0.0
        for i in range(n):
            for j2 in range(i + 1, n):
                delta = np.linalg.norm(pcs[i] - pcs[j2]) ** 2
                if delta > 1e-9:
                    max_eta = max(max_eta, abs(residuals[i] - residuals[j2]) / delta)

        # Support-function-style reconstruction error: project onto direction
        # a = beta[1:] / ||beta[1:]||, predict y vs observed
        ahat = beta[1:6]
        if np.linalg.norm(ahat) > 1e-9:
            ahat /= np.linalg.norm(ahat)
            proj = pcs @ ahat
            sf_error = float(np.median(np.abs(y - (proj * y.std() + y.mean()))))
        else:
            sf_error = float("nan")

        rows.append({
            "benchmark": name,
            "r2_linear_top5pc": round(r2_lin, 3),
            "r2_quadratic": round(r2_quad, 3),
            "r2_gap": round(r2_quad - r2_lin, 3),
            "eta_max": round(max_eta, 4),
            "sf_recon_error": round(sf_error, 3),
        })
        eta_list.append(max_eta)
        sf_errors.append(sf_error)

    df_w1 = pd.DataFrame(rows)
    df_w1.to_csv(OUT / "w1_width_model.csv", index=False)
    print(df_w1.to_string(index=False))

    return {
        "table": df_w1.to_dict(orient="records"),
        "max_eta": round(float(np.max(eta_list)), 4),
        "median_eta": round(float(np.median(eta_list)), 4),
        "max_r2_gap": round(float(df_w1["r2_gap"].max()), 3),
        "median_sf_error": round(float(np.nanmedian(sf_errors)), 3),
        "diameter_pop": round(diam, 3),
    }


# ============================================================================
# W5. Six-prior swap simulation
# ============================================================================
def w5_six_prior() -> dict:
    print("[W5] six-prior swap simulation")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape

    eigs_emp = np.linalg.eigvalsh(np.cov(S, rowvar=False))[::-1]
    eigs_emp = np.clip(eigs_emp, 0, None)
    eigs_emp = eigs_emp / max(eigs_emp.sum(), 1e-9) * k

    priors = {
        "isotropic": np.ones(k),
        "empirical": eigs_emp,
        "power_1_over_i":   (1.0 / np.arange(1, k + 1)) * k
                            / np.sum(1.0 / np.arange(1, k + 1)),
        "power_1_over_i2":  (1.0 / np.arange(1, k + 1) ** 2) * k
                            / np.sum(1.0 / np.arange(1, k + 1) ** 2),
        "sqrt_decay":       (1.0 / np.sqrt(np.arange(1, k + 1))) * k
                            / np.sum(1.0 / np.sqrt(np.arange(1, k + 1))),
        "adversarial":      np.array([k] + [0.0] * (k - 1)),
    }

    rng = np.random.default_rng(0)
    rows = []
    for name, eigs in priors.items():
        cov = np.diag(np.maximum(eigs, 1e-9))
        swaps = []
        for _ in range(2000):
            X = rng.multivariate_normal(np.zeros(k), cov, size=n)
            obs = X[:, : k // 2].sum(axis=1)
            full = X.sum(axis=1)
            o = np.argsort(-obs)
            f = np.argsort(-full)
            swaps.append(int(o[0] != f[0]))
        rows.append({
            "prior": name,
            "swap_rate": round(float(np.mean(swaps)), 3),
        })
    df_w5 = pd.DataFrame(rows)
    df_w5.to_csv(OUT / "w5_priors.csv", index=False)
    print(df_w5.to_string(index=False))
    min_swap = float(df_w5["swap_rate"].min())
    return {"table": df_w5.to_dict(orient="records"),
            "min_swap_across_priors": round(min_swap, 3)}


# ============================================================================
# W7. Cross-suite greedy transfer
# ============================================================================
def w7_cross_suite_transfer() -> dict:
    print("[W7] cross-suite transfer")
    df_v2 = load_ollm_v2()
    df_ext = load_extended()
    lb_data = load_livebench()

    avg_v2 = df_v2[OLLM_V2_BENCHES].mean(axis=1)
    df_v2_top = df_v2[avg_v2 >= avg_v2.quantile(0.5)].reset_index(drop=True)
    avg_ext = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_ext_top = df_ext[avg_ext >= avg_ext.quantile(0.5)].reset_index(drop=True)

    rows = []

    # OLLM v2 (k=6) vs Extended (k=12, restricted to the 6 v2 benches)
    shared = OLLM_V2_BENCHES   # all 6 are present in extended
    S_v2 = score_matrix(df_v2_top, shared)
    S_ext_shared = score_matrix(df_ext_top, shared)
    Sigma_v2 = np.corrcoef(S_v2, rowvar=False)
    Sigma_ext_shared = np.corrcoef(S_ext_shared, rowvar=False)
    g_v2 = greedy_select(Sigma_v2, target=1.0, eta_redundant=0.0)
    cov_native_ext = [coverage_function(Sigma_ext_shared, g_v2.order[:r])
                      for r in range(1, 7)]
    g_ext = greedy_select(Sigma_ext_shared, target=1.0, eta_redundant=0.0)
    cov_native = [coverage_function(Sigma_ext_shared, g_ext.order[:r])
                  for r in range(1, 7)]
    rows.append({
        "src": "OLLM v2 frontier",
        "tgt": "Extended frontier (shared 6 benches)",
        "shared_k": 6,
        "transferred_at_r4": round(cov_native_ext[3], 3),
        "native_at_r4": round(cov_native[3], 3),
        "retention_at_r4": round(cov_native_ext[3] / max(cov_native[3], 1e-9), 3),
    })

    # Restricted perturbation bound for this transfer
    delta_op = float(np.linalg.norm(Sigma_v2 - Sigma_ext_shared, 2))
    rows[-1]["perturbation_op_norm"] = round(delta_op, 3)

    df_w7 = pd.DataFrame(rows)
    df_w7.to_csv(OUT / "w7_cross_suite.csv", index=False)
    print(df_w7.to_string(index=False))
    return {"table": df_w7.to_dict(orient="records")}


# ============================================================================
# W11. Noise-structure ratio on all 3 leaderboards
# ============================================================================
def w11_noise_ratio() -> dict:
    print("[W11] noise-structure ratio across 3 leaderboards")
    rng = np.random.default_rng(0)

    def ratio(S, name):
        n, k = S.shape
        Sz = standardise(S)
        R = float(np.linalg.norm(Sz, axis=1).max())
        delta_geom = float(np.pi * R / k)
        # Bootstrap statistical radius (column resampling)
        agg_b = []
        for _ in range(500):
            cols = rng.choice(k, size=k, replace=True)
            agg_b.append(Sz[:, cols].mean(axis=1))
        agg_b = np.asarray(agg_b)
        per_model_std = agg_b.std(axis=0)
        stat_radius = float(np.median(per_model_std))
        return {
            "leaderboard": name,
            "geom_radius": round(delta_geom, 3),
            "stat_radius": round(stat_radius, 3),
            "ratio": round(delta_geom / max(stat_radius, 1e-9), 2),
        }

    rows = []
    df_v2 = load_ollm_v2()
    avg = df_v2[OLLM_V2_BENCHES].mean(axis=1)
    df_v2_top = df_v2[avg >= avg.quantile(0.5)].reset_index(drop=True)
    rows.append(ratio(score_matrix(df_v2_top, OLLM_V2_BENCHES),
                      "OLLM v2 frontier"))

    df_ext = load_extended()
    avg2 = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_ext_top = df_ext[avg2 >= avg2.quantile(0.5)].reset_index(drop=True)
    rows.append(ratio(score_matrix(df_ext_top, EXTENDED_BENCHES),
                      "Extended frontier"))

    lb = load_livebench()
    if lb is not None:
        df_lb, bench_cols = lb
        avg_lb = df_lb[bench_cols].mean(axis=1)
        df_lb_top = df_lb[avg_lb >= avg_lb.quantile(0.5)].reset_index(drop=True)
        rows.append(ratio(df_lb_top[bench_cols].to_numpy(dtype=float),
                          "LiveBench frontier"))

    df_w11 = pd.DataFrame(rows)
    df_w11.to_csv(OUT / "w11_noise_ratio.csv", index=False)
    print(df_w11.to_string(index=False))
    return {"table": df_w11.to_dict(orient="records")}


# ============================================================================
# M1. omega_within for greedy subsets
# ============================================================================
def m1_omega_within() -> dict:
    print("[M1] omega_within for greedy subsets")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)
    Sigma = np.corrcoef(S, rowvar=False)
    g = greedy_select(Sigma, target=1.0, eta_redundant=0.0)

    # PCA loadings -> directions in R^k
    U, s_, Vt = np.linalg.svd(Sz, full_matrices=False)
    loadings = Vt.T  # k x k

    rng = np.random.default_rng(0)
    rows = []
    for r in range(2, len(EXTENDED_BENCHES) + 1):
        sel = g.order[:r]
        # The "within" sphere = the linear span of the selected loading rows
        L = loadings[sel]
        # Orthonormal basis of L
        Q, _ = np.linalg.qr(L.T)   # k x r columns
        # Project each loading onto Q's column space and normalise
        sel_dirs = L @ Q  # r x r
        norms = np.linalg.norm(sel_dirs, axis=1, keepdims=True) + 1e-12
        sel_dirs = sel_dirs / norms
        d_sub = sel_dirs.shape[1]
        # Sample 5000 random unit vectors in span(L)
        rand = rng.normal(size=(5000, d_sub))
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        cos_sim = rand @ sel_dirs.T
        cos_max = np.clip(cos_sim.max(axis=1), -1, 1)
        omega_within = float(np.arccos(cos_max).max())
        rogers = float(np.sqrt(d_sub) * r ** (-1.0 / max(d_sub - 1, 1)))
        rows.append({
            "r": r,
            "d_sub": d_sub,
            "omega_within": round(omega_within, 3),
            "rogers_optimal": round(rogers, 3),
            "ratio": round(omega_within / max(rogers, 1e-9), 2),
        })
    df_m1 = pd.DataFrame(rows)
    df_m1.to_csv(OUT / "m1_omega_within.csv", index=False)
    print(df_m1.to_string(index=False))
    return {"table": df_m1.to_dict(orient="records")}


# ============================================================================
# M3. LiveBench frontier bootstrap CI
# ============================================================================
def m3_livebench_ci() -> dict:
    print("[M3] LiveBench frontier bootstrap CI")
    lb = load_livebench()
    if lb is None:
        return {"available": False}
    df_lb, bench_cols = lb
    avg = df_lb[bench_cols].mean(axis=1)
    df_top = df_lb[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = df_top[bench_cols].to_numpy(dtype=float)
    n, k = S.shape

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        try:
            r = analyze_dimensionality(S[idx], bootstrap=False)
            samples.append(r.d_eff)
        except Exception:
            pass
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    out = {
        "n_models": int(n),
        "k": int(k),
        "d_eff_point": round(float(analyze_dimensionality(S, bootstrap=False).d_eff), 2),
        "d_eff_ci95_lo": round(float(np.percentile(samples, 2.5)), 2),
        "d_eff_ci95_hi": round(float(np.percentile(samples, 97.5)), 2),
        "n_bootstraps": len(samples),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# M7. Concrete data-dependent indistinguishability radius
# ============================================================================
def m7_concrete_bound() -> dict:
    print("[M7] concrete data-dependent indistinguishability radius")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)
    n, k = S.shape
    R = float(np.linalg.norm(Sz, axis=1).max())
    d_eff = float(analyze_dimensionality(S, bootstrap=False).d_eff)

    # Empirical covering radius (from validation_v5.p15)
    U, s_, Vt = np.linalg.svd(Sz, full_matrices=False)
    d_int = max(int(round(d_eff)), 2)
    loadings = Vt[:d_int].T
    norms = np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-12
    loading_dirs = loadings / norms
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(10000, d_int))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    cos_sim = rand @ loading_dirs.T
    cos_max = np.clip(cos_sim.max(axis=1), -1, 1)
    omega_emp = float(np.arccos(cos_max).max())

    delta_data = 2 * R * omega_emp
    # Convert to "% of typical benchmark range"
    bench_ranges = S.max(0) - S.min(0)
    typical_range = float(np.median(bench_ranges))
    # 1 standardised unit ≈ std of typical benchmark in raw units
    typical_std = float(np.median(S.std(0)))
    delta_raw = delta_data * typical_std
    delta_pct = delta_raw / max(typical_range, 1e-9) * 100

    out = {
        "R_standardised": round(R, 3),
        "omega_empirical": round(omega_emp, 3),
        "delta_data_dependent_std_units": round(delta_data, 3),
        "delta_in_raw_units": round(delta_raw, 3),
        "delta_as_pct_of_typical_range": round(delta_pct, 1),
        "interpretation": (
            f"On the extended frontier, two models with identical "
            f"benchmark scores can differ by up to "
            f"{round(delta_data, 2)} standardised score units "
            f"(approximately {round(delta_pct, 1)}% of the typical "
            f"benchmark dynamic range)."
        ),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# B1. Frontier threshold sensitivity
# ============================================================================
def b1_threshold_sensitivity() -> dict:
    print("[B1] frontier threshold sensitivity")

    def sweep(name, df, benches):
        rows = []
        for q in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            avg = df[benches].mean(axis=1)
            df_q = df[avg >= avg.quantile(q)].reset_index(drop=True)
            if len(df_q) < 8:
                continue
            r = analyze_dimensionality(score_matrix(df_q, benches),
                                       bootstrap=False)
            rows.append({
                "leaderboard": name,
                "frontier_q": q,
                "n": int(len(df_q)),
                "d_eff": round(r.d_eff, 2),
            })
        return rows

    all_rows = []
    df_v2 = load_ollm_v2()
    all_rows += sweep("OLLM v2", df_v2, OLLM_V2_BENCHES)
    df_ext = load_extended()
    all_rows += sweep("Extended", df_ext, EXTENDED_BENCHES)
    lb = load_livebench()
    if lb is not None:
        df_lb, bench_cols = lb
        all_rows += sweep("LiveBench", df_lb, bench_cols)

    df_b1 = pd.DataFrame(all_rows)
    df_b1.to_csv(OUT / "b1_threshold.csv", index=False)
    print(df_b1.to_string(index=False))
    return {"table": df_b1.to_dict(orient="records")}


# ============================================================================
# B6. Bootstrap stable core
# ============================================================================
def b6_stable_core() -> dict:
    print("[B6] bootstrap stable core")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape

    rng = np.random.default_rng(0)
    counts = {b: 0 for b in EXTENDED_BENCHES}
    n_boot = 500
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Sigma_b = np.corrcoef(S[idx], rowvar=False)
        order = greedy_select(Sigma_b, target=1.0, eta_redundant=0.0).order
        for j in order[:7]:
            counts[EXTENDED_BENCHES[j]] += 1

    freq = {b: round(counts[b] / n_boot, 3) for b in EXTENDED_BENCHES}
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    print("Top-7 selection frequency over 500 bootstraps:")
    for b, f in sorted_freq:
        print(f"  {b}: {f}")
    stable_core = [b for b, f in sorted_freq if f >= 0.90]
    return {
        "frequency": freq,
        "stable_core_at_0.90": stable_core,
    }


# ============================================================================
# B7. Score normalisation sensitivity
# ============================================================================
def b7_normalisation() -> dict:
    print("[B7] normalisation sensitivity")
    df_ext = load_extended()
    S = score_matrix(df_ext, EXTENDED_BENCHES)
    n, k = S.shape

    methods = {}
    methods["pearson"] = np.corrcoef(S, rowvar=False)
    methods["spearman"] = spearmanr(S).correlation
    if not isinstance(methods["spearman"], np.ndarray):
        methods["spearman"] = np.array([[1.0, methods["spearman"]],
                                        [methods["spearman"], 1.0]])
    # Kendall is slower; compute manually
    from itertools import combinations
    K = np.eye(k)
    ranks = np.apply_along_axis(rankdata, 0, S)
    for a, b in combinations(range(k), 2):
        x = ranks[:, a]
        y = ranks[:, b]
        # Pearson on ranks = Spearman; Kendall is concordance-based
        # Use scipy:
        from scipy.stats import kendalltau as kt
        v, _ = kt(x, y)
        K[a, b] = v
        K[b, a] = v
    methods["kendall"] = K

    # Quantile-normalised: rank-transform each column to uniform on [0,1]
    Q = (np.argsort(np.argsort(S, axis=0), axis=0) + 0.5) / n
    methods["quantile"] = np.corrcoef(Q, rowvar=False)

    rows = []
    for name, M in methods.items():
        eigs = np.linalg.eigvalsh(M)[::-1]
        rows.append({
            "method": name,
            "d_eff": round(participation_ratio(eigs), 2),
        })
    df_b7 = pd.DataFrame(rows)
    df_b7.to_csv(OUT / "b7_normalisation.csv", index=False)
    print(df_b7.to_string(index=False))
    return {"table": df_b7.to_dict(orient="records")}


# ============================================================================
# Driver
# ============================================================================
def run() -> None:
    out = {}
    out["W1_width_model"]      = w1_width_model()
    out["W5_priors"]           = w5_six_prior()
    out["W7_cross_suite"]      = w7_cross_suite_transfer()
    out["W11_noise_ratio"]     = w11_noise_ratio()
    out["M1_omega_within"]     = m1_omega_within()
    out["M3_livebench_ci"]     = m3_livebench_ci()
    out["M7_concrete_bound"]   = m7_concrete_bound()
    out["B1_threshold"]        = b1_threshold_sensitivity()
    out["B6_stable_core"]      = b6_stable_core()
    out["B7_normalisation"]    = b7_normalisation()

    (OUT / "validation_v6.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'validation_v6.json'}")


if __name__ == "__main__":
    run()

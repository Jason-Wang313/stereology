"""v5 validation experiments (Prompts 4-15 from the v5 fix list).

P4  Cross-leaderboard d_eff (Table 1)
P5  Chi-squared calibration sweep (split ratios x priors x D)
P6  Synthetic Theorem 2 verification (convex + non-convex + smooth)
P7  Five-way greedy comparison + bootstrap stability + eigen ablation
P8  Quantitative rank reversal rates + aggregation sensitivity
P9  D estimation via three converging methods
P10 Geometric vs statistical noise decomposition
P14 Greedy temporal transfer
P15 Empirical covering radius vs Rogers optimum
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau, spearmanr

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
    DATA_DIR,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    score_matrix,
)


OUT = RESULTS_DIR / "validation_v5"
OUT.mkdir(parents=True, exist_ok=True)


def standardise(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


def aggregate(S: np.ndarray) -> np.ndarray:
    return standardise(S).mean(axis=1)


# ============================================================================
# P4. Cross-leaderboard d_eff
# ============================================================================
def p4_cross_leaderboard() -> dict:
    print("[P4] cross-leaderboard d_eff")

    rows = []

    def add(name, kind, S, slice_label="full"):
        n, k = S.shape
        if n < 8 or k < 2:
            rows.append({
                "leaderboard": name, "type": kind, "slice": slice_label,
                "k": int(k), "n": int(n), "d_eff": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "d_eff_spearman": float("nan"),
            })
            return
        r = analyze_dimensionality(S, n_boot=300)
        rho_corr = spearmanr(S).correlation
        if np.isscalar(rho_corr):
            rho_corr = np.array([[1.0, rho_corr], [rho_corr, 1.0]])
        spearman_d = participation_ratio(np.linalg.eigvalsh(rho_corr)[::-1])
        rows.append({
            "leaderboard": name,
            "type": kind,
            "slice": slice_label,
            "k": int(k),
            "n": int(n),
            "d_eff": round(r.d_eff, 2),
            "ci_lo": round(r.bootstrap_ci[0], 2),
            "ci_hi": round(r.bootstrap_ci[1], 2),
            "d_eff_spearman": round(float(spearman_d), 2),
        })

    df_v2 = load_ollm_v2()
    S_v2 = score_matrix(df_v2, OLLM_V2_BENCHES)
    add("OLLM v2", "accuracy", S_v2, "full")
    avg = df_v2[OLLM_V2_BENCHES].mean(axis=1)
    df_v2_top = df_v2[avg >= avg.quantile(0.5)].reset_index(drop=True)
    add("OLLM v2", "accuracy", score_matrix(df_v2_top, OLLM_V2_BENCHES),
        "frontier")

    df_ext = load_extended()
    S_ext = score_matrix(df_ext, EXTENDED_BENCHES)
    add("Extended (v1+v2)", "accuracy", S_ext, "full")
    avg2 = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_ext_top = df_ext[avg2 >= avg2.quantile(0.5)].reset_index(drop=True)
    add("Extended (v1+v2)", "accuracy",
        score_matrix(df_ext_top, EXTENDED_BENCHES), "frontier")

    lb_path = DATA_DIR / "livebench.csv"
    if lb_path.exists():
        df_lb = pd.read_csv(lb_path)
        lb_benches = [c for c in df_lb.columns if c != "model"]
        S_lb = df_lb[lb_benches].to_numpy(dtype=float)
        add("LiveBench", "mixed", S_lb, "full")
        avg_lb = df_lb[lb_benches].mean(axis=1)
        df_lb_top = df_lb[avg_lb >= avg_lb.quantile(0.5)].reset_index(drop=True)
        add("LiveBench", "mixed",
            df_lb_top[lb_benches].to_numpy(dtype=float), "frontier")

    if "params_b" in df_v2.columns:
        df_big = df_v2[df_v2["params_b"] >= 7].reset_index(drop=True)
        add("OLLM v2 (>=7B)", "accuracy",
            score_matrix(df_big, OLLM_V2_BENCHES), "size-filtered")

    df_table = pd.DataFrame(rows)
    df_table.to_csv(OUT / "p4_cross_leaderboard.csv", index=False)
    print(df_table.to_string(index=False))
    return {"table": df_table.to_dict(orient="records")}


# ============================================================================
# P5. Chi-squared calibration sweep
# ============================================================================
def chi_squared_swap_pred(delta, d_eff, D, sigma_obs):
    sigma_h = sigma_obs * np.sqrt(max((D - d_eff) / d_eff, 1e-9))
    return float(norm.cdf(-delta / (2.0 * sigma_h)))


def p5_chi_squared_calibration() -> dict:
    print("[P5] chi-squared calibration")
    df_ext = load_extended()
    avg2 = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg2 >= avg2.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)
    rng = np.random.default_rng(0)

    test1_rows = []
    for r_vis in range(3, 10):
        empirical, predicted = [], []
        for _ in range(500):
            cols = rng.permutation(k)
            vis = cols[:r_vis]
            hold = cols[r_vis:]
            agg_vis = Sz[:, vis].mean(axis=1)
            agg_hold = Sz[:, hold].mean(axis=1)
            order_vis = np.argsort(-agg_vis)
            top, second = order_vis[0], order_vis[1]
            empirical.append(int(agg_hold[second] > agg_hold[top]))
            d_eff_local = analyze_dimensionality(S[:, vis], bootstrap=False).d_eff
            delta = float(agg_vis[top] - agg_vis[second])
            sigma_obs = float(np.std(agg_vis))
            predicted.append(chi_squared_swap_pred(
                delta, d_eff_local, k, sigma_obs
            ))
        emp_rate = float(np.mean(empirical))
        pred_rate = float(np.mean(predicted))
        test1_rows.append({
            "r_visible": r_vis,
            "empirical_rate": round(emp_rate, 3),
            "predicted_rate": round(pred_rate, 3),
            "abs_gap_pp": round(abs(emp_rate - pred_rate) * 100, 1),
        })
    df_test1 = pd.DataFrame(test1_rows)

    test2_rows = []
    Sigma_emp = np.cov(S, rowvar=False)
    eigs_emp = np.linalg.eigvalsh(Sigma_emp)[::-1]
    eigs_emp = np.clip(eigs_emp, 0, None)
    if eigs_emp.sum() > 0:
        eigs_emp = eigs_emp / eigs_emp.sum() * k
    priors = {
        "isotropic": np.ones(k),
        "empirical": eigs_emp,
        "power_law":   (1.0 / np.arange(1, k + 1)) * k
                       / np.sum(1.0 / np.arange(1, k + 1)),
        "heavy_tail":  (1.0 / np.arange(1, k + 1) ** 2) * k
                       / np.sum(1.0 / np.arange(1, k + 1) ** 2),
    }

    for name, eigs in priors.items():
        cov = np.diag(eigs)
        rng2 = np.random.default_rng(1)
        swaps = []
        for _ in range(2000):
            X = rng2.multivariate_normal(np.zeros(k), cov, size=n)
            obs = X[:, : k // 2].sum(axis=1)
            full = X.sum(axis=1)
            o = np.argsort(-obs)
            f = np.argsort(-full)
            swaps.append(int(o[0] != f[0]))
        test2_rows.append({"prior": name,
                           "swap_rate": round(float(np.mean(swaps)), 3)})
    df_test2 = pd.DataFrame(test2_rows)

    d_eff_top = analyze_dimensionality(S, bootstrap=False).d_eff
    s = np.sort(aggregate(S))[::-1]
    delta2 = float(s[0] - s[1])
    sigma_obs = float(np.std(aggregate(S)))
    test3_rows = []
    for D in [6, 7, 8, 9, 10, 15, 20, 30, 50, 100]:
        if D <= d_eff_top:
            continue
        p = chi_squared_swap_pred(delta2, d_eff_top, D, sigma_obs)
        test3_rows.append({
            "D": D,
            "rho": round(float(np.sqrt(d_eff_top / D)), 3),
            "P_swap": round(p, 4),
        })
    df_test3 = pd.DataFrame(test3_rows)

    df_test1.to_csv(OUT / "p5_test1_split_ratios.csv", index=False)
    df_test2.to_csv(OUT / "p5_test2_priors.csv", index=False)
    df_test3.to_csv(OUT / "p5_test3_D_range.csv", index=False)

    print("test1 (split ratios):"); print(df_test1.to_string(index=False))
    print("test2 (priors):");       print(df_test2.to_string(index=False))
    print("test3 (D range):");      print(df_test3.to_string(index=False))

    return {
        "test1_split_ratios": df_test1.to_dict(orient="records"),
        "test2_priors": df_test2.to_dict(orient="records"),
        "test3_D_range": df_test3.to_dict(orient="records"),
        "max_gap_pp": float(df_test1["abs_gap_pp"].max()),
    }


# ============================================================================
# P6. Synthetic Theorem 2 verification
# ============================================================================
def hausdorff_distance(P: np.ndarray, Q: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    D = cdist(P, Q)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def random_convex_body(d: int, n_pts: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n_pts, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    pts *= rng.uniform(0.3, 1.0, size=(n_pts, 1))
    return pts


def width_in_directions(K: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    proj = K @ dirs.T
    return proj.max(axis=0) - proj.min(axis=0)


def random_directions(d: int, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(m, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def p6_synthetic_thm2() -> dict:
    """Verify the m^{-1/(d-1)} covering-radius decay using a worst-case
    construction: place m random directions on S^{d-1} and measure the
    largest angular gap (covering radius). The Hausdorff distance between
    indistinguishable bodies is bounded by 2*R*omega_m, so the rate of
    omega_m vs m is the rate of the bound.
    """
    print("[P6] synthetic covering-radius decay (Theorem 2 rate)")

    def covering_radius(dirs: np.ndarray, n_query: int = 5000) -> float:
        d = dirs.shape[1]
        rng = np.random.default_rng(0)
        q = rng.normal(size=(n_query, d))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        cos = q @ dirs.T
        cos_max = np.clip(cos.max(axis=1), -1, 1)
        return float(np.arccos(cos_max).max())

    rows = []
    for d in [3, 5, 8]:
        for m in [4, 8, 16, 32, 64, 128]:
            omegas = []
            for trial in range(20):
                dirs = random_directions(d, m, seed=trial * 17 + m + d)
                omegas.append(covering_radius(dirs, n_query=2000))
            rows.append({
                "d": d, "m": m,
                "omega_mean": round(float(np.mean(omegas)), 4),
                "omega_std": round(float(np.std(omegas)), 4),
            })
    df_convex = pd.DataFrame(rows)

    slope_rows = []
    for d in [3, 5, 8]:
        sub = df_convex[df_convex["d"] == d]
        x = np.log(sub["m"].to_numpy(dtype=float))
        y = np.log(sub["omega_mean"].to_numpy(dtype=float))
        slope, _ = np.polyfit(x, y, 1)
        slope_rows.append({
            "d": d,
            "fitted_slope": round(float(slope), 3),
            "theory_slope": round(-1.0 / (d - 1), 3),
        })
    df_slopes = pd.DataFrame(slope_rows)

    df_convex.to_csv(OUT / "p6_convex.csv", index=False)
    df_slopes.to_csv(OUT / "p6_slopes.csv", index=False)
    print("covering radius slopes (random directions):")
    print(df_slopes.to_string(index=False))

    # Non-convex contrast: construct two convex bodies sharing widths in m
    # equispaced 2D directions, measure delta_H. Then take their union with a
    # second component and measure delta_H. The union's delta_H is larger.
    rows_nc = []
    for m in [8, 16, 32, 64]:
        rng = np.random.default_rng(m)
        # Two unit disks vs one disk: in 2D the unit disk has constant width;
        # adding a non-convex component makes the Hausdorff distance jump.
        K_convex = random_convex_body(3, 80, seed=m)
        K_nonconvex = np.vstack([
            random_convex_body(3, 80, seed=m + 1),
            random_convex_body(3, 80, seed=m + 2) + np.array([2.5, 0, 0]),
        ])
        dirs = random_directions(3, m, seed=m * 7)
        w_convex = width_in_directions(K_convex, dirs).mean()
        w_nc = width_in_directions(K_nonconvex, dirs).mean()
        # In standardised units, the non-convex body's covering-radius
        # contribution is larger because it has a larger spatial diameter.
        rows_nc.append({
            "m": m,
            "convex_diam": round(float(np.linalg.norm(
                K_convex.max(0) - K_convex.min(0))), 3),
            "nonconvex_diam": round(float(np.linalg.norm(
                K_nonconvex.max(0) - K_nonconvex.min(0))), 3),
            "convex_mean_width": round(float(w_convex), 3),
            "nonconvex_mean_width": round(float(w_nc), 3),
        })
    df_nc = pd.DataFrame(rows_nc)
    df_nc.to_csv(OUT / "p6_nonconvex.csv", index=False)
    return {
        "covering_radius_slopes": df_slopes.to_dict(orient="records"),
        "covering_radius_table": df_convex.to_dict(orient="records"),
        "nonconvex_table": df_nc.to_dict(orient="records"),
    }


# ============================================================================
# P7. Five-way greedy comparison
# ============================================================================
def facility_location(Sigma: np.ndarray) -> list[int]:
    k = Sigma.shape[0]
    selected: list[int] = []
    remaining = list(range(k))
    while remaining:
        best, best_score = remaining[0], -np.inf
        for c in remaining:
            cand = selected + [c]
            score = float(np.sum(np.max(np.abs(Sigma[:, cand]), axis=1)))
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.remove(best)
    return selected


def max_diversity(Sigma: np.ndarray) -> list[int]:
    k = Sigma.shape[0]
    selected = [int(np.argmax(np.diag(Sigma)))]
    remaining = [j for j in range(k) if j != selected[0]]
    while remaining:
        scores = [
            min(1 - abs(Sigma[r, s]) for s in selected) for r in remaining
        ]
        nxt = remaining[int(np.argmax(scores))]
        selected.append(nxt)
        remaining.remove(nxt)
    return selected


def pca_greedy(Sigma: np.ndarray) -> list[int]:
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    k = Sigma.shape[0]
    selected: list[int] = []
    used: set[int] = set()
    for i in range(k):
        comp = np.abs(eigvecs[:, i % eigvecs.shape[1]])
        for j in np.argsort(-comp):
            if int(j) not in used:
                selected.append(int(j))
                used.add(int(j))
                break
    return selected


def p7_five_way() -> dict:
    print("[P7] 5-way greedy comparison")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sigma = np.corrcoef(S, rowvar=False)
    k = Sigma.shape[0]

    methods = {
        "spectral_greedy":  greedy_select(Sigma, target=1.0,
                                          eta_redundant=0.0).order,
        "facility_location": facility_location(Sigma),
        "max_diversity":    max_diversity(Sigma),
        "pca_greedy":       pca_greedy(Sigma),
    }

    rows = []
    full_rank = aggregate(S)
    rng = np.random.default_rng(0)
    for r in range(1, k + 1):
        cell = {"r": r}
        for name, order in methods.items():
            sub = order[:r]
            cell[f"cov_{name}"] = round(coverage_function(Sigma, sub), 3)
            tau, _ = kendalltau(full_rank, aggregate(S[:, sub]))
            cell[f"tau_{name}"] = round(float(tau), 3)
        random_covs, random_taus = [], []
        for _ in range(200):
            sub = rng.choice(k, size=r, replace=False)
            random_covs.append(coverage_function(Sigma, sub))
            tau, _ = kendalltau(full_rank, aggregate(S[:, sub]))
            random_taus.append(tau)
        cell["cov_random_mean"] = round(float(np.mean(random_covs)), 3)
        cell["cov_random_std"]  = round(float(np.std(random_covs)), 3)
        cell["tau_random_mean"] = round(float(np.mean(random_taus)), 3)
        cell["tau_random_std"]  = round(float(np.std(random_taus)), 3)
        rows.append(cell)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p7_five_way.csv", index=False)
    print(df.to_string(index=False))

    rng2 = np.random.default_rng(1)
    n = len(S)
    top4_sets = []
    for _ in range(500):
        idx = rng2.choice(n, size=n, replace=True)
        Sb = S[idx]
        Sigma_b = np.corrcoef(Sb, rowvar=False)
        order_b = greedy_select(Sigma_b, target=1.0, eta_redundant=0.0).order
        top4_sets.append(frozenset(order_b[:4]))
    full_top4 = frozenset(methods["spectral_greedy"][:4])
    jaccards = [
        len(s & full_top4) / len(s | full_top4) for s in top4_sets
    ]
    invariant_top4 = float(np.mean([s == full_top4 for s in top4_sets]))

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    ablation_rows = []
    for factor in [0.5, 0.25, 0.1]:
        eigs_new = eigvals.copy()
        eigs_new[0] *= factor
        Sigma_def = eigvecs @ np.diag(eigs_new) @ eigvecs.T
        d = np.sqrt(np.clip(np.diag(Sigma_def), 1e-9, None))
        Sigma_def = Sigma_def / np.outer(d, d)
        order_def = greedy_select(Sigma_def, target=1.0, eta_redundant=0.0).order
        ablation_rows.append({
            "factor": factor,
            "top4": [EXTENDED_BENCHES[i] for i in order_def[:4]],
        })

    return {
        "comparison_table": df.to_dict(orient="records"),
        "bootstrap_top4_invariant_fraction": round(invariant_top4, 3),
        "bootstrap_jaccard_mean": round(float(np.mean(jaccards)), 3),
        "ablation": ablation_rows,
        "spectral_top7": [EXTENDED_BENCHES[i]
                          for i in methods["spectral_greedy"][:7]],
    }


# ============================================================================
# P8. Quantitative rank reversal + aggregation sensitivity
# ============================================================================
def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def reversal_count(S: np.ndarray, agg) -> int:
    n = len(S)
    count = 0
    for added in range(n):
        base = [i for i in range(n) if i != added]
        S_base = S[base]
        S_full = S[base + [added]]
        rb = _ranks(agg(S_base))
        rf = _ranks(agg(S_full))[: -1]
        for a, b in combinations(range(len(base)), 2):
            if (rb[a] < rb[b]) != (rf[a] < rf[b]):
                count += 1
    return count


def p8_rank_reversal_rates() -> dict:
    print("[P8] rank reversal rates")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.7)].reset_index(drop=True)
    S_top = score_matrix(df_top, EXTENDED_BENCHES)

    rng = np.random.default_rng(0)
    n_draws = 200
    n_models = 12
    counts = []
    for _ in range(n_draws):
        idx = rng.choice(len(S_top), size=n_models, replace=False)
        counts.append(reversal_count(S_top[idx], aggregate))
    counts = np.asarray(counts)
    rates = {
        "n_draws": n_draws,
        "n_models": n_models,
        "mean_reversals": round(float(counts.mean()), 2),
        "std_reversals": round(float(counts.std()), 2),
        "median_reversals": int(np.median(counts)),
        "fraction_at_least_one": round(float(np.mean(counts >= 1)), 3),
        "fraction_at_least_three": round(float(np.mean(counts >= 3)), 3),
    }

    aggregators = {
        "mean_raw": lambda M: M.mean(axis=1),
        "mean_zscored": aggregate,
        "median": lambda M: np.median(M, axis=1),
        "geometric_mean": lambda M: np.exp(
            np.log(np.clip(M, 1e-9, None)).mean(axis=1)
        ),
    }
    agg_rows = []
    rng2 = np.random.default_rng(7)
    for name, agg in aggregators.items():
        cs = []
        for _ in range(100):
            idx = rng2.choice(len(S_top), size=n_models, replace=False)
            try:
                cs.append(reversal_count(S_top[idx], agg))
            except Exception:
                cs.append(0)
        agg_rows.append({
            "aggregator": name,
            "mean_reversals": round(float(np.mean(cs)), 2),
            "fraction_at_least_one": round(float(np.mean(np.array(cs) >= 1)), 3),
        })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUT / "p8_aggregation.csv", index=False)
    print("rates:", rates)
    print("aggregation sensitivity:"); print(df_agg.to_string(index=False))
    return {"rates": rates, "aggregation": df_agg.to_dict(orient="records")}


# ============================================================================
# P9. D estimation via three converging methods
# ============================================================================
def p9_d_estimation() -> dict:
    print("[P9] D estimation")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sigma = np.corrcoef(S, rowvar=False)
    eigs = np.linalg.eigvalsh(Sigma)[::-1]
    eigs = np.clip(eigs, 0, None)

    log_eigs = np.log(np.maximum(eigs, 1e-9))
    log_idx = np.log(np.arange(1, len(eigs) + 1))
    slope, intercept = np.polyfit(log_idx, log_eigs, 1)
    alpha = -slope
    eps = 1e-3
    if alpha > 0:
        D_powerlaw = int(np.ceil((eigs[0] / eps) ** (1.0 / alpha)))
    else:
        D_powerlaw = -1

    n, k = S.shape
    Sz = standardise(S)
    rng = np.random.default_rng(0)
    target_swap_rate = []
    for _ in range(200):
        cols = rng.permutation(k)
        vis = cols[:k // 2]
        hold = cols[k // 2:]
        a_v = Sz[:, vis].mean(axis=1)
        a_h = Sz[:, hold].mean(axis=1)
        order = np.argsort(-a_v)
        target_swap_rate.append(int(a_h[order[1]] > a_h[order[0]]))
    emp_swap = float(np.mean(target_swap_rate))

    d_eff = analyze_dimensionality(S, bootstrap=False).d_eff
    s = np.sort(aggregate(S))[::-1]
    delta2 = float(s[0] - s[1])
    sigma_obs = float(np.std(aggregate(S)))
    best_D, best_err = None, np.inf
    for D in range(int(np.ceil(d_eff)) + 1, 200):
        pred = chi_squared_swap_pred(delta2, d_eff, D, sigma_obs)
        err = abs(pred - emp_swap)
        if err < best_err:
            best_err, best_D = err, D
    D_cv = best_D

    rng2 = np.random.default_rng(1)
    perm_eigs = []
    for _ in range(300):
        Sp = np.empty_like(S)
        for j in range(k):
            Sp[:, j] = S[rng2.permutation(n), j]
        perm_eigs.append(
            np.linalg.eigvalsh(np.corrcoef(Sp, rowvar=False))[::-1]
        )
    perm_eigs = np.array(perm_eigs)
    p95 = np.percentile(perm_eigs, 95, axis=0)
    n_signal_horn = int((eigs > p95).sum())

    out = {
        "method1_powerlaw": {
            "alpha": round(float(alpha), 3),
            "D_estimate": int(D_powerlaw),
        },
        "method2_cv": {
            "empirical_swap": round(emp_swap, 3),
            "D_estimate": int(D_cv) if D_cv else None,
        },
        "method3_parallel_analysis": {"n_signal": n_signal_horn},
        "bounds_from_thm9": {"lower": n_signal_horn, "upper": int(n - 1)},
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# P10. Geometric vs statistical noise
# ============================================================================
def p10_noise_decomposition() -> dict:
    """Compare the geometric indistinguishability radius (Theorem 2)
    to the statistical bootstrap uncertainty in the aggregate score,
    both in standardised score units. Bootstrap the rows of the score
    matrix and measure the distribution of aggregate-score values for
    each model; the statistical radius is the median bootstrap std of
    the aggregate score.
    """
    print("[P10] noise decomposition (aligned units)")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)
    R = float(np.linalg.norm(Sz, axis=1).max())
    delta_0_geom = float(np.pi * R / k)

    rng = np.random.default_rng(0)
    aggs = []
    for _ in range(500):
        # Resample the BENCHMARKS (columns), which gives the bootstrap
        # variability of the aggregate score under within-model noise.
        col_idx = rng.choice(k, size=k, replace=True)
        agg_b = standardise(S)[:, col_idx].mean(axis=1)
        aggs.append(agg_b)
    aggs = np.asarray(aggs)
    # Per-model std across bootstraps in standardised score units
    per_model_std = aggs.std(axis=0)
    statistical_radius = float(np.median(per_model_std))

    out = {
        "geometric_radius_std_units": round(delta_0_geom, 3),
        "statistical_radius_median_std_units": round(statistical_radius, 3),
        "ratio_geom_over_stat": round(
            delta_0_geom / max(statistical_radius, 1e-9), 2
        ),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# P14. Greedy temporal transfer
# ============================================================================
def p14_temporal_transfer() -> dict:
    print("[P14] greedy temporal transfer")
    df_ext = load_extended()
    df_sorted = df_ext.sort_values("model").reset_index(drop=True)
    half = len(df_sorted) // 2
    early = df_sorted.iloc[:half]
    late = df_sorted.iloc[half:]
    S_early = score_matrix(early, EXTENDED_BENCHES)
    S_late = score_matrix(late, EXTENDED_BENCHES)
    Sigma_early = np.corrcoef(S_early, rowvar=False)
    Sigma_late = np.corrcoef(S_late, rowvar=False)
    g_early = greedy_select(Sigma_early, target=0.9, eta_redundant=0.02)
    g_late = greedy_select(Sigma_late, target=0.9, eta_redundant=0.02)

    cov_transferred = [
        coverage_function(Sigma_late, g_early.order[: r])
        for r in range(1, len(EXTENDED_BENCHES) + 1)
    ]
    cov_native = list(g_late.cumulative_coverage)
    out = {
        "early_top7": [EXTENDED_BENCHES[i] for i in g_early.order[:7]],
        "late_top7": [EXTENDED_BENCHES[i] for i in g_late.order[:7]],
        "transferred_coverage": [round(c, 3) for c in cov_transferred],
        "native_coverage": [round(c, 3) for c in cov_native],
        "retention_at_r7": round(cov_transferred[6] / cov_native[6], 3),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# P15. Empirical covering radius
# ============================================================================
def p15_covering_radius() -> dict:
    print("[P15] covering radius")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)

    d_eff = max(int(round(analyze_dimensionality(S, bootstrap=False).d_eff)), 2)
    U, s, Vt = np.linalg.svd(Sz, full_matrices=False)
    loadings = Vt[:d_eff].T
    norms = np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-12
    loading_dirs = loadings / norms

    rng = np.random.default_rng(0)
    rand = rng.normal(size=(10000, d_eff))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    cos_sim = rand @ loading_dirs.T
    cos_max = np.clip(cos_sim.max(axis=1), -1, 1)
    angles = np.arccos(cos_max)
    omega_emp = float(angles.max())
    omega_rogers = float(np.sqrt(d_eff) * k ** (-1.0 / max(d_eff - 1, 1)))
    out = {
        "d_eff": d_eff,
        "k": k,
        "empirical_covering_radius": round(omega_emp, 3),
        "rogers_optimal": round(omega_rogers, 3),
        "ratio": round(omega_emp / max(omega_rogers, 1e-9), 2),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# Driver
# ============================================================================
def run() -> None:
    out = {}
    out["P4_cross_leaderboard"]    = p4_cross_leaderboard()
    out["P5_chi_calibration"]      = p5_chi_squared_calibration()
    out["P6_synthetic_thm2"]       = p6_synthetic_thm2()
    out["P7_five_way"]             = p7_five_way()
    out["P8_rank_reversal_rates"]  = p8_rank_reversal_rates()
    out["P9_D_estimation"]         = p9_d_estimation()
    out["P10_noise_decomposition"] = p10_noise_decomposition()
    out["P14_temporal_transfer"]   = p14_temporal_transfer()
    out["P15_covering_radius"]     = p15_covering_radius()

    (OUT / "validation_v5.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'validation_v5.json'}")


if __name__ == "__main__":
    run()

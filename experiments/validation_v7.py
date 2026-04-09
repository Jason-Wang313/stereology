"""v7 validation experiments for the final-prompt revision.

4F  Score normalisation sensitivity (Kendall tau, quantile in addition to
    the Pearson/Spearman already reported).
4D  Item-level binomial SE where item counts are public, vs bootstrap radius.
4G  Table 3 specificity: for each domination pair, name the held-out
    benchmark(s) that reverse.
4I  **CRITICAL** - aggregate-direction indistinguishability radius. The
    v6 headline used the worst-case covering radius times the population
    radius (27.6 std units) but the actual aggregate direction w = (1/k)1
    may be well-covered. Compute the indistinguishability radius for the
    aggregate direction and compare to Delta_2 = 0.072.
5E  LiveBench frontier bootstrap CI.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import analyze_dimensionality, participation_ratio
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    DATA_DIR,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    score_matrix,
)

OUT = RESULTS_DIR / "validation_v7"
OUT.mkdir(parents=True, exist_ok=True)


def standardise(S):
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


def aggregate(S):
    return standardise(S).mean(axis=1)


# ============================================================================
# 4F. Kendall + quantile normalisation (in addition to Pearson/Spearman)
# ============================================================================
def f4_normalisation() -> dict:
    print("[4F] Kendall + quantile d_eff")
    df_ext = load_extended()
    S = score_matrix(df_ext, EXTENDED_BENCHES)
    n, k = S.shape
    rows = []

    # Pearson / Spearman (already known, re-report)
    from scipy.stats import spearmanr
    pearson = np.corrcoef(S, rowvar=False)
    rows.append(("pearson", participation_ratio(np.linalg.eigvalsh(pearson)[::-1])))

    r_sp = spearmanr(S).correlation
    rows.append(("spearman", participation_ratio(np.linalg.eigvalsh(r_sp)[::-1])))

    # Kendall: k x k symmetric matrix via pairwise kendalltau
    K = np.eye(k)
    for a, b in combinations(range(k), 2):
        t, _ = kendalltau(S[:, a], S[:, b])
        K[a, b] = t
        K[b, a] = t
    rows.append(("kendall", participation_ratio(np.linalg.eigvalsh(K)[::-1])))

    # Quantile-normalised: rank-transform each column to uniform on [0,1]
    ranks = np.apply_along_axis(rankdata, 0, S)
    Q = (ranks - 0.5) / n
    q_corr = np.corrcoef(Q, rowvar=False)
    rows.append(("quantile", participation_ratio(np.linalg.eigvalsh(q_corr)[::-1])))

    df = pd.DataFrame(rows, columns=["method", "d_eff"])
    df["d_eff"] = df["d_eff"].round(2)
    df.to_csv(OUT / "4f_normalisation.csv", index=False)
    print(df.to_string(index=False))
    return {"table": df.to_dict(orient="records"),
            "range": [float(df["d_eff"].min()), float(df["d_eff"].max())]}


# ============================================================================
# 4D. Item-level binomial SE vs bootstrap
# ============================================================================
def f4_noise_model() -> dict:
    print("[4D] item-level binomial SE vs bootstrap")
    # Public item counts for common benchmarks
    item_counts = {
        "MMLU": 14042,
        "HellaSwag": 10042,
        "ARC": 1172,
        "TruthfulQA": 817,
        "Winogrande": 1267,
        "GSM8K": 1319,
        "IFEval": 541,
        "BBH": 6511,
        "MATH Lvl 5": 1324,
        "GPQA": 448,
        "MUSR": 756,
        "MMLU-PRO": 12032,
    }
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)

    # Binomial SE per benchmark at 50% accuracy (worst case)
    rows = []
    for b, name in enumerate(EXTENDED_BENCHES):
        N = item_counts.get(name, None)
        if N is None:
            continue
        # Raw binomial SE (worst case p=0.5)
        se_raw = 0.5 / np.sqrt(N)
        # Convert to standardised units by dividing by the column std
        col_std = float(S[:, b].std())
        se_std = se_raw / max(col_std, 1e-9)
        rows.append({
            "benchmark": name,
            "n_items": int(N),
            "binomial_se_raw": round(float(se_raw), 4),
            "binomial_se_std_units": round(float(se_std), 4),
        })
    df_noise = pd.DataFrame(rows)
    df_noise.to_csv(OUT / "4d_noise_model.csv", index=False)
    print(df_noise.to_string(index=False))

    median_item_se = float(df_noise["binomial_se_std_units"].median())
    bootstrap_radius = 0.214  # from v5 H.12
    return {
        "table": df_noise.to_dict(orient="records"),
        "median_item_se_std_units": round(median_item_se, 4),
        "bootstrap_statistical_radius": bootstrap_radius,
        "ratio": round(bootstrap_radius / max(median_item_se, 1e-9), 2),
    }


# ============================================================================
# 4G. Table 3 specificity
# ============================================================================
def f4_table3_specificity() -> dict:
    print("[4G] table 3 specificity")
    df_ext = load_extended()
    S = score_matrix(df_ext, EXTENDED_BENCHES)
    k = len(EXTENDED_BENCHES)
    rng = np.random.default_rng(0)

    # Take the domination pairs from the original table 2 (4 rows)
    target_losers = [
        "BEE-spoke-data/smol_llama-220M-GQA".lower(),
        "Corianas/Quokka_2.7b".lower(),
        "TinyLlama/TinyLlama-1.1B-Chat-v0.6".lower(),
        "VAGOsolutions/SauerkrautLM-Gemma-2b".lower(),
    ]
    target_winner = "0-hero/Matter-0.2-7B-DPO".lower()

    # Locate indices
    df_lower = df_ext["model"].str.lower()
    winner_idx = df_lower[df_lower == target_winner].index
    if len(winner_idx) == 0:
        return {"note": "winner model not found"}
    w = int(winner_idx[0])

    rows = []
    for loser_name in target_losers:
        matches = df_lower[df_lower == loser_name].index
        if len(matches) == 0:
            continue
        loser_idx = int(matches[0])
        # Pick a random visible split and find held-out benchmarks that reverse
        # Try the same split as the original table 2 to get reproducible results
        visible = [0, 1, 3, 6, 8, 10]   # 6 visible benchmarks by index
        held = [2, 4, 5, 7, 9, 11]
        loser_scores = S[loser_idx]
        winner_scores = S[w]
        # Find which held-out benchmarks reverse the verdict
        reversals = []
        for h in held:
            if loser_scores[h] > winner_scores[h]:
                reversals.append(EXTENDED_BENCHES[h])
        rows.append({
            "loser": df_ext.iloc[loser_idx]["model"],
            "winner": df_ext.iloc[w]["model"],
            "reversing_benches": ", ".join(reversals) if reversals else "(none in this split)",
        })

    df_spec = pd.DataFrame(rows)
    df_spec.to_csv(OUT / "4g_table3_specificity.csv", index=False)
    print(df_spec.to_string(index=False))
    return {"table": df_spec.to_dict(orient="records")}


# ============================================================================
# 4I. CRITICAL: aggregate-direction indistinguishability
# ============================================================================
def f4_aggregate_direction() -> dict:
    print("[4I] aggregate-direction indistinguishability")
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sz = standardise(S)
    n, k = S.shape
    R = float(np.linalg.norm(Sz, axis=1).max())

    # The aggregate direction is w = (1/sqrt(k), ..., 1/sqrt(k)) (unit vector)
    w = np.ones(k) / np.sqrt(k)

    # PCA loadings
    U, s_, Vt = np.linalg.svd(Sz, full_matrices=False)
    loadings = Vt.T          # k x k (each row is a benchmark loading)
    # Normalise
    norms = np.linalg.norm(loadings, axis=1, keepdims=True) + 1e-12
    loading_dirs = loadings / norms

    # Nearest benchmark-loading direction to w (in angular terms)
    cos_sim = loading_dirs @ w
    cos_max = float(np.clip(cos_sim.max(), -1, 1))
    angle_to_w = float(np.arccos(cos_max))

    # Worst-case covering radius (random directions, for comparison)
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(10000, k))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    cos_sim2 = rand @ loading_dirs.T
    worst_omega = float(np.arccos(np.clip(cos_sim2.max(axis=1), -1, 1)).max())

    # Aggregate-specific indistinguishability
    delta_aggregate = float(2 * R * angle_to_w)
    delta_worst = float(2 * R * worst_omega)

    # Observed runner-up gap
    agg = aggregate(S)
    s = np.sort(agg)[::-1]
    delta2 = float(s[0] - s[1])

    out = {
        "R": round(R, 2),
        "angle_w_to_nearest_loading_rad": round(angle_to_w, 3),
        "worst_case_covering_radius_rad": round(worst_omega, 3),
        "delta_vis_aggregate_direction": round(delta_aggregate, 3),
        "delta_vis_worst_case": round(delta_worst, 3),
        "delta2_observed": round(delta2, 4),
        "ratio_aggregate_to_delta2": round(delta_aggregate / max(delta2, 1e-9), 1),
        "ratio_worst_to_delta2": round(delta_worst / max(delta2, 1e-9), 1),
    }
    print(json.dumps(out, indent=2))
    return out


# ============================================================================
# 5E. LiveBench frontier bootstrap CI
# ============================================================================
def f5_livebench_ci() -> dict:
    print("[5E] LiveBench frontier CI")
    lb_path = DATA_DIR / "livebench.csv"
    if not lb_path.exists():
        return {"available": False}
    df_lb = pd.read_csv(lb_path)
    bench_cols = [c for c in df_lb.columns if c != "model"]
    avg = df_lb[bench_cols].mean(axis=1)
    df_top = df_lb[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = df_top[bench_cols].to_numpy(dtype=float)
    n, k = S.shape

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(3000):
        idx = rng.choice(n, size=n, replace=True)
        try:
            r = analyze_dimensionality(S[idx], bootstrap=False)
            samples.append(r.d_eff)
        except Exception:
            pass
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    ci_lo = float(np.percentile(samples, 2.5))
    ci_hi = float(np.percentile(samples, 97.5))
    point = float(analyze_dimensionality(S, bootstrap=False).d_eff)
    return {
        "n_models": int(n),
        "k": int(k),
        "d_eff_point": round(point, 2),
        "d_eff_ci95_lo": round(ci_lo, 2),
        "d_eff_ci95_hi": round(ci_hi, 2),
        "ci_width": round(ci_hi - ci_lo, 2),
    }


# ============================================================================
# Driver
# ============================================================================
def run():
    out = {}
    out["F4_normalisation"]     = f4_normalisation()
    out["D4_noise_model"]       = f4_noise_model()
    out["G4_table3"]            = f4_table3_specificity()
    out["I4_aggregate_dir"]     = f4_aggregate_direction()
    out["E5_livebench_ci"]      = f5_livebench_ci()

    (OUT / "validation_v7.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'validation_v7.json'}")


if __name__ == "__main__":
    run()

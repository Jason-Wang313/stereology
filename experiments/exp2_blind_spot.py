"""Experiment 2: Quantify the indistinguishability blind spot.

Generates:
- Figure 3: Predicted vs actual divergence (held-out validation).
- Figure 4: Top-1 ranking unreliability vs effective dimensionality ratio.
- results/exp2_blind_spot.json with the indistinguishability radius and
  ranking-pair violation rate per population slice.

Held-out experiment
-------------------
For each random split of the 12 extended benchmarks into a "visible"
half (V) and a "held-out" half (H), we:
1. Aggregate-rank models on V.
2. Compute the visible aggregate-score gap for every model pair.
3. Compute the held-out aggregate-score divergence for every pair.
4. Theory predicts the held-out divergence is bounded by
   epsilon + pi * R / m, where R is the standardised capability radius
   and m is the visible benchmark count.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import analyze_dimensionality
from src.theorem2 import (
    capability_radius_estimate,
    lipschitz_bound,
    minimum_benchmarks,
    ranking_unreliability,
)
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    save_fig,
    score_matrix,
    setup_matplotlib,
)


def standardise(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


# ----------------------------------------------------------------------------
# Held-out divergence
# ----------------------------------------------------------------------------
def heldout_divergence(
    S: np.ndarray, n_splits: int = 200, seed: int = 0
) -> pd.DataFrame:
    """For random visible/held-out splits, compute pair gaps + theory bound."""
    rng = np.random.default_rng(seed)
    n, k = S.shape
    Sz = standardise(S)
    R_full = float(np.linalg.norm(Sz, axis=1).max())
    half = k // 2

    rows = []
    for split in range(n_splits):
        cols = rng.permutation(k)
        vis = sorted(cols[:half].tolist())
        hold = sorted(cols[half:].tolist())
        Sv = Sz[:, vis]
        Sh = Sz[:, hold]
        agg_v = Sv.mean(axis=1)
        agg_h = Sh.mean(axis=1)
        # all pairs
        ii, jj = np.triu_indices(n, k=1)
        gap_v = np.abs(agg_v[ii] - agg_v[jj])
        gap_h = np.abs(agg_h[ii] - agg_h[jj])
        bound = lipschitz_bound(epsilon=0.0, R=R_full, m=len(vis))
        rows.append({
            "split": split,
            "n_pairs": len(ii),
            "median_visible_gap": float(np.median(gap_v)),
            "median_heldout_gap": float(np.median(gap_h)),
            "max_heldout_divergence": float(np.max(np.abs(agg_v - agg_h))),
            "predicted_bound": bound,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Ranking-reliability scan
# ----------------------------------------------------------------------------
def ranking_curve(S: np.ndarray, m_list: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, k = S.shape
    Sz = standardise(S)
    R = float(np.linalg.norm(Sz, axis=1).max())
    rows = []
    for m in m_list:
        if m > k:
            continue
        # Average over 30 random benchmark subsets
        violations = []
        for _ in range(30):
            cols = rng.choice(k, size=m, replace=False)
            sub = Sz[:, cols]
            r = ranking_unreliability(sub, R=R)
            violations.append(r.pair_violation_rate)
        rows.append({
            "m": m,
            "mean_violation_rate": float(np.mean(violations)),
            "std_violation_rate": float(np.std(violations)),
            "delta_0": float(np.pi * R / m),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figure_divergence(div_df: pd.DataFrame, save_name="fig3_divergence") -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(div_df["median_visible_gap"], div_df["median_heldout_gap"],
               s=14, alpha=0.5, color="C0",
               label=f"random splits (n={len(div_df)})")
    pmin = float(min(div_df["median_visible_gap"].min(),
                     div_df["median_heldout_gap"].min()))
    pmax = float(max(div_df["median_visible_gap"].max(),
                     div_df["median_heldout_gap"].max()))
    grid = np.linspace(0, pmax * 1.05, 50)
    ax.plot(grid, grid, "--", color="grey", lw=0.7, label="$y = x$")
    bound = float(div_df["predicted_bound"].iloc[0])
    ax.axhline(bound, color="C3", linestyle=":", lw=1,
               label=f"Lipschitz bound $\\pi R / m = {bound:.2f}$")
    ax.set_xlabel("median visible-half pair gap")
    ax.set_ylabel("median held-out-half pair gap")
    ax.set_title("Held-out divergence vs theory bound")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


def figure_ranking(curve_df: pd.DataFrame, save_name="fig4_ranking") -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.errorbar(curve_df["m"], curve_df["mean_violation_rate"],
                yerr=curve_df["std_violation_rate"], marker="o",
                color="C0", lw=1, capsize=3, label="empirical")
    # Theory: violation rate decays as ~1/m (Lipschitz)
    if len(curve_df) > 1:
        m0 = curve_df["m"].iloc[-1]
        v0 = curve_df["mean_violation_rate"].iloc[-1]
        ax.plot(curve_df["m"], v0 * m0 / curve_df["m"], "--", color="C3", lw=0.8,
                label="theory $\\propto 1/m$")
    ax.set_xlabel("number of visible benchmarks $m$")
    ax.set_ylabel("pair violation rate")
    ax.set_title("Top-1 unreliability vs benchmark count")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run() -> dict:
    out: dict = {}

    df_ext = load_extended()
    S_full = score_matrix(df_ext, EXTENDED_BENCHES)

    # Use the frontier slice (top 50%) where d_eff ~ 5
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S_top = score_matrix(df_top, EXTENDED_BENCHES)

    print(f"extended n={len(df_ext)}  top50%={len(df_top)}")
    R_top = capability_radius_estimate(S_top)
    print(f"R (top50%) = {R_top:.2f}")

    # 1. Held-out validation on the frontier slice
    div_df = heldout_divergence(S_top, n_splits=300, seed=0)
    div_df.to_csv(RESULTS_DIR / "exp2_divergence.csv", index=False)
    print(f"  median observed heldout gap: "
          f"{div_df['median_heldout_gap'].mean():.3f}")
    print(f"  predicted bound: {div_df['predicted_bound'].iloc[0]:.3f}")
    out["heldout"] = {
        "n_splits": len(div_df),
        "mean_visible_gap": float(div_df["median_visible_gap"].mean()),
        "mean_heldout_gap": float(div_df["median_heldout_gap"].mean()),
        "lipschitz_bound": float(div_df["predicted_bound"].iloc[0]),
        "fraction_within_bound": float(
            (div_df["median_heldout_gap"] <= div_df["predicted_bound"]).mean()
        ),
        "R_estimate": R_top,
    }

    # 2. Top-1 ranking reliability across m
    m_list = list(range(2, 13))
    rank_df = ranking_curve(S_top, m_list)
    rank_df.to_csv(RESULTS_DIR / "exp2_ranking.csv", index=False)
    out["ranking"] = rank_df.to_dict(orient="records")

    # 3. Per-leaderboard summary using the headline ranking-unreliability bound
    summaries = {}
    for name, df, benches in [
        ("ollm_v2", load_ollm_v2(), OLLM_V2_BENCHES),
        ("extended", df_ext, EXTENDED_BENCHES),
        ("extended_top50", df_top, EXTENDED_BENCHES),
    ]:
        S = score_matrix(df, benches)
        R = capability_radius_estimate(S)
        rinfo = ranking_unreliability(S, R=R)
        dim = analyze_dimensionality(S, bootstrap=False)
        summaries[name] = {
            "n_models": len(S),
            "k_benchmarks": S.shape[1],
            "R": R,
            "delta_0": rinfo.indistinguishability_radius,
            "pair_violation_rate": rinfo.pair_violation_rate,
            "top1_upper_bound": rinfo.upper_bound_on_top1_correct,
            "d_eff": dim.d_eff,
        }
        print(f"  {name}: viol={rinfo.pair_violation_rate:.3f}  "
              f"top1<={rinfo.upper_bound_on_top1_correct:.3f}  "
              f"d_eff={dim.d_eff:.2f}")
    out["summaries"] = summaries

    figure_divergence(div_df)
    figure_ranking(rank_df)

    out_path = RESULTS_DIR / "exp2_blind_spot.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    run()

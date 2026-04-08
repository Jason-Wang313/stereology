"""Experiment 1: Effective dimensionality of public benchmark suites.

Generates:
- Figure 1: eigenvalue spectrum with the Marchenko-Pastur threshold
  (scree plot) for OLLM v2 and the extended 12-benchmark matrix.
- Figure 2: PCA biplot of benchmarks coloured by benchmark family.
- results/exp1_dimensionality.json with d_eff, signal counts, and CIs
  across multiple population slices.

Key empirical question:
    Is d_eff in [3, 5] for major leaderboards?

Result reported in paper: yes, when restricted to frontier models;
the full population is dominated by a single g-factor (small <-> large).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import analyze_dimensionality
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V1_BENCHES,
    OLLM_V2_BENCHES,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    save_fig,
    score_matrix,
    setup_matplotlib,
)


# ----------------------------------------------------------------------------
# Slicing helpers
# ----------------------------------------------------------------------------
def quantile_slice(df: pd.DataFrame, benches, q: float) -> pd.DataFrame:
    avg = df[list(benches)].mean(axis=1)
    return df[avg >= avg.quantile(q)].reset_index(drop=True)


def make_slices(df, benches):
    return {
        "all": df.reset_index(drop=True),
        "params>=7B": df[df["params_b"] >= 7].reset_index(drop=True),
        "top50pct": quantile_slice(df, benches, 0.50),
        "top25pct": quantile_slice(df, benches, 0.75),
    }


# ----------------------------------------------------------------------------
# Figure 1: scree plot
# ----------------------------------------------------------------------------
def figure_scree(results, save_name="fig1_scree") -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), sharey=False)
    panels = [
        ("ollm_v2", "all", "Open LLM Leaderboard v2 (n = 458, k = 6)"),
        ("extended", "all", "Extended (n = 295, k = 12)"),
    ]
    for ax, (ds, slc, title) in zip(axes, panels):
        r = results[ds][slc]
        eigs = np.asarray(r["eigenvalues"])
        idx = np.arange(1, len(eigs) + 1)
        ax.bar(idx, eigs, width=0.6, alpha=0.7, color="C0", label="eigenvalue")
        ax.axhline(r["mp_threshold"], color="C3", linestyle="--", linewidth=1,
                   label=f"MP threshold $\\lambda_+ = {r['mp_threshold']:.2f}$")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7,
                   label="$\\lambda = 1$ (Kaiser)")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel(r"$\lambda_i$")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=False)
        ax.text(
            0.97, 0.55,
            f"$d_{{\\mathrm{{eff}}}}$ = {r['d_eff']:.2f}\n"
            f"signal eigs: {r['n_signal_eigs']}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5),
        )
    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: PCA biplot of benchmarks
# ----------------------------------------------------------------------------
def figure_biplot(df, benches, save_name="fig2_biplot") -> None:
    setup_matplotlib()
    S = score_matrix(df, benches)
    Sz = (S - S.mean(0)) / S.std(0)
    U, s, Vt = np.linalg.svd(Sz, full_matrices=False)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    # Project models
    coords = U[:, :2] * s[:2]
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.35, color="grey",
               label="models")

    # Project benchmarks (loading vectors)
    loadings = Vt[:2].T * s[:2] / np.sqrt(len(S))
    fam_color = {
        "v2_reasoning": "C0",
        "v2_knowledge": "C1",
        "v1_kb": "C2",
        "v1_reasoning": "C3",
    }
    family = {
        "IFEval": "v2_knowledge", "BBH": "v2_reasoning",
        "MATH Lvl 5": "v2_reasoning", "GPQA": "v2_knowledge",
        "MUSR": "v2_reasoning", "MMLU-PRO": "v2_knowledge",
        "ARC": "v1_reasoning", "HellaSwag": "v1_kb",
        "MMLU": "v1_kb", "TruthfulQA": "v1_kb",
        "Winogrande": "v1_kb", "GSM8K": "v1_reasoning",
    }
    scale = 0.95 * np.abs(coords).max() / np.abs(loadings).max()
    for j, b in enumerate(benches):
        c = fam_color[family[b]]
        ax.annotate(
            "", xy=(loadings[j, 0] * scale, loadings[j, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
        )
        ax.text(loadings[j, 0] * scale * 1.06,
                loadings[j, 1] * scale * 1.06,
                b, color=c, fontsize=7,
                ha="center", va="center")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({s[0] ** 2 / (s ** 2).sum() * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({s[1] ** 2 / (s ** 2).sum() * 100:.1f}%)")
    ax.set_title(f"PCA biplot, n={len(S)}, k={len(benches)}")

    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run() -> dict:
    df_v2 = load_ollm_v2()
    df_ext = load_extended()
    print(f"OLLM v2: {df_v2.shape}; extended: {df_ext.shape}")

    out: dict = {"ollm_v2": {}, "extended": {}}

    for slc_name, slc_df in make_slices(df_v2, OLLM_V2_BENCHES).items():
        S = score_matrix(slc_df, OLLM_V2_BENCHES)
        if len(S) < 8:
            continue
        r = analyze_dimensionality(S, n_boot=500, seed=0)
        out["ollm_v2"][slc_name] = {
            "n_models": r.n_models,
            "k_benchmarks": r.k_benchmarks,
            "eigenvalues": r.eigenvalues.tolist(),
            "explained_variance": r.explained_variance.tolist(),
            "d_eff": r.d_eff,
            "d_eff_mp": r.d_eff_mp,
            "mp_threshold": r.mp_threshold,
            "n_signal_eigs": r.n_signal_eigs,
            "bootstrap_ci_d_eff": list(r.bootstrap_ci),
            "bootstrap_ci_d_eff_mp": list(r.bootstrap_ci_mp),
        }
        print(f"  v2 [{slc_name}] n={r.n_models}  d_eff={r.d_eff:.2f}  "
              f"CI95={[round(x,2) for x in r.bootstrap_ci]}  "
              f"signal={r.n_signal_eigs}")

    for slc_name, slc_df in make_slices(df_ext, EXTENDED_BENCHES).items():
        S = score_matrix(slc_df, EXTENDED_BENCHES)
        if len(S) < 8:
            continue
        r = analyze_dimensionality(S, n_boot=500, seed=0)
        out["extended"][slc_name] = {
            "n_models": r.n_models,
            "k_benchmarks": r.k_benchmarks,
            "eigenvalues": r.eigenvalues.tolist(),
            "explained_variance": r.explained_variance.tolist(),
            "d_eff": r.d_eff,
            "d_eff_mp": r.d_eff_mp,
            "mp_threshold": r.mp_threshold,
            "n_signal_eigs": r.n_signal_eigs,
            "bootstrap_ci_d_eff": list(r.bootstrap_ci),
            "bootstrap_ci_d_eff_mp": list(r.bootstrap_ci_mp),
        }
        print(f"  ext[{slc_name}] n={r.n_models}  d_eff={r.d_eff:.2f}  "
              f"CI95={[round(x,2) for x in r.bootstrap_ci]}  "
              f"signal={r.n_signal_eigs}")

    figure_scree(out)
    figure_biplot(load_extended(), EXTENDED_BENCHES, save_name="fig2_biplot")

    out_path = RESULTS_DIR / "exp1_dimensionality.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    run()

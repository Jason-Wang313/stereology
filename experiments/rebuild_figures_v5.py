"""Rebuild Figure 1 (2x2 scree) and Figure 3 right panel (gap/delta0 ratio).

Outputs:
  figures/fig1_scree.{pdf,png} - 2x2 grid (full + frontier for both suites)
  figures/fig4_ranking.{pdf,png} - replaced with gap/delta0(m) ratio plot
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem1 import analyze_dimensionality, mp_threshold
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    load_extended,
    load_ollm_v2,
    save_fig,
    score_matrix,
    setup_matplotlib,
)


def standardise(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(0)) / np.where(S.std(0) > 1e-12, S.std(0), 1.0)


def panel(ax, S, title):
    r = analyze_dimensionality(S, bootstrap=False)
    eigs = r.eigenvalues
    idx = np.arange(1, len(eigs) + 1)
    ax.bar(idx, eigs, width=0.6, alpha=0.75, color="C0")
    ax.axhline(r.mp_threshold, color="C3", linestyle="--", lw=1)
    ax.axhline(1.0, color="grey", linestyle=":", lw=0.7)
    ax.set_xlabel("eigenvalue index", fontsize=9)
    ax.set_ylabel(r"$\lambda_i$", fontsize=9)
    ax.set_title(title, fontsize=9)
    # All annotations in a single text box at upper-right to avoid overlap
    info = (
        f"MP $\\lambda_+ = {r.mp_threshold:.2f}$ (---)\n"
        f"Kaiser $\\lambda = 1$ (...)\n"
        f"$d_{{\\mathrm{{eff}}}} = {r.d_eff:.2f}$, "
        f"signal: {r.n_signal_eigs}"
    )
    ax.text(
        0.97, 0.95, info,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey",
                  lw=0.5, alpha=0.9),
    )


def fig1_scree_2x2():
    setup_matplotlib()
    df_v2 = load_ollm_v2()
    avg_v2 = df_v2[OLLM_V2_BENCHES].mean(axis=1)
    df_v2_top = df_v2[avg_v2 >= avg_v2.quantile(0.5)].reset_index(drop=True)
    df_ext = load_extended()
    avg_ext = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_ext_top = df_ext[avg_ext >= avg_ext.quantile(0.5)].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.6), sharey=False)
    panel(axes[0, 0],
          score_matrix(df_v2, OLLM_V2_BENCHES),
          f"Open LLM v2 (full, n={len(df_v2)}, k=6)")
    panel(axes[0, 1],
          score_matrix(df_ext, EXTENDED_BENCHES),
          f"Extended (full, n={len(df_ext)}, k=12)")
    panel(axes[1, 0],
          score_matrix(df_v2_top, OLLM_V2_BENCHES),
          f"Open LLM v2 (frontier, n={len(df_v2_top)}, k=6)")
    panel(axes[1, 1],
          score_matrix(df_ext_top, EXTENDED_BENCHES),
          f"Extended (frontier, n={len(df_ext_top)}, k=12)")
    fig.tight_layout()
    save_fig(fig, "fig1_scree")
    plt.close(fig)
    print("wrote figures/fig1_scree.{pdf,png}")


def fig4_gap_over_delta0():
    """Replace fig4_ranking with median pair gap / delta_0(m) ratio."""
    setup_matplotlib()
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    n, k = S.shape
    Sz = standardise(S)
    R = float(np.linalg.norm(Sz, axis=1).max())

    rng = np.random.default_rng(0)
    m_vals = list(range(2, k + 1))
    means = []
    stds = []
    for m in m_vals:
        ratios = []
        for _ in range(200):
            cols = rng.choice(k, size=m, replace=False)
            sub = Sz[:, cols].mean(axis=1)
            ii, jj = np.triu_indices(n, k=1)
            gap = np.abs(sub[ii] - sub[jj])
            delta_0 = np.pi * R / m
            ratios.append(np.median(gap) / delta_0)
        means.append(np.mean(ratios))
        stds.append(np.std(ratios))

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.errorbar(m_vals, means, yerr=stds, marker="o", color="C0",
                lw=1, capsize=3, label="empirical")
    ax.axhline(1.0, color="C3", linestyle="--", lw=0.8,
               label="$\\delta_0$ threshold")
    ax.set_xlabel("number of visible benchmarks $m$")
    ax.set_ylabel("median pair gap $/\\;\\delta_0(m)$")
    ax.set_title("Pair separation vs indistinguishability radius")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig4_ranking")
    plt.close(fig)
    print("wrote figures/fig4_ranking.{pdf,png}")


def fig_greedy_combined():
    """Combined 3-panel figure: coverage curve + blind spot before/after.

    Replaces the separate fig5_exttop_curve + fig6_exttop_blindspot with a
    single figure that has proper width ratios so the right panels are legible.
    """
    setup_matplotlib()
    from src.theorem3 import coverage_function, greedy_select, uncovered_directions

    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    S = score_matrix(df_top, EXTENDED_BENCHES)
    Sigma = np.corrcoef(S, rowvar=False)
    k = Sigma.shape[0]

    g = greedy_select(Sigma, target=0.9, eta_redundant=0.02)
    rng = np.random.default_rng(0)
    rand_curve = np.zeros(k)
    for _ in range(200):
        order = rng.permutation(k)
        for i in range(1, k + 1):
            rand_curve[i - 1] += coverage_function(Sigma, order[:i])
    rand_curve /= 200

    eigvals_full = np.linalg.eigvalsh(Sigma)[::-1]
    eigvals_full = np.clip(eigvals_full, 0, None)
    bs_empty = uncovered_directions(Sigma, [])
    bs_greedy = uncovered_directions(Sigma, g.minimum_subset)

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 3.8),
        gridspec_kw={"width_ratios": [2.2, 1, 1]},
    )

    # Panel 1: coverage curve
    ax = axes[0]
    x = np.arange(1, k + 1)
    ax.plot(x, g.cumulative_coverage, marker="o", color="C0", label="greedy")
    ax.plot(x, rand_curve, marker="s", color="C1", lw=1, ms=4,
            label="random (mean of 200)")
    ax.axhline(0.9, color="grey", linestyle="--", lw=0.7, label=r"$\tau = 0.9$")
    ax.set_xlabel("number of benchmarks selected")
    ax.set_ylabel(r"coverage $f(T)$")
    ax.set_title("Greedy coverage (extended frontier)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)

    # Panels 2-3: blind spot before / after
    ymax = max(eigvals_full.max() * 1.05, 0.1)
    for ax, title_label, vals in [
        (axes[1], "before: 0 benchmarks", bs_empty.uncovered_eigenvalues),
        (axes[2], f"after: {len(g.minimum_subset)} greedy",
         bs_greedy.uncovered_eigenvalues),
    ]:
        idx = np.arange(1, len(vals) + 1)
        ax.bar(idx, vals, color="C3", alpha=0.7)
        ax.set_xlabel("uncovered direction", fontsize=9)
        ax.set_ylabel("eigen-mass", fontsize=9)
        ax.set_title(title_label, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.text(
            0.95, 0.90,
            f"total: {vals.sum():.2f}\nfrac: {vals.sum() / eigvals_full.sum():.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5),
        )

    fig.tight_layout()
    # Save as BOTH the old filenames so the LaTeX references work
    save_fig(fig, "fig5_exttop_curve")
    # Also save as standalone fig6 for backward compat
    save_fig(fig, "fig_greedy_combined")
    plt.close(fig)
    print("wrote figures/fig5_exttop_curve.{pdf,png} (combined 3-panel)")
    print("wrote figures/fig_greedy_combined.{pdf,png}")


if __name__ == "__main__":
    fig1_scree_2x2()
    fig4_gap_over_delta0()
    fig_greedy_combined()

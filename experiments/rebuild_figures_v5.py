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
    ax.axhline(r.mp_threshold, color="C3", linestyle="--", lw=1,
               label=f"$\\lambda_+ = {r.mp_threshold:.2f}$")
    ax.axhline(1.0, color="grey", linestyle=":", lw=0.7,
               label="Kaiser ($\\lambda=1$)")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(title, fontsize=9)
    ax.text(
        0.97, 0.85,
        f"$d_{{\\mathrm{{eff}}}} = {r.d_eff:.2f}$\nsignal eigs: {r.n_signal_eigs}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7)


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


if __name__ == "__main__":
    fig1_scree_2x2()
    fig4_gap_over_delta0()

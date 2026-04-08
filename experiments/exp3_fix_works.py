"""Experiment 3: Greedy benchmark selection (Theorem 3) reduces the blind spot.

Generates:
- Figure 5: cumulative coverage curve (greedy vs random vs optimal-on-PCA).
- Figure 6: before/after blind-spot comparison (uncovered eigenvalue mass).
- results/exp3_fix_works.json with the recommended benchmark subset.

Story: "20 benchmarks but really only need 7." On the extended 12-benchmark
matrix the greedy algorithm reaches 90% coverage with about half of the
benchmarks; the rest are redundant. We also show that the eigen-mass
of the uncovered directions shrinks dramatically after greedy selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theorem3 import (
    coverage_function,
    dimension_bounds,
    greedy_select,
    uncovered_directions,
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


def random_curve(Sigma: np.ndarray, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Mean cumulative coverage of random benchmark orderings."""
    rng = np.random.default_rng(seed)
    k = Sigma.shape[0]
    out = np.zeros(k)
    for _ in range(n_perm):
        order = rng.permutation(k)
        for i in range(1, k + 1):
            out[i - 1] += coverage_function(Sigma, order[:i])
    return out / n_perm


def figure_curve(
    name: str, benches: list[str], Sigma: np.ndarray,
    greedy_res, random_mean: np.ndarray, save_name: str,
) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    k = Sigma.shape[0]
    x = np.arange(1, k + 1)
    ax.plot(x, greedy_res.cumulative_coverage, marker="o", color="C0",
            label="greedy")
    ax.plot(x, random_mean, marker="s", color="C1", lw=1, ms=4,
            label="random (mean of 200)")
    ax.axhline(0.9, color="grey", linestyle="--", lw=0.7,
               label="$\\tau = 0.9$")
    ax.set_xlabel("number of benchmarks selected")
    ax.set_ylabel("coverage $f(T)$")
    ax.set_title(f"{name}: greedy coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


def figure_blind_spot_compare(
    Sigma: np.ndarray,
    benches: list[str],
    selected: list[int],
    save_name: str,
) -> None:
    setup_matplotlib()
    eigvals_full = np.linalg.eigvalsh(Sigma)[::-1]
    eigvals_full = np.clip(eigvals_full, 0, None)

    bs = uncovered_directions(Sigma, selected)
    bs_empty = uncovered_directions(Sigma, [])

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), sharey=True)
    titles = [
        ("before: 0 benchmarks", bs_empty.uncovered_eigenvalues),
        (f"after: {len(selected)} greedy benchmarks",
         bs.uncovered_eigenvalues),
    ]
    for ax, (title, vals) in zip(axes, titles):
        idx = np.arange(1, len(vals) + 1)
        ax.bar(idx, vals, color="C3", alpha=0.7)
        ax.set_xlabel("uncovered direction index")
        ax.set_ylabel("uncovered eigen-mass")
        ax.set_title(title)
        ax.set_ylim(0, max(eigvals_full.max() * 1.05, 0.1))
        ax.text(
            0.97, 0.92,
            f"total: {vals.sum():.2f}\n"
            f"frac: {vals.sum() / eigvals_full.sum():.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5),
        )
    fig.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


def analyse(name: str, df, benches: list[str], save_suffix: str) -> dict:
    S = score_matrix(df, benches)
    Sigma = np.corrcoef(S, rowvar=False)
    g = greedy_select(Sigma, target=0.9, eta_redundant=0.02)
    db = dimension_bounds(S)
    print(f"\n{name}: n={len(S)}  k={len(benches)}")
    print(f"  greedy order: {[benches[i] for i in g.order]}")
    print(f"  marginal gains: {[round(x,3) for x in g.marginal_gains]}")
    print(f"  cum coverage:   {[round(x,3) for x in g.cumulative_coverage]}")
    print(f"  min subset @ tau=0.9: "
          f"{[benches[i] for i in g.minimum_subset]}")
    print(f"  redundancies (gain<0.02): "
          f"{[benches[i] for i in g.redundancies]}")
    print(f"  D bounds: {db.n_signal} <= D <= {db.upper}")

    rand = random_curve(Sigma, n_perm=200, seed=0)
    figure_curve(name, benches, Sigma, g, rand,
                 save_name=f"fig5_{save_suffix}_curve")
    figure_blind_spot_compare(Sigma, benches, g.minimum_subset,
                              save_name=f"fig6_{save_suffix}_blindspot")

    return {
        "n_models": int(len(S)),
        "k_benchmarks": int(len(benches)),
        "greedy_order": [benches[i] for i in g.order],
        "marginal_gains": list(map(float, g.marginal_gains)),
        "cumulative_coverage": list(map(float, g.cumulative_coverage)),
        "min_subset_for_90pct": [benches[i] for i in g.minimum_subset],
        "redundancies": [benches[i] for i in g.redundancies],
        "D_lower_bound": db.n_signal,
        "D_upper_bound": db.upper,
        "lambda_plus": db.lambda_plus,
        "random_mean_curve": list(map(float, rand)),
    }


def run() -> dict:
    out = {
        "ollm_v2": analyse("OLLM v2", load_ollm_v2(), OLLM_V2_BENCHES, "v2"),
        "extended": analyse("Extended", load_extended(), EXTENDED_BENCHES, "ext"),
    }
    # Frontier slice
    df_ext = load_extended()
    avg = df_ext[EXTENDED_BENCHES].mean(axis=1)
    df_top = df_ext[avg >= avg.quantile(0.5)].reset_index(drop=True)
    out["extended_top50"] = analyse(
        "Extended top50%", df_top, EXTENDED_BENCHES, "exttop"
    )

    out_path = RESULTS_DIR / "exp3_fix_works.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    run()

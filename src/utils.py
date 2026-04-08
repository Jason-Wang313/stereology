"""Shared utilities for the STEREOLOGY package.

Includes:
- Data loaders for OLLM v2 and extended score matrices.
- Score-matrix preprocessing (centering, standardisation, correlation).
- Plot styling helpers shared by all experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical benchmark name lists
OLLM_V2_BENCHES: list[str] = [
    "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO",
]
OLLM_V1_BENCHES: list[str] = [
    "ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K",
]
EXTENDED_BENCHES: list[str] = OLLM_V2_BENCHES + OLLM_V1_BENCHES


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
def load_ollm_v2(official_only: bool = True) -> pd.DataFrame:
    """Load Open LLM Leaderboard v2 score matrix."""
    fname = "ollm_v2.csv" if official_only else "ollm_v2_top200.csv"
    path = DATA_DIR / fname
    df = pd.read_csv(path)
    return df


def load_extended(official_only: bool = False) -> pd.DataFrame:
    """Load 12-benchmark extended score matrix (OLLM v1 + v2 join)."""
    fname = "extended_official.csv" if official_only else "extended.csv"
    return pd.read_csv(DATA_DIR / fname)


def load_arena() -> pd.DataFrame:
    """Load Chatbot Arena Bradley-Terry scores."""
    return pd.read_csv(DATA_DIR / "arena_elo.csv")


def score_matrix(df: pd.DataFrame, benches: Sequence[str]) -> np.ndarray:
    """Extract a (n_models x k_benchmarks) numeric array."""
    return df[list(benches)].to_numpy(dtype=float)


# ----------------------------------------------------------------------------
# Statistical primitives
# ----------------------------------------------------------------------------
def correlation_matrix(S: np.ndarray) -> np.ndarray:
    """Pearson correlation across benchmarks (columns)."""
    return np.corrcoef(S, rowvar=False)


def eigendecomp(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with eigenvalues sorted descending."""
    w, v = np.linalg.eigh((M + M.T) / 2.0)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


# ----------------------------------------------------------------------------
# Plot styling
# ----------------------------------------------------------------------------
def setup_matplotlib() -> None:
    """Apply consistent styling for all paper figures."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    mpl.rcParams.update({
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str) -> None:
    """Save figure as both PDF and PNG into figures/."""
    base = FIGURES_DIR / name
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))

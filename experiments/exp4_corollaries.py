"""Experiment 4: Corollaries in the wild.

Generates:
- Table 1 (results/exp4_table1_rank_reversals.csv): documented rank
  reversals where adding a single model flips an existing pair, predicted
  by Corollary 2.2 (rank reversal possible iff d_eff < n - 1).
- Table 2 (results/exp4_table2_domination.csv): pairs of models where
  benchmark domination on one half of the suite is reversed on the held-out
  half (Busemann-Petty analogue, Proposition in proofs/corollary_proofs.tex).
- results/exp4_corollaries.json with summary statistics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corollaries import (
    benchmark_dominated_pairs,
    domination_under_holdout,
    rank_reversals_on_addition,
    _ranks_array,
)
from src.theorem1 import analyze_dimensionality
from src.theorem2 import rank_reversal_susceptible
from src.utils import (
    EXTENDED_BENCHES,
    OLLM_V2_BENCHES,
    RESULTS_DIR,
    load_extended,
    load_ollm_v2,
    score_matrix,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def small_population(df, benches, n=12, seed=0):
    """Take a small homogeneous frontier slice for the rank-reversal experiment.

    With ``n`` models, rank reversals are visually clean and
    Corollary 2.2 (d_eff < n - 1) becomes a sharp prediction.
    """
    rng = np.random.default_rng(seed)
    avg = df[benches].mean(axis=1)
    df_top = df[avg >= avg.quantile(0.7)].reset_index(drop=True)
    if len(df_top) <= n:
        return df_top
    idx = rng.choice(len(df_top), size=n, replace=False)
    return df_top.iloc[sorted(idx)].reset_index(drop=True)


def aggregator_normalised(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((M - mu) / sd).mean(axis=1)


# ----------------------------------------------------------------------------
# Table 1: rank reversals
# ----------------------------------------------------------------------------
def table1_rank_reversals(df, benches, n_models: int = 12) -> pd.DataFrame:
    sub = small_population(df, benches, n=n_models)
    S = score_matrix(sub, benches)
    n = len(sub)
    dim = analyze_dimensionality(S, bootstrap=False)
    susceptible = rank_reversal_susceptible(dim.d_eff, n)
    print(f"  rank-reversal pop: n={n}  d_eff={dim.d_eff:.2f}  "
          f"d_eff < n-1? {susceptible}")

    reversals = rank_reversals_on_addition(
        S, aggregator=aggregator_normalised, max_triples=10000
    )
    rows = []
    for r in reversals:
        rows.append({
            "model_a": sub.iloc[r.a_idx]["model"],
            "model_b": sub.iloc[r.b_idx]["model"],
            "added": sub.iloc[r.added_idx]["model"],
            "rank_a_before": r.rank_a_before,
            "rank_b_before": r.rank_b_before,
            "rank_a_after": r.rank_a_after,
            "rank_b_after": r.rank_b_after,
        })
    return pd.DataFrame(rows), {
        "n_models": int(n),
        "d_eff": float(dim.d_eff),
        "predicted_susceptible": bool(susceptible),
        "n_reversals_observed": len(rows),
    }


# ----------------------------------------------------------------------------
# Table 2: held-out benchmark domination
# ----------------------------------------------------------------------------
def table2_domination(df, benches: list[str], n_splits: int = 50,
                      seed: int = 0) -> pd.DataFrame:
    """Find pairs where pointwise domination on visible benchmarks is
    contradicted by at least one held-out benchmark.
    """
    rng = np.random.default_rng(seed)
    S = score_matrix(df, benches)
    k = len(benches)
    half = k // 2

    rows: list[dict] = []
    seen = set()
    for split in range(n_splits):
        cols = rng.permutation(k)
        held = sorted(cols[:half].tolist())
        vis = sorted(cols[half:].tolist())
        failures = domination_under_holdout(S, held)
        for p in failures:
            key = (min(p.a_idx, p.b_idx), max(p.a_idx, p.b_idx))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "split": split,
                "loser_visible": df.iloc[p.a_idx]["model"],
                "winner_visible": df.iloc[p.b_idx]["model"],
                "visible_benches": ",".join(benches[c] for c in vis),
                "heldout_benches": ",".join(benches[c] for c in held),
                "n_visible_dominated": int((p.margins > 0).sum()),
                "max_margin": float(p.margins.max()),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run() -> dict:
    out: dict = {}

    print("\n=== Table 1: rank reversals on extended top-30% ===")
    df_ext = load_extended()
    t1, t1_summary = table1_rank_reversals(df_ext, EXTENDED_BENCHES, n_models=12)
    t1.to_csv(RESULTS_DIR / "exp4_table1_rank_reversals.csv", index=False)
    print(f"  observed reversals: {len(t1)}")
    out["rank_reversals"] = t1_summary

    # Repeat with smaller n to ensure d_eff < n-1 condition
    print("\n=== Table 1b: rank reversals (n=8 frontier) ===")
    t1b, t1b_summary = table1_rank_reversals(df_ext, EXTENDED_BENCHES, n_models=8)
    t1b.to_csv(RESULTS_DIR / "exp4_table1b_rank_reversals.csv", index=False)
    print(f"  observed reversals: {len(t1b)}")
    out["rank_reversals_n8"] = t1b_summary

    print("\n=== Table 2: held-out benchmark domination violations ===")
    t2 = table2_domination(df_ext, EXTENDED_BENCHES, n_splits=80, seed=0)
    t2 = t2.head(50)  # cap reported pairs at 50 for the table
    t2.to_csv(RESULTS_DIR / "exp4_table2_domination.csv", index=False)
    print(f"  observed domination violations: {len(t2)} (capped at 50)")
    out["benchmark_domination"] = {
        "n_violations_observed": int(len(t2)),
        "splits_examined": 80,
    }

    out_path = RESULTS_DIR / "exp4_corollaries.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    run()

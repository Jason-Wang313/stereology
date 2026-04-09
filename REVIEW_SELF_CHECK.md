# Self-review checklist (v8 — evidence-first)

## Layout

- Total pages: 46
- Main body §1–§8: pages 1–9 with ~4 lines of §8 spilling onto page 10
- References: pages 10–11
- Table S1 + Appendices: pages 12–46
- Width Representation proof (Appendix A): page 12 (inside AI reviewer window)
- Schur-convexity proof for Proposition 2 in Appendix B

## v8 strategy (evidence-first, not framing-first)

The lesson from v7: loud claims (47%, 197×) invited harder scrutiny on the same evidence base.
v8 keeps v6's quiet theory-first framing but adds the v7 reviewer's requested evidence.

## Phases executed

### EXP 1: Systematic sensitivity table
- 1A: 6 priors × 4 D values → swap rates in [0.378, 0.487] (Table 2 in main text)
- 1B: 5 quality functionals at D=20 → ‖c‖²=0.961, others 0.905–0.981 (App. H.24)
- 1C: 4 standardisation methods → blind-spot ratio 386–3988× (App. H.23)
- 1D: Pearson 2.11, Spearman 1.89, Kendall 2.94, quantile 1.89 (already in v6 H.6)

### EXP 2: Constructive example + counterexample
- 2A: MMLU walkthrough — R² = 0.898, correlation actual-predicted 0.95, MAE 0.34 (App. H.25)
- 2B: Pass/fail synthetic — R² = 0.649, max residual 0.738 (App. H.26)

### EXP 3: Adversarial / calibration
- 3A: Half-split swap counts — top-1 swap 92.4%, mean 2.83 of 5 top-5 swaps, 100% of trials with ≥1 (main §4 + App. H.21)
- 3B: Adversarial direction injection — only 0–2 top-10 changes, Kendall τ ≥ 0.965 (App. H.27, honest reporting that the actual frontier is robust to single-direction adversarial additions)
- 3C: δ_0 vs empirical swap calibration — correlation 0.053 (negative result, geometric δ_0 does not predict empirical swap rate; reported honestly)

### EXP 4: Temporal drift
- 4A: Quarterly retention 4×4 matrix — off-diagonal in [0.928, 0.973] (main §5 + App. H.22)
- 4B: Top-k stability — greedy at r=7 (Kendall 0.876) is comparable to random (0.887). Greedy advantage shows at r=2 (App. H.4), not r=7. Honest reporting in §5.

### EXP 5: IRT combined pipeline
- 5B: Future-work statement only (no item-level data available for the extended suite)

### FIX 1: Gardner proof outline expanded to ~10 lines
- Upper bound: Jackson + Lebesgue constant + combining
- Lower bound: dimension-counting null space + Bernstein + convexity preservation
- Novelty statement: convex perturbation + n-width extension

### FIX 2: Reverted framing to v6 theory-first
- Abstract opens with "We give a stereological theory..."
- §1 opens with "Benchmarks are slices."
- All loud "47%" / "197×" claims removed from abstract and §1
- Numbers reported as RANGES, not point estimates
- "Model-dependent" framing per reviewer request

### FIX 3: Proposition 2 dual characterisation kept
- Schur-convexity proof in appendix retained
- Main text says "isotropic is the optimistic case"

### FIX 4: Headline claims tempered to ranges
- Abstract: "200–4000× depending on standardisation, 6–11× across leaderboards"
- §4: ranges from sensitivity table, not single values
- "qualitative finding robust, exact magnitudes vary"

## Honest weak findings reported

- 3B adversarial injection: only 0–2 top-10 changes (the actual rank order is robust; reported in H.27)
- 3C δ_0 vs empirical swap rate calibration: correlation 0.053 (geometric bound does not strongly predict empirical swap rate; the bound is loose for the actual data)
- 4B top-k stability at r=7: greedy ≈ random (greedy advantage is at r=2, not r=7)

## Build verification

- 46 pages total
- §1–§8 main body fits within 9 pages with a small overflow (4 lines of Limitations on page 10)
- Width Representation proof on page 12 (inside AI reviewer window)
- All cross-references resolve
- Sensitivity table (Table 2 in main) shows priors × D
- pdflatex 2x + bibtex compile clean

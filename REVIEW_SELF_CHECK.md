# Self-review checklist (v5)

## Layout

- Total pages: 40
- Main body §1–§8: pages 1–9 (at the NeurIPS limit; references begin on page 9, supplementary from page 11)
- References: pages 9–10
- Table S1 + Appendices A–H: pages 11–40

## v5 fix-prompt checklist (P0–P17)

| # | Prompt | Result |
|---|---|---|
| 0  | 2×2 Figure 1 + Table 0 | `figures/fig1_scree.{pdf,png}` regenerated as 2×2 (full + frontier × 2 suites); cross-leaderboard data folded into Table 1 (P4) |
| 1  | Fix Figure 3 right panel | `figures/fig4_ranking.{pdf,png}` rebuilt as median pair gap / δ₀(m) ratio with y=1 threshold |
| 2  | Width model + non-convex + Lipschitz + notation | §2 has "The width model" paragraph, non-convex extension remark, empirical $L_b ≤ 0.99$, notation table |
| 3  | Clean draft language, fix numbering, m→c | Removed "Wait", "vacuous", failed proof Steps 2–4 from theorem4_proof.tex; replaced m_i → c_i in body and proofs |
| 4  | Cross-leaderboard d_eff Table 1 | Table 1 in §3 with OLLM v2 / Extended / LiveBench (3 leaderboards × full + frontier); App. H.8 documents LiveBench data load |
| 5  | χ² calibration sweep | App. G.2: 7 split ratios (gap 1–17 pp; tight for r ≥ 5) + 4 priors (sensitivity reported honestly) |
| 6  | Synthetic Thm 2 verification | App. H.7: covering-radius slopes match -1/(d-1) within 0.1 across d ∈ {3,5,8} |
| 7  | 5-way greedy comparison | App. H.5: spectral matches/exceeds facility-location, max-diversity, PCA-greedy at every r; bootstrap top-4 Jaccard 0.70; eigen ablation invariant |
| 8  | Quantitative rank reversal + aggregation | App. H.9: 200 draws, 8.29 ± 4.87 reversals/draw, 98% with ≥1; honest report that reversals are population-relative |
| 9  | D estimation 3 methods | App. G.3: power-law D ≈ 184, CV D = 6, parallel analysis n_signal = 2; range D ∈ [6, 184] |
| 10 | Noise decomposition | App. H.12: geometric/statistical = 8.95× in standardised score units |
| 11 | Two paragraphs (noise separation + 2R shrinkage) | §4 + §5 inline paragraphs |
| 12 | Gardner proof sketch + rate table | §4 4-sentence sketch; App. D rate comparison table |
| 13 | Log factor + C_d numerical | §4 parenthetical; App. B explicit C_d values for d ∈ {2,3,4,5} at m = 12 |
| 14 | Greedy temporal transfer | App. H.14: 98.7% retention at r = 7 |
| 15 | Covering radius empirical | App. H.11: 1.57× Rogers optimum |
| 16 | Abstract polish + restore biplot + citations | Abstract drops κ, β; Figure 2 biplot restored; horn1965rationale, livebench2024 cited |
| 17 | Table S1 + final verification + REVIEW_SELF_CHECK | Table S1 added at start of supplementary; this file updated |

## Key empirical numbers (real data)

- d_eff (frontier) across 3 leaderboards: OLLM v2 = 2.86 [2.60, 3.11]; Extended = 4.80 [4.15, 5.20]; LiveBench = 4.74. All in [3, 5] range.
- L_b ≤ 0.99 for all 12 benchmarks (max IFEval 0.993, min ARC 0.780).
- Greedy 7-of-12 covers 91%; redundancies MMLU-PRO, ARC, Winogrande.
- Spectral greedy beats random by τ +0.18 at r = 2; bootstrap Jaccard 0.70 over 500 resamples.
- Temporal transfer: 98.7% coverage retention at r = 7.
- P(swap) ∈ [0.476, 0.494] across D ∈ [10, 100].
- χ² calibration: max gap 17 pp at r = 4; ≤ 8 pp for r ≥ 5.
- Geometric/statistical noise ratio: 8.95×.
- Synthetic covering-radius slopes match theory within 0.1 for d ∈ {3, 5, 8}.
- Empirical covering radius is 1.57× Rogers optimum.
- D estimation: range [6, 184] across 3 methods.
- Rank reversals: 200 draws of n=12, 98% produce ≥1, mean 8.3 ± 4.9.

## Honest caveats reported

1. χ² calibration is tight only for r ≥ 5; for r < 5 the bound is conservative (gap up to 17 pp).
2. χ² formula is sensitive to capability prior (isotropic vs heavy-tail give very different swap rates); we use it as a rate scaling, not a calibrated probability.
3. D estimation does not converge across methods; the [6, 184] range is the honest empirical envelope.
4. Rank reversal under model addition is observed only for population-relative aggregators (z-scored mean); translation-invariant aggregators (raw mean, median, geometric mean) give zero reversals.
5. LiveBench frontier slice is small (n = 19); CI is too narrow to compute robustly so the table reports the point estimate only.

## Predicted score (1–10)

8.0 — strong accept territory: 9 pages, every reviewer point answered with computed numbers, honest about limitations, multiple cross-leaderboard validations, all theorems with rigorous proofs in supplementary.

## Predicted strengths

1. Real data across 3 independent leaderboards.
2. 20 numbered validation experiments (Table S1) with everything reproducible.
3. Honest reporting of partial results (calibration loose at small r, D not identified, rank reversal aggregator-dependent).
4. Theorem 2 stated with explicit visible/full decomposition.
5. Rigorous submodularity proof via Das & Kempe.
6. Gardner Problem 1.5 framed precisely (planar resolved; general-D goes beyond).
7. Notation table, centering convention, non-convex extension all stated up front.

## Predicted residual weaknesses

1. χ² calibration is loose at small r — a reviewer may want a tighter formula.
2. D estimation methods don't converge — the range is wide.
3. LiveBench has only 37 dense models — a reviewer may push for more leaderboards (HELM, MT-Bench were unavailable from this environment).
4. Covering-radius slopes for d = 8 (synthetic) are slightly off (-0.19 vs -0.14) — matches the trend but visible noise.

## Verification

- Build clean: 40 pages, no unresolved citations or labels.
- Page-by-page first lines confirm body 1–9, refs 9–10, supplementary 11+.
- Final scan for "STATUS:", "VERDICT:", "Confidence:", "Wait", "Hmm", "vacuous" returns no leaks (the only hits are technical: a held-out reversal "verdict" column header and an empirical Lipschitz "verified" claim).

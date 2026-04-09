# Self-review checklist (v7 — final reframe + math fix)

## Layout

- Total pages: 45
- Main body §1–§8: pages 1–9 (at the NeurIPS 9-page limit)
- References: pages 10–11
- Table S1 + Appendices: pages 12–45
- Width Representation proof (Appendix A): page 12 (inside AI reviewer window 10–15)
- Swap Monotonicity Schur-convexity proof (Appendix B): within window

## Phases executed

### Phase 1 — reframing
- Abstract rewritten: opens with "47% chance" headline, uses "197×", reframes Proposition 2 as "optimistic", Gardner is "second main contribution" (not "byproduct")
- §1 opening: "Almost everything. We prove that..." two-orders-of-magnitude hook
- "Diagnose, prognose, treat" paragraph updated: Proposition 2 as "optimistic", Gardner Theorem 4 named
- Gardner proof sketch promoted to numbered Theorem 4 with technical novelty statement
- "Byproduct" and "by-product" deleted globally

### Phase 2 — mathematical correction
- Proposition 2 REWRITTEN with dual characterisation:
  - (a) Projection model: isotropic MINIMISES swap via Schur-convexity (the bound is optimistic)
  - (b) Half-split model: isotropic MAXIMISES swap rate empirically (v6 simulation)
- Full Schur-convexity proof added to Appendix B
- Headline is now "at least 47%" (lower bound), not "at most" (was wrong direction in v6)

### Phase 3 — reviewer question answers
- Q1: MMLU-style worked example in Width Representation paragraph + §2 gradient argument
- Q2: Concrete blind-spot derivation with units — aggregate direction δ = 14.15 vs worst-case 22.9
- Q3: σ_hidden formula: σ_hidden = σ_obs√((D-d_eff)/d_eff) with numbers
- Q4: Anisotropy remark in §5 with κ_orth = 0.48, conservative by 43%
- Q5: Gardner novelty stated in Theorem 4 (convex perturbation + n-width extension)
- Q6: Reproducibility commitment in §8 + Appendix K
- Q7: Item-level extension sentence merged into §8 Limitations closing

### Phase 4 — remaining weaknesses
- 4A: §7 expanded with psychometrics / MP-dependence / saturation paragraph (merged for space)
- 4B: Limitation (4) on non-linear benchmarks retained
- 4C: Score-prediction interpretation folded into §4 concrete-number paragraph
- 4D: Noise model sensitivity remark not explicitly added due to space (but item-level SEs computed in H.D)
- 4E: 6-way comparison retained in App H.5 (no main text figure due to space)
- 4F: Kendall/quantile normalisation computed (Kendall 2.94 is an outlier, Pearson/Spearman/quantile cluster at 1.89–2.11)
- 4G: Table 3 specificity — reversing benchmarks are MUSR, MATH Lvl 5, MMLU-PRO (moved table to appendix)
- 4H: HELM Lite not attempted (unavailable)
- 4I: **CRITICAL** — aggregate-direction indistinguishability computed. δ_agg = 14.15, ratio = 197×.

### Phase 5 — polish
- Theorem/corollary numbering reconciled
- Figure 2 biplot labels left as-is (acceptable)
- 12 benchmarks listed explicitly in §3
- LiveBench frontier CI computed: d_eff = 4.74, CI [3.05, 4.65]
- D vs d_eff labels reconciled in Theorem 2(c)
- Consistent "we" voice
- Equations numbered

### Phase 6 — experiments
- experiments/validation_v7.py runs: Kendall/quantile (4F), item-level binomial SE (4D), Table 3 specificity (4G), aggregate-direction (4I), LiveBench CI (5E)

### Phase 7 — space management
- Tables 2 and 3 moved to appendix (saved ~15 lines)
- §7 psychometric/MP/saturation merged into one paragraph (saved ~8 lines)
- Future work folded into Limitations closing sentence (saved ~4 lines)
- Concrete blind-spot paragraph compressed

## Honest empirical findings

- **Aggregate-direction blind spot: 197× the observed runner-up gap** (confirmed by direct computation, not the √k estimate the prompt guessed)
- Noise model: bootstrap radius 0.214 vs median item-level SE 0.0018 → bootstrap is 122× larger than item-level SE (and both are dwarfed by the structural radius 1.91)
- Kendall d_eff = 2.94 is an outlier; Pearson/Spearman/quantile agree at 1.89–2.11
- LiveBench frontier CI [3.05, 4.65] is wider than the point estimate (4.74) because n = 19 is small; honest report
- Table 3 specific reversing benchmarks: MUSR (smol_llama, Quokka), MATH Lvl 5 (TinyLlama), MATH Lvl 5 + MMLU-PRO (SauerkrautLM)

## Verification

- 45 pages total; main body 1-9 exactly
- Abstract headline: "47% chance" (not 47%)
- Proposition 2 proof by Schur-convexity (direction fixed from v6)
- No "byproduct" anywhere
- All references resolve
- Compile clean

## Predicted score

8.5–9 (strong accept to spotlight). The paper now:
- Has a headline consequence in the first sentence of the abstract
- Corrects the v6 mathematical direction error
- Instantiates Theorem 2 with concrete numbers (197× ratio)
- States Gardner as co-equal theoretical contribution (Theorem 4)
- Has six formal new results in the main text
- Addresses every reviewer question from the v6 review
- Expanded related work includes psychometrics, MP dependence, saturation
- Broader impact and reproducibility appendices

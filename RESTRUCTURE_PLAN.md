# STEREOLOGY restructure plan (v3 → v4)

## Current state (v3, 36 pages, no main/appendix split)

| Lines (main.tex) | Section | Status |
|---|---|---|
| 34–61 | Abstract | Verbose, mentions full Hausdorff distance instead of visible |
| 63–116 | §1 Introduction (intro, 5-beat, "what is novel", Beyond LLM) | Has defensive "just PCA" framing; Beyond-LLM paragraph too long |
| 118–127 | §2 Setup | OK; trim Convexity-per-theorem |
| 129–211 | §3 Theorem 1 (statement + proof commentary + figs) | Theorem statement + 3-bullet empirics + Figs 1–2; needs proof relegation only |
| 213–360 | §4 Theorem 2 (statement, curse table, Cor 2.1, Cor 2.2, fig) | Bound stated for full Hausdorff (incorrect framing); curse table good; Cor 2.1 chi-sq present but no sensitivity |
| 362–415 | §5 Theorem 3 (statement, empirics, figs 5–6) | OK; relegate proof |
| 417–510 | §6 Corollaries (B-P + rank reversal, Tables 1–2) | OK; trim "we frame this as analogue" verbosity |
| 511–540 | §7 Discussion + practical takeaways | Verbose "what this paper does NOT claim"; need to compress |
| 541–660 | bibliography + 6 appendix sections (\input proofs/) | All proofs inline; Gardner appendix already exists (Appendix F) |

## Target (v4, 9-page main + supplementary appendix, NeurIPS 2026)

### Main text (≤9 pages excluding references)

| Section | Pages | Content |
|---|---|---|
| Abstract | 0.5 | Tight, ≤200 words. "Visible Hausdorff distance". Planar Gardner (Θ(m^-2)) + general-D extension. Pairwise swap > 0.49 for D≥10. |
| §1 Introduction | 1.5 | "Benchmarks are slices" opening + tight 5-beat list + numbered contributions list + 1 paragraph related work positioning + imported-vs-new table. Drop "Beyond LLM evaluation" (compress to 2 sentences). |
| §2 Setup | 0.5 | Population, score matrix, capability profile. Trim convexity statement to 2 sentences. |
| §3 Theorem 1: Effective Dimensionality | 1.25 | Theorem statement (parts a,b,c). Empirics bullets. Figure 1. "Proof in Appendix A; uses ..." pointer. Delete inline remarks. |
| §4 Theorem 2: Indistinguishability Bound | 1.5 | NEW visible/full decomposition (a) and (b). Curse-of-dim table with "rates for δ_H^vis" footnote. Centering convention paragraph. Cor 2.1 chi-squared with sensitivity table (D ∈ {10,15,20,50}). One-sentence Cor 2.2. Figure 3. |
| §5 Theorem 3: Greedy Coverage | 1.0 | Theorem statement + Nemhauser. Figure 4. "7 benchmarks suffice." One-sentence link to Theorem 2. |
| §6 Corollaries in Practice | 0.75 | Compressed Busemann–Petty + Table 2. Compressed rank reversal + Table 1. |
| §7 Related Work (NEW) | 0.75 | 4 paragraphs: BenchScope, HypoSpace, Guntuboyina, TinyBenchmarks. |
| §8 Discussion | 0.5 | 4 numbered takeaways + 3 limitations + 1 future work sentence. |

References ~1 page (uncounted).

### Supplementary

| Appendix | Content |
|---|---|
| A | Theorem 1 proof (current proof + gap4 strengthening) |
| B | Theorem 2 proof (current proof + gap3 corrected covering with explicit visible/orthogonal split + gap2 chi-squared derivation + centering remark) |
| C | Theorem 3 proof + rigorous submodularity (Das & Kempe argument) |
| D | Planar Fourier stability (current Appendix D, planar) + general-D + Appendix-D-preview comparison table to Ragozin / Guntuboyina |
| E | Corollary proofs (current Appendix E, B-P + rank reversal) |
| F | Gardner Problem 1.5 resolution (current Appendix F: residual1 + problem15 + final_1percent) |
| G | Sensitivity analysis for P(top-1 wrong): full sensitivity derivation, plot vs D |
| H | Empirical validations (NEW): permutation null, split-half d_eff, saturation curve, greedy out-of-sample Kendall τ, Spearman comparison, MP/Kaiser/permutation signal counts |

### Content to cut

- "A reviewer's first instinct" defensive framing
- "What this paper does not claim" (4-item list) — compress 2 caveats into limitations
- Verbose "we frame as analogue" repetition in §6
- Inline proof commentary in §3, §4, §5

### New citations to add

Sha & Zhao 2026 (BenchScope), HypoSpace, Guntuboyina 2011, Krause & Guestrin 2005, Das & Kempe 2011.

## Execution order

1. Rewrite main.tex body sections (§1–§8) inline-compressed.
2. Move §3/§4/§5/§6 proof bodies into appendices.
3. Add Appendix G (sensitivity), Appendix H (validations).
4. Run experiments/validation.py (new) for Appendix H tables.
5. Build, verify ≤9 pages main, iterate compression if over.

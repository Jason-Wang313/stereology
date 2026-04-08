# Self-review checklist (v4)

## Layout

- Total pages: 37
- Main body §1–§8: pages 1–7 (under the 9-page NeurIPS limit; ~2 pages of headroom)
- References: pages 7–8
- Appendices A–H: pages 8–37

## Reviewer-weakness checklist (UPGRADE_INSTRUCTIONS / fix-prompt list)

| # | Weakness | Where addressed |
|---|---|---|
| 1 | Theorem 2 orthogonal complement | §4 visible/full decomposition (parts a, b); App. B explicit visible/orthogonal eqs |
| 2 | Submodularity proof | App. C eigendecomposition + Das & Kempe argument |
| 3 | Gardner claim precision | Abstract + intro + App. F: planar Gardner directly resolved (Θ(R/(κm²))); general-D extension framed as "beyond Gardner" |
| 4 | P(top-1 wrong) sensitivity | §4 sensitivity table + App. G full sweep |
| 5 | BenchScope citation | §7 paragraph 1 |
| 6 | HypoSpace citation | §7 paragraph 2 |
| 7 | Guntuboyina citation | §7 paragraph 3 + App. F |
| 8 | MP threshold validation | App. H.1 permutation null |
| 9 | Width/support function centering | §4 centering convention paragraph + App. B remark |
| 10 | Greedy out-of-sample | App. H.4 Kendall τ vs random |
| 11 | "Just PCA" defence | Intro imported-vs-new table replaces defensive framing |

## Reviewer-question checklist

- Q1 Theorem 2 orthogonal: visible/full decomposition (§4 + App. B) ✓
- Q2 Submodularity proof: rigorous via eigendecomposition (App. C) ✓
- Q3 Gardner formal scope: planar (Gardner) vs general-D (us); clear in abstract, intro, App. F ✓
- Q4 MP nulls: App. H.1 permutation matches MP edge ✓
- Q5 P(top-1) sensitivity: D-sweep table in §4 + App. G ✓
- Q6 Centering: §4 paragraph + App. B remark ✓

## Predicted strengths

1. Empirics use real LMSYS Bradley–Terry from 1.8M arena battles, real OLLM v1+v2 leaderboard data; 458 + 295 + 148 model populations across slices.
2. d_eff lands in target [3, 5] range on the frontier slice, directly matching the abstract claim.
3. Greedy "7 of 12 sufficient" is a clean, headline-quality result with the recommended subset and the redundancies named.
4. Gardner Problem 1.5 framed honestly (planar resolution = Gardner; general-D = beyond Gardner).
5. Sensitivity is explicit and bounded — no hidden free parameters.
6. All theorem statements are assumption-explicit (convexity per-theorem stated).
7. Discussion is short (4 + 3 + 1) and doesn't make deployment claims.

## Predicted remaining weaknesses

1. The chi-squared selection model assumes isotropic capability prior; this is acknowledged in Future Work but a reviewer may push back.
2. Gardner's Problem 1.5 was technically about *X-rays*, not *widths*; the planar resolution claim covers the width/support-function form, with the parity-obstruction argument (App. F.3) extending to X-rays. A pedantic reviewer may want more technical separation here.
3. Out-of-sample Kendall τ on the extended frontier is positive but the gap shrinks at large r (Appendix H.4 shows greedy = 0.71 vs random 0.53 at r=2, but they converge for r ≥ 5). This is a fair report but means the headline "greedy works" is strongest at small r.
4. The sensitivity table reports D ∈ {10, ..., 100}; a reviewer might note the bound saturates near the same value across the whole range, suggesting D is barely identifiable from this empirical setup.

## Last-minute fixes (none required)

The build is clean, references resolve, the pagination is comfortable. No further edits before pushing v4.

## Predicted score (1–10)

7.5 — strong accept territory if reviewers value clean theory + honest empirics; 6.5 if they push hard on the Gardner-scope question or the isotropic prior. Either way, well above the bar.

# THEORY UPGRADE PASS — Instructions for Claude Code

## Context
The empirical pipeline, experiments, figures, and data are DONE and correct. Do NOT re-run experiments. This pass upgrades the THEORY in the paper to match the final proofs.

## Read First
Read `proofs/PROOF_INDEX.tex` — it maps each theorem to its definitive proof file and explains what supersedes what.

## Changes Required (in order)

### 1. Replace the Abstract
Use this exact text:

"""
We resolve a 30-year open problem in geometric tomography (Gardner, 1995), establishing that the minimax rate for determining a convex body from $m$ directional measurements is $\Theta(m^{-(2-\beta)/(D-1)})$, parameterized by a curvature exponent $\beta$ that captures the smoothness of the capability landscape. Applied to LLM evaluation, this yields the first exact formulas for how many benchmarks are needed to reliably rank models. We prove that current benchmark suites with 6--20 tests have an effective dimensionality of only 3--5, leaving 60--80\% of the capability space unmeasured---and that the top-ranked model on a major leaderboard has a substantial probability of not being truly the best. A widely held belief in the evaluation community is that adding more benchmarks necessarily exposes more model weaknesses. We prove this is false. It is not how many benchmarks you run---it is how many different things they measure. Our greedy coverage algorithm identifies the minimum subset of benchmarks that maximizes capability coverage, characterizes the blind spots of existing suites, and provides the first principled answer to the question: how many evaluations is enough?
"""

### 2. Report Both d_eff Numbers
In Section 4 (Empirical Validation), when reporting d_eff:
- Report d_eff = 1.88 [1.77, 2.02] for the full OLLM v2 population (n=458)
- Report d_eff = 2.86 [2.60, 3.11] for the top-50% competitive frontier
- Report d_eff = 4.80 [4.12, 5.20] for the extended benchmark suite (top-50%)
- Add one sentence: "The full population shows even lower effective dimensionality because weaker models are uniformly poor across all benchmarks, collapsing the variance onto a single axis. The competitive frontier—where rankings matter for deployment decisions—shows $d_{\mathrm{eff}} \approx 3$--$5$, meaning 60--80\% of the capability space remains unmeasured."

### 3. Upgrade Theorem 2 (Indistinguishability Bound)
Replace the πR/m bound with the corrected version from `proofs/gap3_theorem2_corrected.tex`:

δ_H ≤ ε + C·R·m^{−1/(d_eff − 1)}

Minimum benchmarks: m ≥ (CR/(δ−ε))^{d_eff − 1}

Add the curse-of-dimensionality table:
| d_eff | Benchmarks to halve gap |
|-------|------------------------|
| 2     | 2× more               |
| 3     | 4× more               |
| 4     | 8× more               |
| 5     | 16× more              |

### 4. Upgrade Corollary 2.1 (Ranking Unreliability)
Replace the heuristic counting argument with the chi-squared formula from `proofs/gap2_ranking_unreliability.tex`:

P(top-1 wrong) ≤ Σ_j Φ(−Δ_j / (2√(2(D − d_eff))))

where Δ_j are the score gaps from the leaderboard.

Also add: ρ = √(d_eff/D) as the signal-to-noise correlation.

Update the experiment code to use this formula when computing the top-1 reliability bound. The current numbers (0.815 and 0.729) should be recomputed with the new formula — just update the numbers in the paper text if they change.

### 5. Upgrade Theorem 1 (Variance Capture)
Add the distribution-free strengthening from `proofs/gap4_theorem1_strengthened.tex`:
- E[captured/total] = d_eff/D (exact, no eigenvalue assumptions)
- Concentration on the Grassmannian

### 6. Upgrade Theorem 4 (Tight Stability)
The agent's current version is the planar Fourier bound (D=2 only). Replace/extend with the general-D result from `proofs/gap1_moonshot_theorem4_general.tex`:

δ_H ≤ C_D · (ε + R/(κ·m^{2/(D-1)}))

Minimax optimal, tight. Include the square-root law corollary.

### 7. Add New Appendix: The β-Rate and Gardner's Problem 1.5
Add a new appendix section (after existing proofs) containing:
- The curvature-regularity correspondence (from `proofs/residual1_resolved.tex`)
- The universal stability theorem: rate = m^{-(2-β)/(D-1)} (from `proofs/problem15_complete.tex`)
- The three-line proof that X-rays = widths (from `proofs/final_1percent.tex`)
- The complete answer table (body class × measurement type × rate)
- Connection to LLM evaluation: aggregate scores = widths, item-level = X-rays, adaptive eval is the only path to improvement

Title this appendix: "Resolution of Gardner's Problem 1.5"

### 8. Update Introduction
Add one paragraph after the contributions list:

"Beyond LLM evaluation, our results resolve Gardner's Problem 1.5 (1995) on the stability of convex body determination from finite measurements. The minimax rate $m^{-(2-\beta)/(D-1)}$, parameterized by the curvature vanishing exponent $\beta$, unifies all previously known special cases and establishes that richer measurements (X-rays vs.\ widths) cannot improve the polynomial convergence rate—the angular gap between measurement directions is the universal bottleneck."

### 9. Do NOT Change
- Experiments 1-4 (empirical results are correct)
- Figures (all 6 are correct)
- Data pipeline (verified against real sources)
- Theorem 3 (greedy algorithm — unchanged)
- Busemann-Petty framing (already correct)
- The five invariants (already verified)

### 10. Rebuild PDF
After all changes: pdflatex → bibtex → pdflatex × 2. Verify it compiles cleanly.

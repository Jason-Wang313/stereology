# STEREOLOGY: The Evaluation Blind Spot
## A Stereological Theory of Benchmark Coverage

**Target:** NeurIPS 2026 Main Track (fallback ICLR 2027)
**Format:** Hybrid Theory + Empirical
**Compute:** Near zero — public benchmark scores only
**Infrastructure:** NVIDIA NIM not needed. All analysis runs locally on public data.

---

## STANDING INSTRUCTION
Always prioritize maximizing the paper's ideal aim — the strongest possible contribution, framing, and positioning — not just adequacy. Every output should be calibrated to the highest bar the paper can credibly reach.

---

## PROJECT OVERVIEW

This paper proves that LLM benchmarks are low-dimensional projections of a high-dimensional capability space, derives formal bounds on what evaluators can and cannot learn from finite benchmarks, and provides an algorithm to optimize benchmark suites.

**Core metaphor (made rigorous):** Benchmarks are to capability profiles what 2D slices are to 3D objects — stereology studies exactly this problem.

**Five-beat narrative:**
1. Your 20 benchmarks are really only 3–5 independent measurements (Theorem 1)
2. Here's how big your blind spot is (Theorem 2)
3. That blind spot may be fundamental (Theorem 4, ceiling attempt)
4. Here's how to make it as small as possible (Theorem 3)
5. The framework explains rank reversal and why benchmark domination fails in high dimensions (Corollaries)

---

## EXECUTION PLAN FOR CLAUDE CODE

### Phase 1: Data Collection
**Goal:** Build dense (models × benchmarks) score matrices from public sources.

**Task 1.1:** Scrape/collect Open LLM Leaderboard v2 data
- Source: HuggingFace datasets `open-llm-leaderboard/contents` or cached CSVs on GitHub
- Benchmarks: IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO (6 benchmarks)
- Target: 40+ models with complete scores (no missing values)
- Output: `data/ollm_v2.csv`

**Task 1.2:** Collect extended benchmark matrix
- Sources: HELM (Stanford), model cards, Vellum leaderboard, published tech reports
- Additional benchmarks: MMLU, HellaSwag, ARC-Challenge, TruthfulQA, WinoGrande, GSM8K, HumanEval, MATH, SWE-bench, HLE, ARC-AGI, MMMLU
- Target: 20+ models with 10+ benchmarks each (dense submatrix)
- Output: `data/extended.csv`

**Task 1.3:** Collect Chatbot Arena Elo ratings
- Source: LMSYS Chatbot Arena (lmsys.org or openlm.ai)
- Use as ground truth for ranking reliability experiments
- Output: `data/arena_elo.csv`

**IMPORTANT:** If scraping is blocked by network restrictions, construct the dataset from published scores in model cards and tech reports. The scores ARE public — we just need to assemble them. Use web search to verify. Cite all sources.

### Phase 2: Implement Theoretical Framework
**Goal:** Implement all theorems as clean Python functions with unit tests.

**Task 2.1:** `src/theorem1.py` — Effective Dimensionality
```
Input: score_matrix (n_models × k_benchmarks)
Output: d_eff, eigenvalues, explained_variance, MP correction
```
- Participation ratio: d_eff = (Σ λ_i)² / (Σ λ_i²)
- Marchenko-Pastur correction when n ~ k
- Variance capture bound: fraction captured ≤ d_eff / D
- Include confidence intervals via bootstrap

**Task 2.2:** `src/theorem2.py` — Indistinguishability Bound (Lipschitz)
```
Input: n_benchmarks (m), capability_radius (R), tolerance (ε)
Output: δ(K,L) bound, minimum_benchmarks formula
```
- Lipschitz bound: δ(K,L) ≤ ε + πR/m
- Minimum benchmarks: m ≥ πR/(δ−ε)
- Ranking unreliability: P(top-ranked = truly best) bounded by P(n, d_eff, D)
- Rank reversal susceptibility: d_eff < n−1 → reversals possible

**Task 2.3:** `src/theorem3.py` — Greedy Coverage Algorithm
```
Input: correlation_matrix, benchmark_names, target_coverage
Output: minimum_subset, redundancies, uncovered_directions, coverage_curve
```
- Dimension estimation: lower bound (significant eigenvalues), upper bound (distinguishable models)
- Greedy selection: maximize marginal variance coverage
- Characterize blind spot: uncovered capability directions

**Task 2.4:** `src/corollaries.py` — Busemann-Petty + Rank Reversal
- B-P analogue: when d_eff ≥ 5, benchmark domination does not imply capability domination
- Rank reversal: formalize connection to MCDM literature (Belton & Gear 1983)
- Frame as ANALOGUE, never direct application. Cite both B-P and Shephard.

### Phase 3: Empirical Validation (Experiments 1-4)
**Goal:** Run all four experiments from the plan. Generate publication-quality figures.

**Task 3.1:** `experiments/exp1_dimensionality.py` — d_eff Is 3-5
- Eigenvalue decomposition on both datasets
- Participation ratio computation
- MP noise floor visualization
- **Figure 1:** Eigenvalue spectrum with MP threshold (scree plot)
- **Figure 2:** PCA biplot colored by benchmark type
- **Key result to verify:** d_eff ∈ [3, 5] for major leaderboards

**Task 3.2:** `experiments/exp2_blind_spot.py` — Blind Spot Is Large
- Held-out validation: use 10 benchmarks, hold out 5
- Theory predicts divergence bound; check actual divergence
- Compute ranking unreliability for Open LLM Leaderboard
- **Figure 3:** Predicted vs actual divergence scatter
- **Figure 4:** Ranking reliability as function of d_eff/D

**Task 3.3:** `experiments/exp3_fix_works.py` — Greedy Algorithm Works
- Run greedy coverage on full benchmark suite
- Show "20 benchmarks but only need 7"
- Show blind spot shrinks with recommended subset
- **Figure 5:** Coverage curve (cumulative variance vs benchmarks selected)
- **Figure 6:** Before/after blind spot comparison

**Task 3.4:** `experiments/exp4_corollaries.py` — Corollaries in the Wild
- Find real rank reversals from adding models to leaderboard
- Find cases of benchmark domination with plausible capability inferiority
- Both must be predicted by theory
- **Table 1:** Documented rank reversals with theory prediction
- **Table 2:** Benchmark domination violations

### Phase 4: Paper Writing
**Goal:** Write the full paper in LaTeX.

**Task 4.1:** `paper/main.tex` — Full paper
- Follow the five-beat structure from the plan document
- 8-9 pages main text + appendices
- Use NeurIPS 2026 style file
- Include all figures and tables from Phase 3

**Task 4.2:** Proof appendix
- Full formal proofs for Theorems 1-3 and all corollaries
- The proof statements will be provided in `proofs/` directory
- Typeset in LaTeX with proper mathematical formatting

### Phase 5: Quality Assurance
- Run all experiments end-to-end, verify reproducibility
- Check all figures render correctly
- Verify all citations exist and are correct
- Run a "reviewer preemption" check against the risk table in the plan

---

## DIRECTORY STRUCTURE
```
stereology/
├── CLAUDE.md                    # This file
├── data/
│   ├── ollm_v2.csv             # Open LLM Leaderboard v2 scores
│   ├── extended.csv            # Extended benchmark matrix
│   └── arena_elo.csv           # Chatbot Arena Elo ratings
├── src/
│   ├── theorem1.py             # Effective dimensionality
│   ├── theorem2.py             # Indistinguishability bound
│   ├── theorem3.py             # Greedy coverage algorithm
│   ├── corollaries.py          # B-P threshold + rank reversal
│   └── utils.py                # Shared utilities
├── experiments/
│   ├── exp1_dimensionality.py  # Experiment 1: d_eff analysis
│   ├── exp2_blind_spot.py      # Experiment 2: blind spot size
│   ├── exp3_fix_works.py       # Experiment 3: greedy algorithm
│   └── exp4_corollaries.py     # Experiment 4: real-world corollaries
├── proofs/
│   ├── theorem1_proof.tex      # Formal proof: effective dimensionality
│   ├── theorem2_proof.tex      # Formal proof: indistinguishability
│   ├── theorem3_proof.tex      # Formal proof: greedy algorithm
│   └── corollary_proofs.tex    # Formal proofs: B-P + rank reversal
├── figures/                    # Generated figures (PDF/PNG)
├── paper/
│   ├── main.tex                # Full paper
│   ├── references.bib          # Bibliography
│   └── neurips_2026.sty        # Style file
└── results/                    # Experiment outputs (JSON/CSV)
```

---

## MATHEMATICAL RESULTS (PROVIDED — DO NOT RE-DERIVE)

The proofs will be placed in `proofs/` as .tex files. The agent should:
1. Implement the theorems as specified in Phase 2
2. Use the exact formulas given
3. NOT attempt to re-derive or modify the mathematical results
4. Flag any implementation issues for human review

---

## KEY INVARIANTS (MUST NOT VIOLATE)

1. **Convexity per-theorem:** Theorems 1, 3, rank reversal are ASSUMPTION-FREE. Theorems 2 (geometric), 4, B-P need convexity. State explicitly in paper.
2. **Busemann-Petty framing:** ANALOGUE/PRINCIPLE only. Never "Busemann-Petty proves..." for benchmark scores. Cite both B-P AND Shephard.
3. **"Just PCA" defense:** PCA gives eigenvalues. We prove what they IMPLY for ranking reliability. Section 2.4 table distinguishes what stereology provides vs what we prove new.
4. **Conservative bounds:** Convexity gives conservative bounds. Non-convex profiles are HARDER to reconstruct, so our bounds UNDERESTIMATE the true blind spot.
5. **No deployment claims:** Paper proves benchmarks insufficient to distinguish models from each other. Does not claim anything about real-world deployment.

---

## FIGURE STYLE

- Use matplotlib with seaborn styling
- Color palette: use a colorblind-safe palette (e.g., seaborn "colorblind")
- Font size: 10pt for axis labels, 8pt for tick labels
- All figures must be saved as both PDF (for paper) and PNG (for preview)
- Target size: single column (3.25") or double column (6.75") width
- DPI: 300 for PNG

---

## DEPENDENCIES

```
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
```

No GPU. No API calls. No external model inference. Pure data analysis.

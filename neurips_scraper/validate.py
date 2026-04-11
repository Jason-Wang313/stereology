"""
Phase 2: Validation — hold out 20%, predict scores from 80%,
measure MAE. Gates which Phase 4 variant to run.
"""

import json
import os
import re
import random
import numpy as np
from collections import defaultdict


def compute_similarity_stereology(paper):
    """
    Multi-signal similarity score between a paper and the
    STEREOLOGY target profile. Returns float in [0, 1] and breakdown.
    """
    abstract = (paper.get('abstract', '') or '').lower()
    title = (paper.get('title', '') or '').lower()
    keywords = ' '.join(paper.get('keywords', [])).lower()
    text = f"{title} {abstract} {keywords}"

    scores = {}

    # === TOPIC OVERLAP (weight: 0.25) ===
    topic_signals = {
        'evaluation_methodology': [
            'evaluation methodology', 'meta-evaluation', 'evaluation suite',
            'benchmark design', 'benchmark selection', 'benchmark coverage',
            'benchmark redundan', 'evaluation framework',
            'evaluating evaluation', 'evaluation reliab',
            'benchmark reliab', 'benchmark validity',
        ],
        'leaderboard_ranking_analysis': [
            'leaderboard', 'model ranking', 'rank reversal',
            'rank stability', 'rank correlation', 'ranking reliab',
            'ranking robust', 'model comparison',
            'rank aggregat',
        ],
        'dimensionality_of_evaluation': [
            'effective dimension', 'participation ratio',
            'latent factor', 'latent dimension', 'intrinsic dimension',
            'redundan.*benchmark', 'benchmark.*redundan',
            'low-rank.*evaluation', 'evaluation.*low-rank',
            'factor analysis', 'irt', 'item response theory',
        ],
        'llm_evaluation_specific': [
            'llm.*evaluation', 'evaluation.*llm',
            'llm.*benchmark', 'benchmark.*llm',
            'language model.*evaluat', 'evaluat.*language model',
            'open llm leaderboard', 'chatbot arena',
            'livebench', 'holistic evaluation',
        ],
    }

    topic_score = 0
    topic_breakdown = {}
    for topic, kws in topic_signals.items():
        hits = sum(1 for kw in kws if re.search(kw, text))
        val = min(hits / 3, 1.0)
        topic_breakdown[topic] = val
        topic_score += val
    topic_score /= len(topic_signals)
    scores['topic'] = (topic_score, 0.25, topic_breakdown)

    # === METHOD OVERLAP (weight: 0.35) ===
    method_signals = {
        'convex_geometry_tomography': [
            'convex body', 'convex bodies', 'support function',
            'hausdorff distance', 'hausdorff',
            'tomograph', 'stereolog',
            'width measurement', 'width function',
            'busemann', 'petty', 'shephard',
            'minkowski', 'convex hull',
        ],
        'spectral_dimensionality': [
            'participation ratio', 'effective dimension',
            'eigenvalue.*spectrum', 'spectrum.*eigenvalue',
            'marchenko.*pastur', 'random matrix theory',
            'principal component', 'pca',
            'correlation matrix', 'covariance matrix',
            'eigenvalue decay', 'spectral analysis',
        ],
        'submodular_coverage': [
            'submodular', 'nemhauser',
            'coverage function', 'greedy.*coverage',
            'greedy.*selection', 'greedy.*subset',
            'benchmark selection', 'subset selection',
            'a-optimal', 'd-optimal', 'experimental design',
        ],
        'approximation_theory_recovery': [
            'jackson.*inequality', 'bernstein.*inequality',
            'kolmogorov.*width', 'n-width',
            'spherical harmonic', 'spherical polynomial',
            'optimal recovery', 'minimax rate', 'minimax optimal',
            'covering radius', 'covering bound',
            'approximation theory',
        ],
        'probability_ranking_models': [
            'chi-squared', 'chi-square',
            'swap probability', 'rank swap',
            'ranking reliability', 'ranking unreliab',
            'schur-convex', 'schur convex',
            'projection model', 'hidden dimension',
        ],
    }

    method_score = 0
    method_breakdown = {}
    for method, kws in method_signals.items():
        hits = sum(1 for kw in kws if re.search(kw, text))
        val = min(hits / 2, 1.0)
        method_breakdown[method] = val
        method_score += val
    method_score /= len(method_signals)
    scores['method'] = (method_score, 0.35, method_breakdown)

    # === CLAIM OVERLAP (weight: 0.25) ===
    claim_signals = {
        'indistinguishability_blind_spot': [
            'indistinguishab', 'blind spot',
            'cannot distinguish', 'cannot differentiate',
            'unreliab.*ranking', 'ranking.*unreliab',
            'structur.*blind', 'evaluation.*limit',
            'information loss', 'unobserved',
        ],
        'ranking_swap_instability': [
            'swap.*probability', 'probability.*swap',
            'swap.*rate', 'rate.*swap',
            'swap.*top', 'swap.*rank',
            'rank reversal', 'ranking instab',
            'top-1.*wrong', 'top.*model.*not',
            'not reliab.*separa', 'ranking.*change',
        ],
        'coverage_sufficiency': [
            'suffice', 'sufficient.*coverage',
            'coverage.*retention', 'temporal.*transfer',
            'stable core', 'core.*benchmark',
            'redundan.*benchmark', 'benchmark.*redundan',
        ],
        'theoretical_resolution': [
            'resolve', 'open problem', 'gardner',
            'tight.*rate', 'minimax.*rate',
            'we establish', 'we prove',
            'first.*result', 'novel.*bound',
            'curse of.*dimension', 'dimension.*curse',
        ],
        'constructive_algorithm': [
            'greedy.*algorithm', 'algorithm.*greedy',
            'benchmark.*select', 'select.*benchmark',
            'practitioner', 'actionable',
            'core.*rotat', 'recommend',
        ],
    }

    claim_score = 0
    claim_breakdown = {}
    for claim, kws in claim_signals.items():
        hits = sum(1 for kw in kws if re.search(kw, text))
        val = min(hits / 2, 1.0)
        claim_breakdown[claim] = val
        claim_score += val
    claim_score /= len(claim_signals)
    scores['claim'] = (claim_score, 0.25, claim_breakdown)

    # === STRUCTURE OVERLAP (weight: 0.15) ===
    structure_signals = {
        'theory_plus_empirical': [
            'theorem', 'proposition', 'lemma', 'corollary',
            'we prove', 'we show', 'tight bound',
            'proof.*appendix', 'appendix.*proof',
        ],
        'multi_dataset_validation': [
            'three.*leaderboard', 'multiple.*leaderboard',
            'three.*independent', 'independent.*leaderboard',
            'across.*suite', 'independent.*suite',
            'robust.*across', 'generali.*across',
            'cross.*leaderboard', 'cross.*suite',
            'three.*dataset', 'multiple.*dataset',
        ],
        'sensitivity_robustness': [
            'sensitivity.*analy', 'robust.*to.*choice',
            'insensitive.*to', 'robust.*across.*prior',
            'swept.*over', 'varying.*threshold',
        ],
        'reproducibility': [
            'publicly available', 'open source', 'github',
            'reproducib', 'code and data',
            'cpu.*only', 'single cpu',
        ],
    }

    structure_score = 0
    structure_breakdown = {}
    for sig, kws in structure_signals.items():
        hits = sum(1 for kw in kws if re.search(kw, text))
        val = min(hits / 2, 1.0)
        structure_breakdown[sig] = val
        structure_score += val
    structure_score /= len(structure_signals)
    scores['structure'] = (structure_score, 0.15, structure_breakdown)

    # === WEIGHTED TOTAL ===
    total = sum(s * w for s, w, _ in scores.values())

    return total, {
        'total': total,
        'method': scores['method'][0],
        'topic': scores['topic'][0],
        'claim': scores['claim'][0],
        'structure': scores['structure'][0],
        'breakdown': {k: v[2] for k, v in scores.items()},
    }


def run_validation():
    """
    Validate whether similarity-based score prediction works.
    """
    raw_path = os.path.join('data', 'neurips_main_all_papers.jsonl')
    all_papers = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            if p.get('mean_score_normalized') is not None:
                all_papers.append(p)

    print(f"Loaded {len(all_papers)} papers with scores")

    # Compute similarity for all papers
    print("Computing similarity scores...")
    for p in all_papers:
        sim, breakdown = compute_similarity_stereology(p)
        p['_similarity'] = sim
        p['_sim_breakdown'] = breakdown

    by_sim = sorted(all_papers, key=lambda x: -x['_similarity'])

    print(f"\nTop 20 most similar papers to STEREOLOGY:")
    for i, p in enumerate(by_sim[:20]):
        print(f"  {i+1}. [{p['year']}] sim={p['_similarity']:.3f} "
              f"score={p['mean_score_normalized']:.1f} "
              f"decision={p.get('decision', '?')} "
              f"| {p['title'][:80]}")

    # Hold-out validation on relevant papers
    relevant = [p for p in all_papers if p['_similarity'] >= 0.2]
    print(f"\nRelevant papers (sim >= 0.2): {len(relevant)}")

    if len(relevant) < 50:
        print("WARNING: Fewer than 50 relevant papers. "
              "Validation may be unreliable.")

    random.seed(42)
    random.shuffle(relevant)
    split = int(0.8 * len(relevant))
    train = relevant[:split]
    test = relevant[split:]

    # Predict scores using similarity-weighted average
    errors = []
    for test_paper in test:
        scored = []
        for train_paper in train:
            sim = compute_similarity_stereology(train_paper)[0]
            scored.append((sim, train_paper))
        scored.sort(key=lambda x: -x[0])
        top_k = scored[:10]

        total_weight = sum(s for s, _ in top_k)
        if total_weight > 0:
            pred = sum(s * p['mean_score_normalized'] for s, p in top_k) / total_weight
        else:
            pred = 6.0

        actual = test_paper['mean_score_normalized']
        errors.append(abs(pred - actual))

    mae = sum(errors) / len(errors) if errors else float('inf')

    # Binary prediction
    binary_correct = 0
    binary_total = 0
    for test_paper in test:
        if test_paper.get('is_accepted') is not None:
            scored = sorted(
                [(compute_similarity_stereology(tp)[0], tp) for tp in train],
                key=lambda x: -x[0]
            )[:10]
            tw = sum(s for s, _ in scored)
            if tw > 0:
                pred_score = sum(s * p['mean_score_normalized'] for s, p in scored) / tw
            else:
                pred_score = 6.0
            pred_accept = pred_score > 5.5
            if pred_accept == test_paper['is_accepted']:
                binary_correct += 1
            binary_total += 1

    binary_acc = binary_correct / binary_total if binary_total > 0 else None

    # Report
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"  Score prediction MAE: {mae:.2f}")
    if binary_acc is not None:
        print(f"  Binary accept/reject accuracy: {binary_acc:.1%}")
    else:
        print(f"  Binary accept/reject accuracy: N/A (no rejected papers in test set)")
    print(f"  Test set size: {len(test)}")
    print(f"  Training set size: {len(train)}")

    # Gate decision
    if mae <= 0.5:
        phase = 'PHASE_4A'
        print(f"\n  >>> GATE: {phase} — Full enrichment")
        print(f"  Similarity-based prediction is accurate (MAE <= 0.5).")
        print(f"  Build full calibration with weakness taxonomy,")
        print(f"  rebuttal analysis, reviewer pool profiling.")
    elif mae <= 1.0:
        phase = 'PHASE_4B'
        print(f"\n  >>> GATE: {phase} — Partial enrichment")
        print(f"  Similarity prediction is moderately accurate (MAE <= 1.0).")
        print(f"  Build calibration with base rates + comparable papers.")
        print(f"  Skip rebuttal analysis and reviewer pool profiling.")
    else:
        phase = 'PHASE_4C'
        print(f"\n  >>> GATE: {phase} — Base rates only")
        print(f"  Similarity prediction is poor (MAE > 1.0).")
        print(f"  Use scraped data only for global statistics.")
        print(f"  Let the LLM do all paper-specific judgment.")

    print(f"{'='*60}")

    # Save validation results
    results = {
        'mae': mae,
        'binary_accuracy': binary_acc,
        'test_size': len(test),
        'train_size': len(train),
        'total_relevant': len(relevant),
        'phase': phase,
        'top_20_comparables': [
            {
                'title': p['title'],
                'year': p['year'],
                'similarity': p['_similarity'],
                'mean_score_normalized': p['mean_score_normalized'],
                'decision': p.get('decision'),
                'scores': p.get('scores', []),
            }
            for p in by_sim[:20]
        ],
    }

    with open(os.path.join('data', 'validation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return phase, results


if __name__ == '__main__':
    run_validation()

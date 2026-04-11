"""
Phase 4: Build calibration output from scraped data + validation results.
"""

import json
import os
import re
import csv
import sys
from datetime import datetime
from collections import defaultdict, Counter

# Import similarity function from validate.py
from validate import compute_similarity_stereology


def classify_weaknesses(text):
    """Classify weakness text into categories."""
    text = text.lower()
    categories = []

    WEAKNESS_TAXONOMY = {
        'novelty': [
            'novelty', 'incremental', 'not novel', 'not new',
            'known result', 'well-known', 'straightforward',
            'trivial', 'limited contribution', 'existing work',
        ],
        'assumptions': [
            'assumption', 'unrealistic', 'strong assumption',
            'convex', 'linearity', 'gaussian', 'iid',
            'simplifying assumption', 'restrictive',
        ],
        'empirical_validation': [
            'empirical', 'experiment', 'synthetic',
            'real-world', 'practical', 'limited experiment',
            'more experiment', 'additional experiment',
            'toy', 'simulated',
        ],
        'clarity_writing': [
            'clarity', 'writing', 'notation', 'unclear',
            'confusing', 'hard to follow', 'poorly written',
            'readability', 'presentation', 'dense',
        ],
        'related_work': [
            'related work', 'prior work', 'missing reference',
            'comparison', 'not cited', 'not discussed',
            'relevant work', 'literature',
        ],
        'proof_correctness': [
            'proof', 'correctness', 'error in', 'mistake',
            'incorrect', 'flawed', 'gap in proof',
            'not rigorous', 'hand-wav',
        ],
        'significance': [
            'significance', 'impact', 'important', 'relevance',
            'practical implication', 'usefulness', 'motivation',
            'why does this matter', 'narrow',
        ],
        'scalability': [
            'scalab', 'scale', 'large-scale', 'computational',
            'complexity', 'efficient', 'runtime',
        ],
        'reproducibility': [
            'reproducib', 'code', 'implementation', 'detail',
            'missing detail', 'not reproducib',
        ],
        'overclaiming': [
            'overclaim', 'overstat', 'too strong', 'bold claim',
            'not supported', 'exaggerat', 'misleading',
        ],
    }

    for cat, keywords in WEAKNESS_TAXONOMY.items():
        if any(kw in text for kw in keywords):
            categories.append(cat)

    return categories


def classify_strengths(text):
    """Classify strength text into categories."""
    text = text.lower()
    categories = []

    STRENGTH_TAXONOMY = {
        'novel_contribution': [
            'novel', 'new', 'original', 'first', 'innovative',
            'creative', 'unique',
        ],
        'theoretical_depth': [
            'theoretical', 'rigorous', 'formal', 'proof',
            'tight bound', 'minimax', 'fundamental',
        ],
        'strong_empirics': [
            'thorough experiment', 'comprehensive', 'extensive',
            'convincing experiment', 'well-designed experiment',
            'strong empirical',
        ],
        'well_written': [
            'well-written', 'well written', 'clear', 'well-organized',
            'easy to follow', 'well-presented', 'readable',
        ],
        'practical_impact': [
            'practical', 'useful', 'applicable', 'impact',
            'relevant', 'actionable', 'important problem',
        ],
        'interesting_finding': [
            'interesting', 'surprising', 'insightful', 'thought-provoking',
            'intriguing', 'compelling',
        ],
    }

    for cat, keywords in STRENGTH_TAXONOMY.items():
        if any(kw in text for kw in keywords):
            categories.append(cat)

    return categories


def build_calibration(phase='PHASE_4B'):
    """Build the calibration JSON based on the gate phase."""
    raw_path = os.path.join('data', 'neurips_main_all_papers.jsonl')
    all_papers = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            all_papers.append(json.loads(line))

    validation = json.load(
        open(os.path.join('data', 'validation_results.json'))
    )
    scale_info = json.load(
        open(os.path.join('data', 'scale_info.json'))
    )

    # Compute similarity for all papers
    for p in all_papers:
        sim, breakdown = compute_similarity_stereology(p)
        p['_similarity'] = sim
        p['_sim_breakdown'] = breakdown

    # Global statistics
    by_year = defaultdict(list)
    for p in all_papers:
        by_year[p['year']].append(p)

    global_stats = {
        'total_papers': len(all_papers),
        'total_reviews': sum(len(p['reviews']) for p in all_papers),
        'by_year': {},
        'score_to_acceptance_rate': {},
        'decision_distribution': {},
        'data_limitations': {
            'survivorship_bias': (
                "NeurIPS main track only publishes accepted paper reviews "
                "(2021-2023, 2025). 2024 has a small opt-in rejected subset. "
                "The weakness taxonomy is biased toward survivable weaknesses."
            ),
            'rejected_papers_available': False,
            'rejected_opt_in_year': 2024,
        },
    }

    for year in sorted(by_year.keys()):
        papers = by_year[year]
        scores_list = [p['mean_score_normalized'] for p in papers
                  if p.get('mean_score_normalized') is not None]
        accepted = sum(1 for p in papers if p.get('is_accepted'))
        rejected = sum(1 for p in papers if p.get('is_accepted') is False)

        mean_s = sum(scores_list) / len(scores_list) if scores_list else None
        std_s = (
            (sum((s - mean_s)**2 for s in scores_list) / len(scores_list)) ** 0.5
            if len(scores_list) > 1 and mean_s is not None else None
        )

        global_stats['by_year'][year] = {
            'n_papers': len(papers),
            'n_reviews': sum(len(p['reviews']) for p in papers),
            'n_accepted': accepted,
            'n_rejected': rejected,
            'mean_score': mean_s,
            'std_score': std_s,
            'scale': scale_info.get(str(year), {}),
        }

    # Score-to-decision mapping
    score_buckets = defaultdict(lambda: {'accepted': 0, 'total': 0})
    for p in all_papers:
        if p.get('is_accepted') is not None and p.get('mean_score_normalized'):
            bucket = round(p['mean_score_normalized'])
            score_buckets[bucket]['total'] += 1
            if p['is_accepted']:
                score_buckets[bucket]['accepted'] += 1

    for bucket in sorted(score_buckets.keys()):
        d = score_buckets[bucket]
        global_stats['score_to_acceptance_rate'][bucket] = {
            'n': d['total'],
            'acceptance_rate': d['accepted'] / d['total'] if d['total'] > 0 else None,
            'caveat': 'biased — mostly accepted papers in sample' if bucket > 4 else None,
        }

    # Decision distribution
    decisions = Counter(p.get('decision') for p in all_papers if p.get('decision'))
    global_stats['decision_distribution'] = dict(decisions)

    # Comparable papers
    comparable = sorted(all_papers, key=lambda x: -x['_similarity'])
    comparable_above_threshold = [p for p in comparable if p['_similarity'] >= 0.2]

    for p in comparable_above_threshold:
        if p['_similarity'] >= 0.5:
            p['_tier'] = 'A_highly_similar'
        elif p['_similarity'] >= 0.35:
            p['_tier'] = 'B_moderately_similar'
        else:
            p['_tier'] = 'C_weakly_similar'

    tier_a = [p for p in comparable_above_threshold if p['_tier'] == 'A_highly_similar']
    tier_b = [p for p in comparable_above_threshold if p['_tier'] == 'B_moderately_similar']
    tier_c = [p for p in comparable_above_threshold if p['_tier'] == 'C_weakly_similar']

    # Weakness/strength analysis — run on ALL papers with review text
    # (not just comparables, since most comparables are 2025 without text)
    weakness_counts = Counter()
    strength_counts = Counter()
    weakness_by_decision = defaultdict(lambda: defaultdict(int))
    papers_with_text = 0

    for p in all_papers:
        has_text = False
        for r in p['reviews']:
            wk = r.get('weaknesses', '') or ''
            st = r.get('strengths', '') or ''
            # Also try review_text field if strengths/weaknesses are empty
            if not wk and not st:
                rt = r.get('review_text', '') or ''
                if rt:
                    wk = rt  # classify the full review text as weakness source
                    st = rt  # and strength source
                    has_text = True

            if not wk and not st:
                continue
            has_text = True

            w_cats = classify_weaknesses(wk)
            s_cats = classify_strengths(st)

            for cat in w_cats:
                weakness_counts[cat] += 1
                if p.get('decision'):
                    weakness_by_decision[cat][p['decision']] += 1

            for cat in s_cats:
                strength_counts[cat] += 1

        if has_text:
            papers_with_text += 1

    print(f"  Weakness analysis: {papers_with_text} papers with review text analyzed")

    def paper_to_output(p, detail_level='full'):
        """Convert paper dict to output format."""
        base = {
            'title': p['title'],
            'year': p['year'],
            'similarity': round(p['_similarity'], 3),
            'tier': p.get('_tier', 'unknown'),
            'decision': p.get('decision'),
            'is_accepted': p.get('is_accepted'),
            'scores_raw': p.get('scores', []),
            'mean_score_normalized': (
                round(p['mean_score_normalized'], 2)
                if p.get('mean_score_normalized') else None
            ),
        }

        if detail_level == 'full':
            base['reviews'] = [
                {
                    'score': r.get('score'),
                    'score_normalized': r.get('score_normalized'),
                    'confidence': r.get('confidence'),
                    'strengths': r.get('strengths', '')[:2000],
                    'weaknesses': r.get('weaknesses', '')[:2000],
                    'summary': r.get('summary', '')[:1000],
                    'was_edited': r.get('was_edited', False),
                    'weakness_categories': classify_weaknesses(r.get('weaknesses', '')),
                    'strength_categories': classify_strengths(r.get('strengths', '')),
                }
                for r in p['reviews']
            ]
            base['meta_review'] = p.get('meta_review')
            base['abstract'] = p.get('abstract', '')[:1000]
        elif detail_level == 'summary':
            base['reviews'] = [
                {
                    'score': r.get('score'),
                    'score_normalized': r.get('score_normalized'),
                    'confidence': r.get('confidence'),
                    'weakness_categories': classify_weaknesses(r.get('weaknesses', '')),
                    'strength_categories': classify_strengths(r.get('strengths', '')),
                    'weaknesses_excerpt': r.get('weaknesses', '')[:300],
                    'strengths_excerpt': r.get('strengths', '')[:300],
                }
                for r in p['reviews']
            ]
        else:
            base['reviews'] = [
                {
                    'score_normalized': r.get('score_normalized'),
                    'weakness_categories': classify_weaknesses(r.get('weaknesses', '')),
                }
                for r in p['reviews']
            ]

        return base

    calibration = {
        'metadata': {
            'scrape_date': datetime.now().isoformat()[:10],
            'target_paper': 'STEREOLOGY — The Evaluation Blind Spot',
            'target_venue': 'NeurIPS 2026 Main Track',
            'pipeline_version': '1.0',
            'validation_mae': validation.get('mae'),
            'validation_phase': phase,
            'data_scope': (
                'NeurIPS main track 2021-2025. '
                'Accepted papers all years. '
                'Opt-in rejected papers 2024 only. '
                'Survivorship bias present for 2021-2023 and 2025.'
            ),
        },
        'global_stats': global_stats,
        'comparable_papers': {
            'total_above_threshold': len(comparable_above_threshold),
            'tier_A_count': len(tier_a),
            'tier_B_count': len(tier_b),
            'tier_C_count': len(tier_c),
            'papers': (
                [paper_to_output(p, 'full') for p in tier_a] +
                [paper_to_output(p, 'summary') for p in tier_b[:50]] +
                [paper_to_output(p, 'stats') for p in tier_c[:100]]
            ),
        },
        'weakness_analysis': {
            'caveat': (
                'Based primarily on accepted papers. '
                'Weaknesses shown are those that SURVIVED review, '
                'not those that killed papers (except 2024 opt-in rejected).'
            ),
            'weakness_frequency': dict(weakness_counts.most_common()),
            'weakness_by_decision': {
                cat: dict(decisions)
                for cat, decisions in weakness_by_decision.items()
            },
            'strength_frequency': dict(strength_counts.most_common()),
        },
    }

    # Context window management
    cal_json = json.dumps(calibration)
    n_tokens_est = len(cal_json) // 4
    print(f"\nCalibration JSON size: {len(cal_json):,} chars (~{n_tokens_est:,} tokens)")

    if n_tokens_est > 100000:
        print("  WARNING: Calibration file exceeds 100k tokens.")
        print("  Trimming Tier C papers and review excerpts...")
        calibration['comparable_papers']['papers'] = (
            [paper_to_output(p, 'full') for p in tier_a[:20]] +
            [paper_to_output(p, 'summary') for p in tier_b[:30]] +
            [paper_to_output(p, 'stats') for p in tier_c[:50]]
        )
        cal_json = json.dumps(calibration)
        n_tokens_est = len(cal_json) // 4
        print(f"  Trimmed to: {len(cal_json):,} chars (~{n_tokens_est:,} tokens)")

    # Save
    out_path = os.path.join('data', 'neurips_calibration_stereology.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)

    print(f"\nCalibration saved to: {out_path}")

    # Summary CSV
    csv_path = os.path.join('data', 'neurips_comparable_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'year', 'similarity', 'score_norm',
                         'decision', 'tier', 'title'])
        for i, p in enumerate(comparable_above_threshold[:100]):
            writer.writerow([
                i + 1, p['year'], f"{p['_similarity']:.3f}",
                f"{p.get('mean_score_normalized', 0):.1f}",
                p.get('decision', '?'), p.get('_tier', '?'),
                p['title'][:100],
            ])

    print(f"Summary CSV saved to: {csv_path}")

    return calibration


if __name__ == '__main__':
    val_path = os.path.join('data', 'validation_results.json')
    if os.path.exists(val_path):
        val = json.load(open(val_path))
        phase = val.get('phase', 'PHASE_4C')
    else:
        print("ERROR: Run validate.py first!")
        sys.exit(1)

    build_calibration(phase)

"""
Phase 1: DOWNLOAD NeurIPS data from existing datasets.
No scraping needed — pre-built datasets exist.

Sources:
  1. PaperCopilot (GitHub JSON): 2021-2025, scores + decisions + abstracts + keywords
     Direct download, ~30MB per year, instant.
  2. nhop/OpenReview (HuggingFace): 2021-2024, full review text + scores + decisions
     ~1.2GB, streamed + filtered for NeurIPS.

Strategy: Download PaperCopilot for all years first (fast),
then enrich with nhop review text (slower but has strengths/weaknesses).
"""

import json
import os
import re
import sys
import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SCORE_FIELDS = ['rating', 'overall', 'recommendation', 'score',
                'overall_assessment', 'Overall', 'overall_score', 'quality', 'Quality']
CONF_FIELDS = ['confidence', 'Confidence', 'reviewer_confidence']


def extract_score(content, fields):
    if not content: return None
    for f in fields:
        if f in content:
            v = content[f]
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, str):
                m = re.match(r'^(\d+(?:\.\d+)?)', v.strip())
                if m: return float(m.group(1))
    return None


# ════════════════════════════════════════════════════════════
# SOURCE 1: PaperCopilot (instant download, all 5 years)
# ════════════════════════════════════════════════════════════

PAPERCOPILOT_URLS = {
    2021: 'https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2021.json',
    2022: 'https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2022.json',
    2023: 'https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2023.json',
    2024: 'https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2024.json',
    2025: 'https://raw.githubusercontent.com/papercopilot/paperlists/main/nips/nips2025.json',
}


def download_papercopilot(year):
    """Download one year from PaperCopilot. Returns list of paper dicts."""
    url = PAPERCOPILOT_URLS[year]
    cache = os.path.join(CHECKPOINT_DIR, f'papercopilot_{year}.json')

    # Use cache if exists
    if os.path.exists(cache):
        with open(cache, encoding='utf-8') as f:
            raw = json.load(f)
        print(f"  [{year}] PaperCopilot: loaded from cache ({len(raw)} papers)")
        return raw

    print(f"  [{year}] PaperCopilot: downloading...")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print(f"  [{year}] PaperCopilot: FAILED (HTTP {r.status_code})")
        return []

    raw = r.json()
    if isinstance(raw, dict) and 'data' in raw:
        raw = raw['data']

    # Cache it
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False)

    print(f"  [{year}] PaperCopilot: {len(raw)} papers")
    return raw


def parse_papercopilot(raw_papers, year):
    """Convert PaperCopilot format to our standard format."""
    papers = []
    for p in raw_papers:
        status = (p.get('status') or '').lower()
        if 'oral' in status:
            decision = 'oral'
            is_accepted = True
        elif 'spotlight' in status:
            decision = 'spotlight'
            is_accepted = True
        elif 'poster' in status:
            decision = 'poster'
            is_accepted = True
        elif 'reject' in status or 'withdrawn' in status:
            decision = 'reject'
            is_accepted = False
        else:
            decision = status or None
            is_accepted = None

        # Parse ratings: "6;7;7" → [6.0, 7.0, 7.0]
        rating_str = p.get('rating', '')
        scores = []
        if rating_str:
            for s in str(rating_str).split(';'):
                s = s.strip()
                if s:
                    try:
                        scores.append(float(s))
                    except ValueError:
                        pass

        # Parse confidence
        conf_str = p.get('confidence', '')
        confidences = []
        if conf_str:
            for c in str(conf_str).split(';'):
                c = c.strip()
                if c:
                    try:
                        confidences.append(float(c))
                    except ValueError:
                        pass

        # Build review stubs (scores only, no text from this source)
        reviews = []
        for i, sc in enumerate(scores):
            reviews.append({
                'review_id': f'{year}_{p.get("id", i)}_{i}',
                'score': sc,
                'confidence': confidences[i] if i < len(confidences) else None,
                'strengths': '',
                'weaknesses': '',
                'summary': '',
                'questions': '',
                'limitations': '',
                'was_edited': False,
                'tcdate': None,
                'tmdate': None,
            })

        keywords = p.get('keywords', '')
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(';') if k.strip()]

        paper_id = p.get('id') or p.get('openreview', '') or f'pc_{year}_{len(papers)}'
        # Extract openreview forum ID from URL if available
        or_url = p.get('openreview', '')
        forum_id = ''
        if or_url and 'id=' in or_url:
            forum_id = or_url.split('id=')[-1].split('&')[0]
        if not forum_id:
            forum_id = paper_id

        papers.append({
            'paper_id': paper_id,
            'forum_id': forum_id,
            'title': p.get('title', ''),
            'abstract': p.get('abstract', ''),
            'keywords': keywords,
            'year': year,
            'venue': 'neurips_main',
            'decision': decision,
            'is_accepted': is_accepted,
            'reviews': reviews,
            'meta_review': None,
            'num_reviews': len(reviews),
            'scores': scores,
            'mean_score': sum(scores) / len(scores) if scores else None,
            '_source': 'papercopilot',
        })

    return papers


# ════════════════════════════════════════════════════════════
# SOURCE 2: nhop/OpenReview (full review text)
# ════════════════════════════════════════════════════════════

def download_nhop_reviews():
    """Stream nhop/OpenReview, filter NeurIPS, return dict of forum_id → reviews."""
    cache = os.path.join(CHECKPOINT_DIR, 'nhop_neurips_reviews.jsonl')

    if os.path.exists(cache):
        reviews_by_id = {}
        with open(cache, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    reviews_by_id[rec['submission_id']] = rec
                except: pass
        print(f"  nhop: loaded {len(reviews_by_id)} NeurIPS papers from cache")
        return reviews_by_id

    print("  nhop/OpenReview: streaming dataset (filtering NeurIPS)...")
    try:
        from datasets import load_dataset
        ds = load_dataset('nhop/OpenReview', split='train', streaming=True)
    except Exception as e:
        print(f"  nhop: Failed to load: {e}")
        return {}

    reviews_by_id = {}
    count = 0
    t0 = time.time()

    with open(cache, 'w', encoding='utf-8') as f:
        for row in ds:
            venue = row.get('venue', '')
            if 'NeurIPS' not in venue and 'NIPS' not in venue and 'nips' not in venue.lower():
                continue

            sub_id = row.get('openreview_submission_id', '') or row.get('paperhash', '')
            reviews_raw = row.get('reviews', [])

            parsed_reviews = []
            for rv in (reviews_raw or []):
                if isinstance(rv, dict):
                    # nhop format: review text in 'review' field, may contain
                    # strengths/weaknesses as part of the text or as separate fields
                    review_text = str(rv.get('review', '') or '')
                    # Try to extract strengths/weaknesses from review text
                    strengths = ''
                    weaknesses = ''
                    if review_text and isinstance(review_text, str):
                        s_match = re.search(r'(?:strengths?|pros?)[\s:]*\n(.*?)(?=\n\s*(?:weakness|cons?|question|limitation|suggestion|improvement|concern)|\Z)',
                                          review_text, re.IGNORECASE | re.DOTALL)
                        w_match = re.search(r'(?:weakness|weaknesses|cons?|improvement|concern)[\s:]*\n(.*?)(?=\n\s*(?:question|limitation|suggestion|overall|rating|score|summary|minor|typo)|\Z)',
                                          review_text, re.IGNORECASE | re.DOTALL)
                        if s_match:
                            strengths = s_match.group(1).strip()[:3000]
                        if w_match:
                            weaknesses = w_match.group(1).strip()[:3000]

                    parsed_reviews.append({
                        'review_id': rv.get('review_id', ''),
                        'score': rv.get('score'),
                        'confidence': rv.get('confidence'),
                        'strengths': strengths,
                        'weaknesses': weaknesses,
                        'summary': '',
                        'review_text': review_text[:5000],
                        'clarity': rv.get('clarity'),
                        'correctness': rv.get('correctness'),
                        'novelty': rv.get('novelty'),
                        'impact': rv.get('impact'),
                    })

            rec = {
                'submission_id': sub_id,
                'title': row.get('title', ''),
                'abstract': row.get('abstract', ''),
                'venue': venue,
                'decision': row.get('decision'),
                'decision_text': row.get('decision_text', ''),
                'mean_score': row.get('mean_score'),
                'reviews': parsed_reviews,
            }

            reviews_by_id[sub_id] = rec
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + '\n')
            count += 1

            if count % 500 == 0:
                elapsed = time.time() - t0
                print(f"  nhop: {count} NeurIPS papers found ({elapsed:.0f}s)")

    print(f"  nhop: DONE — {count} NeurIPS papers")
    return reviews_by_id


# ════════════════════════════════════════════════════════════
# MERGE + ENRICH
# ════════════════════════════════════════════════════════════

def enrich_with_nhop(papers, nhop_data):
    """Enrich PaperCopilot papers with nhop review text where available."""
    matched = 0

    # Build lookup by title (normalized)
    nhop_by_title = {}
    for rec in nhop_data.values():
        t = (rec.get('title') or '').strip().lower()
        if t:
            nhop_by_title[t] = rec

    for paper in papers:
        title_norm = (paper.get('title') or '').strip().lower()

        # Try forum_id match first, then title match
        nhop_rec = nhop_data.get(paper['forum_id']) or nhop_by_title.get(title_norm)

        if nhop_rec and nhop_rec.get('reviews'):
            nhop_reviews = nhop_rec['reviews']
            # Match by index (both ordered by reviewer)
            for i, review in enumerate(paper['reviews']):
                if i < len(nhop_reviews):
                    nr = nhop_reviews[i]
                    # Enrich with text
                    if nr.get('strengths'):
                        review['strengths'] = nr['strengths']
                    if nr.get('weaknesses'):
                        review['weaknesses'] = nr['weaknesses']
                    if nr.get('review_text'):
                        review['review_text'] = nr['review_text']
                    # Use nhop score if PaperCopilot score is missing
                    if review['score'] is None and nr.get('score') is not None:
                        review['score'] = nr['score']
            matched += 1

    print(f"  Enriched {matched}/{len(papers)} papers with review text from nhop")
    return papers


# ════════════════════════════════════════════════════════════
# NORMALIZATION
# ════════════════════════════════════════════════════════════

def detect_scale(papers):
    scores = [r.get('score') for p in papers for r in p.get('reviews', []) if r.get('score') is not None]
    if not scores: return None, None, "no scores"
    mn, mx = min(scores), max(scores)
    if mx <= 6: return 1, 6, "1-6"
    if mx <= 10: return 1, 10, "1-10"
    return mn, mx, f"[{mn},{mx}]"

def norm10(s, smin, smax):
    if smax == 10: return s
    if smax == 6: return 1 + (s - 1) * 1.8
    if smax == smin: return 5.0
    return 1 + (s - smin) / (smax - smin) * 9


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    years = [2021, 2022, 2023, 2024, 2025]

    print("=" * 60)
    print("NEURIPS DATA DOWNLOADER")
    print("Source 1: PaperCopilot (GitHub) — scores, decisions, abstracts")
    print("Source 2: nhop/OpenReview (HF) — full review text")
    print("=" * 60)

    # ─── Step 1: Download PaperCopilot (all 5 years, parallel) ───
    print(f"\n--- STEP 1: PaperCopilot (all years) ---")
    raw_by_year = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(download_papercopilot, y): y for y in years}
        for fut in as_completed(futs):
            y = futs[fut]
            try:
                raw_by_year[y] = fut.result()
            except Exception as e:
                print(f"  [{y}] FAILED: {e}")
                raw_by_year[y] = []

    # Parse all
    all_papers = []
    for y in years:
        parsed = parse_papercopilot(raw_by_year.get(y, []), y)
        all_papers.extend(parsed)
        accepted = sum(1 for p in parsed if p.get('is_accepted'))
        rejected = sum(1 for p in parsed if p.get('is_accepted') is False)
        print(f"  [{y}] {len(parsed)} papers (accepted={accepted}, rejected={rejected})")

    pc_time = time.time() - t0
    print(f"\n  PaperCopilot done in {pc_time:.0f}s — {len(all_papers)} total papers")

    # ─── Step 2: Download nhop review text ───
    print(f"\n--- STEP 2: nhop/OpenReview (review text enrichment) ---")
    nhop_data = download_nhop_reviews()

    if nhop_data:
        all_papers = enrich_with_nhop(all_papers, nhop_data)

    # ─── Step 3: Normalize scores ───
    print(f"\n--- STEP 3: Normalize scores ---")
    by_year = defaultdict(list)
    for p in all_papers:
        by_year[p['year']].append(p)

    # Known scale overrides — NeurIPS changed scale in 2025
    KNOWN_SCALES = {
        2025: (1, 6, "1-6 (2025 system)"),
        # 2021-2024 are 1-10
    }

    scale_info = {}
    for y in sorted(by_year):
        if y in KNOWN_SCALES:
            smin, smax, desc = KNOWN_SCALES[y]
            # Cap outlier scores (D&B track leakage)
            for p in by_year[y]:
                for r in p.get('reviews', []):
                    if r.get('score') is not None and r['score'] > smax:
                        r['score'] = float(smax)
        else:
            smin, smax, desc = detect_scale(by_year[y])
        scale_info[y] = (smin, smax, desc)
        print(f"  [{y}] scale={desc}")

    for p in all_papers:
        y = p['year']
        smin, smax, _ = scale_info.get(y, (1, 10, ''))
        if smin and smax:
            for r in p.get('reviews', []):
                if r.get('score') is not None:
                    r['score_raw'] = r['score']
                    r['score_normalized'] = norm10(r['score'], smin, smax)
            ns = [r['score_normalized'] for r in p.get('reviews', [])
                  if r.get('score_normalized') is not None]
            p['mean_score_normalized'] = sum(ns)/len(ns) if ns else None

    # ─── Step 4: Save ───
    print(f"\n--- STEP 4: Save ---")
    raw_path = os.path.join(OUTPUT_DIR, 'neurips_main_all_papers.jsonl')
    with open(raw_path, 'w', encoding='utf-8') as f:
        for p in all_papers:
            f.write(json.dumps(p, default=str, ensure_ascii=False) + '\n')

    with open(os.path.join(OUTPUT_DIR, 'scale_info.json'), 'w') as f:
        json.dump({str(k): {'min': v[0], 'max': v[1], 'desc': v[2]}
                   for k, v in scale_info.items()}, f, indent=2)

    # ─── Summary ───
    total_r = sum(len(p.get('reviews', [])) for p in all_papers)
    with_text = sum(1 for p in all_papers
                    for r in p.get('reviews', [])
                    if r.get('strengths') or r.get('weaknesses') or r.get('review_text'))
    accepted = sum(1 for p in all_papers if p.get('is_accepted'))
    rejected = sum(1 for p in all_papers if p.get('is_accepted') is False)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"  Papers: {len(all_papers)}")
    print(f"  Reviews: {total_r} ({with_text} with text)")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")
    print(f"  Output: {raw_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    run()

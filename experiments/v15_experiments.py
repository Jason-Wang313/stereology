"""v15: All Block A experiments — run as single script for reliability."""
import numpy as np, pandas as pd, json, os
from scipy.stats import spearmanr, kendalltau, norm
from scipy.cluster.vq import kmeans2
from itertools import combinations

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

bench_cols = ['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO',
              'ARC','HellaSwag','MMLU','TruthfulQA','Winogrande','GSM8K']
v2_cols = bench_cols[:6]
v1_cols = bench_cols[6:]

def d_eff(S):
    S_std = (S - S.mean(0)) / (S.std(0) + 1e-10)
    C = np.corrcoef(S_std.T)
    ev = np.linalg.eigvalsh(C)[::-1]
    return float((ev.sum())**2 / (ev**2).sum())

# Load core data
ext = pd.read_csv('data/extended.csv')
ext_scores = ext[bench_cols].apply(pd.to_numeric, errors='coerce').dropna()
S_all = ext_scores.values
means = S_all.mean(1)
front_mask = means >= np.median(means)
S_front = S_all[front_mask]
n_front, k = S_front.shape
S_f_std = (S_front - S_front.mean(0)) / S_front.std(0)
models_front = ext.loc[ext_scores.index[front_mask], 'model'].values

# ================================================================
# A1: Epoch AI dense submatrix
# ================================================================
print('=== A1: Epoch AI Dense Submatrix ===')
epoch = pd.read_csv('data/epoch_ai_merged.csv', index_col=0)
if 'coverage' in epoch.columns:
    epoch = epoch.drop('coverage', axis=1)

# Find largest dense submatrix with >=20 models
bench_counts = epoch.notna().sum().sort_values(ascending=False)
selected_bench = []
for b in bench_counts.index:
    candidate = selected_bench + [b]
    complete = epoch[candidate].dropna()
    if len(complete) >= 20:
        selected_bench = candidate
    if len(selected_bench) >= 25:
        break

dense = epoch[selected_bench].dropna()
n_d, k_d = dense.shape
print(f'  Dense submatrix: {n_d} models x {k_d} benchmarks')
print(f'  Benchmarks: {selected_bench[:10]}...')

if n_d >= 10 and k_d >= 5:
    d_epoch = d_eff(dense.values)
    print(f'  d_eff = {d_epoch:.2f}')

    # Domain tagging
    domains_found = set()
    domain_map = {
        'coding': ['aider_polyglot','swe_bench','cybench','terminalbench','webdev_arena'],
        'math': ['math_level_5','gsm8k','frontiermath','otis_mock_aime','gpqa_diamond'],
        'reasoning': ['bbh','arc_agi','simplebench','hle'],
        'knowledge': ['mmlu','trivia_qa','open_book_qa','bool_q','hella_swag','piqa','wino_grande','lambada'],
        'multimodal': ['video_mme','geobench','vpct'],
        'agents': ['apex_agents','metr_time_horizons','deepresearchbench'],
        'writing': ['lech_mazur_writing','fictionlivebench'],
    }
    for b in selected_bench:
        for dom, kws in domain_map.items():
            if any(kw in b for kw in kws):
                domains_found.add(dom)

    epoch_result = {
        'n': n_d, 'k': k_d, 'd_eff': d_epoch,
        'benchmarks': selected_bench,
        'domains': list(domains_found),
        'n_domains': len(domains_found),
    }
else:
    epoch_result = {'n': n_d, 'k': k_d, 'd_eff': None, 'error': 'too small'}

with open('results/epoch_dense.json', 'w') as f:
    json.dump(epoch_result, f, indent=2)
print(f'  Domains: {epoch_result.get("domains", [])}')
print(f'  Saved: results/epoch_dense.json')

# ================================================================
# A2: Rank equivalence classes
# ================================================================
print('\n=== A2: Rank Equivalence Classes ===')
# Use sigma_hidden from Q5 results
sigma_h = 1.003  # from reviewer_fixes.py Q5 output

top20_idx = np.argsort(-S_f_std.mean(axis=1))[:20]
S_top20 = S_f_std[top20_idx]
top20_names = models_front[top20_idx]

# Compute pairwise P(swap)
n_top = 20
swap_prob = np.zeros((n_top, n_top))
for i in range(n_top):
    for j in range(n_top):
        if i == j:
            swap_prob[i, j] = 0.5
        else:
            gap = abs(S_top20[i].mean() - S_top20[j].mean())
            swap_prob[i, j] = float(norm.cdf(-gap / (2 * sigma_h)))

# Group into equivalence classes (P(swap) > 0.4 → indistinguishable)
threshold = 0.40
classes = []
assigned = set()
rank_order = np.argsort(-S_top20.mean(axis=1))

for idx in rank_order:
    if idx in assigned:
        continue
    cls = [idx]
    assigned.add(idx)
    for other in rank_order:
        if other in assigned:
            continue
        # Check if indistinguishable with ALL members of current class
        if all(swap_prob[idx, m] > threshold for m in cls):
            cls.append(other)
            assigned.add(other)
    classes.append(cls)

print(f'  {len(classes)} equivalence classes from {n_top} models:')
equiv_result = {'classes': [], 'threshold': threshold, 'sigma_h': sigma_h}
for ci, cls in enumerate(classes):
    names = [str(top20_names[i])[:40] for i in cls]
    scores = [float(S_top20[i].mean()) for i in cls]
    print(f'    Class {ci+1}: {names}')
    equiv_result['classes'].append({
        'class': ci + 1,
        'models': names,
        'mean_scores': scores,
        'size': len(cls),
    })

with open('results/rank_equivalence.json', 'w') as f:
    json.dump(equiv_result, f, indent=2)

# ================================================================
# A3: Three non-overlapping suites
# ================================================================
print('\n=== A3: Three Non-Overlapping Suites ===')
v2_idx = [bench_cols.index(c) for c in v2_cols]
v1_idx = [bench_cols.index(c) for c in v1_cols]

d_v2 = d_eff(S_front[:, v2_idx])
d_v1 = d_eff(S_front[:, v1_idx])

lb = pd.read_csv('data/livebench.csv')
lb_bench = [c for c in lb.columns if c != 'model']
lb_scores = lb[lb_bench].apply(pd.to_numeric, errors='coerce').dropna().values
d_lb = d_eff(lb_scores)

# Bootstrap CIs
np.random.seed(42)
def bootstrap_deff(S, n_boot=500):
    ds = []
    for _ in range(n_boot):
        idx = np.random.choice(len(S), len(S), replace=True)
        ds.append(d_eff(S[idx]))
    return np.percentile(ds, [2.5, 97.5])

ci_v2 = bootstrap_deff(S_front[:, v2_idx])
ci_v1 = bootstrap_deff(S_front[:, v1_idx])
ci_lb = bootstrap_deff(lb_scores)

three_result = {
    'v2': {'benchmarks': v2_cols, 'n': n_front, 'd_eff': d_v2, 'ci': ci_v2.tolist()},
    'v1': {'benchmarks': v1_cols, 'n': n_front, 'd_eff': d_v1, 'ci': ci_v1.tolist()},
    'livebench': {'benchmarks': lb_bench, 'n': len(lb_scores), 'd_eff': d_lb, 'ci': ci_lb.tolist()},
}
print(f'  v2 (6 bench): d_eff={d_v2:.2f} [{ci_v2[0]:.2f}, {ci_v2[1]:.2f}]')
print(f'  v1 (6 bench): d_eff={d_v1:.2f} [{ci_v1[0]:.2f}, {ci_v1[1]:.2f}]')
print(f'  LB (7 bench): d_eff={d_lb:.2f} [{ci_lb[0]:.2f}, {ci_lb[1]:.2f}]')

with open('results/three_independent.json', 'w') as f:
    json.dump(three_result, f, indent=2)

# ================================================================
# A4: Multi-population d_eff
# ================================================================
print('\n=== A4: Multi-Population d_eff ===')
multi = {
    'extended_frontier': {'n': n_front, 'd_eff': d_eff(S_front)},
    'extended_full': {'n': len(S_all), 'd_eff': d_eff(S_all)},
}

# OLLM v2 full
try:
    from results_cache import ollm_full_deff
except:
    ollm = pd.read_csv('data/ollm_v2.csv')
    ollm_s = ollm[v2_cols].apply(pd.to_numeric, errors='coerce').dropna().values
    multi['ollm_v2_full'] = {'n': len(ollm_s), 'd_eff': d_eff(ollm_s)}

# Epoch AI (overlapping benchmarks)
overlap_bench = [b for b in ['gpqa_diamond', 'mmlu', 'gsm8k', 'math_level_5', 'hella_swag', 'bbh']
                 if b in epoch.columns]
if len(overlap_bench) >= 4:
    epoch_overlap = epoch[overlap_bench].dropna()
    if len(epoch_overlap) >= 15:
        multi['epoch_ai'] = {'n': len(epoch_overlap), 'd_eff': d_eff(epoch_overlap.values),
                             'benchmarks': overlap_bench}

for name, data in multi.items():
    print(f'  {name}: n={data["n"]}, d_eff={data["d_eff"]:.2f}')

with open('results/multi_population.json', 'w') as f:
    json.dump(multi, f, indent=2, default=str)

# ================================================================
# A5: Explicit constants
# ================================================================
print('\n=== A5: Explicit Constants ===')
Sigma = np.corrcoef(S_f_std.T)
ev = np.linalg.eigvalsh(Sigma)[::-1]
d = d_eff(S_front)
R = float(np.max(np.linalg.norm(S_f_std, axis=1)))

# Empirical covering radius (from paper: 1.57× Rogers)
omega_emp = 1.57  # × Rogers optimum
# Rogers optimum for d_eff-1 sphere with m=12 points
# Rogers bound: omega ~ (d/m)^{1/(d-1)} for m points on S^{d-1}
d_int = int(np.round(d))
m = k
omega_rogers = (d_int / m) ** (1.0 / max(d_int - 1, 1))
omega_actual = omega_emp * omega_rogers

# delta_H = 2R * omega_actual (from practitioner formula)
delta_H = 2 * R * omega_actual

# Back out the constant: delta_H = C * R * m^{-1/(d-1)}
rate = m ** (-1.0 / max(d - 1, 0.5))
C_empirical = delta_H / (R * rate)

# Worst-case constant from Rogers covering theorem
# C_worst = 2 * omega_worst where omega_worst is the Rogers bound
C_worst = 2 * omega_rogers / rate * m ** (1.0 / max(d - 1, 0.5))

const_result = {
    'd_eff': float(d), 'R': R, 'k': k,
    'omega_emp_ratio': omega_emp,
    'omega_rogers': float(omega_rogers),
    'delta_H': float(delta_H),
    'C_empirical': float(C_empirical),
    'C_worst': float(C_worst) if C_worst > 0 else None,
    'ratio': float(C_empirical / C_worst) if C_worst > 0 else None,
}
print(f'  R={R:.2f}, d_eff={d:.2f}, k={k}')
print(f'  delta_H = {delta_H:.2f}')
print(f'  C_empirical = {C_empirical:.2f}')

with open('results/explicit_constants.json', 'w') as f:
    json.dump(const_result, f, indent=2)

# ================================================================
# A6: Monoculture ranking impact
# ================================================================
print('\n=== A6: Shared Structure Ranking Impact ===')
S_v2 = S_f_std[:, v2_idx]
S_v1 = S_f_std[:, v1_idx]

# PC1 scores on each suite
_, evec_v2 = np.linalg.eigh(np.corrcoef(S_v2.T))
_, evec_v1 = np.linalg.eigh(np.corrcoef(S_v1.T))
pc1_v2 = S_v2 @ evec_v2[:, -1]  # last eigenvec = largest eigenvalue
pc1_v1 = S_v1 @ evec_v1[:, -1]

# Ranking by PC1 on each suite
rank_pc1_v2 = np.argsort(np.argsort(-pc1_v2))
rank_pc1_v1 = np.argsort(np.argsort(-pc1_v1))

# Full ranking on each suite
rank_full_v2 = np.argsort(np.argsort(-S_v2.mean(1)))
rank_full_v1 = np.argsort(np.argsort(-S_v1.mean(1)))

tau_pc1, _ = kendalltau(rank_pc1_v2, rank_pc1_v1)
tau_full, _ = kendalltau(rank_full_v2, rank_full_v1)

# Among top-20
top20_v2 = np.argsort(-S_v2.mean(1))[:20]
tau_top20_pc1, _ = kendalltau(rank_pc1_v2[top20_v2], rank_pc1_v1[top20_v2])
tau_top20_full, _ = kendalltau(rank_full_v2[top20_v2], rank_full_v1[top20_v2])

mono_result = {
    'tau_pc1_all': float(tau_pc1),
    'tau_full_all': float(tau_full),
    'tau_pc1_top20': float(tau_top20_pc1),
    'tau_full_top20': float(tau_top20_full),
}
print(f'  PC1 ranking agreement (all): tau={tau_pc1:.3f}')
print(f'  Full ranking agreement (all): tau={tau_full:.3f}')
print(f'  PC1 ranking agreement (top-20): tau={tau_top20_pc1:.3f}')
print(f'  Full ranking agreement (top-20): tau={tau_top20_full:.3f}')

with open('results/monoculture_ranking_impact.json', 'w') as f:
    json.dump(mono_result, f, indent=2)

# ================================================================
# A1 addendum: Convexity — BIC comparison (1 Gaussian vs 2 clusters)
# ================================================================
print('\n=== B1 prep: Convexity BIC Test ===')
from sklearn.mixture import GaussianMixture

bic_1 = GaussianMixture(n_components=1, random_state=42).fit(S_f_std).bic(S_f_std)
bic_2 = GaussianMixture(n_components=2, random_state=42).fit(S_f_std).bic(S_f_std)
bic_3 = GaussianMixture(n_components=3, random_state=42).fit(S_f_std).bic(S_f_std)

bic_improvement = (bic_1 - bic_2) / abs(bic_1) * 100

print(f'  BIC(1 Gaussian): {bic_1:.0f}')
print(f'  BIC(2 clusters): {bic_2:.0f}')
print(f'  BIC(3 clusters): {bic_3:.0f}')
print(f'  Improvement 1→2: {bic_improvement:.1f}%')
print(f'  Best: {"1 Gaussian" if bic_1 <= bic_2 else "2+ clusters"}')

convex_result = {
    'bic_1': float(bic_1), 'bic_2': float(bic_2), 'bic_3': float(bic_3),
    'improvement_pct': float(bic_improvement),
    'best_model': '1_gaussian' if bic_1 <= bic_2 else '2_clusters',
}
with open('results/convexity_bic.json', 'w') as f:
    json.dump(convex_result, f, indent=2)

# ================================================================
print('\n' + '='*60)
print('ALL A-BLOCK EXPERIMENTS COMPLETE')
print('='*60)

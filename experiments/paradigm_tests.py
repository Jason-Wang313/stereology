"""Phase 2: Paradigm shift quick-tests A-D."""
import numpy as np, pandas as pd, json, os, glob

bench_cols = ['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO',
              'ARC','HellaSwag','MMLU','TruthfulQA','Winogrande','GSM8K']
shared_cols = bench_cols[:6]  # OLLM v2 benchmarks

def d_eff(S):
    if S.shape[0] < S.shape[1]: return float('nan')
    C = np.corrcoef(S.T)
    ev = np.linalg.eigvalsh(C)[::-1]
    return (ev.sum())**2 / (ev**2).sum()

ext = pd.read_csv('data/extended.csv')
ext_scores = ext[bench_cols].apply(pd.to_numeric, errors='coerce').dropna()
ext_all = ext_scores.values
ext_params = ext.loc[ext_scores.index, 'params_b'].apply(pd.to_numeric, errors='coerce')

ollm = pd.read_csv('data/ollm_v2.csv')
ollm_scores = ollm[shared_cols].apply(pd.to_numeric, errors='coerce').dropna()
ollm_all = ollm_scores.values

ext_front = ext_all[ext_all.mean(1) >= np.median(ext_all.mean(1))]
ollm_front = ollm_all[ollm_all.mean(1) >= np.median(ollm_all.mean(1))]

results = {}

# === TEST A ===
print('=== TEST A: Cross-Suite PC Alignment ===')
shared_idx = [bench_cols.index(c) for c in shared_cols]
ext_sh = ext_front[:, shared_idx]
ext_sh_std = (ext_sh - ext_sh.mean(0)) / ext_sh.std(0)
ollm_std = (ollm_front - ollm_front.mean(0)) / ollm_front.std(0)

C_e = np.corrcoef(ext_sh_std.T)
C_o = np.corrcoef(ollm_std.T)
_, ve = np.linalg.eigh(C_e); ve = ve[:,::-1]
_, vo = np.linalg.eigh(C_o); vo = vo[:,::-1]

cosines = [abs(np.dot(ve[:,i], vo[:,i])) for i in range(6)]
for i in range(6):
    print(f'  PC{i+1}: cosine = {cosines[i]:.3f}')

# Best-match alignment
for i in range(3):
    best = max(range(6), key=lambda j: abs(np.dot(ve[:,i], vo[:,j])))
    bc = abs(np.dot(ve[:,i], vo[:,best]))
    print(f'  ext PC{i+1} best-match: ollm PC{best+1} cos={bc:.3f}')

results['test_a'] = {
    'pc1_cosine': float(cosines[0]),
    'signal': bool(cosines[0] > 0.8),
    'verdict': 'INVEST' if cosines[0] > 0.8 else 'SKIP'
}
print(f'  VERDICT: {results["test_a"]["verdict"]}')

# === TEST B ===
print('\n=== TEST B: d_eff vs Model Scale ===')
params = ext_params.loc[ext_scores.index].values
valid = ~np.isnan(params)
S_v = ext_all[valid]; p_v = params[valid]

bins = [(0,7,'<7B'), (7,30,'7-30B'), (30,70,'30-70B'), (70,1000,'>70B')]
d_list = []
for lo, hi, label in bins:
    mask = (p_v >= lo) & (p_v < hi)
    n = mask.sum()
    if n < 10:
        print(f'  {label}: n={n} (skip)')
        d_list.append(None)
        continue
    S_b = S_v[mask]
    m = S_b.mean(1)
    f = m >= np.median(m)
    if f.sum() < 8: f = np.ones(len(S_b), dtype=bool)
    S_bf = S_b[f]
    S_bs = (S_bf - S_bf.mean(0)) / (S_bf.std(0) + 1e-10)
    d = d_eff(S_bs)
    d_list.append(d)
    print(f'  {label}: n_front={f.sum()}, d_eff={d:.2f}')

d_clean = [x for x in d_list if x is not None]
mono = all(d_clean[i] <= d_clean[i+1] for i in range(len(d_clean)-1)) if len(d_clean) >= 2 else False
spread = max(d_clean) - min(d_clean) if len(d_clean) >= 2 else 0
results['test_b'] = {
    'd_effs': d_list, 'monotone': bool(mono), 'spread': float(spread),
    'signal': bool(mono and spread >= 0.5),
    'verdict': 'INVEST' if mono and spread >= 0.5 else 'SKIP'
}
print(f'  Monotone: {mono}, Spread: {spread:.2f}, VERDICT: {results["test_b"]["verdict"]}')

# === TEST C ===
print('\n=== TEST C: Herding / Convergence ===')
models = ext.loc[ext_scores.index, 'model'].values
def gen(name):
    n = str(name).lower()
    if any(x in n for x in ['llama-2','falcon','mpt-','bloom']): return 'gen1'
    if any(x in n for x in ['llama-3-','mistral-7b','gemma-7b','phi-3','qwen2-']): return 'gen2'
    if any(x in n for x in ['llama-3.1','llama-3.2','llama-3.3','qwen2.5','gemma-2-','phi-4']): return 'gen3'
    return 'unknown'

gens = np.array([gen(m) for m in models])
gen_results = {}
for g in ['gen1','gen2','gen3']:
    mask = gens == g
    n = mask.sum()
    if n < 10:
        print(f'  {g}: n={n} (skip)')
        continue
    S_g = ext_all[mask]
    m = S_g.mean(1); f = m >= np.median(m)
    if f.sum() < 8: f = np.ones(n, dtype=bool)
    S_gf = S_g[f]
    S_gs = (S_gf - S_gf.mean(0)) / (S_gf.std(0) + 1e-10)
    d = d_eff(S_gs)
    v = S_gf.std(0).mean()
    gen_results[g] = {'n': int(f.sum()), 'd_eff': float(d), 'mean_std': float(v)}
    print(f'  {g}: n={int(f.sum())}, d_eff={d:.2f}, score_std={v:.2f}')

results['test_c'] = {'generations': gen_results, 'verdict': 'MANUAL_CHECK'}
print(f'  VERDICT: MANUAL_CHECK (see values above)')

# === TEST D: Self-Application ===
print('\n=== TEST D: Self-Application ===')
# This needs a matrix with rows=resamples, cols=experiment metrics
# Use bootstrap: for each of 100 resamples of the frontier population,
# recompute ALL key metrics → rows are resamples, columns are metrics
np.random.seed(42)
n_boot = 100
n_front, k = ext_front.shape
metric_matrix = []

for b in range(n_boot):
    idx = np.random.choice(n_front, n_front, replace=True)
    S_b = ext_front[idx]
    S_s = (S_b - S_b.mean(0)) / (S_b.std(0) + 1e-10)
    C = np.corrcoef(S_s.T)
    ev = np.linalg.eigvalsh(C)[::-1]

    deff = (ev.sum())**2 / (ev**2).sum()
    R = np.max(np.linalg.norm(S_s, axis=1))
    mean_score = S_b.mean()
    std_score = S_b.std()
    top_ev = ev[0]
    ev_ratio = ev[0] / ev[1] if ev[1] > 0 else 0

    # Swap rate (quick: random 6/6 split)
    perm = np.random.permutation(k)
    vis = perm[:k//2]; hid = perm[k//2:]
    r_vis = np.argsort(-S_s[:, vis].mean(1))
    r_hid = np.argsort(-S_s[:, hid].mean(1))
    swap_frac = (r_vis[:10] != r_hid[:10]).mean()

    metric_matrix.append([deff, R, mean_score, std_score, top_ev, ev_ratio, swap_frac])

M = np.array(metric_matrix)  # (100, 7)
print(f'  Bootstrap metric matrix: {M.shape}')
d_self = d_eff(M)
print(f'  d_eff of validation metrics: {d_self:.2f} (out of {M.shape[1]} metrics)')

results['test_d'] = {
    'd_eff': float(d_self),
    'n_metrics': M.shape[1],
    'signal': bool(d_self > M.shape[1] * 0.4),
    'verdict': 'INVEST' if d_self > M.shape[1] * 0.4 else 'SKIP'
}
print(f'  VERDICT: {results["test_d"]["verdict"]}')

# === SUMMARY ===
print('\n' + '='*60)
print('PARADIGM SHIFT QUICK-TEST RESULTS')
print('='*60)
for t, r in results.items():
    print(f'  {t}: {r["verdict"]}')

with open('results/paradigm_tests.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved: results/paradigm_tests.json')

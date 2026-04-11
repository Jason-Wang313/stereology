"""Phase 1 experiments: Q2 (de-dup), Q3 (LiveBench CI), Q5 (anisotropic), Q7 (standardization)."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, norm
sns.set_style('whitegrid'); sns.set_palette('colorblind')

bench_cols = ['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO',
              'ARC','HellaSwag','MMLU','TruthfulQA','Winogrande','GSM8K']
ollm_cols = bench_cols[:6]
v1_cols = bench_cols[6:]

def d_eff(C):
    ev = np.linalg.eigvalsh(C)[::-1]
    return (ev.sum())**2 / (ev**2).sum()

ext = pd.read_csv('data/extended.csv')
ext_scores = ext[bench_cols].apply(pd.to_numeric, errors='coerce').dropna()
ext_all = ext_scores.values
n_all, k = ext_all.shape
means = ext_all.mean(1)
front = means >= np.median(means)
ext_front = ext_all[front]

results = {}

# === Q2: DE-DUPLICATION ===
print('=== Q2: Model De-Duplication ===')
models = ext.loc[ext_scores.index, 'model'].values[front]
S = ext_front

# Group by family prefix (first 2 path components or first 20 chars)
def family(name):
    parts = str(name).split('/')
    if len(parts) >= 2:
        return parts[0] + '/' + parts[1].split('-')[0]
    return str(name)[:20]

families = [family(m) for m in models]
unique_fams = set(families)
print(f'  Models: {len(models)}, Families: {len(unique_fams)}')

# Keep best per family
best_idx = []
for fam in unique_fams:
    indices = [i for i, f in enumerate(families) if f == fam]
    best = max(indices, key=lambda i: S[i].mean())
    best_idx.append(best)

S_dedup = S[best_idx]
n_dedup = len(S_dedup)
print(f'  After de-dup: {n_dedup} models')

S_std = (ext_front - ext_front.mean(0)) / ext_front.std(0)
S_dedup_std = (S_dedup - S_dedup.mean(0)) / (S_dedup.std(0) + 1e-10)

d_orig = d_eff(np.corrcoef(S_std.T))
d_dedup = d_eff(np.corrcoef(S_dedup_std.T))
print(f'  d_eff original: {d_orig:.2f}')
print(f'  d_eff de-duped: {d_dedup:.2f}')
print(f'  Change: {d_dedup - d_orig:+.2f}')

results['q2'] = {'n_orig': len(models), 'n_dedup': n_dedup,
                  'n_families': len(unique_fams),
                  'd_eff_orig': d_orig, 'd_eff_dedup': d_dedup}

# === Q3: LIVEBENCH BOOTSTRAP CIs ===
print('\n=== Q3: LiveBench Bootstrap CIs ===')
lb = pd.read_csv('data/livebench.csv')
lb_bench = [c for c in lb.columns if c != 'model']
lb_scores = lb[lb_bench].apply(pd.to_numeric, errors='coerce').dropna()
S_lb = lb_scores.values
n_lb, k_lb = S_lb.shape

# Frontier
lb_means = S_lb.mean(1)
lb_front = lb_means >= np.median(lb_means)
S_lb_f = S_lb[lb_front]
n_lb_f = lb_front.sum()
print(f'  LiveBench frontier: n={n_lb_f}, k={k_lb}')

S_lb_std = (S_lb_f - S_lb_f.mean(0)) / (S_lb_f.std(0) + 1e-10)
d_lb_orig = d_eff(np.corrcoef(S_lb_std.T))
print(f'  d_eff: {d_lb_orig:.2f}')

np.random.seed(42)
d_boots = []
for _ in range(500):
    idx = np.random.choice(n_lb_f, n_lb_f, replace=True)
    S_b = S_lb_f[idx]
    S_bs = (S_b - S_b.mean(0)) / (S_b.std(0) + 1e-10)
    d_boots.append(d_eff(np.corrcoef(S_bs.T)))

d_boots = np.array(d_boots)
ci_lo, ci_hi = np.percentile(d_boots, [2.5, 97.5])
print(f'  Bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]')

# Horn's parallel analysis
n_horn = 500
horn_maxeigs = []
for _ in range(n_horn):
    R = np.random.randn(n_lb_f, k_lb)
    C = np.corrcoef(R.T)
    horn_maxeigs.append(np.linalg.eigvalsh(C)[-1])
horn_95 = np.percentile(horn_maxeigs, 95)
print(f'  Horn 95th percentile max eigenvalue: {horn_95:.2f}')

real_eigs = np.linalg.eigvalsh(np.corrcoef(S_lb_std.T))[::-1]
n_sig_horn = sum(1 for e in real_eigs if e > horn_95)
print(f'  Signal eigenvalues (Horn): {n_sig_horn}')

results['q3'] = {'d_eff': d_lb_orig, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                  'horn_95': horn_95, 'n_signal_horn': n_sig_horn}

# === Q5: ANISOTROPIC CALIBRATION ===
print('\n=== Q5: Anisotropic Calibration ===')
S_f = ext_front
S_f_std = (S_f - S_f.mean(0)) / S_f.std(0)
n_f = len(S_f)

vis_idx = [bench_cols.index(c) for c in ollm_cols]
hid_idx = [bench_cols.index(c) for c in v1_cols]

S_vis = S_f_std[:, vis_idx]
S_hid = S_f_std[:, hid_idx]

Sigma_hid = np.cov(S_hid.T)
sigma_hid_iso = np.sqrt(np.trace(Sigma_hid) / len(hid_idx))
sigma_hid_aniso = np.sqrt(np.trace(Sigma_hid @ Sigma_hid) / np.trace(Sigma_hid))

# Predict swap rates
vis_scores = S_vis.mean(1)
vis_rank = np.argsort(np.argsort(-vis_scores))
hid_scores = S_hid.mean(1)
hid_rank = np.argsort(np.argsort(-hid_scores))

# Empirical swap rate (top-20 pairs)
top20 = np.argsort(-vis_scores)[:20]
emp_swaps = 0
emp_total = 0
for i in range(len(top20)):
    for j in range(i+1, len(top20)):
        a, b = top20[i], top20[j]
        vg = vis_scores[a] - vis_scores[b]
        hg = hid_scores[a] - hid_scores[b]
        if np.sign(vg) != np.sign(hg): emp_swaps += 1
        emp_total += 1

emp_rate = emp_swaps / emp_total if emp_total > 0 else 0

# Isotropic prediction
gaps = []
for i in range(len(top20)):
    for j in range(i+1, len(top20)):
        gaps.append(abs(vis_scores[top20[i]] - vis_scores[top20[j]]))
gaps = np.array(gaps)
iso_pred = norm.cdf(-gaps / (2 * sigma_hid_iso)).mean()
aniso_pred = norm.cdf(-gaps / (2 * sigma_hid_aniso)).mean()

print(f'  Sigma_hid trace: {np.trace(Sigma_hid):.2f}')
print(f'  sigma_iso: {sigma_hid_iso:.3f}, sigma_aniso: {sigma_hid_aniso:.3f}')
print(f'  Empirical swap rate (top-20): {emp_rate:.3f}')
print(f'  Isotropic prediction: {iso_pred:.3f}')
print(f'  Anisotropic prediction: {aniso_pred:.3f}')

results['q5'] = {'sigma_iso': sigma_hid_iso, 'sigma_aniso': sigma_hid_aniso,
                  'emp_swap': emp_rate, 'iso_pred': iso_pred, 'aniso_pred': aniso_pred}

# === Q7: STANDARDIZATION SENSITIVITY FIGURE ===
print('\n=== Q7: Standardization Sensitivity ===')
suites = {
    'OLLM v2': (pd.read_csv('data/ollm_v2.csv')[ollm_cols].apply(pd.to_numeric, errors='coerce').dropna().values, ollm_cols),
    'Extended': (ext_front, bench_cols),
}

methods = ['z-score', 'min-max', 'rank', 'raw']
fig, ax = plt.subplots(figsize=(6.75, 3.0))
x = np.arange(len(methods))
width = 0.35

for si, (suite_name, (S_suite, cols)) in enumerate(suites.items()):
    means_s = S_suite.mean(1)
    front_s = means_s >= np.median(means_s)
    S_sf = S_suite[front_s]
    n_sf = front_s.sum()

    ratios = []
    for method in methods:
        if method == 'z-score':
            S_t = (S_sf - S_sf.mean(0)) / S_sf.std(0)
        elif method == 'min-max':
            S_t = (S_sf - S_sf.min(0)) / (S_sf.max(0) - S_sf.min(0) + 1e-10)
        elif method == 'rank':
            S_t = np.zeros_like(S_sf)
            for c in range(S_sf.shape[1]):
                S_t[:, c] = np.argsort(np.argsort(S_sf[:, c])).astype(float) / n_sf
        else:  # raw
            S_t = S_sf.copy()

        C = np.corrcoef(S_t.T)
        ev = np.linalg.eigvalsh(C)[::-1]
        de = (ev.sum())**2 / (ev**2).sum()
        R = np.max(np.linalg.norm(S_t - S_t.mean(0), axis=1))
        omega = 1.57  # approximate from paper
        delta_vis = 2 * R * omega
        delta_2 = np.sort(S_t.mean(1))[-1] - np.sort(S_t.mean(1))[-2]
        ratio = delta_vis / (delta_2 + 1e-10)
        ratios.append(ratio)
        print(f'  {suite_name} {method}: d_eff={de:.2f}, R={R:.1f}, delta_vis={delta_vis:.1f}, ratio={ratio:.0f}x')

    ax.bar(x + si*width, ratios, width, label=suite_name)

ax.set_xticks(x + width/2)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel(r'$\delta_H^{vis} / \Delta_2$', fontsize=10)
ax.set_title('Standardization sensitivity', fontsize=10)
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig('figures/fig14_standardization.pdf', dpi=300)
fig.savefig('figures/fig14_standardization.png', dpi=300)
plt.close()
print('  Saved fig14_standardization')

results['q7'] = 'figure saved'

# === SAVE ===
with open('results/reviewer_fixes.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nAll Phase 1 experiments done. Saved: results/reviewer_fixes.json')

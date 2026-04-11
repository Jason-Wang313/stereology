"""Test A full investment: Evaluation Monoculture analysis."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid'); sns.set_palette('colorblind')

bench_cols = ['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO',
              'ARC','HellaSwag','MMLU','TruthfulQA','Winogrande','GSM8K']
shared_cols = bench_cols[:6]

ext = pd.read_csv('data/extended.csv')
ext_scores = ext[bench_cols].apply(pd.to_numeric, errors='coerce').dropna()
ext_all = ext_scores.values
ext_front = ext_all[ext_all.mean(1) >= np.median(ext_all.mean(1))]

ollm = pd.read_csv('data/ollm_v2.csv')
ollm_scores = ollm[shared_cols].apply(pd.to_numeric, errors='coerce').dropna()
ollm_all = ollm_scores.values
ollm_front = ollm_all[ollm_all.mean(1) >= np.median(ollm_all.mean(1))]

lb = pd.read_csv('data/livebench.csv')
lb_bench = [c for c in lb.columns if c != 'model']
lb_scores = lb[lb_bench].apply(pd.to_numeric, errors='coerce').dropna()
lb_all = lb_scores.values
lb_front = lb_all[lb_all.mean(1) >= np.median(lb_all.mean(1))]

def get_pcs(S):
    S_std = (S - S.mean(0)) / (S.std(0) + 1e-10)
    C = np.corrcoef(S_std.T)
    ev, evec = np.linalg.eigh(C)
    return ev[::-1], evec[:, ::-1]

# === 1. Cross-suite alignment in shared benchmark space ===
print('=== Cross-Suite PC Alignment (shared 6 benchmarks) ===')
shared_idx = [bench_cols.index(c) for c in shared_cols]
ext_shared = ext_front[:, shared_idx]
ev_e, vec_e = get_pcs(ext_shared)
ev_o, vec_o = get_pcs(ollm_front)

# Alignment matrix (absolute cosine similarity)
n_pcs = 6
align_matrix = np.zeros((n_pcs, n_pcs))
for i in range(n_pcs):
    for j in range(n_pcs):
        align_matrix[i, j] = abs(np.dot(vec_e[:, i], vec_o[:, j]))

print('Alignment matrix (rows=Extended PCs, cols=OLLM PCs):')
for i in range(n_pcs):
    row = ' '.join(f'{align_matrix[i,j]:.3f}' for j in range(n_pcs))
    print(f'  ext PC{i+1}: [{row}]')

# === 2. Bootstrap CI on PC1 alignment ===
print('\n=== Bootstrap CI on PC1 Alignment ===')
np.random.seed(42)
n_boot = 500
pc1_cosines = []
for _ in range(n_boot):
    idx_e = np.random.choice(len(ext_shared), len(ext_shared), replace=True)
    idx_o = np.random.choice(len(ollm_front), len(ollm_front), replace=True)
    _, ve = get_pcs(ext_shared[idx_e])
    _, vo = get_pcs(ollm_front[idx_o])
    pc1_cosines.append(abs(np.dot(ve[:, 0], vo[:, 0])))

pc1_cosines = np.array(pc1_cosines)
ci_lo, ci_hi = np.percentile(pc1_cosines, [2.5, 97.5])
print(f'  PC1 alignment: {abs(np.dot(vec_e[:,0], vec_o[:,0])):.3f} [{ci_lo:.3f}, {ci_hi:.3f}]')

# === 3. Null model ===
print('\n=== Null Model (random rotation) ===')
null_cosines = []
for _ in range(10000):
    # Random unit vector in R^6
    v1 = np.random.randn(6); v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(6); v2 /= np.linalg.norm(v2)
    null_cosines.append(abs(np.dot(v1, v2)))
null_cosines = np.array(null_cosines)
null_95 = np.percentile(null_cosines, 95)
print(f'  Null expected |cosine|: {null_cosines.mean():.3f} (95th: {null_95:.3f})')
print(f'  Observed PC1: {abs(np.dot(vec_e[:,0], vec_o[:,0])):.3f}')
print(f'  p-value: {(null_cosines >= abs(np.dot(vec_e[:,0], vec_o[:,0]))).mean():.6f}')

# === 4. Name the shared eigenstructure ===
print('\n=== Shared Eigenstructure Interpretation ===')
# Average the two suites' PCs (they're nearly identical)
avg_pc1 = (vec_e[:, 0] + vec_o[:, 0]) / 2
avg_pc1 /= np.linalg.norm(avg_pc1)
print('  Shared PC1 loadings (g-factor):')
for i, bench in enumerate(shared_cols):
    print(f'    {bench:15s}: {avg_pc1[i]:+.3f}')

avg_pc2 = (vec_e[:, 1] + vec_o[:, 1]) / 2
avg_pc2 /= np.linalg.norm(avg_pc2)
print('  Shared PC2 loadings (second axis):')
for i, bench in enumerate(shared_cols):
    print(f'    {bench:15s}: {avg_pc2[i]:+.3f}')

# === 5. Heatmap figure ===
fig, ax = plt.subplots(figsize=(4.0, 3.5))
im = ax.imshow(align_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(n_pcs)); ax.set_xticklabels([f'OLLM PC{i+1}' for i in range(n_pcs)], fontsize=7, rotation=45)
ax.set_yticks(range(n_pcs)); ax.set_yticklabels([f'Ext PC{i+1}' for i in range(n_pcs)], fontsize=7)
for i in range(n_pcs):
    for j in range(n_pcs):
        ax.text(j, i, f'{align_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if align_matrix[i,j] > 0.7 else 'black')
ax.set_title('Cross-suite PC alignment\n(|cosine similarity|)', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig('figures/fig15_monoculture.pdf', dpi=300)
fig.savefig('figures/fig15_monoculture.png', dpi=300)
plt.close()
print('\nSaved fig15_monoculture')

# === 6. Save results ===
results = {
    'alignment_matrix': align_matrix.tolist(),
    'pc1_cosine': float(abs(np.dot(vec_e[:,0], vec_o[:,0]))),
    'pc1_bootstrap_ci': [float(ci_lo), float(ci_hi)],
    'null_95': float(null_95),
    'null_pvalue': float((null_cosines >= abs(np.dot(vec_e[:,0], vec_o[:,0]))).mean()),
    'shared_pc1_loadings': {bench: float(avg_pc1[i]) for i, bench in enumerate(shared_cols)},
    'shared_pc2_loadings': {bench: float(avg_pc2[i]) for i, bench in enumerate(shared_cols)},
}
with open('results/monoculture.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved: results/monoculture.json')

"""Plot train/val loss for every HW1 run and print a summary table (reads nanoGPT/out-hw1/<run>/history.json)."""
import json, os
import matplotlib.pyplot as plt

ROOT = 'nanoGPT/out-hw1'
RUNS = ['baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa']
COMPARISONS = {'q4_1_baseline': ['baseline'], 'q4_2_rmsnorm': ['baseline', 'rmsnorm'],
               'q4_3_swiglu': ['baseline', 'swiglu'], 'q4_4_posenc': ['baseline', 'nope', 'rope'],
               'q4_5_gqa': ['baseline', 'gqa']}

data = {r: json.load(open(f'{ROOT}/{r}/history.json')) for r in RUNS if os.path.exists(f'{ROOT}/{r}/history.json')}
os.makedirs('plots', exist_ok=True)

for fname, runs in COMPARISONS.items():
    runs = [r for r in runs if r in data]
    if not runs:
        continue
    plt.figure(figsize=(6, 4))
    for i, r in enumerate(runs):
        h = data[r]['history']
        it = [p['iter'] for p in h]
        plt.plot(it, [p['train'] for p in h], f'C{i}--', alpha=0.7, label=f'{r} train')
        plt.plot(it, [p['val'] for p in h], f'C{i}-', label=f'{r} val')
    plt.xlabel('iteration'); plt.ylabel('cross-entropy loss'); plt.ylim(1.0, 2.8)
    plt.legend(); plt.grid(alpha=0.3); plt.title(fname)
    plt.tight_layout(); plt.savefig(f'plots/{fname}.png', dpi=150); plt.close()

print('| run | params (M) | best val | final train | final val |\n|---|---|---|---|---|')
for r, d in data.items():
    h = d['history']
    print(f"| {r} | {d['n_params']/1e6:.2f} | {min(p['val'] for p in h):.4f} | {h[-1]['train']:.4f} | {h[-1]['val']:.4f} |")

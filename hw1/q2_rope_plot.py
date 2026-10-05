"""Q2: RoPE attention score A(Δ) = (1/64) Σ_{m=0}^{63} cos(Δ·θ_m), θ_m = 10000^(-2m/128), for q = k = 1/√128·[1,...,1]."""
import numpy as np
import matplotlib.pyplot as plt

d, base = 128, 10000.0
theta = base ** (-np.arange(d // 2) * 2 / d)          # 64 frequencies
delta = np.arange(0, 65537)
A = np.cos(np.outer(delta, theta)).mean(axis=1)       # (1/64) Σ cos(Δ θ_m)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(delta, A, lw=0.4)
axes[0].set(xlabel='|i - j|', ylabel=r'$A_{i,j}$', title='RoPE score vs. distance (linear x)')
axes[1].semilogx(delta[1:], A[1:], lw=0.6)
axes[1].set(xlabel='|i - j| (log)', ylabel=r'$A_{i,j}$', title='RoPE score vs. distance (log x)')
for ax in axes:
    ax.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
plt.savefig('plots/q2_rope.png', dpi=150)
print(f"A(0)={A[0]:.3f}  A(100)={A[100]:.3f}  A(1000)={A[1000]:.3f}  "
      f"mean|A| over [10000,65536]={np.abs(A[10000:]).mean():.3f}  min={A.min():.3f} at Δ={A.argmin()}")

# CS6180 Homework 1

Code: `q2_rope_plot.py`, `plot_losses.py`, `make_pdf.py`, and modified `nanoGPT/` (`model.py`, `train.py`, `config/hw1_base.py`, `run_hw1.sh`).

---

## Q1. Convolution

**1.1** Output size = (4 − 3)/1 + 1 = **2×2**.

**1.2** im2col turns each 3×3 patch into a column, so X_col is 9×4. Column (r,c) is vec(X[r:r+3, c:c+3]).
With w = vec(W)ᵀ (1×9): **Y = w · X_col** (1×4), reshaped to 2×2.

GPU advantage: the convolution becomes one big dense GEMM. That means highly optimized BLAS kernels (cuBLAS, Tensor Cores), regular memory access and high parallelism, and many filters and images handled in a single call. Cost: extra memory, because pixels are duplicated (36 vs 16 entries).

**1.3** With g = vec(G):

  ∂L/∂X_col = wᵀ g (9×4),  **∂L/∂X = col2im(wᵀ g)**, where col2im sums overlapping entries.

Equivalently, this is a full convolution of the zero-padded G with the 180°-rotated kernel:

  **∂L/∂X = conv(pad(G, 2), rot180(W))**, i.e. ∂L/∂X[p,q] = Σ_{r,c} G[r,c] · W[p−r, q−c].



---

## Q2. RoPE

**2.1** (R_i q)ᵀ(R_j k) = qᵀ R_{j−i} k. For each of the 64 rotated pairs, with a = 1/√128:

  [a, a] · Rot(Δθ_m) · [a, a]ᵀ = 2a² cos(Δθ_m)  (the sin terms cancel).

Summing over the pairs:

  **A_{i,j} = (1/64) Σ_{m=0}^{63} cos((i − j) θ_m),  θ_m = 10000^(−2m/128)**

This is the score before the 1/√d scaling and softmax.

**2.2**
![RoPE](plots/q2_rope.png)

**2.3** The score decays quickly with distance. Beyond a few thousand tokens it only oscillates around 0, so far positions carry no useful signal. Also, positions longer than the training length produce rotation angles the model never saw, so RoPE does not extrapolate past its training context.

Mitigations:
- Position Interpolation (rescale positions)
- NTK-aware scaling / a larger RoPE base
- YaRN
- Fine-tuning on longer sequences
- ALiBi as an alternative

---

## Q3. Grammar error correction model

**Architecture.** A Transformer encoder-decoder (like BART/T5):
- **Tokenizer:** BPE, 32k vocab.
- **Embedding:** token embedding (d = 768) plus positional encoding.
- **Encoder (6–12 layers):** bidirectional multi-head self-attention, then an FFN (768→3072→768, GELU), each with LayerNorm and a residual. It reads the whole erroneous sentence.
- **Decoder (6–12 layers):** causal self-attention, cross-attention to the encoder output, and an FFN, each with LayerNorm and a residual.
- **Output:** linear layer + softmax over the vocabulary.

```
wrong sentence → [Embedding] → [Encoder × N] ─┐
                                              ▼
shifted target → [Embedding] → [Decoder × N (self-attn, cross-attn, FFN)] → [Linear + Softmax] → corrected sentence
```

**Input / output.** Input: the ungrammatical sentence. Output: the corrected sentence, generated token by token (a copy if it is already correct).

**Training:**
- **Data:** start from a pretrained BART/T5, train on synthetic error pairs, then fine-tune on real GEC data (Lang-8, FCE, W&I+LOCNESS).
- **Loss:** token-level cross-entropy with teacher forcing (label smoothing 0.1).
- **Optimizer:** AdamW (β = 0.9/0.98, weight decay 0.01), lr 3e-5 for fine-tuning with 1k warmup steps then linear decay, batch ~64 sentences, ~20k–50k steps, early stopping on the dev set.
- **Inference / evaluation:** beam search at inference; F0.5 (ERRANT) for evaluation.

---

## Q4. Mini-LLM (nanoGPT)

**Setup.** All runs share one config, `nanoGPT/config/hw1_base.py`:
- Scaled-down `train_shakespeare_char.py` so all six runs fit on an 8GB Apple M1 (MPS): 4 layers, 4 heads, d = 256, context 128, batch 32, dropout 0.2.
- 3000 iterations, lr 1e-3 with cosine decay to 1e-4.
- Same seed (1337) for every run.

Each variant changes **one** flag in `model.py` relative to the baseline. The learning rate was **not tuned**: every run uses 1e-3.

Run with `cd nanoGPT && python data/shakespeare_char/prepare.py && ./run_hw1.sh`, then `python plot_losses.py`. Logs are in `logs/`, loss history in `nanoGPT/out-hw1/*/history.json`.

| Run | Change | Params (M, excluding position embeddings) | Final train | Final val |
|---|---|---|---|---|
| baseline | — | 3.16 | 1.3585 | 1.5602 |
| rmsnorm | LayerNorm → RMSNorm | 3.16 | 1.3612 | 1.5643 |
| swiglu | GELU MLP → SwiGLU | 3.23 | 1.3112 | 1.5353 |
| nope | no positional encoding | 3.16 | 1.6520 | 1.8393 |
| rope | learned absolute → RoPE | 3.16 | 1.2939 | **1.5170** |
| gqa | MHA → GQA (4 query heads, 2 KV heads) | 2.90 | 1.3679 | 1.5674 |

**4.1 Baseline.** Final val loss is 1.560.
![baseline](plots/q4_1_baseline.png)

**4.2 RMSNorm.** nanoGPT uses **pre-LayerNorm**: `x = x + attn(ln_1(x))`, `x = x + mlp(ln_2(x))`, plus a final `ln_f`. Replacing every LayerNorm with RMSNorm (no mean subtraction, no bias) gives a val loss of 1.564 versus 1.560, essentially the same. RMSNorm is slightly cheaper with no loss in quality.
![rmsnorm](plots/q4_2_rmsnorm.png)

**4.3 SwiGLU.** The MLP uses **GELU** with a 4d hidden layer. SwiGLU computes `W2(SiLU(W1 x) ⊙ W3 x)` with hidden size ≈ 8/3·d, rounded to 704, which gives roughly the same parameter count (+2%). Val loss improves to 1.535 (−0.025); the gating helps.
![swiglu](plots/q4_3_swiglu.png)

**4.4 Positional encoding.** The model uses **learned absolute positional embeddings** (`wpe`).
- **NoPE** (remove `wpe`) is much worse: 1.839, +0.28. Causal masking gives only weak implicit position information, which is not enough for this small model.
- **RoPE** (rotate q and k in every attention layer, base 10000) is the best run: 1.517, −0.043. Relative-position information injected at every layer helps character-level modeling.

![posenc](plots/q4_4_posenc.png)

**4.5 GQA.** Keys and values are projected to 2 heads, and each KV head is shared by 2 query heads (`repeat_interleave`). This cuts attention K/V parameters and the KV cache in half (−8% total parameters). Val loss is 1.567 versus 1.560: almost no quality loss for a smaller model and cheaper inference.
![gqa](plots/q4_5_gqa.png)

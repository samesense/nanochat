# Attention, extracted from nanochat

> *Interview prompt: "Pull the attention out of this repo and explain it."*

This folder is my answer. I took the attention layer out of `nanochat/gpt.py`,
stripped away everything that isn't attention (the optimizer, dtype plumbing,
value-embedding gates, the Flash-Attention kernel), and rebuilt it as a single
pure-PyTorch file that runs on a laptop CPU.

```
learning/attention/
├── README.md      ← this write-up
├── attention.py   ← the extracted layer (~180 lines, runs on CPU)
└── demo.py        ← runnable checks: shapes, causality, KV cache, sliding window
```

Run it:

```bash
pip install torch      # not vendored in this repo
cd learning/attention
python demo.py
```

---

## 1. Where the code lives in the repo

The attention itself is one class, `CausalSelfAttention` in
[`nanochat/gpt.py`](../../nanochat/gpt.py) (lines ~65–126). It leans on three
collaborators:

| Concern | In the repo | Why it's separate |
|---|---|---|
| The actual softmax-attention kernel | [`nanochat/flash_attention.py`](../../nanochat/flash_attention.py) | Uses Flash-Attention 3 on Hopper GPUs, falls back to PyTorch SDPA everywhere else. |
| Position encoding (RoPE) | `apply_rotary_emb` + `_precompute_rotary_embeddings` in `gpt.py` | Shared setup, applied inside the layer. |
| Inference state | `KVCache` in [`nanochat/engine.py`](../../nanochat/engine.py) | Lives with the serving engine, not the model. |

My `attention.py` folds the essential parts of all three into one place and
swaps the Flash-Attention backend for `F.scaled_dot_product_attention`, which is
numerically the same thing but portable.

---

## 2. What attention actually computes

For each token, attention asks: *"given what I'm looking for, which other tokens
matter, and what do I take from them?"* Three projections of the input `x`:

- **Query** `q` — what this token is looking for
- **Key** `k` — what each token offers as a match signal
- **Value** `v` — what each token actually contributes if matched

The output for token *i* is a weighted average of all values, where the weights
come from how well *i*'s query matches each token's key:

```
attn(Q, K, V) = softmax( Q Kᵀ / √d  +  mask ) V
```

- `Q Kᵀ` — every query dotted with every key → a (T × T) score matrix.
- `/ √d` — scale so the dot products don't blow up the softmax as `d` grows.
- `mask` — `−∞` on positions a token isn't allowed to see (see causality below).
- `softmax` — turn scores into a probability distribution over tokens.
- `… V` — average the values under that distribution.

"Multi-head" just means we do this `H` times in parallel on `d`-dim slices of the
vector and concatenate, so different heads can specialize (syntax, coreference,
position, …).

This repo never materializes that (T × T) matrix in Python — Flash Attention /
SDPA fuse the whole expression into one kernel — but the math is exactly the
above.

---

## 3. Walking the layer, step by step

Following `forward()` in `attention.py`:

**1. Project and split into heads.** `c_q/c_k/c_v` are bias-free linears. We
reshape to `(B, T, H, D)` so the head axis is explicit.

**2. Rotary position embeddings (RoPE).** Plain attention is permutation-
invariant — it has no idea what order tokens came in. RoPE fixes this by
*rotating* each query and key by an angle proportional to its position. Because a
rotation by `θ_i` on the query and `θ_j` on the key leaves their dot product
depending only on `θ_i − θ_j`, the model sees **relative** position for free.
That's why there are no learned positional embeddings anywhere in this model.
(Note: RoPE rotates `q` and `k` only — never `v`.)

**3. QK-norm.** Before the dot product we RMS-normalize each query and key
vector. This caps how large the attention logits can get, which keeps training
stable (a known failure mode is logits drifting large and the softmax
saturating). The repo then multiplies both by `1.2` to *sharpen* the
distribution slightly — a tuned knob.

**4. Grouped-Query Attention (GQA).** Queries use `n_head` heads but keys/values
use fewer, `n_kv_head`. Several query heads share one KV head. Quality is nearly
unchanged, but the KV **cache** (the memory bottleneck at inference) shrinks by
`n_head / n_kv_head`. With `n_head=8, n_kv_head=2` that's a 4× smaller cache.
`F.scaled_dot_product_attention(..., enable_gqa=True)` handles the broadcast.

**5. Causal + sliding-window masking.** This is a decoder — token *i* may only
attend to tokens `≤ i`, otherwise it would peek at the answer it's trying to
predict. That's the causal mask. Optionally a **sliding window** further limits a
token to the last `window` keys; in the full model most layers use a short
window (cheaper, local) and a few use full context, set by the `window_pattern`
string (`"SSSL"` = three short layers then one long, tiled). `demo.py` shows both
masks empirically.

**6. Re-assemble and project.** Concatenate the heads back to `(B, T, n_embd)`
and pass through `c_proj` into the residual stream.

---

## 4. The KV cache (why inference is fast)

At generation time we emit one token at a time. The keys and values for all
earlier tokens never change, so recomputing them every step is pure waste —
`O(T²)` over a full decode. `KVCache` stores K and V per layer and per position;
each step computes K/V for just the *new* token, appends it, and attends against
the whole stored history. This is the single biggest reason transformer
inference is tractable, and it's also *why* GQA matters: the cache is what eats
memory bandwidth, so making it smaller directly speeds up decoding.

`demo.py`'s third check feeds a sequence through the cache one token at a time
and confirms it reproduces the full-sequence forward pass to within `1e-4`.

---

## 5. What I deliberately left out

These live in `gpt.py` but are *additions around* attention, not attention
itself — I dropped them to keep the core legible:

- **Value embeddings / ResFormer gate** (`ve`, `ve_gate`) — mixes a learned
  per-token value directly into `v` through an input-dependent gate.
- **Flash Attention 3 kernel** — a faster fused implementation; SDPA computes the
  same result.
- **Smear, backout, per-layer `resid`/`x0` lambdas** — residual-stream tricks at
  the `GPT` level, not inside the attention layer.
- **bf16/fp8 dtype handling, meta-device init, the Muon/AdamW optimizer.**

Everything in §2–§4 is the part you'd be expected to reproduce on a whiteboard;
this folder is that part, made runnable.

---

## 6. Quick reference: tensor shapes

```
x            (B, T, n_embd)              layer input
q            (B, T, n_head,    head_dim) queries  (H heads)
k, v         (B, T, n_kv_head, head_dim) keys/values (fewer heads: GQA)
cos, sin     (1, T, 1, head_dim/2)       rotary tables, broadcast over B and H
scores       (B, H, T, T)                Q·Kᵀ, never materialized in practice
y → c_proj   (B, T, n_embd)              layer output, back to residual stream
```

`B`=batch, `T`=sequence length, `H`=`n_head`, `D`=`head_dim`=`n_embd/n_head`.

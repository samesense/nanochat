# Single-Head Attention (one batch, one head)

The simplest possible version: **one sequence, one head, plain 2D matrices.** No
batch axis, no head axis, no transposes. The point is to see the core matrix
math without any bookkeeping.

```python
scores  = q @ k.T / sqrt(C)   # (T, T)  every query · every key
weights = softmax(scores)      # (T, T)  each row sums to 1
out     = weights @ v          # (T, C)  blend the values
```

## The shapes, end to end

With `T` tokens and embedding dim `C`:

| tensor | shape | meaning |
|---|---|---|
| `x` | `(T, C)` | input — one row per token |
| `q`, `k`, `v` | `(T, C)` | each token's query / key / value |
| `scores` | `(T, T)` | `scores[i, j]` = token *i*'s query · token *j*'s key |
| `weights` | `(T, T)` | `scores` after softmax over each row |
| `out` | `(T, C)` | `weights @ v` — each token's new representation |

The whole mechanism is three matrix multiplies and one softmax.

## Reading the `(T, T)` matrix

`scores = q @ k.T` is the heart of it. Multiplying `(T, C) @ (C, T)` pairs every
query with every key, so `scores[i, j]` is the dot product of token *i*'s query
with token *j*'s key — a single number saying "how relevant is token *j* to
token *i*?"

`softmax(scores, dim=-1)` runs **along each row**, turning row *i* into a
probability distribution: how token *i* divides its attention across all tokens.
That's why every row sums to 1. The demo prints this matrix so you can see it.

`weights @ v` then takes, for each token *i*, a weighted average of all value
vectors using row *i* as the weights — `(T, T) @ (T, C) -> (T, C)`.

## How this grows up

This is deliberately the floor. Each step toward the real model adds one axis or
one rule:

- **Multiple heads** — split `C` into `H` slices, run this in parallel on each,
  concatenate. Adds a head axis `(H, T, D)` and a transpose to merge back.
- **Batching** — process many sequences at once. Adds a leading batch axis
  `(B, H, T, D)`. The `@` and softmax are unchanged; only the shapes grow.
- **Causal mask** — set `scores[i, j] = -inf` for `j > i` so a token can't see
  the future. See `../causal-attention/`.
- **RoPE, QK-norm, GQA, KV cache** — the nanochat additions, in `../attention.py`.

Nothing about the core math changes as you add these — the extra dimensions just
ride along in front. That's the whole reason to start here.

Run:

```bash
python demo.py
```

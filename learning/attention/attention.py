"""
Self-contained causal self-attention, extracted and trimmed from nanochat/gpt.py.

The goal of this file is *pedagogy*: everything you need to understand one
attention layer lives here, in pure PyTorch, with no Flash-Attention dependency
and no distributed/training machinery. It runs on CPU.

What's kept (the parts that define "attention" in this repo):
  - Q/K/V projections
  - Rotary position embeddings (RoPE)
  - QK-norm (RMSNorm on queries and keys before the dot product)
  - Grouped-Query Attention (GQA): fewer K/V heads than Q heads
  - Causal masking + sliding-window masking
  - A KV cache for autoregressive (one-token-at-a-time) inference

What's dropped (orthogonal to attention itself):
  - Flash Attention 3 kernels  -> replaced by F.scaled_dot_product_attention
  - Value embeddings / ResFormer gate, smear, backout, per-layer lambdas
  - fp8/bf16 dtype juggling, meta-device init, the optimizer

Run `python demo.py` for a guided walkthrough.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def rms_norm(x):
    """RMSNorm with no learnable scale. Normalizes the last dim to unit RMS.

    Unlike LayerNorm there is no mean-subtraction and no bias: we just divide by
    the root-mean-square of the vector. Cheap and works well for transformers.
    """
    return F.rms_norm(x, (x.size(-1),))


def precompute_rotary(seq_len, head_dim, base=10000.0, device="cpu"):
    """Precompute the cos/sin tables that RoPE rotates Q and K by.

    Idea: pair up the head dimensions and treat each pair as a 2D point. At
    position `t` we rotate that point by an angle `t * inv_freq`, where each
    pair has its own frequency. Low-index pairs rotate fast, high-index pairs
    rotate slowly, so the model can read both fine and coarse relative offsets.

    Returns cos, sin of shape (1, seq_len, 1, head_dim/2) ready to broadcast
    over (batch, time, head, dim).
    """
    # one frequency per *pair* of channels -> head_dim/2 frequencies
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)              # (seq_len, head_dim/2)
    cos, sin = freqs.cos(), freqs.sin()
    # add batch and head axes so they broadcast against (B, T, H, D)
    return cos[None, :, None, :], sin[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    """Rotate the per-head vectors in x by the angles encoded in cos/sin.

    x is (B, T, H, D). We split D into two halves (x1, x2), treat them as the
    real/imaginary parts of D/2 complex numbers, and multiply by e^{i*theta}:
        (x1 + i x2) * (cos + i sin)  ->  (x1 cos - x2 sin) + i(x1 sin + x2 cos)
    This makes the dot product q.k depend only on the *relative* position of the
    two tokens, which is exactly the inductive bias we want.
    """
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=3)


def sdpa(q, k, v, window, enable_gqa):
    """Scaled dot-product attention with causal + optional sliding-window mask.

    Inputs are in (B, H, T, D) layout (PyTorch SDPA's native layout).
      window < 0   -> full causal context
      window >= 0  -> each query may look back at most `window` keys

    `enable_gqa=True` lets SDPA broadcast a small number of K/V heads across a
    larger number of Q heads (Grouped-Query Attention).
    """
    Tq, Tk = q.size(2), k.size(2)

    # Fast path: full causal attention over a same-length sequence (training).
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # General path: build an explicit boolean mask. This also covers KV-cache
    # decoding, where Tq (new tokens) != Tk (everything seen so far).
    device = q.device
    # row i is absolute query position (Tk - Tq) + i ; col j is key position j
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx                       # causal: can't see the future
    if 0 <= window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)  # and not too far in the past
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# -----------------------------------------------------------------------------
# A minimal KV cache for inference
# -----------------------------------------------------------------------------

class KVCache:
    """Stores keys/values for every position seen so far, for one attention layer.

    During generation we feed one token at a time. Without a cache we'd recompute
    K and V for the entire prefix on every step (O(T^2) wasted work). The cache
    keeps them around so each step only computes K/V for the *new* token.
    """

    def __init__(self, batch_size, n_kv_head, head_dim, max_seq_len, device, dtype):
        self.k = torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim, device=device, dtype=dtype)
        self.pos = 0  # number of tokens currently stored

    def append(self, k_new, v_new):
        """Write the new token(s) into the cache and return the full K/V so far."""
        T = k_new.size(1)
        self.k[:, self.pos:self.pos + T] = k_new
        self.v[:, self.pos:self.pos + T] = v_new
        self.pos += T
        return self.k[:, :self.pos], self.v[:, :self.pos]


# -----------------------------------------------------------------------------
# The attention layer
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """One multi-head causal self-attention layer (GQA + RoPE + QK-norm).

    Mirrors nanochat/gpt.py:CausalSelfAttention, minus the value-embedding gate
    and the Flash-Attention backend.
    """

    def __init__(self, n_embd, n_head, n_kv_head):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head

        # Q gets a full set of heads; K and V get n_kv_head heads (GQA). When
        # n_kv_head < n_head the cache is smaller and decoding is more bandwidth
        # efficient -- several query heads share one key/value head.
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin, window=-1, kv_cache=None):
        B, T, C = x.size()

        # 1) Project to queries, keys, values and split into heads.
        #    Layout (B, T, H, D) keeps the head axis explicit.
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2) RoPE: inject *relative* position info into q and k (not v).
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 3) QK-norm: normalize each query/key vector before the dot product.
        #    This bounds the logits and stabilizes training. The 1.2 factor
        #    sharpens the softmax a touch (split as 1.2 on q and 1.2 on k).
        q, k = rms_norm(q), rms_norm(k)
        q, k = q * 1.2, k * 1.2

        # 4) If decoding with a cache, append new K/V and pull the full history.
        if kv_cache is not None:
            k, v = kv_cache.append(k, v)

        # 5) Attention. SDPA wants (B, H, T, D), so transpose in and back out.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        enable_gqa = self.n_kv_head != self.n_head
        y = sdpa(q, k, v, window, enable_gqa)
        y = y.transpose(1, 2)                       # back to (B, T, H, D)

        # 6) Concatenate heads and project back into the residual stream.
        y = y.contiguous().view(B, T, C)
        return self.c_proj(y)

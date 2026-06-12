"""
Causal scaled dot-product self-attention.

This is the same as vanilla attention, except we mask out future keys before the
softmax. Token i can attend only to tokens 0..i, which is the rule needed for an
autoregressive decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device):
    """Lower-triangular mask where True means attention is allowed."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def scaled_dot_product_causal_attention(q, k, v):
    """Compute softmax(mask(QK^T / sqrt(d)))V.

    q, k, v are shaped (B, H, T, D).
    """
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))
    mask = causal_mask(q.size(2), q.device)
    scores = scores.masked_fill(~mask, -torch.inf)
    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out, weights


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def _split_heads(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, D)

    def forward(self, x, return_weights=False):
        q = self._split_heads(self.c_q(x))
        k = self._split_heads(self.c_k(x))
        v = self._split_heads(self.c_v(x))

        y, weights = scaled_dot_product_causal_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)

        if return_weights:
            return y, weights
        return y

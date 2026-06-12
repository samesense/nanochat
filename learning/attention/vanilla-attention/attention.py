"""
Vanilla scaled dot-product self-attention.

This is the smallest useful attention implementation:
  1) project x into Q, K, V
  2) compute scores = Q K^T / sqrt(head_dim)
  3) softmax scores into attention weights
  4) use weights to average V

There is deliberately no causal mask here, so every token can attend to every
other token, including tokens that appear later in the sequence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    """Compute softmax(QK^T / sqrt(d))V.

    q, k, v are shaped (B, H, T, D), where:
      B = batch size
      H = number of heads
      T = sequence length
      D = per-head dimension
    """
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))
    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out, weights


class VanillaSelfAttention(nn.Module):
    """Multi-head self-attention without any mask."""

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

        y, weights = scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)

        if return_weights:
            return y, weights
        return y

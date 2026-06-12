"""
Single-head, single-batch vanilla self-attention.

Stripped to the bone so the matrix math is the only thing on screen: no batch
axis, no head axis, no transposes to undo. Everything is a 2D matrix.

Shapes throughout (T = number of tokens, C = embedding dim):
    x        (T, C)     input: one row per token
    q, k, v  (T, C)     each token's query / key / value
    scores   (T, T)     scores[i, j] = how much token i attends to token j
    weights  (T, T)     scores after softmax over each row (rows sum to 1)
    out      (T, C)     weights @ v: each token's new representation

Read `attention()` top to bottom; that is the whole idea.
"""

import math

import torch
import torch.nn as nn


def softmax(scores):
    """Softmax over the last dimension, written out by hand.

        softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    turns a row of arbitrary numbers into a probability distribution: every
    entry is positive and the row sums to 1.

    The one subtlety is numerical stability. exp() overflows for large inputs,
    so we first subtract each row's max. This changes nothing mathematically --
    exp(x_i - m) / sum_j exp(x_j - m) cancels the exp(-m) top and bottom and
    equals the formula above -- but now the largest exponent is exp(0) = 1, so
    nothing overflows.
    """
    scores = scores - scores.max(dim=-1, keepdim=True).values  # (T, 1) broadcast
    exp = scores.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def attention(q, k, v):
    """Vanilla scaled dot-product attention on plain (T, C) matrices.

    Returns (out, weights) so the demo can print the attention matrix.
    """
    C = q.size(-1)

    # scores[i, j] = dot(q_i, k_j). q @ k.T pairs every query with every key.
    # (T, C) @ (C, T) -> (T, T)
    scores = q @ k.T

    # Scale so the scores don't grow with C (keeps the softmax from saturating).
    scores = scores / math.sqrt(C)

    # Softmax over each ROW: row i becomes a probability distribution describing
    # how token i splits its attention across all tokens j. Rows sum to 1.
    weights = softmax(scores)

    # Blend the value vectors using those weights.
    # (T, T) @ (T, C) -> (T, C)
    out = weights @ v
    return out, weights


class SingleHeadSelfAttention(nn.Module):
    """One attention head, batch size 1. Input and output are both (T, C)."""

    def __init__(self, n_embd):
        super().__init__()
        # Learn how to turn each token into its query, key, and value.
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        # Final projection back into the residual stream.
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, return_weights=False):
        # x is (T, C) -- a single sequence, no batch dimension.
        q = self.c_q(x)   # (T, C)
        k = self.c_k(x)   # (T, C)
        v = self.c_v(x)   # (T, C)

        out, weights = attention(q, k, v)
        out = self.c_proj(out)

        if return_weights:
            return out, weights
        return out

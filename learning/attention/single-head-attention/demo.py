"""
Run with:
    python demo.py

Prints the actual (T, T) attention matrix so you can see the matrix math, and
checks the two invariants that define vanilla attention:
  1) shapes: (T, C) in -> (T, C) out, with a (T, T) attention matrix
  2) every row of the attention matrix is a probability distribution (sums to 1)
"""

import torch
import torch.nn.functional as F

from attention import SingleHeadSelfAttention, attention, softmax

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)


# 0) Our hand-written softmax matches PyTorch's ------------------------------
# We only use the library version here as a correctness oracle, not in attention.
probe = torch.randn(4, 4) * 50  # large values: exercises the overflow-safe path
assert torch.allclose(softmax(probe), F.softmax(probe, dim=-1), atol=1e-6)
print("[softmax] hand-written softmax matches torch (even for large inputs)")

T, C = 4, 8  # 4 tokens, 8-dim embeddings -- small enough to print
x = torch.randn(T, C)
attn = SingleHeadSelfAttention(n_embd=C).eval()


# 1) Run it and look at the attention matrix --------------------------------
with torch.no_grad():
    y, weights = attn(x, return_weights=True)

print(f"[shapes]  x {tuple(x.shape)} -> y {tuple(y.shape)},  weights {tuple(weights.shape)}")
print("\n[attention matrix] weights[i, j] = how much token i attends to token j")
print(weights)
assert y.shape == x.shape
assert weights.shape == (T, T)


# 2) Each row is a probability distribution ---------------------------------
row_sums = weights.sum(dim=-1)
print(f"\n[rows]    row sums (should all be 1.0): {row_sums.tolist()}")
assert torch.allclose(row_sums, torch.ones(T), atol=1e-6)


# 3) Sanity-check the bare formula on hand-made vectors ----------------------
# Two tokens are identical, one is different. The identical pair should attend
# to each other more strongly than to the odd one out.
q = k = v = torch.tensor([
    [1.0, 0.0],   # token 0
    [1.0, 0.0],   # token 1  (same direction as token 0)
    [0.0, 1.0],   # token 2  (orthogonal)
])
_, w = attention(q, k, v)
print("\n[formula] q=k=v, tokens 0 and 1 point the same way, token 2 is orthogonal:")
print(w)
assert w[0, 1] > w[0, 2], "token 0 should attend to its twin more than the odd one"

print("\nAll single-head attention checks passed.")

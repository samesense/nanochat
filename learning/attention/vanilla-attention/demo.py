"""
Run with:
    python demo.py
"""

import torch

from attention import VanillaSelfAttention


torch.manual_seed(0)

B, T, C, H = 2, 6, 16, 4
x = torch.randn(B, T, C)
attn = VanillaSelfAttention(n_embd=C, n_head=H).eval()


# 1) Shape check ---------------------------------------------------------------
with torch.no_grad():
    y, weights = attn(x, return_weights=True)

print(f"[shapes]  x {tuple(x.shape)} -> y {tuple(y.shape)}")
print(f"[weights] attention weights {tuple(weights.shape)}")
assert y.shape == x.shape
assert weights.shape == (B, H, T, T)


# 2) Attention weights are probabilities --------------------------------------
row_sums = weights.sum(dim=-1)
max_prob_error = (row_sums - 1).abs().max().item()
print(f"[weights] max |row_sum - 1| = {max_prob_error:.2e}")
assert max_prob_error < 1e-6


# 3) No causal mask ------------------------------------------------------------
# Editing the final token may affect earlier outputs because vanilla attention
# allows every query to look at every key/value.
with torch.no_grad():
    x2 = x.clone()
    x2[:, -1] += 10.0
    y2 = attn(x2)

changed = ((y - y2).abs().sum(dim=-1) > 1e-6)[0].int().tolist()
print(f"[unmasked] positions changed by editing last token: {changed}")
assert any(changed[:-1]), "expected at least one earlier token to use the final token"

print("\nAll vanilla attention checks passed.")

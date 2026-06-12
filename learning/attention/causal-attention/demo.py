"""
Run with:
    python demo.py
"""

import torch

from attention import CausalSelfAttention


torch.manual_seed(0)

B, T, C, H = 2, 6, 16, 4
x = torch.randn(B, T, C)
attn = CausalSelfAttention(n_embd=C, n_head=H).eval()


# 1) Shape check ---------------------------------------------------------------
with torch.no_grad():
    y, weights = attn(x, return_weights=True)

print(f"[shapes]  x {tuple(x.shape)} -> y {tuple(y.shape)}")
print(f"[weights] attention weights {tuple(weights.shape)}")
assert y.shape == x.shape
assert weights.shape == (B, H, T, T)


# 2) Mask check ----------------------------------------------------------------
# Future-token weights must be exactly zero after softmax.
future_weights = weights.masked_select(torch.ones(T, T, dtype=torch.bool).triu(1))
max_future_weight = future_weights.max().item()
print(f"[mask]    max attention weight on future tokens = {max_future_weight:.2e}")
assert max_future_weight == 0.0


# 3) Causality check -----------------------------------------------------------
# Editing the final token must not affect any earlier output.
with torch.no_grad():
    x2 = x.clone()
    x2[:, -1] += 10.0
    y2 = attn(x2)

changed = ((y - y2).abs().sum(dim=-1) > 1e-6)[0].int().tolist()
print(f"[causal]  positions changed by editing last token: {changed}")
assert not any(changed[:-1]), "past outputs changed after editing a future token"
assert changed[-1], "the edited token's own output should change"

print("\nAll causal attention checks passed.")

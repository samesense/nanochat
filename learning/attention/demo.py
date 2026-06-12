"""
Guided demo of the extracted attention layer.

    pip install torch
    python demo.py

It checks the things an interviewer would actually probe:
  1) Shapes flow through correctly (GQA: many Q heads, few K/V heads).
  2) The manual attention formula matches PyTorch SDPA.
  3) Causality holds: a token's output never depends on future tokens.
  4) The KV cache gives the same answer as a full forward pass, token by token.
  5) Sliding-window masking affects only the expected local range.
"""

import torch
from attention import CausalSelfAttention, precompute_rotary, KVCache

torch.manual_seed(0)

B, T = 2, 16          # batch, sequence length
n_embd = 64
n_head = 8            # query heads
n_kv_head = 2         # key/value heads  -> 4 query heads share each KV head (GQA)
head_dim = n_embd // n_head

attn = CausalSelfAttention(n_embd, n_head, n_kv_head).eval()
cos, sin = precompute_rotary(T, head_dim)
x = torch.randn(B, T, n_embd)


# 1) Shapes -------------------------------------------------------------------
with torch.no_grad():
    y = attn(x, cos, sin, window=-1)
print(f"[shapes]   in {tuple(x.shape)} -> out {tuple(y.shape)}  "
      f"(Q heads={n_head}, KV heads={n_kv_head})")
assert y.shape == x.shape


# 2) Manual attention == SDPA --------------------------------------------------
# The layer normally uses PyTorch's fused scaled_dot_product_attention. For
# learning, attention.py also includes a direct implementation of:
#   softmax(Q K^T / sqrt(d) + mask) V
with torch.no_grad():
    y_manual = attn(x, cos, sin, window=-1, use_manual=True)
backend_diff = (y - y_manual).abs().max().item()
print(f"[manual]   max |SDPA - manual| = {backend_diff:.2e}")
assert backend_diff < 1e-5, "manual attention diverged from SDPA"
print("[manual]   explicit attention formula matches SDPA")


# 3) Causality ----------------------------------------------------------------
# Perturb only the LAST token of the input. Outputs at every earlier position
# must be byte-for-byte identical, because causal attention can't look forward.
with torch.no_grad():
    x2 = x.clone()
    x2[:, -1] += 10.0  # large perturbation at the final position
    y2 = attn(x2, cos, sin, window=-1)
changed = (y - y2).abs().sum(dim=-1) > 1e-6        # (B, T): did this position change?
print(f"[causal]   positions changed by editing last token: "
      f"{changed[0].int().tolist()}")
assert not changed[:, :-1].any(), "causality violated: past attended to the future!"
print("[causal]   only the last position changed -> causal mask is correct")


# 4) KV cache == full forward -------------------------------------------------
# Feed the sequence one token at a time through the cache and confirm we recover
# the same outputs as the single full-sequence forward pass above.
with torch.no_grad():
    cache = KVCache(B, n_kv_head, head_dim, max_seq_len=T, device="cpu", dtype=x.dtype)
    incremental = []
    for t in range(T):
        cos_t, sin_t = cos[:, t:t + 1], sin[:, t:t + 1]   # rotary slice for this position
        yt = attn(x[:, t:t + 1], cos_t, sin_t, window=-1, kv_cache=cache)
        incremental.append(yt)
    y_cached = torch.cat(incremental, dim=1)
max_diff = (y - y_cached).abs().max().item()
print(f"[kvcache]  max |full - incremental| = {max_diff:.2e}")
assert max_diff < 1e-4, "KV cache diverged from the full forward pass"
print("[kvcache]  incremental decoding matches the full forward pass")


# 5) Sliding window ----------------------------------------------------------
# With window=4, a query may only look back 4 keys. Editing a token should now
# affect at most the next `window` positions, not the whole tail.
with torch.no_grad():
    yw = attn(x, cos, sin, window=4)
    xw = x.clone(); xw[:, 4] += 10.0
    yw2 = attn(xw, cos, sin, window=4)
changed_w = ((yw - yw2).abs().sum(dim=-1) > 1e-6)[0].int().tolist()
print(f"[window=4] positions changed by editing token 4: {changed_w}")
expected_w = [0] * T
for i in range(4, 4 + 4 + 1):
    expected_w[i] = 1
assert changed_w == expected_w, "sliding window mask affected the wrong positions"
print("[window=4] only token 4 and the next 4 positions changed")

print("\nAll checks passed.")

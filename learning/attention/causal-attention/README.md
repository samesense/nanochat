# Causal Attention

This folder implements the decoder version of scaled dot-product self-attention:

```python
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
scores = scores.masked_fill(future_positions, -inf)
weights = softmax(scores, dim=-1)
out = weights @ v
```

The causal mask is the only conceptual difference from vanilla attention. It
prevents token `i` from attending to tokens after `i`, which prevents leakage
when training or generating autoregressive language models.

Run:

```bash
python demo.py
```

The demo checks three things: shapes, that future-token weights are exactly zero
after the softmax, and that editing the last token never changes an earlier
output (causality).

## Boundaries: what this deliberately leaves out

This is the minimal teaching version. To keep the core legible it omits several
things a production decoder needs — each handled one level up in the nanochat
extraction (`../attention.py`):

- **Positional information.** Causal attention is still permutation-equivariant
  *within* the allowed set — the mask imposes ordering but not distance. A real
  decoder adds positional encoding (RoPE in nanochat) so the model knows *how
  far* apart two tokens are, not just which came first.
- **Memory.** This materializes the full `(T × T)` score matrix, which is `O(T²)`
  in memory and OOMs on long sequences. Flash Attention / online-softmax compute
  the same result without ever storing that matrix.
- **No attention dropout**, and the three Q/K/V projections could be fused into a
  single `c_attn` linear — both standard in real implementations.
- **The mask is rebuilt every forward.** Cheap here, but it should be a cached
  `register_buffer`, or you can skip it entirely with
  `F.scaled_dot_product_attention(..., is_causal=True)`.

One thing it gets *right* and is worth saying out loud: the diagonal is always
unmasked, so no softmax row is entirely `-inf` → the output is always
well-defined (no NaNs). That edge case is a classic source of bugs.

## Follow-up questions an interviewer might ask

Use these to pressure-test your understanding of the code above:

1. **Why divide by `sqrt(head_dim)` and not `sqrt(n_embd)`?** What happens to the
   softmax distribution if you drop the scaling entirely as `head_dim` grows?
2. **Why mask with `-inf` before the softmax** instead of zeroing the weights
   after it? (Hint: post-softmax zeroing breaks the normalization.)
3. **What guarantees no row of the score matrix is fully masked,** and why would
   that matter numerically?
4. **This has no positional encoding — so what does the causal mask alone tell
   the model about order?** Where does the rest of the positional signal come
   from?
5. **How would you make this not materialize the `(T × T)` matrix** for a
   100k-token sequence? Sketch the online-softmax / Flash-Attention idea.
6. **At generation time you decode one token at a time — what's wasteful about
   re-running this `forward` each step,** and how does a KV cache fix it?
7. **How does this change for Grouped-Query Attention** (fewer K/V heads than Q
   heads), and what specifically does that buy you at inference?


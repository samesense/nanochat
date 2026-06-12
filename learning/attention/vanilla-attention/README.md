# Vanilla Attention

This folder implements plain scaled dot-product self-attention. Scaled dot-product self-attention is the mechanism at the heart of a transformer. For each token, it creates three vectors: a query (what am I looking for?), a key (what do I offer as a match?), and a value (what do I contribute if matched?). The query is compared to the keys of all tokens by taking a dot product. These dot products give scores that say how much each query token should attend to key tokens: does this key match my query? Those scores are scaled by dividing by the square root of the dimension to keep them stable, then passed through a softmax to form weights. Finally, these weights are used to blend the values, producing a weighted sum that becomes the new representation of that token. Since it’s self-attention, each token attends to the full set, including itself, in that same sequence.

Why divide by the square root of the dimension specifically? A dot product of two `d`-dimensional vectors with unit-variance components has variance roughly `d`, so the raw scores grow like `√d` as the head dimension grows. Dividing by `√d` rescales the logits back to variance ≈ 1. Without it, large logits push the softmax into a saturated, near one-hot region where its gradients vanish — which stalls training. Keeping the logits well-scaled (and the attention weights correspondingly less peaky) is what keeps gradients healthy.

```python
scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(scores, dim=-1)
out = weights @ v
```

There is no mask. Every token can attend to every other token, including future
tokens. That is useful for encoders and for learning the core attention formula,
but it is not valid for autoregressive language modeling by itself.

Run:

```bash
python demo.py
```


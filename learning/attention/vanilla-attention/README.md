# Vanilla Attention

This folder implements plain scaled dot-product self-attention:

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


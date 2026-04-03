# Attention From Scratch — Implementation Plan

## Background

Attention lets each token look at every other token and decide what's relevant. Given a sequence of token vectors, each token produces a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what do I offer?").

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

The dot product Q·K^T measures similarity between tokens. Dividing by √d_k prevents the dot products from getting too large (which would make softmax saturate). The result is a weighted sum of Values.

## Steps — `attention.py`

### 1. Q, K, V projections
- Three linear layers: `W_Q`, `W_K`, `W_V`, each `(dim, dim)`
- Input `x: (batch, seq_len, dim)` → three tensors of same shape
- These are learned — the model figures out what to put in Q vs K vs V

### 2. Scaled dot-product attention
- `scores = Q @ K.transpose(-2, -1) / sqrt(d_k)`
- `scores` shape: `(batch, seq_len, seq_len)` — every token's similarity to every other
- `weights = softmax(scores, dim=-1)` — normalize to probabilities
- `output = weights @ V` — weighted sum of values

### 3. Causal mask (for autoregressive / decoder models)
- Lower-triangular mask: token at position i can only attend to positions 0..i
- Set future positions to -inf before softmax so they become 0
- Without this: model can cheat by looking at the answer

### 4. `SingleHeadAttention(nn.Module)`
- Wraps steps 1-3 into a module
- `__init__(self, dim: int)` — creates W_Q, W_K, W_V
- `forward(self, x, causal: bool = True)` — full attention pass

### 5. Verification
- Check output shape matches input shape
- Check causal mask: attention weights should be zero above the diagonal
- Check attention weights sum to 1 across each row (valid probability distribution)

# Transformer Block From Scratch — Implementation Plan

## Goal

Build a minimal pre-norm transformer block on top of the attention work you
already have:
- multi-head self-attention
- RoPE
- causal masking
- residual connections
- feedforward network
- layer normalization

The target is a small standalone module that takes:

```python
x: (batch, seq_len, dim)
```

and returns:

```python
y: (batch, seq_len, dim)
```

## Step 1: Create a standalone `transformers` mini-project

Files:
- `transformers/PLAN.md`
- `transformers/pyproject.toml`
- `transformers/transformer.py`

Dependencies:
- `torch`
- optionally `numpy` if needed for clean local runtime behavior

Reason:
- keep `transformers` separate from `attention`, `clip`, and the other exercises
- match the structure you now have in `gelu`, `swiglu`, and `attention`

## Step 2: Reuse or inline the minimal attention pieces

The transformer block needs:
- multi-head attention
- causal masking
- optional RoPE

Decision:
- either copy the minimal attention code into `transformer.py`
- or import from `attention` only if you want an intentional dependency

Recommended for this repo:
- keep `transformers` self-contained
- inline the minimal `MultiHeadAttention` implementation

## Step 3: Implement the feedforward network

Start with the standard transformer MLP:

```python
Linear(dim, 4 * dim) -> GELU -> Linear(4 * dim, dim)
```

This gives the second half of the transformer block after attention.

Keep it minimal:
- no dropout yet
- no SwiGLU yet
- just the canonical FFN

## Step 4: Implement pre-norm `TransformerBlock`

Structure:

```python
x = x + attn(ln1(x))
x = x + ffn(ln2(x))
```

Components:
- `ln1 = LayerNorm(dim)`
- `attn = MultiHeadAttention(...)`
- `ln2 = LayerNorm(dim)`
- `ffn = FeedForward(dim)`

Why pre-norm:
- simpler and stable
- matches modern transformer practice
- easier to reason about during learning

## Step 5: Support causal mode

Expose:

```python
forward(self, x, causal: bool = False)
```

and pass `causal` into attention.

This lets the same block work for:
- bidirectional encoder-style experiments
- autoregressive decoder-style experiments

## Step 6: Add a tiny runnable example

In `__main__`, create:
- a small toy input tensor
- a `TransformerBlock(dim=..., num_heads=...)`
- a forward pass
- printed shapes

Example checks:
- input shape
- output shape
- confirm output shape matches input shape

## Step 7: Add basic correctness checks

Minimum sanity checks:
- `dim % num_heads == 0`
- output shape equals input shape
- causal path runs without error
- RoPE path runs without error

If you want lightweight tests later, add:
- `transformers/test_transformer.py`

## Step 8: Optional follow-ups after the minimal block works

Natural next extensions:
- attention weight return path
- padding mask support
- dropout
- SwiGLU FFN variant
- tiny transformer stack with multiple blocks
- token embedding + final LM head

# GELU From Scratch — Implementation Plan

## Background

GELU (Gaussian Error Linear Unit) soft-gates inputs by their percentile under a standard normal distribution, unlike ReLU which hard-gates at zero.

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the CDF of the standard normal. The tanh approximation:

```
GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
```

## Steps — `gelu.py`

### 1. Swish / SiLU helper
- `swish(x) = x · σ(x)` where σ is sigmoid
- Useful comparison point and building block
- Implement as a standalone function

### 2. `gelu_exact(x)`
- Uses the error function: `GELU(x) = 0.5x · (1 + erf(x / √2))`
- `torch.erf` gives erf directly
- This is the mathematically precise version

### 3. `gelu_tanh_approx(x)`
- `0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))`
- The fast approximation used in practice (GPT-2, early BERT)
- Constants: `√(2/π) ≈ 0.7978845608`

### 4. `GELU(nn.Module)`
- Wraps the above into a proper PyTorch module
- `__init__(self, approximate: bool = False)` — selects exact vs tanh
- `forward(self, x)` — dispatches to the right function

### 5. Verification
- Compare output of our `gelu_exact` vs `torch.nn.functional.gelu` — should match
- Compare `gelu_tanh_approx` vs `F.gelu(x, approximate="tanh")` — should match
- Simple forward pass: `nn.Linear → GELU → nn.Linear` on random input

## Steps — `notebook.py` (marimo)

### 1. Formula and intuition
- Markdown cell explaining GELU and why it matters

### 2. Plot GELU vs ReLU vs Sigmoid
- Side-by-side to show GELU's smooth, slightly negative region

### 3. The gating intuition
- Plot x, Φ(x), and x·Φ(x) separately
- Shows how the normal CDF acts as a soft gate

### 4. Exact vs tanh approximation
- Overlay both curves + plot the absolute difference
- Shows the approximation is very tight (max error ~0.004)

### 5. Derivatives
- Plot GELU'(x) vs ReLU'(x)
- Key insight: smooth gradient everywhere, no dead neurons

### 6. Forward pass demo
- Tiny `Linear → GELU → Linear` network
- Pass random data, show output distribution

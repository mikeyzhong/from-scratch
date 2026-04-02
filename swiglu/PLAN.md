# SwiGLU From Scratch — Implementation Plan

## Background

SwiGLU is a gated FFN variant introduced by Noam Shazeer (2020). It replaces the standard transformer FFN block (`Linear → GELU → Linear`) with a gated design using Swish as the activation:

```
SwiGLU(x, W, V, b, c) = (Swish(xW + b)) ⊙ (xV + c)
```

The full FFN becomes: `SwiGLU(x, W, V) · W₂`

This is what LLaMA, PaLM, Mistral, and Gemma use. It consistently outperforms GELU FFNs at the same compute budget.

### Why three matrices instead of two?

A standard FFN has two matrices: up-project (d → 4d) and down-project (4d → d). SwiGLU needs three: W (gate projection), V (value projection), and W₂ (down projection). To keep parameter count comparable, the hidden dim is typically scaled to `(2/3) · 4d` ≈ `8d/3`, often rounded to a multiple of 256.

## Steps — `swiglu.py`

### 1. `swish(x)` activation
- `swish(x) = x · σ(x)` where σ is sigmoid
- Import from `gelu.swish` or reimplement (it's one line)
- This is the element-wise activation used inside the gate

### 2. `SwiGLU(nn.Module)` — the gated layer
- `__init__(self, dim: int, hidden_dim: int, bias: bool = False)`
  - `self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)` — gate projection
  - `self.w_up = nn.Linear(dim, hidden_dim, bias=bias)` — value/up projection
  - `self.w_down = nn.Linear(hidden_dim, dim, bias=bias)` — down projection
- `forward(self, x)`
  - `gate = swish(self.w_gate(x))`
  - `up = self.w_up(x)`
  - `return self.w_down(gate * up)`
- Note: most LLaMA-style implementations omit bias (bias=False)

### 3. `SwiGLUFFN(nn.Module)` — drop-in transformer FFN replacement
- Wraps SwiGLU with the `(2/3) · 4d` hidden dim scaling
- `__init__(self, dim: int, multiple_of: int = 256)`
  - Compute `hidden_dim = int(2 * (4 * dim) / 3)`
  - Round up to nearest `multiple_of`
- Shows how it's sized in practice (LLaMA convention)

### 4. Standard FFN for comparison
- `StandardFFN(nn.Module)`: `Linear(d, 4d) → GELU → Linear(4d, d)`
- Used in the notebook to compare against SwiGLU

### 5. Verification
- Forward pass with random input: check output shape matches input shape
- Compare parameter counts: SwiGLU (3 matrices at 8d/3) vs standard FFN (2 matrices at 4d)
- Gradient flow: backward pass should work without errors

## Steps — `notebook.py` (marimo)

### 1. What is SwiGLU?
- Markdown explaining the motivation and formula
- How it relates to GELU (both are smooth activations, but SwiGLU adds learned gating)

### 2. Visualize the gating mechanism
- For a fixed input x, show the gate values `swish(xW)` and up values `xV` side by side
- Then show the element-wise product — illustrates which features get passed vs suppressed

### 3. Parameter count comparison
- Table/bar chart: SwiGLU vs standard GELU FFN at various model dims (512, 1024, 4096)
- Show they're roughly comparable when using the 2/3 scaling

### 4. Output distribution comparison
- Pass same random input through both SwiGLU FFN and standard GELU FFN
- Histogram of output values — shows SwiGLU tends to produce sharper distributions

### 5. Gradient norms
- Compare gradient norms flowing back through SwiGLU vs standard FFN
- Key insight: gating helps with gradient flow

### 6. Drop-in replacement demo
- Tiny 2-layer transformer-style stack: `LayerNorm → FFN → residual`
- Swap between SwiGLU and standard FFN, show both work identically as drop-in replacements

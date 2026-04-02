import torch
import torch.nn as nn


# Step 1: Swish / SiLU — the activation used inside the SwiGLU gate
# swish(x) = x · σ(x)
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# Step 2: SwiGLU — one extra learned gate on top of a standard FFN
# SwiGLU(x) = swish(x @ W_gate) ⊙ (x @ W_up), then projected down by W_down
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(swish(self.w_gate(x)) * self.w_up(x))


# Step 3: Drop-in transformer FFN with LLaMA-style hidden dim scaling
# 3 matrices at (2/3)*4d ≈ same param count as 2 matrices at 4d
class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 256, bias: bool = False):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.swiglu = SwiGLU(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)

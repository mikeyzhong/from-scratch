import math

import torch
import torch.nn as nn


# Step 1: Swish / SiLU — uses sigmoid as a soft gate
# swish(x) = x · σ(x)
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# Step 2: Exact GELU using the error function
# GELU(x) = x · Φ(x) = 0.5x · (1 + erf(x / √2))
def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Step 3: Tanh approximation (used in GPT-2, early BERT)
# GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


# Step 4: nn.Module wrapper
class GELU(nn.Module):
    def __init__(self, approximate: bool = False):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate:
            return gelu_tanh_approx(x)
        return gelu_exact(x)

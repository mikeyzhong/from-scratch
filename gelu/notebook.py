import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import torch
    import matplotlib.pyplot as plt
    from gelu import swish, gelu_exact, gelu_tanh_approx, GELU

    return GELU, gelu_exact, gelu_tanh_approx, math, mo, plt, swish, torch


@app.cell
def _(mo):
    mo.md(r"""
    # GELU From Scratch

    **GELU** (Gaussian Error Linear Unit) soft-gates inputs by their percentile
    under a standard normal distribution:

    $$\text{GELU}(x) = x \cdot \Phi(x) = 0.5x \cdot (1 + \text{erf}(x / \sqrt{2}))$$

    where $\Phi(x)$ is the CDF of the standard normal. Unlike ReLU's hard cutoff at zero,
    GELU smoothly interpolates between suppressing and passing inputs.
    """)
    return


@app.cell
def _(gelu_exact, plt, torch):
    _x = torch.linspace(-4, 4, 1000)
    _gelu = gelu_exact(_x)
    _relu = torch.relu(_x)
    _sigmoid = torch.sigmoid(_x)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_x, _gelu, linewidth=2, label="GELU")
    _ax.plot(_x, _relu, linewidth=2, label="ReLU", linestyle="--", alpha=0.7)
    _ax.plot(_x, _sigmoid, linewidth=2, label="Sigmoid", linestyle="--", alpha=0.7)
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.axvline(0, color="gray", linewidth=0.5)
    _ax.set_xlabel("x")
    _ax.set_ylabel("activation(x)")
    _ax.set_title("GELU vs ReLU vs Sigmoid")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.set_ylim(-0.5, 3.0)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Gating Intuition

    GELU = $x \cdot \Phi(x)$. The normal CDF $\Phi(x)$ acts as a soft gate:
    inputs likely to be positive pass through, inputs likely to be negative get suppressed.
    """)
    return


@app.cell
def _(gelu_exact, math, plt, torch):
    _x = torch.linspace(-4, 4, 1000)
    _phi = 0.5 * (1.0 + torch.erf(_x / math.sqrt(2.0)))

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    _axes[0].plot(_x, _x, linewidth=2, color="steelblue")
    _axes[0].set_title("x (input)")
    _axes[0].grid(True, alpha=0.3)
    _axes[0].axhline(0, color="gray", linewidth=0.5)

    _axes[1].plot(_x, _phi, linewidth=2, color="orange")
    _axes[1].set_title("Φ(x) (gate)")
    _axes[1].set_ylim(-0.1, 1.1)
    _axes[1].grid(True, alpha=0.3)

    _axes[2].plot(_x, gelu_exact(_x), linewidth=2, color="green")
    _axes[2].set_title("x · Φ(x) = GELU(x)")
    _axes[2].grid(True, alpha=0.3)
    _axes[2].axhline(0, color="gray", linewidth=0.5)

    for _ax in _axes:
        _ax.set_xlabel("x")
        _ax.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exact vs Tanh Approximation

    The tanh approximation $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$
    was used in GPT-2 because `erf` wasn't well-optimized on GPUs at the time.
    """)
    return


@app.cell
def _(gelu_exact, gelu_tanh_approx, plt, torch):
    _x = torch.linspace(-4, 4, 1000)
    _exact = gelu_exact(_x)
    _approx = gelu_tanh_approx(_x)
    _diff = (_exact - _approx).abs()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))

    _axes[0].plot(_x, _exact, linewidth=2, label="Exact (erf)")
    _axes[0].plot(_x, _approx, linewidth=2, linestyle="--", label="Tanh approx")
    _axes[0].set_title("Both GELU variants")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    _axes[1].plot(_x, _diff, linewidth=2, color="red")
    _axes[1].set_title(f"Absolute difference (max={_diff.max():.6f})")
    _axes[1].set_ylabel("|exact - approx|")
    _axes[1].grid(True, alpha=0.3)

    for _ax in _axes:
        _ax.set_xlabel("x")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Derivatives

    GELU has a smooth derivative everywhere — no hard kink like ReLU at 0,
    and no dead neurons (derivative is never permanently 0 for negative inputs).
    """)
    return


@app.cell
def _(gelu_exact, plt, torch):
    _x = torch.linspace(-4, 4, 1000)
    _eps = 1e-5
    _d_gelu = (gelu_exact(_x + _eps) - gelu_exact(_x - _eps)) / (2 * _eps)
    _d_relu = (torch.relu(_x + _eps) - torch.relu(_x - _eps)) / (2 * _eps)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(_x, _d_gelu, linewidth=2, label="GELU'(x)")
    _ax.plot(_x, _d_relu, linewidth=2, linestyle="--", alpha=0.7, label="ReLU'(x)")
    _ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="ideal = 1.0")
    _ax.set_xlabel("x")
    _ax.set_ylabel("derivative")
    _ax.set_title("Activation Derivatives")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.set_ylim(-0.3, 1.5)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Forward Pass Demo

    A tiny `Linear → GELU → Linear` network on random input,
    showing how GELU reshapes the output distribution.
    """)
    return


@app.cell
def _(GELU, plt, torch):
    torch.manual_seed(42)
    _net = torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        GELU(),
        torch.nn.Linear(32, 8),
    )
    _input = torch.randn(256, 8)
    _output = _net(_input)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(_input.detach().flatten().numpy(), bins=50, alpha=0.7, edgecolor="black")
    _axes[0].set_title("Input distribution")
    _axes[0].set_xlabel("value")

    _axes[1].hist(_output.detach().flatten().numpy(), bins=50, alpha=0.7, edgecolor="black", color="orange")
    _axes[1].set_title("Output distribution (after Linear → GELU → Linear)")
    _axes[1].set_xlabel("value")

    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()

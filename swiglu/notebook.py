import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    from swiglu import swish, SwiGLU, SwiGLUFFN

    return F, SwiGLU, SwiGLUFFN, mo, nn, plt, swish, torch


@app.cell
def _(mo):
    mo.md(r"""
    # SwiGLU From Scratch

    **SwiGLU** replaces the standard transformer FFN with a gated design:

    $$\text{Standard: } \text{GELU}(xW_{\text{up}}) \cdot W_{\text{down}}$$

    $$\text{SwiGLU: } (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}) \cdot W_{\text{down}}$$

    The key addition is $W_{\text{gate}}$ — a learned projection that decides
    **which features to keep**, independently from what those features are.
    Used in LLaMA, PaLM, Mistral, and most modern LLMs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Gating Mechanism

    For a fixed input, we can see how the gate path and value path produce
    independent vectors, and how their element-wise product selectively
    suppresses features.
    """)
    return


@app.cell
def _(SwiGLU, plt, swish, torch):
    torch.manual_seed(7)
    _dim, _hidden = 8, 12
    _layer = SwiGLU(_dim, _hidden)
    _x = torch.randn(1, _dim)

    _gate_raw = _layer.w_gate(_x).detach().squeeze()
    _gate_act = swish(_gate_raw)
    _value = _layer.w_up(_x).detach().squeeze()
    _product = _gate_act * _value

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))

    _axes[0].bar(range(_hidden), _gate_raw.numpy(), color="#1f6feb", alpha=0.8)
    _axes[0].set_title("x·W_gate (raw)")
    _axes[0].set_ylabel("value")

    _axes[1].bar(range(_hidden), _gate_act.numpy(), color="#58a6ff", alpha=0.8)
    _axes[1].set_title("swish(x·W_gate)")
    _axes[1].set_ylabel("gate strength")

    _axes[2].bar(range(_hidden), _value.detach().numpy(), color="#d29922", alpha=0.8)
    _axes[2].set_title("x·W_up (features)")

    _colors = ["#3fb950" if abs(g) > 0.15 else "#f8514966" for g in _gate_act]
    _axes[3].bar(range(_hidden), _product.numpy(), color=_colors, alpha=0.8)
    _axes[3].set_title("gate × value (output)")

    for _ax in _axes:
        _ax.set_xlabel("hidden dim")
        _ax.grid(True, alpha=0.2)
        _ax.axhline(0, color="gray", linewidth=0.5)

    _fig.suptitle("SwiGLU gating: gate decides which features survive", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Parameter Count Comparison

    SwiGLU has 3 matrices instead of 2. To match the standard FFN's parameter
    count, the hidden dim is scaled to $\frac{2}{3} \times 4d$, rounded to a
    multiple of 256.
    """)
    return


@app.cell
def _(nn, plt, SwiGLUFFN):
    _dims = [256, 512, 1024, 2048, 4096]

    _std_params = []
    _swiglu_params = []
    for _d in _dims:
        _std = nn.Sequential(nn.Linear(_d, 4 * _d, bias=False), nn.GELU(), nn.Linear(4 * _d, _d, bias=False))
        _swi = SwiGLUFFN(_d)
        _std_params.append(sum(p.numel() for p in _std.parameters()))
        _swiglu_params.append(sum(p.numel() for p in _swi.parameters()))

    _x_pos = range(len(_dims))
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _bar_w = 0.35
    _bars1 = _ax.bar([x - _bar_w / 2 for x in _x_pos], [p / 1e6 for p in _std_params],
                     _bar_w, label="Standard FFN (2 matrices @ 4d)", color="#d29922", alpha=0.8)
    _bars2 = _ax.bar([x + _bar_w / 2 for x in _x_pos], [p / 1e6 for p in _swiglu_params],
                     _bar_w, label="SwiGLU FFN (3 matrices @ ⅔·4d)", color="#3fb950", alpha=0.8)

    _ax.set_xticks(list(_x_pos))
    _ax.set_xticklabels([str(d) for d in _dims])
    _ax.set_xlabel("Model dimension (d)")
    _ax.set_ylabel("Parameters (millions)")
    _ax.set_title("Parameter count: Standard FFN vs SwiGLU FFN")
    _ax.legend()
    _ax.grid(True, alpha=0.2, axis="y")

    for _b1, _b2 in zip(_bars1, _bars2):
        _ratio = _b2.get_height() / _b1.get_height() * 100
        _ax.text(_b2.get_x() + _b2.get_width() / 2, _b2.get_height(),
                 f"{_ratio:.0f}%", ha="center", va="bottom", fontsize=8, color="#8b949e")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Output Distribution Comparison

    Same random input through both FFN types — SwiGLU's gating
    tends to produce a sharper, more selective output distribution.
    """)
    return


@app.cell
def _(nn, plt, SwiGLUFFN, torch):
    torch.manual_seed(42)
    _d = 256
    _std_ffn = nn.Sequential(nn.Linear(_d, 4 * _d, bias=False), nn.GELU(), nn.Linear(4 * _d, _d, bias=False))
    _swi_ffn = SwiGLUFFN(_d)
    _x = torch.randn(512, _d)

    _std_out = _std_ffn(_x).detach().flatten().numpy()
    _swi_out = _swi_ffn(_x).detach().flatten().numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(_std_out, bins=80, alpha=0.7, edgecolor="black", color="#d29922")
    _axes[0].set_title("Standard FFN output")
    _axes[0].set_xlabel("value")
    _axes[0].set_ylabel("count")

    _axes[1].hist(_swi_out, bins=80, alpha=0.7, edgecolor="black", color="#3fb950")
    _axes[1].set_title("SwiGLU FFN output")
    _axes[1].set_xlabel("value")

    for _ax in _axes:
        _ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gradient Norms

    Comparing gradient magnitudes flowing back through each FFN type.
    Healthier gradients = more stable training.
    """)
    return


@app.cell
def _(nn, plt, SwiGLUFFN, torch):
    torch.manual_seed(0)
    _d = 256
    _n_trials = 50

    _std_norms = []
    _swi_norms = []
    for _ in range(_n_trials):
        _x = torch.randn(32, _d, requires_grad=True)

        _std_ffn = nn.Sequential(nn.Linear(_d, 4 * _d, bias=False), nn.GELU(), nn.Linear(4 * _d, _d, bias=False))
        _std_out = _std_ffn(_x)
        _std_out.sum().backward()
        _std_norms.append(_x.grad.norm().item())

        _x2 = torch.randn(32, _d, requires_grad=True)
        _swi_ffn = SwiGLUFFN(_d)
        _swi_out = _swi_ffn(_x2)
        _swi_out.sum().backward()
        _swi_norms.append(_x2.grad.norm().item())

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_std_norms, label="Standard FFN", color="#d29922", alpha=0.8, linewidth=1.5)
    _ax.plot(_swi_norms, label="SwiGLU FFN", color="#3fb950", alpha=0.8, linewidth=1.5)
    _ax.set_xlabel("Trial")
    _ax.set_ylabel("Input gradient norm")
    _ax.set_title("Gradient norms: Standard FFN vs SwiGLU")
    _ax.legend()
    _ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Drop-in Replacement Demo

    Both FFN types work as drop-in replacements in a transformer-style
    block: `LayerNorm → FFN → residual`. Same interface, same shapes.
    """)
    return


@app.cell
def _(nn, plt, SwiGLUFFN, torch):
    torch.manual_seed(99)
    _d = 128

    class _TransformerBlock(nn.Module):
        def __init__(self, ffn):
            super().__init__()
            self.norm = nn.LayerNorm(_d)
            self.ffn = ffn

        def forward(self, x):
            return x + self.ffn(self.norm(x))

    _std_block = _TransformerBlock(
        nn.Sequential(nn.Linear(_d, 4 * _d, bias=False), nn.GELU(), nn.Linear(4 * _d, _d, bias=False))
    )
    _swi_block = _TransformerBlock(SwiGLUFFN(_d))

    _x = torch.randn(64, 16, _d)

    _std_out = _std_block(_x)
    _swi_out = _swi_block(_x)

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    _axes[0].hist(_x.detach().flatten().numpy(), bins=60, alpha=0.7, edgecolor="black", color="#8b949e")
    _axes[0].set_title(f"Input {list(_x.shape)}")

    _axes[1].hist(_std_out.detach().flatten().numpy(), bins=60, alpha=0.7, edgecolor="black", color="#d29922")
    _axes[1].set_title(f"Standard FFN output {list(_std_out.shape)}")

    _axes[2].hist(_swi_out.detach().flatten().numpy(), bins=60, alpha=0.7, edgecolor="black", color="#3fb950")
    _axes[2].set_title(f"SwiGLU FFN output {list(_swi_out.shape)}")

    for _ax in _axes:
        _ax.set_xlabel("value")
        _ax.grid(True, alpha=0.2)

    _fig.suptitle("Drop-in replacement: same input shape in, same output shape out", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()

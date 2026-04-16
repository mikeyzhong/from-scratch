# ResNet From Scratch

## Background

ResNet (He et al., 2015) introduced **skip connections** that let gradients
flow directly through the network, enabling training of very deep networks
(100+ layers) where previous architectures failed due to vanishing gradients.

The key insight -- the **residual mapping**:

    y = F(x) + x

Instead of learning H(x) directly, the layers learn the residual
F(x) = H(x) - x. If the optimal mapping is close to identity, it is easier
for the network to push F(x) toward zero than to learn an identity mapping
from scratch.

Two block types handle different depth regimes:

- **BasicBlock**: two 3x3 convs (ResNet-18, 34)
- **Bottleneck**: 1x1 -> 3x3 -> 1x1 convs, reducing computation via
  channel compression (ResNet-50, 101, 152)

## Architecture Variants

| Model      | Block      | Layers per stage | Total layers | Params |
|------------|------------|------------------|--------------|--------|
| ResNet-18  | BasicBlock | [2, 2, 2, 2]    | 18           | 11.7M  |
| ResNet-34  | BasicBlock | [3, 4, 6, 3]    | 34           | 21.8M  |
| ResNet-50  | Bottleneck | [3, 4, 6, 3]    | 50           | 25.6M  |
| ResNet-101 | Bottleneck | [3, 4, 23, 3]   | 101          | 44.5M  |
| ResNet-152 | Bottleneck | [3, 8, 36, 3]   | 152          | 60.2M  |

## Implementation Steps

1. **Conv helpers** -- `conv3x3` and `conv1x1` with `bias=False`
   (redundant when followed by BatchNorm)

2. **BasicBlock** -- two 3x3 convs with BN, ReLU, and identity shortcut.
   Optional downsample when spatial dims or channels change.

3. **Bottleneck** -- 1x1 (reduce) -> 3x3 -> 1x1 (expand) with
   `expansion = 4`. More parameter-efficient than two full 3x3 convs.

4. **ResNet class** -- stem (7x7 conv stride 2 + maxpool) followed by
   4 stages built via `_make_layer()`. Global avg pool -> FC classifier.
   Kaiming/He initialization throughout.

5. **Factory functions** -- `resnet18()` through `resnet152()`

6. **Verification** -- shape checks, parameter counts, gradient flow

## Verification

```bash
uv run python resnet.py
```

Expected: all 5 variants produce correct output shapes, parameter counts
match known values, and gradients flow to the first conv layer.

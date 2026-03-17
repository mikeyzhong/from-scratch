# ViT-Mini — Implementation Plan

## Config
- Input: 32×32×3 (CIFAR-10)
- Patch size: 4×4 → 8×8 grid = 64 patches
- Embed dim: 192
- Depth: 4 transformer blocks
- Heads: 3 (64 dim per head)
- MLP ratio: 4× (192 → 768 → 192)

## Steps

### 1. PatchEmbedding
- `Conv2d(3, 192, kernel_size=4, stride=4)` — projects each 4×4 patch to 192-d
- Reshape from (B, 192, 8, 8) → (B, 64, 192)

### 2. MultiHeadSelfAttention
- Three linear projections: Q, K, V each (192 → 192)
- Split into 3 heads (64 dim per head)
- Scaled dot-product: softmax(QK^T / sqrt(64)) · V
- Output projection: Linear(192 → 192)

### 3. TransformerBlock (pre-norm)
- LayerNorm → MHSA → residual add
- LayerNorm → MLP → residual add
- MLP: Linear(192 → 768) → GELU → Linear(768 → 192)

### 4. ViT assembly
- PatchEmbedding
- Prepend learnable CLS token: `Parameter(zeros(1, 1, 192))`
- Add learnable positional embeddings: `Parameter(zeros(1, 65, 192))` (64 patches + 1 CLS)
- 4× TransformerBlock
- Final LayerNorm
- Output: CLS token (B, 192), patch tokens (B, 64, 192)

### 5. Sanity check
- Add `Linear(192, 10)` on CLS token, train on CIFAR-10
- Should hit ~85% in ~50 epochs

# Toy DINOv2 — Implementation Plan

## Dataset
- **CIFAR-10**: 60k 32×32 images, instant download, good for quick iteration
- **STL-10** (stretch): 96×96, includes 100k unlabeled images designed for self-supervised learning

## Steps

### 1. ViT backbone
- Patch embedding (4×4 patches for CIFAR-10, 8×8 for STL-10)
- CLS token + learnable positional embeddings
- 4-6 transformer blocks, dim 192-384, 6 heads
- Output: CLS token embedding

### 2. Student-teacher framework (EMA)
- Two copies of the ViT: student (gradient-trained) and teacher (EMA of student)
- EMA schedule: start ~0.996, cosine anneal toward 1.0

### 3. DINO projection head
- MLP: hidden → 2048 → 256 → 65536 prototypes
- L2-normalize before the last linear layer

### 4. Multi-crop augmentation
- 2 global crops (50-100% of image) → both student and teacher
- Several local crops (20-50%) → student only
- Augmentations: color jitter, gaussian blur, horizontal flip, solarization

### 5. Self-distillation loss
- Teacher: center + sharpen with low temperature (0.04)
- Student: sharpen with higher temperature (0.1)
- Loss = cross-entropy(teacher_softmax, student_softmax) over all cross-view pairs
- Update centering vector via EMA of teacher outputs (prevents collapse)

### 6. Training loop
- AdamW, cosine LR schedule with warmup
- ~100-300 epochs on CIFAR-10
- Batch size 256-512

### 7. Evaluation
- Linear probe: freeze backbone, train linear classifier on CLS token
- k-NN (k=20) on feature space for quick sanity checks

## DINOv2 additions (layer on after basic DINO works)
- **iBOT**: masked patch prediction loss — predict masked patch tokens via the teacher
- **KoLeo regularizer**: encourages uniform feature distribution in the batch

## Implementation order
1. ViT forward pass → verify shapes
2. Multi-crop augmentation pipeline
3. Student-teacher with EMA updates
4. DINO loss with centering
5. Train + evaluate with k-NN
6. Add iBOT masked patch loss (DINOv2)
7. Add KoLeo regularizer (DINOv2)

## Key gotchas
- **Collapse**: centering + sharpening in the teacher is critical — without it all outputs converge
- Start without iBOT, get self-distillation working first, then add masked image modeling

# Toy 3D Gaussian Splatting — Implementation Plan

## Background

3D Gaussian Splatting represents a scene as a cloud of anisotropic 3D Gaussians, each with a position, covariance (rotation + scale), opacity, and color. To render, we project each Gaussian into 2D image space and alpha-blend them front-to-back. The whole pipeline is differentiable, so we can fit the Gaussians to a set of posed images by gradient descent.

The trick that makes this fast and high-quality:
- **Explicit primitives** (no MLP forward pass per ray, unlike NeRF)
- **Splatting** (rasterize, don't ray-march)
- **Anisotropic Gaussians** parameterized as `Σ = R S S^T R^T` so we optimize a valid covariance

For a *toy* version we drop the CUDA tile-rasterizer and densification heuristics, and just write a slow but correct PyTorch pipeline that fits a small scene.

## Scope of the toy

- Single scene fit (overfit), not generalization
- ~10k–50k Gaussians (no adaptive density control needed at this size)
- Pure PyTorch — no custom CUDA kernel
- Render at low resolution (e.g. 200×200) so a per-pixel loop / broadcasted sort is tolerable
- Color = per-Gaussian RGB (skip spherical harmonics in v1, add later)

## Dataset

- **NeRF synthetic "lego"** or **"chair"** — small, posed, clean backgrounds, comes with `transforms_*.json` (camera intrinsics + c2w extrinsics)
- 100 train images, 200 test images, white background
- Alternatively a tiny synthetic scene we render ourselves (3 cubes) for the very first sanity check

## Steps — `gaussians.py`, `render.py`, `train.py`

### 1. Gaussian parameters
A `GaussianModel` holding learnable tensors:
- `means`: `(N, 3)` — xyz positions
- `scales_raw`: `(N, 3)` — log-scales (exp to get positive scale)
- `quats_raw`: `(N, 4)` — unnormalized quaternion (normalize before use)
- `opacities_raw`: `(N, 1)` — pre-sigmoid logits
- `colors`: `(N, 3)` — RGB in [0, 1] (or pre-sigmoid logits)

Helpers:
- `get_scales()` → `exp(scales_raw)`
- `get_rotation()` → quaternion → 3×3 rotation matrix
- `get_opacity()` → `sigmoid(opacities_raw)`
- `get_covariance()` → `R diag(s)^2 R^T`

### 2. Initialization
- From a sparse point cloud (COLMAP output, or for synthetic data: random points in the scene bounding box)
- Initial scale = distance to nearest neighbor (so Gaussians cover gaps without huge overlap)
- Random unit quaternion, opacity ~0.1, color from the nearest input pixel (or random)

### 3. Camera + projection
- Pinhole camera: intrinsics `K` (focal, cx, cy), extrinsics world→camera `W = [R|t]`
- For each Gaussian, transform mean to camera space: `μ_cam = R_w μ + t_w`
- Project to image: `μ_pix = K μ_cam / μ_cam.z`
- Cull: drop Gaussians behind the camera or outside the frustum

### 4. Project the 3D covariance to 2D (the EWA splat)
- Build the Jacobian `J` of the perspective projection at `μ_cam`
- World→camera rotation `W` (3×3)
- 2D covariance: `Σ_2D = J W Σ W^T J^T`, then take the upper-left 2×2 block
- Add a small isotropic regularizer (e.g. `+ 0.3 * I`) for numerical stability — this is the "low-pass filter" trick from the paper

### 5. Per-pixel evaluation (the slow toy rasterizer)
For each Gaussian we now have:
- 2D mean `μ_2D` (pixel coords)
- 2D covariance `Σ_2D`
- opacity `α`, color `c`, depth `z`

Render:
1. Sort Gaussians by depth (front to back)
2. For each pixel `p` and each Gaussian, compute:
   - `d = p - μ_2D`
   - `power = -0.5 * d^T Σ_2D^{-1} d`
   - `weight = α * exp(power)`
3. Alpha-composite front-to-back:
   ```
   C = 0; T = 1
   for g in sorted:
       w = weight_g(p)
       C += T * w * c_g
       T *= (1 - w)
       if T < 1e-4: break
   ```
4. Add background color * T (white for the synthetic scenes)

Vectorize across pixels with broadcasting; the per-Gaussian loop is fine for a toy. Skip Gaussians whose 2D extent doesn't touch the pixel grid (3-sigma bbox cull).

### 6. Loss
- L1 between rendered image and ground truth
- + small SSIM term (the paper uses `0.8 * L1 + 0.2 * (1 - SSIM)`)
- Optional: opacity regularizer to encourage sparsity

### 7. Training loop — `train.py`
- AdamW with **per-parameter learning rates** (this matters a lot):
  - means: 1.6e-4 (with exponential decay over training)
  - scales: 5e-3
  - rotations: 1e-3
  - opacities: 5e-2
  - colors: 2.5e-3
- Iterate: pick a random training view → render → loss → backward → step
- ~7k–30k iterations for a toy scene

### 8. Evaluation
- Render held-out test views, compute PSNR
- Save rendered images side-by-side with ground truth
- Lego at 200×200 with ~30k Gaussians should reach ~25–28 PSNR

## Stretch goals (after the basic loop fits a scene)

1. **Adaptive density control**
   - Clone Gaussians with large positional gradients in under-reconstructed regions
   - Split large Gaussians (scale > threshold) into two
   - Prune Gaussians with opacity below a threshold
   - Run every N iterations

2. **Spherical harmonics for view-dependent color**
   - Replace per-Gaussian RGB with degree-3 SH coefficients (`(N, 16, 3)`)
   - Evaluate using the view direction in world space

3. **Tile-based rasterizer**
   - Bin Gaussians into 16×16 pixel tiles by 2D bbox
   - Per-tile sort once, render only relevant Gaussians per pixel
   - This is what makes the real implementation fast — even in pure PyTorch this is a big speedup

4. **Custom CUDA kernel** (out of scope for the toy, but the natural next step)

## Implementation order

1. `gaussians.py`: parameter container + activations (verify shapes, gradients flow)
2. Render a single Gaussian on a blank canvas — eyeball it, check the splat looks elliptical
3. Render 10 hand-placed Gaussians, verify alpha compositing order is correct
4. Hook up a real camera + project covariance — render N random Gaussians from a known viewpoint
5. Fit a single image (no multi-view) — should converge to near-perfect reconstruction
6. Multi-view fit on the lego scene → compute PSNR
7. Add SH colors
8. Add adaptive density control

## Key gotchas

- **Quaternion normalization**: do it inside the forward pass, not as a one-off — otherwise gradients pull it off the unit sphere
- **Covariance must stay PSD**: parameterize via `R S S^T R^T`, never optimize Σ directly
- **Depth sort is per-view**: don't cache it across iterations
- **Front-to-back vs back-to-front**: the formula above is front-to-back with transmittance `T`. Mixing this up gives a plausible-looking but wrong image
- **Background**: synthetic NeRF scenes are on white, real scenes on black — using the wrong one tanks the loss
- **Numerical issues**: add the `+ 0.3 I` to `Σ_2D` and clamp `det(Σ_2D)` away from zero before inverting
- **Learning rates**: the per-parameter LRs above are not optional — a single global LR will either explode the means or freeze the opacities

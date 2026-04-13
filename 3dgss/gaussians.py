import torch
import torch.nn as nn


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (N, 4) -> rotation matrices (N, 3, 3).

    Uses the Hamilton convention: q = (w, x, y, z).
    """
    q = q / q.norm(dim=-1, keepdim=True)  # normalize
    w, x, y, z = q.unbind(dim=-1)

    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    return R


class GaussianModel(nn.Module):
    """A cloud of N learnable 3D Gaussians.

    Each Gaussian has:
      - mean (position):   (N, 3)
      - scale (3 axis lengths): stored as log-scale, exp'd to stay positive
      - rotation (orientation): stored as raw quaternion, normalized in forward
      - opacity: stored as logit, sigmoid'd to [0, 1]
      - color (RGB): stored as logit, sigmoid'd to [0, 1]
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

        # Learnable raw parameters
        self.means = nn.Parameter(torch.randn(n, 3))
        self.scales_raw = nn.Parameter(torch.zeros(n, 3))          # log-space
        self.quats_raw = nn.Parameter(self._random_quaternions(n))  # unnormalized
        self.opacities_raw = nn.Parameter(torch.full((n, 1), -2.0))  # sigmoid(-2) ≈ 0.12
        self.colors_raw = nn.Parameter(torch.zeros(n, 3))          # sigmoid(0) = 0.5

    @staticmethod
    def _random_quaternions(n: int) -> torch.Tensor:
        """Sample roughly-uniform random quaternions."""
        q = torch.randn(n, 4)
        return q / q.norm(dim=-1, keepdim=True)

    # --- activated properties ---

    def get_scales(self) -> torch.Tensor:
        """(N, 3) positive scales."""
        return self.scales_raw.exp()

    def get_rotations(self) -> torch.Tensor:
        """(N, 3, 3) rotation matrices from the raw quaternions."""
        return quaternion_to_rotation_matrix(self.quats_raw)

    def get_opacities(self) -> torch.Tensor:
        """(N, 1) opacities in [0, 1]."""
        return self.opacities_raw.sigmoid()

    def get_colors(self) -> torch.Tensor:
        """(N, 3) RGB colors in [0, 1]."""
        return self.colors_raw.sigmoid()

    def get_covariance(self) -> torch.Tensor:
        """(N, 3, 3) covariance matrices Σ = R S S^T R^T.

        Parameterizing this way guarantees Σ is always symmetric
        positive semi-definite regardless of what the optimizer does.
        """
        R = self.get_rotations()          # (N, 3, 3)
        s = self.get_scales()             # (N, 3)
        S = torch.diag_embed(s)           # (N, 3, 3) — diagonal scale matrix
        M = R @ S                         # (N, 3, 3)
        return M @ M.transpose(-1, -2)    # (N, 3, 3) — R S S^T R^T


# --- Verification ---

if __name__ == "__main__":
    N = 1000
    model = GaussianModel(N)

    print(f"Gaussian count: {model.n}")
    print(f"means:      {model.means.shape}")
    print(f"scales:     {model.get_scales().shape}  min={model.get_scales().min().item():.4f}")
    print(f"rotations:  {model.get_rotations().shape}")
    print(f"opacities:  {model.get_opacities().shape}  mean={model.get_opacities().mean().item():.4f}")
    print(f"colors:     {model.get_colors().shape}  mean={model.get_colors().mean().item():.4f}")
    print(f"covariance: {model.get_covariance().shape}")

    # Check rotation matrices are valid (orthogonal, det=+1)
    R = model.get_rotations()
    eye = torch.eye(3).expand(N, 3, 3)
    ortho_err = (R @ R.transpose(-1, -2) - eye).abs().max().item()
    det = torch.det(R)
    print(f"\nR orthogonality error: {ortho_err:.2e}")
    print(f"R determinants: min={det.min().item():.4f} max={det.max().item():.4f}")

    # Check covariance is symmetric PSD (eigenvalues >= 0)
    cov = model.get_covariance()
    sym_err = (cov - cov.transpose(-1, -2)).abs().max().item()
    eigvals = torch.linalg.eigvalsh(cov)
    print(f"\nΣ symmetry error: {sym_err:.2e}")
    print(f"Σ eigenvalues: min={eigvals.min().item():.6f} (should be ≥ 0)")

    # Check gradients flow through all paths
    loss = (
        model.get_covariance().sum()
        + model.get_opacities().sum()
        + model.get_colors().sum()
    )
    loss.backward()
    for name, p in model.named_parameters():
        grad_ok = p.grad is not None and p.grad.abs().sum() > 0
        print(f"grad flows through {name}: {grad_ok}")

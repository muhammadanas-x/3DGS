import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianModel(nn.Module):

    def __init__(self, num_gaussians=1000, max_sh_degree=3):
        super().__init__()

        self.num_gaussians = num_gaussians
        self.max_sh_degree = max_sh_degree

        # -------------------------------------------------
        # 1. 3D Gaussian means
        # -------------------------------------------------

        self.means = nn.Parameter(torch.randn(num_gaussians, 3, dtype=torch.float32) * 0.5)

        # -------------------------------------------------
        # 2. Anisotropic scales
        # -------------------------------------------------

        self.log_scales = nn.Parameter(torch.full((num_gaussians, 3), -3.0, dtype=torch.float32))

        # -------------------------------------------------
        # 3. Quaternion rotations
        # -------------------------------------------------

        quats = torch.zeros(num_gaussians, 4, dtype=torch.float32)
        quats[:, 0] = 1.0

        self.quats = nn.Parameter(quats)

        # -------------------------------------------------
        # 4. Opacity logits
        # -------------------------------------------------

        self.opacity_logits = nn.Parameter(torch.full((num_gaussians, 1), -2.0, dtype=torch.float32))

        # -------------------------------------------------
        # 5. Spherical Harmonic coefficients
        # -------------------------------------------------

        num_sh_coeffs = (max_sh_degree + 1) ** 2
        num_rest = num_sh_coeffs - 1

        self.sh0 = nn.Parameter(torch.zeros(num_gaussians, 1, 3, dtype=torch.float32))
        self.sh_rest = nn.Parameter(torch.zeros(num_gaussians, num_rest, 3, dtype=torch.float32))


    @property
    def scales(self):
        return torch.exp(self.log_scales)


    @property
    def rotations(self):
        return F.normalize(self.quats, dim=-1)


    @property
    def opacities(self):
        return torch.sigmoid(self.opacity_logits)


    @property
    def sh_coeffs(self):
        return torch.cat([self.sh0, self.sh_rest], dim=1)


    def print_shapes(self):
        print("means:", self.means.shape)
        print("log_scales:", self.log_scales.shape)
        print("scales:", self.scales.shape)
        print("quats:", self.quats.shape)
        print("rotations:", self.rotations.shape)
        print("opacity_logits:", self.opacity_logits.shape)
        print("opacities:", self.opacities.shape)
        print("sh0:", self.sh0.shape)
        print("sh_rest:", self.sh_rest.shape)
        print("sh_coeffs:", self.sh_coeffs.shape)
import torch


_pixel_grid_cache = {}


def create_pixel_grid(height, width, device, dtype=torch.float32):
    key = (height, width, str(device), dtype)

    if key not in _pixel_grid_cache:
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        _pixel_grid_cache[key] = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    return _pixel_grid_cache[key]


def inverse_covariance_2d(covariance):
    a = covariance[:, 0, 0]
    b = covariance[:, 0, 1]
    c = covariance[:, 1, 1]

    det = torch.clamp(a * c - b * b, min=1e-8)

    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det

    return inv_a, inv_b, inv_c


def rasterize_gaussians_batched(means_2d, covariances_2d, colors, opacities, height, width, background):
    device = means_2d.device
    dtype = means_2d.dtype

    pixels = create_pixel_grid(height, width, device, dtype)

    dx = pixels[:, 0].unsqueeze(0) - means_2d[:, 0].unsqueeze(1)
    dy = pixels[:, 1].unsqueeze(0) - means_2d[:, 1].unsqueeze(1)

    inv_a, inv_b, inv_c = inverse_covariance_2d(covariances_2d)

    mahalanobis = (
        inv_a[:, None] * dx * dx
        + 2.0 * inv_b[:, None] * dx * dy
        + inv_c[:, None] * dy * dy
    )

    gaussian_weights = torch.exp(-0.5 * mahalanobis)

    alphas = opacities[:, None] * gaussian_weights
    alphas = alphas.clamp(0.0, 0.99)

    one_minus_alpha = 1.0 - alphas

    prefix = torch.cat([
        torch.ones(1, alphas.shape[1], device=device, dtype=dtype),
        one_minus_alpha[:-1]
    ], dim=0)

    transmittance = torch.cumprod(prefix, dim=0)

    contributions = transmittance * alphas

    image = torch.sum(contributions.unsqueeze(-1) * colors[:, None, :], dim=0)

    remaining_transmittance = torch.prod(one_minus_alpha, dim=0)

    image = image + remaining_transmittance.unsqueeze(-1) * background

    return image.reshape(height, width, 3), remaining_transmittance.reshape(height, width)
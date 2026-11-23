import torch
from scipy.spatial import cKDTree


def create_random_point_cloud(num_points=100_000, extent=1.3):
    points = torch.rand(num_points, 3) * (2.0 * extent) - extent
    colors = torch.rand(num_points, 3)
    return points, colors


def initialize_log_scales(points, k=3):
    points_np = points.detach().cpu().numpy()
    tree = cKDTree(points_np)
    distances, _ = tree.query(points_np, k=k + 1, workers=-1)

    distances = torch.from_numpy(distances[:, 1:]).to(dtype=points.dtype)
    dist2_avg = (distances ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(torch.clamp(dist2_avg, min=1e-12))

    return torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)


def initialize_quaternions(num_points):
    quats = torch.rand(num_points, 4)
    return quats


def initialize_opacity_logits(num_points, initial_opacity=0.1):
    opacity = torch.full((num_points, 1), initial_opacity)
    return torch.logit(opacity)


def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def initialize_sh(colors, max_sh_degree=3):
    num_coeffs = (max_sh_degree + 1) ** 2
    sh = torch.zeros(colors.shape[0], num_coeffs, 3)

    sh[:, 0, :] = rgb_to_sh(colors)

    sh0 = sh[:, :1, :]
    sh_rest = sh[:, 1:, :]

    return sh0, sh_rest
import torch

from renderer.projection import world_to_camera, project_points
from renderer.covariance import build_covariance, world_covariance_to_camera, project_covariance_to_screen
from renderer.gaussian2d import rasterize_gaussians_batched


C0 = 0.28209479177387814


def sh0_to_rgb(sh0):
    return (C0 * sh0.squeeze(1) + 0.5).clamp(0.0, 1.0)


def render_model(model, c2w, K, height, width):
    means_3d = model.means
    scales = model.scales
    quats = model.quats
    opacities = model.opacities.squeeze(-1)
    colors = sh0_to_rgb(model.sh0)

    covariance_world = build_covariance(scales, quats)

    points_camera = world_to_camera(means_3d, c2w)
    means_2d, depth = project_points(means_3d, c2w, K)

    covariance_camera = world_covariance_to_camera(covariance_world, c2w)
    covariance_2d = project_covariance_to_screen(covariance_camera, points_camera, K)

    valid = depth > 1e-4

    means_2d = means_2d[valid]
    covariance_2d = covariance_2d[valid]
    colors = colors[valid]
    opacities = opacities[valid]
    depth = depth[valid]

    eye = torch.eye(2, device=means_3d.device, dtype=means_3d.dtype)
    covariance_2d = covariance_2d + eye.unsqueeze(0) * 1e-4

    sort_idx = torch.argsort(depth)

    means_2d = means_2d[sort_idx]
    covariance_2d = covariance_2d[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]
    depth = depth[sort_idx]

    background = torch.ones(3, device=means_3d.device, dtype=means_3d.dtype)

    rendered, remaining_transmittance = rasterize_gaussians_batched(
        means_2d,
        covariance_2d,
        colors,
        opacities,
        height,
        width,
        background
    )

    debug = {
        "means_2d": means_2d,
        "covariance_2d": covariance_2d,
        "depth": depth,
        "colors": colors,
        "opacities": opacities
    }

    return rendered, debug
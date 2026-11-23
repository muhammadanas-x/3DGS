import torch
import torch.nn.functional as F


def quaternion_to_rotation_matrix(quats):
    quats = F.normalize(quats, dim=-1)

    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]

    R = torch.empty(quats.shape[0], 3, 3, dtype=quats.dtype, device=quats.device)

    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)

    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)

    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return R


def build_covariance(scales, quats):
    R = quaternion_to_rotation_matrix(quats)
    S2 = torch.diag_embed(scales ** 2)
    covariance = R @ S2 @ R.transpose(-1, -2)
    return covariance


def world_covariance_to_camera(covariance_world, c2w):
    w2c = torch.linalg.inv(c2w)
    R_wc = w2c[:3, :3]
    covariance_camera = R_wc.unsqueeze(0) @ covariance_world @ R_wc.T.unsqueeze(0)
    return covariance_camera


def perspective_jacobian(points_camera, K):
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    depth = -z

    fx = K[0, 0]
    fy = K[1, 1]

    J = torch.zeros(points_camera.shape[0], 2, 3, dtype=points_camera.dtype, device=points_camera.device)

    J[:, 0, 0] = fx / depth
    J[:, 0, 2] = fx * x / (depth ** 2)

    J[:, 1, 1] = -fy / depth
    J[:, 1, 2] = -fy * y / (depth ** 2)

    return J


def project_covariance_to_screen(covariance_camera, points_camera, K):
    J = perspective_jacobian(points_camera, K)
    covariance_2d = J @ covariance_camera @ J.transpose(-1, -2)
    return covariance_2d
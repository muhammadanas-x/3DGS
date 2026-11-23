import torch


def world_to_camera(points_world, c2w):
    """
    points_world: [N, 3]
    c2w:          [4, 4]

    Returns points in Blender/OpenGL camera coordinates.
    """

    w2c = torch.linalg.inv(c2w)

    N = points_world.shape[0]

    ones = torch.ones(
        (N, 1),
        dtype=points_world.dtype,
        device=points_world.device
    )

    points_h = torch.cat(
        [points_world, ones],
        dim=1
    )

    points_camera_h = (
        w2c @ points_h.T
    ).T

    return points_camera_h[:, :3]


def project_points(points_world, c2w, K):
    """
    Project world-space points into image pixels.

    NeRF synthetic / Blender camera convention:
        camera looks along -Z.

    points_world: [N, 3]
    c2w:          [4, 4]
    K:            [3, 3]
    """

    points_camera = world_to_camera(
        points_world,
        c2w
    )

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]

    # Blender/OpenGL camera looks down -Z.
    depth = -z

    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    # OpenGL camera:
    #
    # x right
    # y up
    # -z forward
    #
    # Image coordinates:
    #
    # x right
    # y down
    #
    u = fx * (x / depth) + cx

    v = -fy * (y / depth) + cy

    pixels = torch.stack(
        [u, v],
        dim=-1
    )

    return pixels, depth
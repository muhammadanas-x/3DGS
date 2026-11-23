import torch
from torchvision.utils import save_image

from datasets.blender import BlenderDataset
from scene.gaussian_model import GaussianModel
from renderer.projection import world_to_camera, project_points
from renderer.covariance import build_covariance, world_covariance_to_camera, project_covariance_to_screen
from renderer.gaussian2d import create_pixel_grid, evaluate_gaussian_2d, alpha_composite, composite_over_background


device = torch.device("cuda")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))


def sh0_to_rgb(sh0):
    C0 = 0.28209479177387814
    return (C0 * sh0.squeeze(1) + 0.5).clamp(0.0, 1.0)


# ============================================================
# 1. Load ONE training image
# ============================================================

dataset = BlenderDataset("data/lego")
camera = dataset[0]

ground_truth = camera["image"].to(device)
c2w = camera["c2w"].to(device)
K = camera["K"].to(device)

height = camera["height"]
width = camera["width"]

print("GT:", ground_truth.shape)


# ============================================================
# 2. Create Gaussian model
# ============================================================
#
# Keep this small while using our brute-force PyTorch rasterizer.
# ============================================================

model = GaussianModel(num_gaussians=30, max_sh_degree=3).to(device)


# ============================================================
# 3. Extract differentiable parameters
# ============================================================

means_3d = model.means
scales = model.scales
quats = model.quats
opacities = model.opacities.squeeze(-1)
colors = sh0_to_rgb(model.sh0)


# ============================================================
# 4. Construct 3D covariance
# ============================================================

covariance_world = build_covariance(scales, quats)


# ============================================================
# 5. Transform Gaussian centers to camera space
# ============================================================

points_camera = world_to_camera(means_3d, c2w)


# ============================================================
# 6. Project Gaussian centers
# ============================================================

means_2d, depth = project_points(means_3d, c2w, K)


# ============================================================
# 7. Project covariance
# ============================================================

covariance_camera = world_covariance_to_camera(covariance_world, c2w)
covariance_2d = project_covariance_to_screen(covariance_camera, points_camera, K)


# ============================================================
# 8. Visibility filtering
# ============================================================

u = means_2d[:, 0]
v = means_2d[:, 1]

valid = (depth > 0.0) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)

print("Visible:", valid.sum().item(), "/", model.num_gaussians)

means_2d = means_2d[valid]
covariance_2d = covariance_2d[valid]
colors = colors[valid]
opacities = opacities[valid]
depth = depth[valid]

if means_2d.shape[0] == 0:
    raise RuntimeError("No Gaussians are visible in this camera.")


# ============================================================
# 9. Sort front-to-back
# ============================================================

sort_idx = torch.argsort(depth)

means_2d = means_2d[sort_idx]
covariance_2d = covariance_2d[sort_idx]
colors = colors[sort_idx]
opacities = opacities[sort_idx]


# ============================================================
# 10. Rasterize
# ============================================================

pixel_positions = create_pixel_grid(height, width, device=device)

alpha_images = []

for i in range(means_2d.shape[0]):
    weights = evaluate_gaussian_2d(pixel_positions, means_2d[i], covariance_2d[i])
    alpha = opacities[i] * weights.reshape(height, width)
    alpha_images.append(alpha)

alphas = torch.stack(alpha_images, dim=0)


# ============================================================
# 11. Alpha composite
# ============================================================

image, transmittance = alpha_composite(colors, alphas)

background = torch.ones(3, device=device)
rendered = composite_over_background(image, transmittance, background)


# ============================================================
# 12. Convert GT to [H,W,3]
# ============================================================

ground_truth_hwc = ground_truth.permute(1, 2, 0)


# ============================================================
# 13. Compute ONE-image loss
# ============================================================

loss = torch.mean(torch.abs(rendered - ground_truth_hwc))

print("Loss:", loss.item())


# ============================================================
# 14. BACKPROPAGATION
# ============================================================

model.zero_grad()

loss.backward()


# ============================================================
# 15. Inspect gradients
# ============================================================

print()
print("Gradient statistics")

print("means grad:", model.means.grad.abs().mean().item())
print("log_scales grad:", model.log_scales.grad.abs().mean().item())
print("quats grad:", model.quats.grad.abs().mean().item())
print("opacity grad:", model.opacity_logits.grad.abs().mean().item())
print("SH0 grad:", model.sh0.grad.abs().mean().item())

print()
print("First Gaussian mean gradient:")
print(model.means.grad[0])


# ============================================================
# 16. Save current render
# ============================================================

save_image(rendered.detach().permute(2, 0, 1).clamp(0.0, 1.0), "before_training.png")
save_image(ground_truth, "ground_truth.png")
import asyncio
import base64
import io

import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from datasets.blender import BlenderDataset
from scene.gaussian_model import GaussianModel
from renderer.manual_renderer import render_model


DEVICE = torch.device("cuda")
NUM_GAUSSIANS = 5000

TRAIN_WIDTH = 128
TRAIN_HEIGHT = 128

TRAIN_STEPS = 1000

SEND_EVERY = 20


app = FastAPI()


def tensor_to_data_url(image):
    image = image.detach().clamp(0.0, 1.0)
    array = (image * 255.0).byte().cpu().numpy()

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def gaussian_debug_to_json(debug):
    means = debug["means_2d"].detach().cpu()
    covariances = debug["covariance_2d"].detach().cpu()
    depths = debug["depth"].detach().cpu()
    colors = debug["colors"].detach().cpu()
    opacities = debug["opacities"].detach().cpu()

    gaussians = []

    for i in range(means.shape[0]):
        gaussians.append({
            "u": means[i, 0].item(),
            "v": means[i, 1].item(),
            "depth": depths[i].item(),
            "cov": [
                covariances[i, 0, 0].item(),
                covariances[i, 0, 1].item(),
                covariances[i, 1, 1].item()
            ],
            "color": colors[i].tolist(),
            "opacity": opacities[i].item()
        })

    return gaussians


@app.get("/")
async def index():
    return HTMLResponse(Path("viewer.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def train_socket(websocket: WebSocket):
    await websocket.accept()

    print("GPU:", torch.cuda.get_device_name(0))

    dataset = BlenderDataset("data/lego")
    camera = dataset[0]

    ground_truth = camera["image"].unsqueeze(0).to(DEVICE)

    ground_truth = F.interpolate(
        ground_truth,
        size=(TRAIN_HEIGHT, TRAIN_WIDTH),
        mode="bilinear",
        align_corners=False
    )[0]

    c2w = camera["c2w"].to(DEVICE)
    K = camera["K"].clone().to(DEVICE)

    sx = TRAIN_WIDTH / camera["width"]
    sy = TRAIN_HEIGHT / camera["height"]

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    ground_truth_hwc = ground_truth.permute(1, 2, 0)

    model = GaussianModel(NUM_GAUSSIANS, max_sh_degree=3).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": 2e-2},
        {"params": [model.log_scales], "lr": 5e-3},
        {"params": [model.quats], "lr": 2e-3},
        {"params": [model.opacity_logits], "lr": 5e-3},
        {"params": [model.sh0], "lr": 2e-2}
    ])

    target_url = tensor_to_data_url(ground_truth_hwc)

    try:
        for iteration in range(TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)

            rendered, debug = render_model(
                model,
                c2w,
                K,
                TRAIN_HEIGHT,
                TRAIN_WIDTH
            )

            loss = F.l1_loss(rendered, ground_truth_hwc)

            loss.backward()
            optimizer.step()

            if iteration % SEND_EVERY == 0 or iteration == TRAIN_STEPS - 1:
                message = {
                    "iteration": iteration,
                    "loss": loss.item(),
                    "render": tensor_to_data_url(rendered),
                    "gaussians": gaussian_debug_to_json(debug),
                    "visible": len(debug["depth"]),
                    "total": NUM_GAUSSIANS
                }

                if iteration == 0:
                    message["target"] = target_url

                await websocket.send_json(message)
                await asyncio.sleep(0)

        await websocket.send_json({"done": True})

    except WebSocketDisconnect:
        print("Viewer disconnected.")
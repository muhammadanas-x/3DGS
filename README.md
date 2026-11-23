# 3DGS · Single-View Gaussian Splatting

This PyTorch and CUDA demo fits a set of 3D Gaussians to **one target image from one known camera pose**. It uses the first training frame of the synthetic Lego scene and renders from that same camera throughout training. The browser viewer shows the target and the current render side by side, with a live loss readout and a toggleable overlay of the projected Gaussians.

## Why one view?

The goal is to iterate quickly on the renderer and training loop. Restricting training to one fixed camera removes the work of fitting the scene across multiple views. The live demo also downsamples the target to **128 × 128 pixels**. That matters because the PyTorch rasterizer evaluates each visible Gaussian at every output pixel, so reducing the resolution cuts GPU computation and memory use.

The Gaussians still have 3D positions, scales, and rotations. Each training step projects them onto the image plane, sorts them by depth, and blends their colors front to back. PyTorch backpropagates the L1 image loss, and Adam updates the Gaussians. With a single reference image, depth and occluded surfaces are underconstrained. The demo measures how well the Gaussians reproduce the training view.

## Run the live demo

Run these steps from the repository root with a CUDA-capable NVIDIA GPU and a Python environment supported by PyTorch.

1. Install CUDA-enabled `torch` and `torchvision` using the command for your system. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

2. Install the viewer dependencies and confirm that PyTorch can see the GPU. The CUDA check should print `True`.

   ```sh
   python -m pip install pillow fastapi "uvicorn[standard]"
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Download the synthetic Lego scene and copy the `lego` directory into `data/`. Git ignores `data/`, so each clone needs its own copy. The paths below are the files the demo reads from the standard dataset. [NeRF example dataset instructions](https://github.com/bmild/nerf#optimizing-a-nerf)

   ```text
   data/
   └── lego/
       ├── transforms_train.json
       └── train/
           └── r_0.png
   ```

4. Start the server.

   ```sh
   python -m uvicorn live_train:app
   ```

   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to start training. The demo uses 5,000 Gaussians and runs for 1,000 iterations, sending a render every 20 iterations and on the final iteration. To reduce GPU memory use, lower `NUM_GAUSSIANS`, `TRAIN_WIDTH`, or `TRAIN_HEIGHT` in `live_train.py`.

## Project layout

| Path | Role |
| --- | --- |
| `live_train.py` | CUDA training loop and FastAPI WebSocket server. |
| `viewer.html` | Browser viewer for the target, render, loss, and Gaussian overlay. |
| `datasets/blender.py` | Loads target images, camera poses, and camera intrinsics from Blender-format data. |
| `scene/gaussian_model.py` | Defines the learnable 3D Gaussian parameters. |
| `renderer/` | Camera projection, covariance transforms, rasterization, and compositing. |

`train.py` is an earlier render and gradient-check script. It currently imports helpers that are absent from `renderer/gaussian2d.py`. Use `live_train.py` for the demo.

import json
import math
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF


class BlenderDataset:

    def __init__(self, root):
        self.root = Path(root)

        transform_path = self.root / "transforms_train.json"

        with open(transform_path, "r") as f:
            metadata = json.load(f)

        self.camera_angle_x = metadata["camera_angle_x"]
        self.frames = metadata["frames"]

        print(f"Loaded {len(self.frames)} training cameras")


    def __len__(self):
        return len(self.frames)


    def __getitem__(self, index):
        frame = self.frames[index]

        # -------------------------------------------------
        # 1. Load image
        # -------------------------------------------------

        image_path = self.root / (frame["file_path"] + ".png")

        image = Image.open(image_path).convert("RGBA")

        image = TF.to_tensor(image)

        # image shape:
        # [4, H, W]

        rgb = image[:3]
        alpha = image[3:4]

        # Composite transparent background onto white.
        rgb = rgb * alpha + (1.0 - alpha)


        # -------------------------------------------------
        # 2. Camera-to-world transform
        # -------------------------------------------------

        c2w = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32
        )


        # -------------------------------------------------
        # 3. Image dimensions
        # -------------------------------------------------

        _, H, W = rgb.shape


        # -------------------------------------------------
        # 4. Convert horizontal FOV -> focal length
        # -------------------------------------------------

        focal = (
            0.5 * W
            /
            math.tan(
                0.5 * self.camera_angle_x
            )
        )


        # -------------------------------------------------
        # 5. Camera intrinsic matrix
        # -------------------------------------------------

        K = torch.tensor(
            [
                [focal, 0.0, W / 2.0],
                [0.0, focal, H / 2.0],
                [0.0, 0.0, 1.0]
            ],
            dtype=torch.float32
        )


        return {
            "image": rgb,
            "c2w": c2w,
            "K": K,
            "focal": focal,
            "height": H,
            "width": W
        }
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "third_party/voxelize"))

import torch
from PIL import Image
import numpy as np
from direct3d_s2.pipeline import Direct3DS2Pipeline

class Direct3DS2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sdf_resolution": ([512, 1024], {"default": 512}),
                "remesh": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "output.obj"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    FUNCTION = "run"
    CATEGORY = "3D/Direct3D"

    def __init__(self):
        # Path: custom_nodes/ComfyUI_Direct3DS2 → up to models/
        model_base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../models/diffusers/direct3d_s2")
        )

        self.pipeline = Direct3DS2Pipeline.from_pretrained(
            model_base_path,
            subfolder="direct3d-s2-v-1-1",
            local_files_only=True  # Do not try to download
        ).to("cuda:0")


    def run(self, image, sdf_resolution, remesh, output_path):
        # Convert image tensor to a temporary file
        if isinstance(image, torch.Tensor):
            image = image.squeeze().detach().cpu().numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
            temp_path = "temp_input.png"
            image.save(temp_path)
            image_path = temp_path
        else:
            raise ValueError("Image format not supported.")

        mesh = self.pipeline(
            image_path,
            sdf_resolution=sdf_resolution,
            remesh=remesh
        )["mesh"]

        mesh.export(output_path)
        return (output_path,)

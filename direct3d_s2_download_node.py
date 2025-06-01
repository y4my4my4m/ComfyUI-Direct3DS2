import os
from huggingface_hub import snapshot_download

class Direct3DS2ModelDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download_model"
    CATEGORY = "3D/Direct3D"

    def download_model(self):
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/direct3d_s2"))
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            repo_id="wushuang98/Direct3D-S2",
            allow_patterns=["direct3d-s2-v-1-1/*"],
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        return ("Model downloaded successfully.",)

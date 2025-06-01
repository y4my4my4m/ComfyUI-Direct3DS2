import os
from huggingface_hub import snapshot_download

def download_direct3d_s2_model(destination):
    print("Downloading Direct3D-S2 model to:", destination)
    snapshot_download(
        repo_id="wushuang98/Direct3D-S2",
        allow_patterns=["direct3d-s2-v-1-1/*"],
        local_dir=destination,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "../../models/diffusers/direct3d_s2")
    os.makedirs(model_dir, exist_ok=True)
    download_direct3d_s2_model(model_dir)
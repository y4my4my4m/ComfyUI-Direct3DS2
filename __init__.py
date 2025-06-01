from .direct3d_s2_node import Direct3DS2Node
from .direct3d_s2_download_node import Direct3DS2ModelDownloader

NODE_CLASS_MAPPINGS = {
    "Direct3DS2Node": Direct3DS2Node,
    "Direct3DS2ModelDownloader": Direct3DS2ModelDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Direct3DS2Node": "Direct3D-S2 Mesh Generator",
    "Direct3DS2ModelDownloader": "Direct3D-S2 Model Downloader"
}

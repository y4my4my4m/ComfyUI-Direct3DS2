# ComfyUI-Direct3DS2

## WIP - Doesnt work yet

A ComfyUI plugin for generating 3D meshes from 2D images using [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), a gigascale 3D generation model developed by DreamTechAI.

![example](https://raw.githubusercontent.com/DreamTechAI/Direct3D-S2/main/assets/test/13.png)

---

## ✨ Features

- Generate high-quality 3D meshes from single images
- Supports SDF resolutions of 512 or 1024
- Option to remesh the output for optimization
- Outputs `.obj` files directly

---

## 📦 Installation

### 📁 Manual Install

1. Clone this repo or install it through ComfyUI's Node Manager.
2. Drop the folder into your `ComfyUI/custom_nodes/` directory.
3. Restart ComfyUI.

### 🧪 Dependencies

Dependencies are managed via `requirements.txt` and will be installed automatically. If needed, you can install them manually:

```bash
pip install -r requirements.txt

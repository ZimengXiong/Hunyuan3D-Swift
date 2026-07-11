"""Run the reference 2.0 UNet2p5DConditionModel forward and dump it (run in .venv-oracle)."""

import os
import sys
import importlib.util

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNET_DIR = os.path.join(ROOT, "weights", "hunyuan3d-paint-v2-0", "unet")
MODULES_PY = os.path.join(ROOT, "metadata", "Hunyuan3D-2.0-paint", "hunyuan3d-paint-v2-0", "unet", "modules.py")


def load_ref():
    spec = importlib.util.spec_from_file_location("ref_modules", MODULES_PY)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    import json
    cfg = json.load(open(os.path.join(UNET_DIR, "config.json")))
    base = UNet2DConditionModel(**cfg)
    model = ref.UNet2p5DConditionModel(base)
    sd = load_file(os.path.join(UNET_DIR, "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(sd, strict=True)
    return model.eval().float()


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = load_ref()
    torch.manual_seed(0)
    B, N_gen, N_ref, H, W = 1, 2, 1, 8, 8
    sample = torch.randn(B, N_gen, 4, H, W)
    normal = torch.randn(B, N_gen, 4, H, W)
    position = torch.randn(B, N_gen, 4, H, W)
    ref_lat = torch.randn(B, N_ref, 4, H, W)
    ehs = torch.randn(B, 77, 1024)
    cam_gen = torch.tensor([[0, 1]], dtype=torch.long)
    cam_ref = torch.tensor([[0]], dtype=torch.long)
    with torch.no_grad():
        out = model(sample, torch.tensor(10.0), ehs,
                    normal_imgs=normal, position_imgs=position, ref_latents=ref_lat,
                    camera_info_gen=cam_gen, camera_info_ref=cam_ref)[0]  # [(B*N_gen),4,H,W]

    def nhwc(a):
        return np.ascontiguousarray(a.detach().numpy().transpose(0, 1, 3, 4, 2))  # [B,N,C,H,W]->[B,N,H,W,C]

    np.save(os.path.join(out_dir, "sample.npy"), nhwc(sample))
    np.save(os.path.join(out_dir, "normal.npy"), nhwc(normal))
    np.save(os.path.join(out_dir, "position.npy"), nhwc(position))
    np.save(os.path.join(out_dir, "ref_lat.npy"), nhwc(ref_lat))
    np.save(os.path.join(out_dir, "ehs.npy"), ehs.numpy())
    np.save(os.path.join(out_dir, "cam_gen.npy"), cam_gen.numpy())
    np.save(os.path.join(out_dir, "cam_ref.npy"), cam_ref.numpy())
    np.save(os.path.join(out_dir, "out.npy"), np.ascontiguousarray(out.detach().numpy().transpose(0, 2, 3, 1)))
    print("OK ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])

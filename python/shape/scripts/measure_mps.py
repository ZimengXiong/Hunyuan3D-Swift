"""Measure PyTorch-on-MPS peak RAM for a Hunyuan3D shape model: load to mps (fp16)
+ one CFG DiT forward + VAE decode. Reports torch MPS allocator + process RSS.
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/measure_mps.py <model_dir>
"""
import resource
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
import safetensors.torch  # noqa: E402
from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT  # noqa: E402
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain  # noqa: E402
from hy3dshape.models.autoencoders.model import ShapeVAE  # noqa: E402
from hy3dshape.models.conditioner import SingleImageEncoder  # noqa: E402

MD = sys.argv[1]
dev = torch.device("mps")
cfg = yaml.safe_load(open(f"{MD}/config.yaml"))
sd = safetensors.torch.load_file(f"{MD}/model.fp16.safetensors", device="cpu")


def split(p):
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


DiT = HunYuanDiTPlain if cfg["model"]["target"].endswith("HunYuanDiTPlain") else Hunyuan3DDiT
dit = DiT(**cfg["model"]["params"]).eval()
dit.load_state_dict(split("model."), strict=False)
vae = ShapeVAE(**cfg["vae"]["params"]).eval()
vae.load_state_dict(split("vae."), strict=False)
cond = SingleImageEncoder(**cfg["conditioner"]["params"]).eval()
cond.load_state_dict(split("conditioner."), strict=False)

for m in (dit, vae, cond):
    m.to(dev, dtype=torch.float16)
del sd
torch.mps.synchronize()
weights_alloc = torch.mps.current_allocated_memory() / 1e9

nl = cfg["vae"]["params"]["num_latents"]
ctx = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]["config"]["hidden_size"]
with torch.no_grad():
    lat = torch.randn(1, nl, 64, device=dev, dtype=torch.float16)
    contexts = {"main": torch.zeros(2, 1370, ctx, device=dev, dtype=torch.float16)}
    t = torch.tensor([0.5, 0.5], device=dev, dtype=torch.float16)
    v = dit(torch.cat([lat, lat]), t, contexts)
    torch.mps.synchronize()
    dec = vae((1.0 / vae.scale_factor) * lat)
    torch.mps.synchronize()

current = torch.mps.current_allocated_memory() / 1e9
driver = torch.mps.driver_allocated_memory() / 1e9
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"MPS  {MD.split('/')[-1]:24s} fp16: "
      f"weights {weights_alloc:5.2f} GB | mps-current {current:5.2f} GB | "
      f"mps-driver {driver:5.2f} GB | process-RSS {rss:5.2f} GB")

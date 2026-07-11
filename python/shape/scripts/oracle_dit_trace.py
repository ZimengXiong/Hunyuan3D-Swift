"""Single-forward DiT trace: dump intermediate activations from the torch reference.
VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/oracle_dit_trace.py
"""
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
import safetensors.torch  # noqa: E402
from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT, timestep_embedding  # noqa: E402

WD = "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini"
cfg = yaml.safe_load(open(f"{WD}/config.yaml"))
sd = safetensors.torch.load_file(f"{WD}/model.fp16.safetensors", device="cpu")
dit = Hunyuan3DDiT(**cfg["model"]["params"]).eval()
dit.load_state_dict({k[6:]: v.float() for k, v in sd.items() if k.startswith("model.")}, strict=False)
dit.float()

noise = np.load("/tmp/noise.npy")[0:1]                 # [1,512,64]
cond = np.load("/tmp/cond_torch.npy")[0:1]             # [1,1370,1536]
x = torch.from_numpy(noise).float()
ctx = torch.from_numpy(cond).float()
t = torch.tensor([0.5]).float()
D = {}

with torch.no_grad():
    latent = dit.latent_in(x); D["latent_in"] = latent.numpy()
    vec = dit.time_in(timestep_embedding(t, 256, dit.time_factor)); D["vec"] = vec.numpy()
    c = dit.cond_in(ctx); D["cond_in"] = c.numpy()
    for i, blk in enumerate(dit.double_blocks):
        latent, c = blk(img=latent, txt=c, vec=vec, pe=None)
        if i == 0:
            D["db0_img"], D["db0_txt"] = latent.numpy(), c.numpy()
    D["db_img"], D["db_txt"] = latent.numpy(), c.numpy()
    cat = torch.cat((c, latent), 1)
    for i, blk in enumerate(dit.single_blocks):
        cat = blk(cat, vec=vec, pe=None)
        if i == 0:
            D["sb0"] = cat.numpy()
    latent = cat[:, c.shape[1]:, ...]; D["sb"] = latent.numpy()
    out = dit.final_layer(latent, vec); D["out"] = out.numpy()

np.savez("/tmp/dit_trace.npz", **D)
for k, v in D.items():
    print(f"{k:10s} shape {str(tuple(v.shape)):16s} std {v.std():.4f} mean {v.mean():.4f}")
print("TRACE DONE")

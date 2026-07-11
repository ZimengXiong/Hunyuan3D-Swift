"""Convert the v2-1 .ckpt (torch pickle) -> flat fp16 safetensors with
conditioner./model./vae. prefixes that hy3dmlx.convert.load_models expects.

The released inference ckpt is a dict {'model':sd, 'vae':sd, 'conditioner':sd}.
Run with the oracle venv (needs torch):
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/convert_v21_ckpt.py
"""
import collections

import torch
import safetensors.torch

CKPT = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt"
OUT = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.safetensors"

print(">> torch.load (cpu)...", flush=True)
try:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
except Exception as e:  # noqa: BLE001
    print("  weights_only=True failed, retrying False:", str(e)[:80], flush=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

if isinstance(ckpt, dict) and "state_dict" in ckpt and "model" not in ckpt:
    ckpt = ckpt["state_dict"]

flat = {}
subs = [k for k in ("model", "vae", "conditioner") if k in ckpt and isinstance(ckpt[k], dict)]
if subs:  # dict-of-substatedicts layout
    print(f"  layout: dict of sub-state-dicts {subs}", flush=True)
    for sub in subs:
        for k, v in ckpt[sub].items():
            flat[f"{sub}.{k}"] = v
else:  # already-flat layout, possibly with a wrapper prefix
    print("  layout: flat", flush=True)
    flat = {k: v for k, v in ckpt.items()}

# fp16 floats, contiguous
out = {}
for k, v in flat.items():
    if not torch.is_tensor(v):
        continue
    out[k] = (v.half() if v.is_floating_point() else v).contiguous()

pref = collections.Counter(k.split(".")[0] for k in out)
print("  top-level prefixes:", dict(pref), flush=True)
print("  total tensors:", len(out), flush=True)
safetensors.torch.save_file(out, OUT)
print(f"WROTE {OUT}", flush=True)

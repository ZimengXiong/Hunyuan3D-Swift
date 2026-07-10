"""Dump the converted VAE weights (flattened MLX params) + a decode fixture for the
mlx-swift parity gate. Run from repo root: uv run python swift/dump_vae_fixture.py"""
import sys, json, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.vae import AutoencoderKL
from hy3dpaint_mlx.convert import load_torch_weights

PBR = "weights/hunyuan3d-paintpbr-v2-1"
VAEW = "weights/hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors"
vae = AutoencoderKL.from_config(json.load(open(f"{PBR}/vae/config.json")))
load_torch_weights(vae, mx.load(VAEW), renames=[(".to_out.0.", ".to_out.")])

params = dict(tree_flatten(vae.parameters()))
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/vae_weights.safetensors",
                    {k: v.astype(mx.float32) for k, v in params.items()})

rng = np.random.RandomState(0)
z = mx.array(rng.randn(1, 16, 16, 4).astype(np.float32))
img = vae.decode(z); mx.eval(img)
# also gate encode: image -> mean latent
ximg = mx.array(rng.randn(1, 128, 128, 3).astype(np.float32))
mean = vae.encode_mean(ximg); mx.eval(mean)
mx.save_safetensors(f"{FIX}/vae_fixture.safetensors",
                    {"z": z, "img": img, "ximg": ximg, "mean": mean})
print("params:", len(params), "| decode", img.shape, "std", round(float(img.std()), 4),
      "| encode_mean", mean.shape)
print("sample keys:", list(params)[:6])

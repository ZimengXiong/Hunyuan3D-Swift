"""Dump fixtures for the MLX-Swift end-to-end mesh:
  - VAE parity: latents + query points + expected SDF logits
  - full run: DINO cond_cat (demo image) + the initial noise, so Swift's denoise
    trajectory matches Python's (MLX RNG differs across bindings).
  PYTHONPATH=. uv run python scripts/dump_swift_fixtures.py
"""
import numpy as np
import mlx.core as mx

from hy3dmlx.pipeline import Hunyuan3DShapePipeline
from hy3dmlx.sampler import flow_match_sigmas

MD = "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini"
pipe = Hunyuan3DShapePipeline.from_pretrained(MD, dtype=mx.float32, verbose=False)
vae = pipe.vae

# --- VAE parity fixture: decode random-ish latents, query a few points ---
lat = mx.array((np.random.RandomState(3).randn(1, 512, 64) * 0.1).astype(np.float32))
kv = vae.decode(lat)
mx.eval(kv)
q = mx.array((np.random.RandomState(4).rand(1, 256, 3) * 2 - 1).astype(np.float32))
sdf = vae.geo_decoder(q, kv)
mx.eval(sdf)
mx.save_safetensors("/tmp/vae_fixture.safetensors", {"lat": lat, "q": q, "sdf": sdf})
print("vae fixture: lat", lat.shape, "q", q.shape, "sdf", sdf.shape, "range",
      float(sdf.min()), float(sdf.max()))

# --- full-run fixture: real DINO cond for the demo image + fixed initial noise ---
cond_cat = pipe.encode_image("reference/Hunyuan3D-2.1/assets/demo.png")  # [2,1370,1536]
mx.eval(cond_cat)
STEPS = 30
sig, _ = flow_match_sigmas(STEPS)
noise = np.random.RandomState(0).randn(1, 512, 64).astype(np.float32)  # match Swift's loaded noise
mx.save_safetensors("/tmp/run_fixture.safetensors", {
    "cond": cond_cat.astype(mx.float32),
    "noise": mx.array(noise),
    "sigmas": mx.array(sig.astype(np.float32)),
})
print("run fixture: cond", cond_cat.shape, "noise", noise.shape, "sigmas", sig.shape)

"""Dump shape fixtures for the MLX-Swift parity tests:
  - shape_vae_fixture:    latents + query points + expected SDF logits
  - shape_dino_fixture:   preprocessed demo-image pixels + expected DINOv2 hidden state
  - shape_sigmas_fixture: flow-match (30-step) + consistency (8-step) sigma schedules
  - shape_run_fixture:    DINO cond (demo image) + fixed initial noise + sigmas, so Swift's
    denoise trajectory matches Python's (MLX RNG differs across bindings)

Run from python/shape (this repo):
    PYTHONPATH=. uv run python dump_swift_fixtures.py                       # 2mini (shape-small)
    SHAPE_VARIANT=turbo PYTHONPATH=. uv run python dump_swift_fixtures.py  # 2.0-turbo (shape-large)

Env: FIXTURES_OUT output dir (default ./fixtures); SHAPE_MODEL checkpoint dir override;
DEMO_IMAGE conditioning image (default reference/Hunyuan3D-2.1/assets/demo.png).
"""
import os
import numpy as np
import mlx.core as mx

from hy3dmlx.pipeline import Hunyuan3DShapePipeline
from hy3dmlx.preprocess import dino_transform, load_image
from hy3dmlx.sampler import consistency_sigmas, flow_match_sigmas

VARIANTS = {
    "mini":  ("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", "", 512, 30),
    "turbo": ("weights/Hunyuan3D-2/hunyuan3d-dit-v2-0-turbo", "_turbo", 3072, 8),
}
VAR = os.environ.get("SHAPE_VARIANT", "mini")
MD_DEFAULT, SUF, NUM_LATENTS, STEPS = VARIANTS[VAR]
MD = os.environ.get("SHAPE_MODEL", MD_DEFAULT)
FIX = os.environ.get("FIXTURES_OUT", "fixtures")
IMG = os.environ.get("DEMO_IMAGE", "reference/Hunyuan3D-2.1/assets/demo.png")
os.makedirs(FIX, exist_ok=True)

pipe = Hunyuan3DShapePipeline.from_pretrained(MD, dtype=mx.float32, verbose=False)
vae = pipe.vae

# --- VAE parity fixture: decode random-ish latents, query a few points ---
lat = mx.array((np.random.RandomState(3).randn(1, NUM_LATENTS, 64) * 0.1).astype(np.float32))
kv = vae.decode(lat)
mx.eval(kv)
q = mx.array((np.random.RandomState(4).rand(1, 256, 3) * 2 - 1).astype(np.float32))
sdf = vae.geo_decoder(q, kv)
mx.eval(sdf)
mx.save_safetensors(f"{FIX}/shape_vae_fixture{SUF}.safetensors", {"lat": lat, "q": q, "sdf": sdf})
print("vae fixture: lat", lat.shape, "q", q.shape, "sdf", sdf.shape, "range",
      float(sdf.min()), float(sdf.max()))

# --- DINOv2 fixture: preprocessed demo-image pixels + expected last_hidden_state ---
img = load_image(IMG, size=512)                       # [1,3,512,512] in [-1,1]
pixels = dino_transform(img[0]).astype(mx.float32)    # NHWC [1,518,518,3]
out = pipe.dino(pixels)
mx.eval(out)
mx.save_safetensors(f"{FIX}/shape_dino_fixture.safetensors", {"pixels": pixels, "out": out})
print("dino fixture: pixels", pixels.shape, "out", out.shape)

# --- sigma schedules (model-independent gate: flow-match + consistency) ---
fm, _ = flow_match_sigmas(30)
cs, _ = consistency_sigmas(8)
mx.save_safetensors(f"{FIX}/shape_sigmas_fixture.safetensors",
                    {"flowmatch": mx.array(fm), "consistency": mx.array(cs)})
print("sigmas fixture: flowmatch", fm.shape, "consistency", cs.shape)

# --- full-run fixture: real DINO cond for the demo image + fixed initial noise ---
cond_cat = pipe.encode_image(IMG)                     # [2,1370,1536] (cond, uncond)
mx.eval(cond_cat)
guidance_embed = bool(getattr(pipe.dit, "guidance_embed", False))
cond = cond_cat[:1] if guidance_embed else cond_cat   # turbo: conditional only (no CFG)
sig, _ = (consistency_sigmas(STEPS) if guidance_embed else flow_match_sigmas(STEPS))
noise = np.random.RandomState(0).randn(1, NUM_LATENTS, 64).astype(np.float32)
mx.save_safetensors(f"{FIX}/shape_run_fixture{SUF}.safetensors", {
    "cond": cond.astype(mx.float32),
    "noise": mx.array(noise),
    "sigmas": mx.array(sig.astype(np.float32)),
    "guidance": mx.array(np.array([5.0], np.float32)),
})
print("run fixture: cond", cond.shape, "noise", noise.shape, "sigmas", sig.shape,
      "|", "guidance-embed" if guidance_embed else "CFG")

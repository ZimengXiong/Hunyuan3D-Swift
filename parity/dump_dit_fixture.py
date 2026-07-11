"""Dump a shape DiT forward (inputs + expected output) to a safetensors fixture
for the MLX-Swift parity test (gate: DiT forward cosine).

Run from python/shape (this repo):
    PYTHONPATH=. uv run python dump_dit_fixture.py                       # 2mini (shape-small)
    SHAPE_VARIANT=turbo PYTHONPATH=. uv run python dump_dit_fixture.py  # 2.0-turbo (shape-large)

Env: FIXTURES_OUT output dir (default ./fixtures); SHAPE_MODEL checkpoint dir override.
Turbo (guidance_embed) models also dump the guidance token input.
"""
import os
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models

VARIANTS = {
    "mini":  ("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", "", 512),
    "turbo": ("weights/Hunyuan3D-2/hunyuan3d-dit-v2-0-turbo", "_turbo", 3072),
}
VAR = os.environ.get("SHAPE_VARIANT", "mini")
MD_DEFAULT, SUF, NUM_LATENTS = VARIANTS[VAR]
MD = os.environ.get("SHAPE_MODEL", MD_DEFAULT)
FIX = os.environ.get("FIXTURES_OUT", "fixtures")

_, dit, _, _ = load_models(MD, dtype=mx.float32, verbose=False)
rng = np.random.RandomState(0)
x = mx.array(rng.randn(1, NUM_LATENTS, 64).astype(np.float32))
t = mx.array(np.array([0.3], np.float32))
cond = mx.array(rng.randn(1, 1370, 1536).astype(np.float32))
dump = {"x": x, "t": t, "cond": cond}
if bool(getattr(dit, "guidance_embed", False)):
    g = mx.array(np.array([5.0], np.float32))
    v = dit(x, t, cond, guidance=g)
    dump["guidance"] = g
else:
    v = dit(x, t, cond)
mx.eval(v)
dump["v"] = v
os.makedirs(FIX, exist_ok=True)
out = f"{FIX}/shape_dit_fixture{SUF}.safetensors"
mx.save_safetensors(out, dump)
print(f"fixture: {out} | x", x.shape, "t", t.shape, "cond", cond.shape, "v", v.shape,
      "| v std", float(v.std()))

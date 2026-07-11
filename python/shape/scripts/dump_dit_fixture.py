"""Dump a 2mini DiT forward (inputs + expected output) to a safetensors fixture
for the MLX-Swift parity test. PYTHONPATH=. uv run python scripts/dump_dit_fixture.py
"""
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models

_, dit, _, _ = load_models("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", dtype=mx.float32, verbose=False)
rng = np.random.RandomState(0)
x = mx.array(rng.randn(1, 512, 64).astype(np.float32))
t = mx.array(np.array([0.3], np.float32))
cond = mx.array(rng.randn(1, 1370, 1536).astype(np.float32))
v = dit(x, t, cond)
mx.eval(v)
mx.save_safetensors("/tmp/dit_fixture.safetensors",
                    {"x": x, "t": t, "cond": cond, "v": v})
print("fixture: x", x.shape, "t", t.shape, "cond", cond.shape, "v", v.shape,
      "| v std", float(v.std()))

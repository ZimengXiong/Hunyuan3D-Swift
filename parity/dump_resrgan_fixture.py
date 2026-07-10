"""Dump RealESRGAN RRDBNet weights + a fixture for the mlx-swift gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.realesrgan import load_rrdbnet

m = load_rrdbnet("weights/realesrgan/rrdbnet.npz")
params = dict(tree_flatten(m.parameters()))
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/resrgan_weights.safetensors",
                    {k: v.astype(mx.float32) for k, v in params.items()})
rng = np.random.RandomState(0)
x = mx.array(rng.rand(1, 32, 32, 3).astype(np.float32))   # small tile -> 128x128
y = m(x); mx.eval(y)
mx.save_safetensors(f"{FIX}/resrgan_fixture.safetensors", {"x": x, "y": y})
print("params:", len(params), "| x", x.shape, "-> y", y.shape, "std", round(float(y.std()), 4))
print("sample keys:", list(params)[:5])

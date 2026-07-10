"""Dump the base SD2.1 UNet2DConditionModel (random init) + a forward, for the Swift backbone gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.unet import UNet2DConditionModel

mx.random.seed(0)
m = UNet2DConditionModel()                       # base config, no ma/ra/pbr; random init
mx.eval(m.parameters())
params = dict(tree_flatten(m.parameters()))
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/unet_base_weights.safetensors",
                    {k: v.astype(mx.float32) for k, v in params.items()})
rng = np.random.RandomState(0)
sample = mx.array(rng.randn(1, 8, 8, 4).astype(np.float32))
ts = mx.array(np.array([500.0], np.float32))
ctx = mx.array(rng.randn(1, 77, 1024).astype(np.float32))
out = m(sample, ts, ctx); mx.eval(out)
mx.save_safetensors(f"{FIX}/unet_base_fixture.safetensors",
                    {"sample": sample, "ts": ts, "ctx": ctx, "out": out})
print("params:", len(params), "| out", out.shape, "std", round(float(out.std()), 4))

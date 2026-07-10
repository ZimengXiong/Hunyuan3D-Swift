"""Dump DINOv2-giant weights + a forward for the Swift gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from hy3dpaint_mlx.dinov2 import Dinov2Model

dino = Dinov2Model()
dino.update(tree_unflatten([(k, (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v))
                            for k, v in mx.load("weights/dinov2-giant/model.safetensors").items()]))
mx.eval(dino.parameters())
params = dict(tree_flatten(dino.parameters()))
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/dino_weights.safetensors",
                    {k: v.astype(mx.float32) for k, v in params.items()})
rng = np.random.RandomState(0)
px = mx.array(rng.randn(1, 518, 518, 3).astype(np.float32))
out = dino(px); mx.eval(out)
mx.save_safetensors(f"{FIX}/dino_fixture.safetensors", {"px": px, "out": out})
print("params:", len(params), "| out", out.shape, "std", round(float(out.std()), 4))

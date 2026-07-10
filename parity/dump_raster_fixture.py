"""Dump a gpu_raster scene + interpolate for the Swift rasterizer gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from hy3dpaint_mlx.raster import gpu_raster, cr_raster

rng = np.random.RandomState(1)
nv, nf, RES = 60, 100, 128
V = rng.uniform(-0.85, 0.85, (nv, 4)).astype(np.float32); V[:, 3] = 1.0
F = rng.randint(0, nv, (nf, 3)).astype(np.int32)
fi, ba = gpu_raster.rasterize(V, F, RES)
col = rng.randn(nv, 3).astype(np.float32)
interp = cr_raster.interpolate(col, fi, ba, F)
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/raster_fixture.safetensors", {
    "V": mx.array(V), "F": mx.array(F.astype(np.int32)),
    "findices": mx.array(fi.astype(np.int32)), "bary": mx.array(ba),
    "col": mx.array(col), "interp": mx.array(interp)})
print("findices covered:", int((fi > 0).sum()), "/", RES * RES, "| RES", RES)

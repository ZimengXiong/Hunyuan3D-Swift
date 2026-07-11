"""Run the Python pipeline from the SAME fixture (cond+noise) the Swift used, so the two are
directly comparable for parity. Saves a Python mesh at the given resolution.
  PYTHONPATH=. uv run python scripts/python_from_fixture.py <R> <out.glb>
"""
import sys
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models
from hy3dmlx.pipeline import Hunyuan3DShapePipeline
from hy3dmlx.sampler import denoise

R = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/python_from_fixture.glb"
OCTREE = "octree" in sys.argv

dino, dit, vae, cfg = load_models("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", dtype=mx.float32, verbose=False)
pipe = Hunyuan3DShapePipeline(dino, dit, vae, cfg, dtype=mx.float32)

fx = mx.load("/tmp/run_fixture.safetensors")
cond, noise, sig = fx["cond"], fx["noise"], np.array(fx["sigmas"])
full = np.concatenate([sig, [1.0]]).astype(np.float32)

lat = mx.array(np.array(noise))
for i in range(len(sig)):
    dt = float(full[i + 1] - sig[i])
    if dt == 0:
        continue
    t = mx.array([float(sig[i])] * 2, dtype=mx.float32)
    v = dit(mx.concatenate([lat, lat], 0), t, cond)
    vc, vu = mx.split(v, 2, 0)
    lat = lat + dt * (vu + 5.0 * (vc - vu))
    mx.eval(lat)

kv = vae.decode(lat / vae.scale_factor); mx.eval(kv)
if OCTREE:
    grid, bmin, bmax, gs = vae.query_grid_octree(kv, octree_resolution=R)
else:
    grid, bmin, bmax, gs = vae.query_grid(kv, octree_resolution=R, num_chunks=50000)
mesh = pipe._grid_to_mesh(grid, bmin, bmax, gs, 0.0)
mesh.export(OUT)
import numpy as _np
print(f"python R={R} {'octree' if OCTREE else 'dense'}: grid {grid.shape} "
      f"active {int(_np.sum(~_np.isnan(grid)))} -> {len(mesh.vertices)} verts -> {OUT}")

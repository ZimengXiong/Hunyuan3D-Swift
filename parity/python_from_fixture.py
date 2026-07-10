"""Run the Python pipeline from the SAME fixture (cond+noise+sigmas) that Swift uses, so the two
meshes are directly comparable (gate: Chamfer distance / bbox diagonal). Writes the Python
reference mesh as both .glb (for viewers) and .safetensors V/F arrays (consumed by the Swift test).

Run from the Hunyuan3D-Shape-MLX repo root, AFTER dump_swift_fixtures.py:
    PYTHONPATH=. uv run python python_from_fixture.py [R]                       # 2mini
    SHAPE_VARIANT=turbo PYTHONPATH=. uv run python python_from_fixture.py [R]  # 2.0-turbo

R is the octree grid resolution (default 256, matching the Swift e2e test); pass "dense"
to use the dense grid instead of the octree decode.
Env: FIXTURES_OUT fixture dir (default ./fixtures); SHAPE_MODEL checkpoint dir override.
"""
import os
import sys
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models
from hy3dmlx.pipeline import Hunyuan3DShapePipeline

VARIANTS = {
    "mini":  ("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", ""),
    "turbo": ("weights/Hunyuan3D-2/hunyuan3d-dit-v2-0-turbo", "_turbo"),
}
VAR = os.environ.get("SHAPE_VARIANT", "mini")
MD_DEFAULT, SUF = VARIANTS[VAR]
MD = os.environ.get("SHAPE_MODEL", MD_DEFAULT)
FIX = os.environ.get("FIXTURES_OUT", "fixtures")

R = next((int(a) for a in sys.argv[1:] if a.isdigit()), 256)
OCTREE = "dense" not in sys.argv
OUT = f"{FIX}/shape_mesh_python_{VAR}"

dino, dit, vae, cfg = load_models(MD, dtype=mx.float32, verbose=False)
pipe = Hunyuan3DShapePipeline(dino, dit, vae, cfg, dtype=mx.float32)

fx = mx.load(f"{FIX}/shape_run_fixture{SUF}.safetensors")
cond, noise, sig = fx["cond"], fx["noise"], np.array(fx["sigmas"])
guidance = float(np.array(fx["guidance"])[0]) if "guidance" in fx else 5.0
full = np.concatenate([sig, [1.0]]).astype(np.float32)
guidance_embed = bool(getattr(dit, "guidance_embed", False))

lat = mx.array(np.array(noise))
gvec = mx.array(np.array([guidance], np.float32))
for i in range(len(sig)):
    dt = float(full[i + 1] - sig[i])
    if dt == 0:
        continue
    if guidance_embed:                                # turbo: single forward, guidance token
        t = mx.array([float(sig[i])], dtype=mx.float32)
        v = dit(lat, t, cond, guidance=gvec)
    else:                                             # mini / 2.0: CFG over a doubled batch
        t = mx.array([float(sig[i])] * 2, dtype=mx.float32)
        v = dit(mx.concatenate([lat, lat], 0), t, cond)
        vc, vu = mx.split(v, 2, 0)
        v = vu + guidance * (vc - vu)
    lat = lat + dt * v
    mx.eval(lat)

kv = vae.decode(lat / vae.scale_factor); mx.eval(kv)
if OCTREE:
    grid, bmin, bmax, gs = vae.query_grid_octree(kv, octree_resolution=R)
else:
    grid, bmin, bmax, gs = vae.query_grid(kv, octree_resolution=R, num_chunks=50000)
mesh = pipe._grid_to_mesh(grid, bmin, bmax, gs, 0.0)
mesh.export(f"{OUT}.glb")
mx.save_safetensors(f"{OUT}.safetensors", {
    "V": mx.array(np.asarray(mesh.vertices, np.float32)),
    "F": mx.array(np.asarray(mesh.faces, np.int32)),
})
print(f"python {VAR} R={R} {'octree' if OCTREE else 'dense'}: grid {grid.shape} "
      f"active {int(np.sum(~np.isnan(grid)))} -> {len(mesh.vertices)} verts -> {OUT}.glb/.safetensors")

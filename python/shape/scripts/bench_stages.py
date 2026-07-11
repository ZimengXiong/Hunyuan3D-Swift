"""Per-stage wall-clock for the Python MLX pipeline, matching the Swift hy3d-cli timing breakdown
(dino / denoise / grid / MC). Same fixture, same model (2mini), same resolution.
  PYTHONPATH=. uv run python scripts/bench_stages.py <R>
"""
import sys, time
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models
from hy3dmlx.pipeline import Hunyuan3DShapePipeline

R = int(sys.argv[1]) if len(sys.argv) > 1 else 256
dino, dit, vae, cfg = load_models("weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini", dtype=mx.float32, verbose=False)
pipe = Hunyuan3DShapePipeline(dino, dit, vae, cfg, dtype=mx.float32)

nf = mx.load("/tmp/dino_fixture.safetensors")
fx = mx.load("/tmp/run_fixture.safetensors")
pixels, noise, sig = nf["pixels"], fx["noise"], np.array(fx["sigmas"])
full = np.concatenate([sig, [1.0]]).astype(np.float32)


def run(label):
    t = time.time(); emb = dino(pixels); mx.eval(emb)
    cond = mx.concatenate([emb, mx.zeros_like(emb)], 0); mx.eval(cond); tDino = time.time() - t

    t = time.time(); lat = mx.array(np.array(noise))
    for i in range(len(sig)):
        dt = float(full[i + 1] - sig[i])
        if dt == 0:
            continue
        tt = mx.array([float(sig[i])] * 2, dtype=mx.float32)
        v = dit(mx.concatenate([lat, lat], 0), tt, cond)
        vc, vu = mx.split(v, 2, 0)
        lat = lat + dt * (vu + 5.0 * (vc - vu)); mx.eval(lat)
    tDenoise = time.time() - t

    t = time.time()
    kv = vae.decode(lat / vae.scale_factor); mx.eval(kv)
    grid, bmin, bmax, gs = vae.query_grid(kv, octree_resolution=R, num_chunks=50000)
    grid = np.array(grid); tGrid = time.time() - t

    t = time.time(); mesh = pipe._grid_to_mesh(grid, bmin, bmax, gs, 0.0); tMC = time.time() - t
    total = tDino + tDenoise + tGrid + tMC
    print(f"[{label}] R={R}  dino {tDino:.2f}s  denoise {tDenoise:.2f}s  grid {tGrid:.2f}s  "
          f"MC {tMC:.2f}s  | total {total:.2f}s  ({len(mesh.vertices)} verts)")


run("cold")   # includes MLX JIT kernel compilation (matches a Swift cold run)
run("warm")   # steady-state

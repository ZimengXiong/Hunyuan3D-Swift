"""Benchmark the VAE grid query at several num_chunks (lossless tuning).
  PYTHONPATH=. uv run python scripts/bench_grid.py <model_dir> [octree]
"""
import sys
import time

import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models

MD = sys.argv[1]
OCT = int(sys.argv[2]) if len(sys.argv) > 2 else 256

dino, dit, vae, cfg = load_models(MD, dtype=mx.float16, verbose=False)
# realistic latents: one cheap denoise-ish — just use random then decode (timing is query-bound)
lat = (mx.random.normal((1, *vae.latent_shape), key=mx.random.key(0)) * 0.1).astype(mx.float16)
kv = vae.decode(lat / vae.scale_factor); mx.eval(kv)

ref = None
for nc in [8000, 32000, 100000, 300000, 600000]:
    t = time.time()
    grid, *_ = vae.query_grid(kv, octree_resolution=OCT, num_chunks=nc)
    dt = time.time() - t
    # lossless check: identical grid regardless of chunking
    drift = "" if ref is None else f" | max-drift-vs-8000 {np.abs(grid - ref).max():.2e}"
    if ref is None:
        ref = grid
    print(f"{MD.split('/')[-1]:22s} octree {OCT} num_chunks {nc:>7d}: {dt:5.1f}s{drift}", flush=True)

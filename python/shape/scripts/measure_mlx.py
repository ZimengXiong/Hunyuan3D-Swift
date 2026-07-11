"""Measure MLX steady-state RAM: resident weights + inference peak.
  PYTHONPATH=. uv run python scripts/measure_mlx.py <model_dir> <fp16|fp32> [quant 0|4|8]

Loads (optionally quantizing), frees the load/quantize transients, records the
resident weight memory, then resets the peak counter and measures the inference
peak (one CFG DiT forward + VAE decode + a grid chunk).
"""
import gc
import sys

import mlx.core as mx

from hy3dmlx.convert import load_models

MD, DT = sys.argv[1], sys.argv[2]
QUANT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
dtype = {"fp16": mx.float16, "fp32": mx.float32}[DT]


def _clear():
    for fn in (getattr(mx, "clear_cache", None), getattr(getattr(mx, "metal", None), "clear_cache", None)):
        if fn:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


dino, dit, vae, cfg = load_models(MD, dtype=dtype, quantize=QUANT or None, verbose=False)
mx.eval(dino.parameters(), dit.parameters(), vae.parameters())
gc.collect(); _clear()
weights = mx.get_active_memory()          # resident weight footprint (transients freed)
mx.reset_peak_memory()

nl = vae.latent_shape[0]
cond = mx.zeros((2, 1370, dino.hidden), dtype=dtype)
lat = mx.random.normal((1, nl, 64)).astype(dtype)
v = dit(mx.concatenate([lat, lat], 0), mx.array([0.5, 0.5], dtype=dtype), cond); mx.eval(v)
kv = vae.decode(lat / vae.scale_factor); mx.eval(kv)
grid, *_ = vae.query_grid(kv, octree_resolution=128, num_chunks=8000)

peak = mx.get_peak_memory()               # inference-only peak (load transient excluded)
qtag = f"{QUANT}bit" if QUANT else DT
print(f"MLX  {MD.split('/')[-1]:22s} {qtag:5s}: "
      f"resident-weights {weights/1e9:5.2f} GB | inference-peak {peak/1e9:5.2f} GB")

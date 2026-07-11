"""MLX side of v2-1 fast parity (DINOv2-large + single HunYuanDiTPlain forward).
  PYTHONPATH=. uv run python scripts/mlx_compare_v21.py
"""
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models

MD = "weights/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"


def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


dino, dit, vae, cfg = load_models(MD, dtype=mx.float32, verbose=True)

pix = np.load("/tmp/v21_pixels.npy")
d = dino(mx.array(np.transpose(pix, (0, 2, 3, 1)))); mx.eval(d)
d = np.array(d.astype(mx.float32))
dt = np.load("/tmp/v21_dino.npy")
print(f"[DINO-large] cosine {cos(d, dt):.6f}  maxabs {np.abs(d - dt).max():.4f}  std {d.std():.4f}/{dt.std():.4f}")

noise = np.load("/tmp/v21_noise.npy")
cc = mx.array(np.load("/tmp/v21_cond.npy"))
inp = mx.array(np.concatenate([noise, noise], 0))
t = mx.array([0.5, 0.5], dtype=mx.float32)
v = dit(inp, t, cc); mx.eval(v)
v = np.array(v.astype(mx.float32))
vt = np.load("/tmp/v21_dit_v.npy")
print(f"[HunYuanDiTPlain] cosine {cos(v, vt):.6f}  maxabs {np.abs(v - vt).max():.4f}  std {v.std():.4f}/{vt.std():.4f}")

"""MLX side of the DiT single-forward trace; compares to /tmp/dit_trace.npz.
PYTHONPATH=. uv run python scripts/mlx_dit_trace.py
"""
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models
from hy3dmlx.layers import timestep_embedding

ref = np.load("/tmp/dit_trace.npz")
_, dit, _, _ = load_models("weights/Hunyuan3D-2mini", dtype=mx.float32, verbose=False)

noise = np.load("/tmp/noise.npy")[0:1]
cond = np.load("/tmp/cond_torch.npy")[0:1]
x = mx.array(noise)
ctx = mx.array(cond)
t = mx.array([0.5], dtype=mx.float32)


def cmp(name, arr):
    a = np.array(arr.astype(mx.float32))
    r = ref[name]
    af, rf = a.flatten().astype(np.float64), r.flatten().astype(np.float64)
    cos = af @ rf / (np.linalg.norm(af) * np.linalg.norm(rf))
    print(f"{name:10s} cos {cos:.6f}  maxabs {np.abs(a - r).max():.4f}  "
          f"mlx std {a.std():.4f} ref std {r.std():.4f}")


latent = dit.latent_in(x); cmp("latent_in", latent)
vec = dit.time_in(timestep_embedding(t, 256, max_period=dit.time_factor)); cmp("vec", vec)
c = dit.cond_in(ctx); cmp("cond_in", c)
for i, blk in enumerate(dit.double_blocks):
    latent, c = blk(latent, c, vec)
    if i == 0:
        cmp("db0_img", latent); cmp("db0_txt", c)
cmp("db_img", latent); cmp("db_txt", c)
cat = mx.concatenate([c, latent], axis=1)
for i, blk in enumerate(dit.single_blocks):
    cat = blk(cat, vec)
    if i == 0:
        cmp("sb0", cat)
latent = cat[:, c.shape[1]:, ...]; cmp("sb", latent)
out = dit.final_layer(latent, vec); cmp("out", out)

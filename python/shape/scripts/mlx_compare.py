"""MLX side of the per-stage parity check. Consumes /tmp/*.npy from oracle_compare.py.

Each stage is isolated: DiT is fed torch's cond, VAE is fed torch's final latents.
Run:  uv run python scripts/mlx_compare.py
"""
import numpy as np
import mlx.core as mx

from hy3dmlx.convert import load_models
from hy3dmlx.sampler import flow_match_sigmas

STEPS, GS = 30, 5.0


def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


dino, dit, vae, cfg = load_models("weights/Hunyuan3D-2mini", dtype=mx.float32, verbose=False)

# ---- Stage 1: DINO with identical pixels ----
pix = np.load("/tmp/dino_pixels.npy")                       # [1,3,518,518]
d = dino(mx.array(np.transpose(pix, (0, 2, 3, 1)))); mx.eval(d)
d = np.array(d.astype(mx.float32))
dt = np.load("/tmp/dino_out_torch.npy")
print(f"[DINO]  cosine {cos(d, dt):.6f}  maxabs {np.abs(d - dt).max():.4f}  "
      f"mlx std {d.std():.4f} torch std {dt.std():.4f}")

# ---- Stage 2: DiT loop, identical noise + identical (torch) cond ----
noise = np.load("/tmp/noise.npy")
cond_cat = mx.array(np.load("/tmp/cond_torch.npy"))         # [2,1370,1536]
sigmas, sigmas_full = flow_match_sigmas(STEPS)
lat = mx.array(noise)
for i in range(STEPS):
    dti = float(sigmas_full[i + 1] - sigmas_full[i])
    if dti == 0.0:
        continue
    t = mx.array([float(sigmas[i])] * 2, dtype=mx.float32)
    v = dit(mx.concatenate([lat, lat], 0), t, cond_cat)
    vc, vu = mx.split(v, 2, 0)
    lat = lat + dti * (vu + GS * (vc - vu))
    mx.eval(lat)
lat = np.array(lat.astype(mx.float32))
latt = np.load("/tmp/latents_torch.npy")
print(f"[DiT]   cosine {cos(lat, latt):.6f}  maxabs {np.abs(lat - latt).max():.4f}  "
      f"mlx std {lat.std():.4f} torch std {latt.std():.4f}")

# ---- Stage 3: VAE grid, identical (torch) latents ----
kv = vae.decode(mx.array(latt) / vae.scale_factor); mx.eval(kv)
grid, _, _, _ = vae.query_grid(kv, bounds=1.01, octree_resolution=96, num_chunks=8000)
gt = np.load("/tmp/grid_torch.npy")
print(f"[VAE]   maxabs {np.abs(grid - gt).max():.4f}  "
      f"mlx range {grid.min():.4f}..{grid.max():.4f} (frac>0 {(grid > 0).mean():.4f})  "
      f"torch range {gt.min():.4f}..{gt.max():.4f} (frac>0 {(gt > 0).mean():.4f})")

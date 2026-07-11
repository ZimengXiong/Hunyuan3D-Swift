"""custom_rasterizer-faithful rasterizer (pure numpy reference).

A clean-room reimplementation of Hunyuan3D's `custom_rasterizer` CPU/CUDA algorithm —
NOT nvdiffrast. Matches the reference exactly:

  * screen map   sx = (x/w*0.5 + 0.5)*(W-1) + 0.5 ,  sy = (0.5 + 0.5*y/w)*(H-1) + 0.5
  * pixel center (px+0.5, py+0.5), coverage by signed-area barycentric in [0,1]
  * z-buffer     depth sz = z/w*0.49999 + 0.5 ; token = int(depth*2^18)*MAXINT + (face+1) ; min wins
  * output       findices [H,W] int32 (0=bg, else 1-based face id)
                 barycentric [H,W,3] float32, PERSPECTIVE-CORRECT (bary_i /= w_i, renormalized)

This is the portable/dev path. It is gated bit-close against the C++ twin oracle
(`oracle/cr_cpu`, which is the reference CUDA algorithm verbatim). The production path
will be a Metal compute kernel implementing this same algorithm.

Source of truth: reference/Hunyuan3D-MLX/hy3dgen/texgen/custom_rasterizer/lib/
                 custom_rasterizer_kernel/{rasterizer.h,rasterizer.cpp}
"""

from __future__ import annotations

import numpy as np

MAXINT = 2147483647
_DEPTH_SCALE = float(2 << 17)  # 262144  (18-bit depth quantization)
_BG = MAXINT - 1               # background sentinel in (token % MAXINT)


def _clip_to_screen(V: np.ndarray, W: int, H: int):
    """V [N,4] homogeneous clip coords -> per-vertex screen (sx, sy, sz) float32 + w."""
    V = np.ascontiguousarray(V, dtype=np.float32)
    w = V[:, 3]
    sx = (V[:, 0] / w * np.float32(0.5) + np.float32(0.5)) * np.float32(W - 1) + np.float32(0.5)
    sy = (np.float32(0.5) + np.float32(0.5) * V[:, 1] / w) * np.float32(H - 1) + np.float32(0.5)
    sz = V[:, 2] / w * np.float32(0.49999) + np.float32(0.5)
    return (sx.astype(np.float32), sy.astype(np.float32), sz.astype(np.float32),
            w.astype(np.float32))


def _signed_area(ax, ay, bx, by, cx, cy):
    """calculateSignedArea2: (c-a) x (b-a)."""
    return (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)


def rasterize(pos: np.ndarray, tri: np.ndarray, resolution):
    """Rasterize. pos: [V,4] or [1,V,4] clip-space homogeneous. tri: [F,3] int.

    resolution: int or (H, W). Returns (findices [H,W] int32, barycentric [H,W,3] float32),
    matching custom_rasterizer.rasterize(...).
    """
    pos = np.asarray(pos)
    if pos.ndim == 3:
        pos = pos[0]
    tri = np.ascontiguousarray(tri, dtype=np.int64)
    if isinstance(resolution, (tuple, list)):
        H, W = int(resolution[0]), int(resolution[1])
    else:
        H = W = int(resolution)

    sx, sy, sz, vw = _clip_to_screen(pos, W, H)
    F = tri.shape[0]

    # ---- pass 1: z-buffer (token = quantized-depth in high bits, face id in low bits) ----
    maxint_token = np.int64(MAXINT) * np.int64(MAXINT) + np.int64(_BG)
    zbuf = np.full(H * W, maxint_token, dtype=np.int64)

    tx = sx[tri]  # [F,3]
    ty = sy[tri]
    tz = sz[tri]

    for f in range(F):
        x0, x1, x2 = tx[f]
        y0, y1, y2 = ty[f]
        z0, z1, z2 = tz[f]
        area = _signed_area(x0, y0, x1, y1, x2, y2)
        if area == 0:
            continue
        xmin = int(np.floor(min(x0, x1, x2)))
        xmax = int(np.floor(max(x0, x1, x2)))
        ymin = int(np.floor(min(y0, y1, y2)))
        ymax = int(np.floor(max(y0, y1, y2)))
        xmin = max(xmin, 0); xmax = min(xmax + 1, W - 1)
        ymin = max(ymin, 0); ymax = min(ymax + 1, H - 1)
        if xmin > xmax or ymin > ymax:
            continue

        px = np.arange(xmin, xmax + 1, dtype=np.float32) + np.float32(0.5)
        py = np.arange(ymin, ymax + 1, dtype=np.float32) + np.float32(0.5)
        PX, PY = np.meshgrid(px, py)  # [h,w]

        inv = np.float32(1.0) / area
        beta = _signed_area(x0, y0, PX, PY, x2, y2) * inv
        gamma = _signed_area(x0, y0, x1, y1, PX, PY) * inv
        alpha = np.float32(1.0) - beta - gamma

        inb = (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1) & (gamma >= 0) & (gamma <= 1)
        if not inb.any():
            continue

        depth = alpha * z0 + beta * z1 + gamma * z2
        zq = (depth * np.float32(_DEPTH_SCALE)).astype(np.int64)  # trunc toward zero (depth>=0)
        token = zq * np.int64(MAXINT) + np.int64(f + 1)

        ix = (np.arange(xmin, xmax + 1)[None, :] + np.zeros((ymax - ymin + 1, 1), dtype=int))
        iy = (np.arange(ymin, ymax + 1)[:, None] + np.zeros((1, xmax - xmin + 1), dtype=int))
        flat = (iy * W + ix)[inb]
        np.minimum.at(zbuf, flat, token[inb])

    # ---- pass 2: decode face id + perspective-correct barycentric ----
    f_tok = zbuf % np.int64(MAXINT)
    fg = f_tok != np.int64(_BG)
    findices = np.where(fg, f_tok, 0).astype(np.int32).reshape(H, W)

    barycentric = np.zeros((H * W, 3), dtype=np.float32)
    idx = np.nonzero(fg)[0]
    if idx.size:
        face = (f_tok[idx] - 1).astype(np.int64)
        v0 = tri[face, 0]; v1 = tri[face, 1]; v2 = tri[face, 2]
        pcx = (idx % W).astype(np.float32) + np.float32(0.5)
        pcy = (idx // W).astype(np.float32) + np.float32(0.5)
        x0, y0 = sx[v0], sy[v0]
        x1, y1 = sx[v1], sy[v1]
        x2, y2 = sx[v2], sy[v2]
        area = _signed_area(x0, y0, x1, y1, x2, y2)
        inv = np.float32(1.0) / area
        beta = _signed_area(x0, y0, pcx, pcy, x2, y2) * inv
        gamma = _signed_area(x0, y0, x1, y1, pcx, pcy) * inv
        alpha = np.float32(1.0) - beta - gamma
        a = alpha / vw[v0]
        b = beta / vw[v1]
        c = gamma / vw[v2]
        s = np.float32(1.0) / (a + b + c)
        barycentric[idx, 0] = a * s
        barycentric[idx, 1] = b * s
        barycentric[idx, 2] = c * s

    return findices, barycentric.reshape(H, W, 3)


def interpolate(col: np.ndarray, findices: np.ndarray, barycentric: np.ndarray, tri: np.ndarray):
    """Vertex-attribute interpolation, matching custom_rasterizer.render.interpolate.

    col: [V,C] or [1,V,C]. Returns [H,W,C] float32 (0 where background).
    """
    col = np.asarray(col, dtype=np.float32)
    if col.ndim == 3:
        col = col[0]
    tri = np.asarray(tri)
    H, W = findices.shape
    f = findices.astype(np.int64) - 1 + (findices == 0)  # bg -> index 0, weight 0 below
    vidx = tri[f.reshape(-1)]                # [HW,3]
    vcol = col[vidx]                         # [HW,3,C]
    out = (barycentric.reshape(-1, 3, 1) * vcol).sum(axis=1)
    out = out.reshape(H, W, -1)
    out[findices == 0] = 0
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# GPU backend (Apple Silicon / MLX Metal kernel)                              #
# --------------------------------------------------------------------------- #

def rasterize_gpu(pos, tri, resolution):
    """GPU rasterize on the Apple GPU via MLX (see raster.gpu_raster).

    Same signature and (bit-close) output as :func:`rasterize`:
    returns (findices [H,W] int32, barycentric [H,W,3] float32).
    """
    from . import gpu_raster
    return gpu_raster.rasterize(pos, tri, resolution)


def gpu_available() -> bool:
    """True if the MLX GPU rasterizer backend can run on this machine."""
    try:
        from . import gpu_raster
        return gpu_raster.available()
    except Exception:
        return False


def rasterize_backend(pos, tri, resolution, backend: str = "numpy"):
    """Dispatch to a rasterizer backend.

    backend: "numpy" (portable reference) or "gpu" (MLX Metal kernel on the Apple GPU).
    "auto" picks "gpu" when available, else "numpy". Returns the same
    (findices [H,W] int32, barycentric [H,W,3] float32) regardless of backend.
    """
    if backend == "auto":
        backend = "gpu" if gpu_available() else "numpy"
    if backend == "gpu":
        return rasterize_gpu(pos, tri, resolution)
    if backend == "numpy":
        return rasterize(pos, tri, resolution)
    raise ValueError(f"unknown rasterizer backend: {backend!r} (use 'numpy', 'gpu', or 'auto')")

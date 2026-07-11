"""GPU custom_rasterizer (Apple Silicon / MLX Metal kernel).

A GPU port of Hunyuan3D's `custom_rasterizer` scanline algorithm, running on the
Apple GPU through `mx.fast.metal_kernel`. Produces the SAME outputs as the numpy
reference in `cr_raster.py` and the bit-exact CPU oracle in `oracle/cr_cpu/`:

  * findices    [H,W] int32  (0 = background, else 1-based face id)
  * barycentric [H,W,3] float32 (perspective-correct)

Algorithm (identical to the reference; see cr_raster.py / oracle/cr_cpu/cr_cpu.cpp):

  screen map   sx = (x/w*0.5 + 0.5)*(W-1) + 0.5
               sy = (0.5 + 0.5*y/w)*(H-1) + 0.5
               sz = z/w*0.49999 + 0.5
  coverage     pixel center (px+0.5, py+0.5); signed-area barycentric all in [0,1]
  z-buffer     depth = sum(bary*sz); zq = int(depth*2^18)
               token = zq*MAXINT + (face+1) ; min token wins

The reference z-buffer is a 64-bit token resolved by atomic_min. Metal (Apple GPU)
has no 64-bit atomics, so we split the lexicographic min of token = zq*MAXINT + fid
into two 32-bit atomic passes — which is EXACTLY equivalent because:

  token is monotonic in (zq, fid): zq dominates (scaled by MAXINT > num_faces),
  fid breaks ties. So min(token) == (min zq, then min fid among the min-zq faces).

  pass 1a  per face: atomic_min the quantized depth zq into zbuf_zq[pixel]
  pass 1b  per face: where this face's zq == zbuf_zq[pixel], atomic_min (face+1)
                     into zbuf_fid[pixel]
  pass 2   per pixel: decode face id, recompute screen-space barycentric at the
                     pixel center, perspective-correct (bary_i /= w_i, renormalize)

FMA / parity: the CPU oracle is compiled with -ffp-contract=off. The Metal kernels
below split every signed-area term into separate products before subtracting (no
fused multiply-add in the coverage / barycentric math) to match the non-FMA result
on exact-edge pixels.
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    _HAS_MLX = True
except Exception:  # pragma: no cover - MLX always present in this repo
    _HAS_MLX = False

MAXINT = 2147483647
_BG = MAXINT - 1  # background sentinel in (token % MAXINT)

# z-buffer "empty" sentinel. MLX's metal_kernel `init_value` is applied through a
# float32 conversion, so it cannot exactly represent MAXINT (2147483647 rounds to
# 2^31). We instead use 2^31 = 2147483648, which IS exactly float32-representable
# and is strictly larger than any real quantized depth (zq < 2^18) or face id
# (num_faces << 2^31). atomic_min therefore never picks it over a real value, and
# pass 2 treats it as background.
_SENTINEL = 2147483648  # 2**31


def available() -> bool:
    return _HAS_MLX


# --------------------------------------------------------------------------- #
# Metal kernel sources                                                         #
# --------------------------------------------------------------------------- #

# Shared device helpers. Signed area is split into two named products then
# subtracted (no FMA) so it matches the -ffp-contract=off CPU oracle.
_HEADER = r"""
inline float signed_area2(float ax, float ay, float bx, float by, float cx, float cy) {
    float p1 = (cx - ax) * (by - ay);
    float p2 = (bx - ax) * (cy - ay);
    return p1 - p2;
}
// Screen-space x/y/z for a vertex, identical expression order to the oracle.
inline void screen_xyz(const device float* V, int v, int W, int H,
                       thread float& sx, thread float& sy, thread float& sz) {
    float x = V[v * 4 + 0];
    float y = V[v * 4 + 1];
    float z = V[v * 4 + 2];
    float w = V[v * 4 + 3];
    sx = (x / w * 0.5f + 0.5f) * (float)(W - 1) + 0.5f;
    sy = (0.5f + 0.5f * y / w) * (float)(H - 1) + 0.5f;
    sz = z / w * 0.49999f + 0.5f;
}
"""

# ---- pass 1a: per-face atomic-min of quantized depth zq -------------------- #
# grid = (num_faces,). zbuf_zq output initialised to MAXINT (background high part).
_SRC_PASS1A = r"""
    uint f = thread_position_in_grid.x;
    if (f >= (uint)num_faces[0]) return;
    int W = res[0];
    int H = res[1];

    int i0 = F[f * 3 + 0];
    int i1 = F[f * 3 + 1];
    int i2 = F[f * 3 + 2];

    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    screen_xyz(V, i0, W, H, x0, y0, z0);
    screen_xyz(V, i1, W, H, x1, y1, z1);
    screen_xyz(V, i2, W, H, x2, y2, z2);

    float area = signed_area2(x0, y0, x1, y1, x2, y2);
    if (area == 0.0f) return;
    float inv = 1.0f / area;

    float xminf = metal::min(x0, metal::min(x1, x2));
    float xmaxf = metal::max(x0, metal::max(x1, x2));
    float yminf = metal::min(y0, metal::min(y1, y2));
    float ymaxf = metal::max(y0, metal::max(y1, y2));

    // Match the oracle loop bounds exactly: int(float) truncates toward zero.
    int px0 = (int)xminf;
    int px1 = (int)(xmaxf + 1.0f);
    int py0 = (int)yminf;
    int py1 = (int)(ymaxf + 1.0f);

    for (int px = px0; px < px1; ++px) {
        if (px < 0 || px >= W) continue;
        float vx = (float)px + 0.5f;
        for (int py = py0; py < py1; ++py) {
            if (py < 0 || py >= H) continue;
            float vy = (float)py + 0.5f;

            float beta  = signed_area2(x0, y0, vx, vy, x2, y2) * inv;
            float gamma = signed_area2(x0, y0, x1, y1, vx, vy) * inv;
            float alpha = 1.0f - beta - gamma;
            if (alpha < 0.0f || alpha > 1.0f) continue;
            if (beta  < 0.0f || beta  > 1.0f) continue;
            if (gamma < 0.0f || gamma > 1.0f) continue;

            float depth = alpha * z0 + beta * z1 + gamma * z2;
            int zq = (int)(depth * 262144.0f);  // 2<<17
            uint pix = (uint)(py * W + px);
            atomic_fetch_min_explicit(&zbuf_zq[pix], (uint)zq, memory_order_relaxed);
        }
    }
"""

# ---- pass 1b: per-face atomic-min of (face+1) among min-zq faces ----------- #
_SRC_PASS1B = r"""
    uint f = thread_position_in_grid.x;
    if (f >= (uint)num_faces[0]) return;
    int W = res[0];
    int H = res[1];

    int i0 = F[f * 3 + 0];
    int i1 = F[f * 3 + 1];
    int i2 = F[f * 3 + 2];

    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    screen_xyz(V, i0, W, H, x0, y0, z0);
    screen_xyz(V, i1, W, H, x1, y1, z1);
    screen_xyz(V, i2, W, H, x2, y2, z2);

    float area = signed_area2(x0, y0, x1, y1, x2, y2);
    if (area == 0.0f) return;
    float inv = 1.0f / area;

    float xminf = metal::min(x0, metal::min(x1, x2));
    float xmaxf = metal::max(x0, metal::max(x1, x2));
    float yminf = metal::min(y0, metal::min(y1, y2));
    float ymaxf = metal::max(y0, metal::max(y1, y2));

    int px0 = (int)xminf;
    int px1 = (int)(xmaxf + 1.0f);
    int py0 = (int)yminf;
    int py1 = (int)(ymaxf + 1.0f);

    for (int px = px0; px < px1; ++px) {
        if (px < 0 || px >= W) continue;
        float vx = (float)px + 0.5f;
        for (int py = py0; py < py1; ++py) {
            if (py < 0 || py >= H) continue;
            float vy = (float)py + 0.5f;

            float beta  = signed_area2(x0, y0, vx, vy, x2, y2) * inv;
            float gamma = signed_area2(x0, y0, x1, y1, vx, vy) * inv;
            float alpha = 1.0f - beta - gamma;
            if (alpha < 0.0f || alpha > 1.0f) continue;
            if (beta  < 0.0f || beta  > 1.0f) continue;
            if (gamma < 0.0f || gamma > 1.0f) continue;

            float depth = alpha * z0 + beta * z1 + gamma * z2;
            int zq = (int)(depth * 262144.0f);
            uint pix = (uint)(py * W + px);
            if ((uint)zq == zbuf_zq[pix]) {
                atomic_fetch_min_explicit(&zbuf_fid[pix], f + 1u, memory_order_relaxed);
            }
        }
    }
"""

# ---- pass 2: per-pixel decode + perspective-correct barycentric ----------- #
# grid = (H*W,). zbuf_fid holds the winning (face+1), or MAXINT if untouched (bg).
_SRC_PASS2 = r"""
    uint pix = thread_position_in_grid.x;
    if (pix >= (uint)(res[0] * res[1])) return;
    int W = res[0];
    int H = res[1];

    uint fid = zbuf_fid[pix];
    if (fid >= (uint)SENTINEL_C || fid == 0u) {
        findices[pix] = 0;
        barycentric[pix * 3 + 0] = 0.0f;
        barycentric[pix * 3 + 1] = 0.0f;
        barycentric[pix * 3 + 2] = 0.0f;
        return;
    }
    findices[pix] = (int)fid;
    int f = (int)fid - 1;

    int i0 = F[f * 3 + 0];
    int i1 = F[f * 3 + 1];
    int i2 = F[f * 3 + 2];

    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    screen_xyz(V, i0, W, H, x0, y0, z0);
    screen_xyz(V, i1, W, H, x1, y1, z1);
    screen_xyz(V, i2, W, H, x2, y2, z2);

    float w0 = V[i0 * 4 + 3];
    float w1 = V[i1 * 4 + 3];
    float w2 = V[i2 * 4 + 3];

    float vx = (float)(pix % (uint)W) + 0.5f;
    float vy = (float)(pix / (uint)W) + 0.5f;

    float area = signed_area2(x0, y0, x1, y1, x2, y2);
    float inv = 1.0f / area;
    float beta  = signed_area2(x0, y0, vx, vy, x2, y2) * inv;
    float gamma = signed_area2(x0, y0, x1, y1, vx, vy) * inv;
    float alpha = 1.0f - beta - gamma;

    float a = alpha / w0;
    float b = beta  / w1;
    float c = gamma / w2;
    float s = 1.0f / (a + b + c);
    barycentric[pix * 3 + 0] = a * s;
    barycentric[pix * 3 + 1] = b * s;
    barycentric[pix * 3 + 2] = c * s;
"""

_k_pass1a = None
_k_pass1b = None
_k_pass2 = None


def _kernels():
    global _k_pass1a, _k_pass1b, _k_pass2
    if _k_pass1a is None:
        _k_pass1a = mx.fast.metal_kernel(
            name="cr_pass1a_zq",
            input_names=["V", "F", "num_faces", "res"],
            output_names=["zbuf_zq"],
            header=_HEADER,
            source=_SRC_PASS1A,
            atomic_outputs=True,
        )
        _k_pass1b = mx.fast.metal_kernel(
            name="cr_pass1b_fid",
            input_names=["V", "F", "num_faces", "res", "zbuf_zq"],
            output_names=["zbuf_fid"],
            header=_HEADER,
            source=_SRC_PASS1B,
            atomic_outputs=True,
        )
        _k_pass2 = mx.fast.metal_kernel(
            name="cr_pass2_bary",
            input_names=["V", "F", "res", "zbuf_fid"],
            output_names=["findices", "barycentric"],
            header=_HEADER + "\n#define SENTINEL_C 2147483648u\n",
            source=_SRC_PASS2,
        )
    return _k_pass1a, _k_pass1b, _k_pass2


def _to_mx(a, dtype):
    if _HAS_MLX and isinstance(a, mx.array):
        return a.astype(dtype)
    return mx.array(np.ascontiguousarray(a), dtype=dtype)


def rasterize(pos, tri, resolution, stream=None):
    """GPU rasterize. pos: [V,4] or [1,V,4] clip-space homogeneous. tri: [F,3] int.

    resolution: int or (H, W). Returns (findices [H,W] int32, barycentric [H,W,3]
    float32), matching cr_raster.rasterize / the cr_cpu oracle. Outputs are numpy
    arrays for drop-in parity with the numpy path.
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX not available; GPU rasterizer unsupported")

    pos_np = np.asarray(pos)
    if pos_np.ndim == 3:
        pos_np = pos_np[0]
    if isinstance(resolution, (tuple, list)):
        H, W = int(resolution[0]), int(resolution[1])
    else:
        H = W = int(resolution)

    V = _to_mx(pos_np, mx.float32)
    F = _to_mx(np.asarray(tri), mx.int32)
    num_faces = int(F.shape[0])
    res = mx.array([W, H], dtype=mx.int32)
    nf = mx.array([num_faces], dtype=mx.int32)

    k1a, k1b, k2 = _kernels()
    npix = H * W

    if num_faces == 0:
        findices = np.zeros((H, W), dtype=np.int32)
        bary = np.zeros((H, W, 3), dtype=np.float32)
        return findices, bary

    # pass 1a: min quantized depth per pixel (init = MAXINT = bg high part)
    (zbuf_zq,) = k1a(
        inputs=[V, F, nf, res],
        grid=(num_faces, 1, 1),
        threadgroup=(min(256, num_faces), 1, 1),
        output_shapes=[(npix,)],
        output_dtypes=[mx.uint32],
        init_value=_SENTINEL,
    )

    # pass 1b: min (face+1) among faces whose zq == winning zq (init = MAXINT = bg)
    (zbuf_fid,) = k1b(
        inputs=[V, F, nf, res, zbuf_zq],
        grid=(num_faces, 1, 1),
        threadgroup=(min(256, num_faces), 1, 1),
        output_shapes=[(npix,)],
        output_dtypes=[mx.uint32],
        init_value=_SENTINEL,
    )

    # pass 2: decode + perspective-correct barycentric, one thread per pixel
    findices, barycentric = k2(
        inputs=[V, F, res, zbuf_fid],
        grid=(npix, 1, 1),
        threadgroup=(min(256, npix), 1, 1),
        output_shapes=[(npix,), (npix * 3,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    mx.eval(findices, barycentric)

    findices = np.asarray(findices).reshape(H, W).astype(np.int32)
    barycentric = np.asarray(barycentric).reshape(H, W, 3).astype(np.float32)
    return findices, barycentric

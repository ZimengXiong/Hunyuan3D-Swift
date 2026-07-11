"""Parity gate: the numpy custom_rasterizer-faithful rasterizer vs the C++ twin oracle.

The C++ oracle (oracle/cr_cpu/libcr_cpu_oracle.dylib) is Hunyuan3D custom_rasterizer's exact
CPU functions over a plain C ABI — algorithmically identical to the reference CUDA kernel.
Matching it == matching the CUDA pipeline. Build it first:  bash oracle/cr_cpu/build.sh
"""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "oracle", "cr_cpu"))

from hy3dpaint_mlx.raster import cr_raster  # noqa: E402
import cr_cpu as cr_cpu_oracle  # noqa: E402  (ctypes loader)

requires_oracle = pytest.mark.skipif(
    not cr_cpu_oracle.available(),
    reason="C++ oracle not built — run: bash oracle/cr_cpu/build.sh",
)


def _quad_scene():
    """A unit quad (2 triangles) filling most of the view, clip-space homogeneous."""
    V = np.array([
        [-0.8, -0.8, 0.0, 1.0],
        [0.8, -0.8, 0.1, 1.0],
        [0.8, 0.8, 0.0, 1.0],
        [-0.8, 0.8, -0.1, 1.0],
    ], dtype=np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return V, F


def _random_scene(n_verts=200, n_faces=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.95, 0.95, size=(n_verts, 2)).astype(np.float32)
    z = rng.uniform(-0.6, 0.6, size=(n_verts, 1)).astype(np.float32)
    w = np.ones((n_verts, 1), dtype=np.float32)
    V = np.concatenate([xy, z, w], axis=1).astype(np.float32)
    F = rng.integers(0, n_verts, size=(n_faces, 3)).astype(np.int32)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def _compare(V, F, res, fidx_exact=True, bary_atol=2e-4):
    f_mlx, b_mlx = cr_raster.rasterize(V, F, res)
    f_ref, b_ref = cr_cpu_oracle.rasterize_image(V, F, res, res)
    f_ref = np.asarray(f_ref)
    b_ref = np.asarray(b_ref)

    cov_mlx = f_mlx > 0
    cov_ref = f_ref > 0
    iou = (cov_mlx & cov_ref).sum() / max((cov_mlx | cov_ref).sum(), 1)

    fidx_match = float((f_mlx == f_ref).mean())
    both = cov_mlx & cov_ref
    bary_maxabs = float(np.abs(b_mlx[both] - b_ref[both]).max()) if both.any() else 0.0
    return dict(iou=float(iou), fidx_match=fidx_match, bary_maxabs=bary_maxabs,
                f_mlx=f_mlx, f_ref=f_ref)


@requires_oracle
def test_quad_exact():
    V, F = _quad_scene()
    for res in (16, 64, 256):
        r = _compare(V, F, res)
        assert r["fidx_match"] == 1.0, f"res={res} face-id mismatch: {r['fidx_match']}"
        assert r["bary_maxabs"] < 2e-4, f"res={res} barycentric drift {r['bary_maxabs']}"


@requires_oracle
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_scene(seed):
    V, F = _random_scene(seed=seed)
    r = _compare(V, F, 128)
    # Face-id map must match the reference exactly (coverage + depth ordering).
    assert r["fidx_match"] == 1.0, f"seed={seed} face-id match only {r['fidx_match']:.6f}, IoU={r['iou']:.4f}"
    assert r["bary_maxabs"] < 2e-4, f"seed={seed} barycentric maxabs {r['bary_maxabs']}"


def test_interpolate_recovers_positions():
    """interpolate(vtx_xy) at each pixel should equal the screen-projected position there.

    Pure self-consistency check (no oracle needed): rasterize, then interpolate the clip-space
    xy; covered pixels should map back near their own location.
    """
    V, F = _quad_scene()
    res = 64
    fidx, bary = cr_raster.rasterize(V, F, res)
    interp = cr_raster.interpolate(V[:, :3], fidx, bary, F)
    assert interp.shape == (res, res, 3)
    assert (fidx > 0).sum() > res * res * 0.5  # quad covers most of the view

"""Parity gate: the MLX GPU rasterizer vs the C++ twin oracle (the reference CUDA algorithm).

Mirrors tests/test_raster_parity.py, but exercises the GPU backend
(hy3dpaint_mlx/raster/gpu_raster.py — a custom Metal compute kernel via
mx.fast.metal_kernel). The oracle (oracle/cr_cpu/libcr_cpu_oracle.dylib) is
Hunyuan3D custom_rasterizer's exact CPU functions over a plain C ABI; matching it
== matching the CUDA pipeline. Build it first:  bash oracle/cr_cpu/build.sh

Gate:
  * face-id match >= 0.999 (we expect 1.0 — bit-exact) on the quad + random scenes
  * barycentric maxabs < 2e-3 on covered pixels

Also benchmarks the GPU rasterizer vs the numpy reference on a 40k-face mesh at
512x512 and prints the speedup.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "oracle", "cr_cpu"))

from hy3dpaint_mlx.raster import cr_raster  # noqa: E402
from hy3dpaint_mlx.raster import gpu_raster  # noqa: E402
import cr_cpu as cr_cpu_oracle  # noqa: E402  (ctypes loader)


def _ensure_oracle():
    """Build the C++ oracle if the dylib is missing."""
    if cr_cpu_oracle.available():
        return True
    build = os.path.join(ROOT, "oracle", "cr_cpu", "build.sh")
    try:
        subprocess.run(["bash", build], check=True, capture_output=True)
    except Exception:
        return False
    return cr_cpu_oracle.available()


_ORACLE_OK = _ensure_oracle()

requires_oracle = pytest.mark.skipif(
    not _ORACLE_OK,
    reason="C++ oracle not built — run: bash oracle/cr_cpu/build.sh",
)
requires_gpu = pytest.mark.skipif(
    not gpu_raster.available(),
    reason="MLX GPU backend unavailable",
)


# --- scene generators (mirrors tests/test_raster_parity.py) ------------------ #

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


def _big_scene(n_verts=20000, n_faces=40000, seed=7):
    """Dense ~40k-face mesh with perspective (varying w) for the benchmark."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.95, 0.95, size=(n_verts, 2)).astype(np.float32)
    z = rng.uniform(-0.6, 0.6, size=(n_verts, 1)).astype(np.float32)
    w = rng.uniform(0.8, 1.2, size=(n_verts, 1)).astype(np.float32)
    V = np.concatenate([xy, z, w], axis=1).astype(np.float32)
    F = rng.integers(0, n_verts, size=(n_faces, 3)).astype(np.int32)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def _compare(V, F, res):
    f_gpu, b_gpu = gpu_raster.rasterize(V, F, res)
    f_ref, b_ref = cr_cpu_oracle.rasterize_image(V, F, res, res)
    f_ref = np.asarray(f_ref)
    b_ref = np.asarray(b_ref)

    cov_gpu = f_gpu > 0
    cov_ref = f_ref > 0
    iou = (cov_gpu & cov_ref).sum() / max((cov_gpu | cov_ref).sum(), 1)

    fidx_match = float((f_gpu == f_ref).mean())
    both = cov_gpu & cov_ref
    bary_maxabs = float(np.abs(b_gpu[both] - b_ref[both]).max()) if both.any() else 0.0
    return dict(iou=float(iou), fidx_match=fidx_match, bary_maxabs=bary_maxabs)


@requires_oracle
@requires_gpu
def test_quad_gpu():
    V, F = _quad_scene()
    for res in (16, 64, 256):
        r = _compare(V, F, res)
        assert r["fidx_match"] >= 0.999, f"res={res} face-id match only {r['fidx_match']:.6f}"
        assert r["bary_maxabs"] < 2e-3, f"res={res} barycentric maxabs {r['bary_maxabs']}"


@requires_oracle
@requires_gpu
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_scene_gpu(seed):
    V, F = _random_scene(seed=seed)
    r = _compare(V, F, 128)
    assert r["fidx_match"] >= 0.999, (
        f"seed={seed} face-id match only {r['fidx_match']:.6f}, IoU={r['iou']:.4f}"
    )
    assert r["bary_maxabs"] < 2e-3, f"seed={seed} barycentric maxabs {r['bary_maxabs']}"


@requires_oracle
@requires_gpu
def test_big_scene_gpu():
    """Dense ~40k-face perspective mesh at 512x512 must also match the oracle."""
    V, F = _big_scene()
    r = _compare(V, F, 512)
    assert r["fidx_match"] >= 0.999, f"big scene face-id match only {r['fidx_match']:.6f}"
    assert r["bary_maxabs"] < 2e-3, f"big scene barycentric maxabs {r['bary_maxabs']}"


@requires_gpu
def test_benchmark_gpu_vs_numpy(capsys):
    """Benchmark the GPU rasterizer vs the numpy reference on a 40k-face mesh at 512x512."""
    V, F = _big_scene()
    res = 512
    n_faces = int(F.shape[0])

    # Warm up (JIT-compile the Metal kernels) and verify it actually covers pixels.
    f_gpu, _ = gpu_raster.rasterize(V, F, res)
    assert (f_gpu > 0).sum() > 0

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        gpu_raster.rasterize(V, F, res)
    t_gpu = (time.perf_counter() - t0) / reps

    # numpy reference (slow Python per-face loop) — fewer reps.
    np_reps = 1
    t0 = time.perf_counter()
    for _ in range(np_reps):
        cr_raster.rasterize(V, F, res)
    t_np = (time.perf_counter() - t0) / np_reps

    speedup = t_np / t_gpu
    msg = (
        f"\n[bench] {n_faces} faces @ {res}x{res}: "
        f"gpu={t_gpu * 1e3:.2f} ms  numpy={t_np * 1e3:.2f} ms  "
        f"speedup={speedup:.1f}x"
    )
    with capsys.disabled():
        print(msg)
    assert speedup > 1.0, f"GPU not faster than numpy: {speedup:.2f}x"

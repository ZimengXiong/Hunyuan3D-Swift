"""Weight-backed integration gates (slow; skipped if weights absent).

Run: uv run pytest tests/test_models.py -v
"""
import os

import numpy as np
import mlx.core as mx
import pytest

from conftest import available_models  # noqa: E402
from hy3dmlx.convert import load_models

MODELS = available_models()
IDS = list(MODELS)


@pytest.fixture(scope="module")
def loaded():
    cache = {}
    for name, md in MODELS.items():
        cache[name] = load_models(md, dtype=mx.float16, verbose=False)
    return cache


@pytest.mark.skipif(not MODELS, reason="no weights")
@pytest.mark.parametrize("name", IDS)
def test_load_gate_and_forward_finite(loaded, name):
    dino, dit, vae, cfg = loaded[name]
    nl = vae.latent_shape[0]
    cond = mx.zeros((2, 1370, dino.hidden), dtype=mx.float16)
    lat = mx.zeros((2, nl, 64), dtype=mx.float16)
    v = dit(lat, mx.array([0.3, 0.3], dtype=mx.float16), cond); mx.eval(v)
    assert v.shape == (2, nl, 64)
    assert bool(mx.all(mx.isfinite(v)).item())
    kv = vae.decode(mx.zeros((1, nl, 64), dtype=mx.float16)); mx.eval(kv)
    assert bool(mx.all(mx.isfinite(kv)).item())


@pytest.mark.skipif("2mini" not in MODELS, reason="2mini weights needed")
def test_quantized_load_and_forward_finite():
    dino, dit, vae, cfg = load_models(MODELS["2mini"], dtype=mx.float16, quantize=4, verbose=False)
    cond = mx.zeros((2, 1370, dino.hidden), dtype=mx.float16)
    v = dit(mx.zeros((2, 512, 64), dtype=mx.float16), mx.array([0.3, 0.3], dtype=mx.float16), cond)
    mx.eval(v)
    assert bool(mx.all(mx.isfinite(v)).item())


_IMG = "reference/Hunyuan3D-2.1/assets/demo.png"


@pytest.mark.skipif("2mini" not in MODELS or not os.path.exists(_IMG),
                    reason="2mini weights + demo image needed")
def test_octree_mesh_near_lossless():
    """End-to-end: octree decode reproduces the dense mesh (same seed -> same surface)."""
    from scipy.spatial import cKDTree

    from hy3dmlx.pipeline import Hunyuan3DShapePipeline
    p = Hunyuan3DShapePipeline.from_pretrained(MODELS["2mini"], dtype=mx.float16, verbose=False)
    kw = dict(num_inference_steps=8, octree_resolution=128, seed=0, verbose=False)
    dense = p.generate(_IMG, octree_decode=False, **kw)
    octree = p.generate(_IMG, octree_decode=True, **kw)
    a, b = dense.sample(10000), octree.sample(10000)
    chamfer = (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2
    assert chamfer < 0.02, f"octree Chamfer {chamfer:.4f} (bbox extent ~2)"

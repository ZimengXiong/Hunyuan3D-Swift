"""Parity gate: MLX Transformer2DModel + timestep embedding vs diffusers."""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_unflatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hy3dpaint_mlx import attention as A

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_DUMP = os.path.join(ROOT, ".parity_dumps", "attn")

pytestmark = pytest.mark.skipif(not os.path.exists(_ORACLE_PY), reason="oracle venv missing")


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _assign(module, sd):
    flat = []
    for k, v in sd.items():
        a = mx.array(np.asarray(v, np.float32))
        if a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        flat.append((k, a))
    module.update(tree_unflatten(flat))


@pytest.fixture(scope="module")
def dumps():
    os.makedirs(_DUMP, exist_ok=True)
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "attn_oracle.py"), _DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"attn oracle failed:\n{r.stdout}\n{r.stderr}")
    return _DUMP


def test_transformer2d(dumps):
    d = os.path.join(dumps, "transformer2d")
    x = np.load(os.path.join(d, "input.npy"))
    ctx = np.load(os.path.join(d, "context.npy"))
    ref = np.load(os.path.join(d, "output.npy"))
    sd = dict(np.load(os.path.join(d, "weights.npz")))
    m = A.Transformer2DModel(320, 5, 64, 1024, depth=1, groups=32, eps=1e-6)
    _assign(m, sd)
    out = np.asarray(m(mx.array(x), context=mx.array(ctx)), np.float32)
    cos, mab = _cos(out, ref), float(np.abs(out - ref).max())
    print(f"transformer2d cosine={cos:.7f} maxabs={mab:.2e}")
    assert cos > 0.9999 and mab < 3e-3


def test_timestep(dumps):
    d = os.path.join(dumps, "timestep")
    ts = np.load(os.path.join(d, "timesteps.npy"))
    emb_ref = np.load(os.path.join(d, "emb.npy"))
    temb_ref = np.load(os.path.join(d, "temb.npy"))
    sd = dict(np.load(os.path.join(d, "weights.npz")))
    emb = np.asarray(A.timestep_embedding(mx.array(ts), 320, flip_sin_to_cos=True, downscale_freq_shift=0.0), np.float32)
    assert _cos(emb, emb_ref) > 0.99999, f"sinusoid cosine {_cos(emb, emb_ref)}"
    te = A.TimestepEmbedding(320, 1280)
    _assign(te, sd)
    temb = np.asarray(te(mx.array(emb)), np.float32)
    cos, mab = _cos(temb, temb_ref), float(np.abs(temb - temb_ref).max())
    print(f"timestep emb cosine={_cos(emb, emb_ref):.7f}  temb cosine={cos:.7f} maxabs={mab:.2e}")
    assert cos > 0.9999 and mab < 3e-3

"""Parity gate: MLX SD2.1 UNet2DConditionModel vs diffusers (random weights)."""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_unflatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hy3dpaint_mlx.unet import UNet2DConditionModel

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_DUMP = os.path.join(ROOT, ".parity_dumps", "unet")

pytestmark = pytest.mark.skipif(not os.path.exists(_ORACLE_PY), reason="oracle venv missing")


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@pytest.fixture(scope="module")
def dump():
    os.makedirs(_DUMP, exist_ok=True)
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "unet_oracle.py"), _DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"unet oracle failed:\n{r.stdout}\n{r.stderr}")
    return _DUMP


def test_unet(dump):
    x = np.load(os.path.join(dump, "input.npy"))
    ctx = np.load(os.path.join(dump, "context.npy"))
    ref = np.load(os.path.join(dump, "output.npy"))
    sd = dict(np.load(os.path.join(dump, "weights.npz")))

    m = UNet2DConditionModel()
    flat = []
    for k, v in sd.items():
        a = mx.array(np.asarray(v, np.float32))
        if a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        flat.append((k, a))
    m.update(tree_unflatten(flat))
    mx.eval(m.parameters())

    out = np.asarray(m(mx.array(x), 10.0, mx.array(ctx)), np.float32)
    cos, mab = _cos(out, ref), float(np.abs(out - ref).max())
    print(f"unet cosine={cos:.7f} maxabs={mab:.2e}")
    assert cos > 0.9999 and mab < 5e-3

"""Per-primitive parity gate: MLX 2D layers vs diffusers/torch (cross-venv .npy dump).

Generates references by running oracle/layers_oracle.py in .venv-oracle, then builds the MLX
module, loads the same weights, and compares (cosine + maxabs) in fp32.
"""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_unflatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hy3dpaint_mlx import layers2d  # noqa: E402

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_DUMP = os.path.join(ROOT, ".parity_dumps", "layers2d")


@pytest.fixture(scope="module")
def dumps():
    if not os.path.exists(_ORACLE_PY):
        pytest.skip("oracle venv missing — uv venv .venv-oracle && install torch/diffusers")
    os.makedirs(_DUMP, exist_ok=True)
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "layers_oracle.py"), _DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"oracle dump failed:\n{r.stdout}\n{r.stderr}")
    return _DUMP


def _load_case(dump_dir, name):
    d = os.path.join(dump_dir, name)
    x = np.load(os.path.join(d, "input.npy"))
    y = np.load(os.path.join(d, "output.npy"))
    temb = np.load(os.path.join(d, "temb.npy")) if os.path.exists(os.path.join(d, "temb.npy")) else None
    sd = dict(np.load(os.path.join(d, "weights.npz")))
    return x, y, temb, sd


def _assign(module, sd, rename=None):
    """Load a torch state_dict (numpy) into an MLX module. Conv weights NCHW->NHWC."""
    rename = rename or {}
    flat = []
    for k, v in sd.items():
        k = rename.get(k, k)
        a = mx.array(np.asarray(v, dtype=np.float32))
        if a.ndim == 4:  # conv weight [O,I,kH,kW] -> [O,kH,kW,I]
            a = a.transpose(0, 2, 3, 1)
        flat.append((k, a))
    module.update(tree_unflatten(flat))


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _check(y_mlx, y_ref, tag, cos_thr=0.9999, maxabs_thr=2e-3):
    y_mlx = np.asarray(y_mlx, dtype=np.float32)
    cos = _cos(y_mlx, y_ref)
    maxabs = float(np.abs(y_mlx - y_ref).max())
    assert cos > cos_thr, f"{tag}: cosine {cos:.7f} <= {cos_thr}"
    assert maxabs < maxabs_thr, f"{tag}: maxabs {maxabs:.2e} >= {maxabs_thr}"
    return cos, maxabs


def test_groupnorm(dumps):
    x, y, _, sd = _load_case(dumps, "groupnorm")
    m = layers2d.GroupNorm2d(32, 64, eps=1e-6)
    _assign(m, sd)
    out = m(mx.array(x))
    print("groupnorm", _check(out, y, "groupnorm"))


def test_resnet_notemb(dumps):
    x, y, _, sd = _load_case(dumps, "resnet_notemb")
    m = layers2d.ResnetBlock2D(64, 128, temb_channels=None, groups=32, eps=1e-6)
    _assign(m, sd)
    out = m(mx.array(x))
    print("resnet_notemb", _check(out, y, "resnet_notemb"))


def test_resnet_temb(dumps):
    x, y, temb, sd = _load_case(dumps, "resnet_temb")
    m = layers2d.ResnetBlock2D(128, 128, temb_channels=512, groups=32, eps=1e-6)
    _assign(m, sd)
    out = m(mx.array(x), mx.array(temb))
    print("resnet_temb", _check(out, y, "resnet_temb"))


def test_downsample(dumps):
    x, y, _, sd = _load_case(dumps, "downsample")
    m = layers2d.Downsample2D(64, 64, padding=0)  # oracle builds it with padding=0
    # diffusers aliases the conv as both "conv" and legacy "Conv2d_0" (duplicate keys) — assign directly
    w = sd.get("conv.weight", sd.get("Conv2d_0.weight"))
    b = sd.get("conv.bias", sd.get("Conv2d_0.bias"))
    m.conv.weight = mx.array(np.asarray(w, np.float32)).transpose(0, 2, 3, 1)
    m.conv.bias = mx.array(np.asarray(b, np.float32))
    out = m(mx.array(x))
    print("downsample", _check(out, y, "downsample"))


def test_upsample(dumps):
    x, y, _, sd = _load_case(dumps, "upsample")
    m = layers2d.Upsample2D(64, 64)
    _assign(m, sd)
    out = m(mx.array(x))
    print("upsample", _check(out, y, "upsample"))

"""Parity gate: MLX RealESRGAN x4 (RRDBNet) vs the torch reference."""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_PTH = os.path.join(ROOT, "weights", "realesrgan", "RealESRGAN_x4plus.pth")
_NPZ = os.path.join(ROOT, "weights", "realesrgan", "rrdbnet.npz")


@pytest.mark.skipif(not (os.path.exists(_ORACLE_PY) and os.path.exists(_PTH) and os.path.exists(_NPZ)),
                    reason="needs oracle venv + RealESRGAN weights")
def test_rrdbnet():
    from hy3dpaint_mlx.realesrgan import load_rrdbnet
    d = os.path.join(ROOT, ".parity_dumps", "resr")
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "realesrgan_oracle.py"), d],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = load_rrdbnet(_NPZ)
    out = np.asarray(m(mx.array(np.load(f"{d}/in.npy"))), np.float32)
    ref = np.load(f"{d}/out.npy")
    a, b = out.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cos > 0.99999 and np.abs(out - ref).max() < 1e-3

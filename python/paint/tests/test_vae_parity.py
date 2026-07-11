"""Parity gate: MLX AutoencoderKL vs diffusers, using the real small-model VAE weights."""

import os
import sys
import json
import subprocess

import numpy as np
import mlx.core as mx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hy3dpaint_mlx.vae import AutoencoderKL
from hy3dpaint_mlx.convert import load_torch_weights

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_VAE_DIR = os.path.join(ROOT, "weights", "hunyuan3d-paint-v2-0", "vae")
_WEIGHTS = os.path.join(_VAE_DIR, "diffusion_pytorch_model.safetensors")
_DUMP = os.path.join(ROOT, ".parity_dumps", "vae")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_ORACLE_PY) and os.path.exists(_WEIGHTS)),
    reason="needs oracle venv + downloaded VAE weights",
)


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10((np.abs(b).max() ** 2) / mse)


@pytest.fixture(scope="module")
def vae():
    cfg = json.load(open(os.path.join(_VAE_DIR, "config.json")))
    m = AutoencoderKL.from_config(cfg)
    sd = mx.load(_WEIGHTS)
    load_torch_weights(m, sd, renames=[(".to_out.0.", ".to_out.")])
    return m


@pytest.fixture(scope="module")
def dumps():
    os.makedirs(_DUMP, exist_ok=True)
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "vae_oracle.py"), _VAE_DIR, _DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"vae oracle failed:\n{r.stdout}\n{r.stderr}")
    return _DUMP


def test_vae_encode(vae, dumps):
    x = np.load(os.path.join(dumps, "input.npy"))
    ref = np.load(os.path.join(dumps, "encode_mean.npy"))
    out = np.asarray(vae.encode_mean(mx.array(x)), np.float32)
    cos, mab = _cos(out, ref), float(np.abs(out - ref).max())
    print(f"vae.encode  cosine={cos:.7f} maxabs={mab:.2e}")
    assert cos > 0.9999 and mab < 5e-3


def test_vae_decode(vae, dumps):
    z = np.load(os.path.join(dumps, "z.npy"))
    ref = np.load(os.path.join(dumps, "decode.npy"))
    out = np.asarray(vae.decode(mx.array(z)), np.float32)
    cos, psnr = _cos(out, ref), _psnr(out, ref)
    print(f"vae.decode  cosine={cos:.7f} psnr={psnr:.1f}dB")
    assert cos > 0.9999 and psnr > 40

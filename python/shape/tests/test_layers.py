"""Fast parity tests for shared primitives vs numpy reference formulas (no weights)."""
import math

import numpy as np
import mlx.core as mx

from hy3dmlx.layers import (FourierEmbedder, LayerNorm, RMSNorm, gelu_erf, gelu_tanh,
                            timestep_embedding)
from hy3dmlx.models.dinov2 import Dinov2SwiGLUFFN

rng = np.random.RandomState(0)


def _close(a, b, tol=1e-4):
    return np.abs(np.array(a.astype(mx.float32)) - b).max() < tol


def test_layernorm_fp32_matches_numpy():
    x = rng.randn(4, 8, 64).astype(np.float32)
    ln = LayerNorm(64, eps=1e-6, affine=False)
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    ref = (x - mu) / np.sqrt(var + 1e-6)
    assert _close(ln(mx.array(x)), ref)


def test_rmsnorm_matches_numpy():
    x = rng.randn(2, 16, 64).astype(np.float32)
    n = RMSNorm(64)  # scale initialised to ones
    ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    assert _close(n(mx.array(x)), ref)


def test_gelu_variants():
    x = rng.randn(100).astype(np.float32)
    erf_ref = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    assert _close(gelu_erf(mx.array(x)), erf_ref, tol=1e-4)
    tanh_ref = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    assert _close(gelu_tanh(mx.array(x)), tanh_ref, tol=1e-4)


def test_timestep_embedding_matches_reference():
    # reference forward uses max_period = time_factor = 1000 (positional-arg quirk)
    t = np.array([0.0, 0.5, 1.0], np.float32)
    emb = timestep_embedding(mx.array(t), 256, max_period=1000)
    tt = 1000.0 * t
    freqs = np.exp(-math.log(1000) * np.arange(128, dtype=np.float32) / 128)
    args = tt[:, None] * freqs[None]
    ref = np.concatenate([np.cos(args), np.sin(args)], -1)
    assert _close(emb, ref, tol=1e-3)


def test_fourier_embedder_dims_and_values():
    fe = FourierEmbedder(num_freqs=8, include_pi=False)
    assert fe.out_dim == 3 * (8 * 2 + 1) == 51
    x = rng.randn(5, 3).astype(np.float32)
    out = np.array(fe(mx.array(x)).astype(mx.float32))
    assert out.shape == (5, 51)
    # first 3 columns are the raw input (include_input=True)
    assert np.abs(out[:, :3] - x).max() < 1e-5


def test_swiglu_shapes():
    ff = Dinov2SwiGLUFFN(1536, 4096)
    out = ff(mx.zeros((2, 10, 1536)))
    assert out.shape == (2, 10, 1536)

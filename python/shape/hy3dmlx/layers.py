"""Shared MLX primitives, ported 1:1 from the torch reference.

Module attribute names mirror the torch modules exactly so a converted
safetensors checkpoint loads by name with no remapping (except the DINO patch
Conv2d, handled in convert.py). Norms compute internally in fp32 for parity.
"""
import math

import mlx.core as mx
import mlx.nn as nn

SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)


def gelu_tanh(x: mx.array) -> mx.array:
    """torch nn.GELU(approximate='tanh') — used by the DiT MLPs."""
    xf = x.astype(mx.float32)
    inner = SQRT_2_OVER_PI * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + mx.tanh(inner))).astype(x.dtype)


def gelu_erf(x: mx.array) -> mx.array:
    """torch nn.GELU() exact (erf) — used by the VAE MLPs."""
    xf = x.astype(mx.float32)
    return (xf * 0.5 * (1.0 + mx.erf(xf / math.sqrt(2.0)))).astype(x.dtype)


class LayerNorm(nn.Module):
    """fp32-internal LayerNorm. affine=False matches the DiT's elementwise_affine=False."""

    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        xf = x.astype(mx.float32)
        mu = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.var(xf, axis=-1, keepdims=True)  # population variance (ddof=0), matches torch
        xf = (xf - mu) * mx.rsqrt(var + self.eps)
        if self.affine:
            xf = xf * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return xf.astype(dt)


class RMSNorm(nn.Module):
    """torch RMSNorm from hunyuan3ddit.py: scale applied after cast-back."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        xf = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * rrms).astype(dt) * self.scale


def sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float | None = None) -> mx.array:
    """Scaled dot-product attention. q,k,v: [B, H, L, D] -> [B, H, L, D]."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def timestep_embedding(t: mx.array, dim: int = 256, max_period: int = 10000,
                       time_factor: float = 1000.0) -> mx.array:
    """Sinusoidal timestep embedding (cos||sin), matching hunyuan3ddit.timestep_embedding.

    t: [B] fractional in [0,1]. Returns [B, dim] (dim even -> no pad).
    """
    t = time_factor * t.astype(mx.float32)
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
    return emb


class MLPEmbedder(nn.Module):
    """DiT time_in / guidance_in: Linear -> SiLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class FourierEmbedder(nn.Module):
    """VAE positional embedding of query points. num_freqs=8, include_pi=False, include_input=True.

    out_dim = input_dim * (num_freqs*2 + 1) = 3 * 17 = 51.
    """

    def __init__(self, num_freqs: int = 8, include_pi: bool = False,
                 input_dim: int = 3, include_input: bool = True, logspace: bool = True):
        super().__init__()
        if logspace:
            freqs = 2.0 ** mx.arange(num_freqs, dtype=mx.float32)
        else:
            freqs = mx.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs).astype(mx.float32)
        if include_pi:
            freqs = freqs * math.pi
        self._frequencies = freqs  # leading underscore -> not a registered parameter
        self.num_freqs = num_freqs
        self.include_input = include_input
        temp = 1 if include_input or num_freqs == 0 else 0
        self.out_dim = input_dim * (num_freqs * 2 + temp)

    def __call__(self, x: mx.array) -> mx.array:
        if self.num_freqs == 0:
            return x
        embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
        if self.include_input:
            return mx.concatenate([x, mx.sin(embed), mx.cos(embed)], axis=-1)
        return mx.concatenate([mx.sin(embed), mx.cos(embed)], axis=-1)

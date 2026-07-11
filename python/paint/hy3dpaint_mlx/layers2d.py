"""Shared 2D primitives for the SD2.1 VAE + UNet, in MLX (NHWC).

Parity-critical conventions (matched to diffusers / torch):
  * NHWC tensors. Conv weights stored MLX-layout [out, kH, kW, in]; torch [out, in, kH, kW]
    is transposed (0,2,3,1) at load. Linear weights [out, in] are identical.
  * GroupNorm computes in float32 internally (torch GroupNorm is fp32-internal), groups are
    consecutive channels, normalization over (spatial + within-group channels).
  * SiLU nonlinearity (diffusers default).

These gate per-primitive against diffusers' own modules (cross-venv .npy dump). See
oracle/layers_oracle.py + tests/test_layers2d_parity.py.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def silu(x):
    return x * mx.sigmoid(x)


class GroupNorm2d(nn.Module):
    """fp32-internal GroupNorm for NHWC tensors, matching torch.nn.GroupNorm (NCHW)."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x):  # x: [N, H, W, C]
        N, H, W, C = x.shape
        g = self.num_groups
        odt = x.dtype
        xf = x.astype(mx.float32).reshape(N, H, W, g, C // g)
        mean = xf.mean(axis=(1, 2, 4), keepdims=True)
        var = xf.var(axis=(1, 2, 4), keepdims=True)
        xf = (xf - mean) * mx.rsqrt(var + self.eps)
        xn = xf.reshape(N, H, W, C)
        out = xn * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return out.astype(odt)


class ResnetBlock2D(nn.Module):
    """diffusers ResnetBlock2D: norm1→silu→conv1 (+temb)→norm2→silu→conv2 (+skip)."""

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int | None = None,
                 groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = GroupNorm2d(groups, in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels) if temb_channels else None
        self.norm2 = GroupNorm2d(groups, out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                              if in_channels != out_channels else None)

    def __call__(self, x, temb=None):
        h = self.conv1(silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            h = h + self.time_emb_proj(silu(temb))[:, None, None, :]
        h = self.conv2(silu(self.norm2(h)))
        skip = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return h + skip


class Downsample2D(nn.Module):
    """diffusers Downsample2D(use_conv=True), stride-2 3x3 conv.

    padding=1 (UNet, symmetric conv padding) vs padding=0 (VAE, manual asym (0,1,0,1) pad).
    """

    def __init__(self, channels: int, out_channels: int | None = None, padding: int = 1):
        super().__init__()
        out_channels = out_channels or channels
        self.padding = padding
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=padding)

    def __call__(self, x):
        if self.padding == 0:  # VAE: asymmetric pad H,W by (0,1)
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample2D(nn.Module):
    """diffusers Upsample2D(use_conv=True): nearest x2 then 3x3 conv."""

    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):  # NHWC nearest-neighbor upsample x2
        N, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (N, H, 2, W, 2, C)).reshape(N, H * 2, W * 2, C)
        return self.conv(x)


class VAEAttnBlock(nn.Module):
    """Single-head spatial self-attention used in the VAE mid block (diffusers Attention)."""

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = GroupNorm2d(groups, channels, eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x):  # [N,H,W,C]
        N, H, W, C = x.shape
        h = self.group_norm(x).reshape(N, H * W, C)
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        scale = 1.0 / (C ** 0.5)
        attn = mx.softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)
        out = attn @ v
        out = self.to_out(out).reshape(N, H, W, C)
        return x + out

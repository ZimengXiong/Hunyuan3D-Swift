"""Attention / Transformer2D / time-embedding primitives for the SD2.1 UNet (MLX, NHWC).

Mirrors diffusers names: Transformer2DModel(norm, proj_in, transformer_blocks, proj_out);
BasicTransformerBlock(norm1, attn1, norm2, attn2, norm3, ff); Attention(to_q,to_k,to_v,to_out.0);
FeedForward(net.0.proj GEGLU, net.2). use_linear_projection=True (SD2.1) -> proj_in/out are Linear.
"""

from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn

from .layers2d import GroupNorm2d


def timestep_embedding(t, dim, flip_sin_to_cos=True, downscale_freq_shift=0.0, max_period=10000):
    """diffusers get_timestep_embedding. t: [N] -> [N, dim]."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / (half - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = t.astype(mx.float32)[:, None] * emb[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class Attention(nn.Module):
    """Multi-head attention (self if context is None, else cross). diffusers Attention."""

    def __init__(self, query_dim, heads, dim_head, context_dim=None, bias=False, out_bias=True):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=bias)
        self.to_k = nn.Linear(context_dim, inner, bias=bias)
        self.to_v = nn.Linear(context_dim, inner, bias=bias)
        self.to_out = [nn.Linear(inner, query_dim, bias=out_bias)]

    def __call__(self, x, context=None):
        context = x if context is None else context
        B, N, _ = x.shape
        M = context.shape[1]
        H, D = self.heads, self.dim_head
        q = self.to_q(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, M, H, D).transpose(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, M, H, D).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        return self.to_out[0](out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x):
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = dim * mult
        # net.0 = GEGLU, net.1 = Dropout (no params), net.2 = Linear
        self.net = [GEGLU(dim, inner), nn.Identity(), nn.Linear(inner, dim)]

    def __call__(self, x):
        return self.net[2](self.net[0](x))


class BasicTransformerBlock(nn.Module):
    """SD2.1 block; optionally 2.5D (paint): adds reference-attn (RA) + multiview-attn (MA)
    inserted between self-attn (attn1) and cross-attn (attn2), matching Basic2p5DTransformerBlock.
    When `xattn` is None it is exactly the base SD2.1 block."""

    def __init__(self, dim, heads, dim_head, context_dim, eps=1e-5,
                 use_ma=False, use_ra=False, layer_name=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn1 = Attention(dim, heads, dim_head, context_dim=None, bias=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn2 = Attention(dim, heads, dim_head, context_dim=context_dim, bias=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ff = FeedForward(dim)
        self.use_ma, self.use_ra, self.layer_name = use_ma, use_ra, layer_name
        if use_ma:
            self.attn_multiview = Attention(dim, heads, dim_head, context_dim=None, bias=False)
        if use_ra:
            self.attn_refview = Attention(dim, heads, dim_head, context_dim=None, bias=False)

    def __call__(self, x, context=None, xattn=None):
        norm_h = self.norm1(x)
        h = self.attn1(norm_h) + x
        if xattn is not None:
            mode = xattn.get("mode", "")
            n = xattn.get("num_in_batch", 1)
            ced = xattn.get("condition_embed_dict")
            B, L, C = x.shape[0] // n, x.shape[1], x.shape[2]
            if "w" in mode and ced is not None:
                ced[self.layer_name] = norm_h.reshape(B, n * L, C)
            if "r" in mode and self.use_ra and ced is not None:
                ce = ced[self.layer_name]                       # [B, Nref*L, C]
                Lr = ce.shape[1]
                ce = mx.broadcast_to(ce[:, None, :, :], (B, n, Lr, C)).reshape(B * n, Lr, C)
                h = xattn.get("ref_scale", 1.0) * self.attn_refview(norm_h, context=ce) + h
            if n > 1 and self.use_ma:
                mv = norm_h.reshape(B, n * L, C)
                out = self.attn_multiview(mv, context=mv).reshape(B * n, L, C)
                h = xattn.get("mva_scale", 1.0) * out + h
        h = self.attn2(self.norm2(h), context=context) + h
        h = self.ff(self.norm3(h)) + h
        return h


class Transformer2DModel(nn.Module):
    """SD2.1 Transformer2D (use_linear_projection=True). NHWC in/out. Optionally 2.5D."""

    def __init__(self, in_channels, num_heads, dim_head, context_dim, depth=1, groups=32, eps=1e-6,
                 use_ma=False, use_ra=False, layer_name=None, pbr=None):
        super().__init__()
        inner = num_heads * dim_head
        self.norm = GroupNorm2d(groups, in_channels, eps)
        self.proj_in = nn.Linear(in_channels, inner)
        if pbr is not None:
            from .attention_pbr import PBRTransformerBlock
            self.transformer_blocks = [
                PBRTransformerBlock(inner, num_heads, dim_head, context_dim, layer_name=layer_name, **pbr)
                for _ in range(depth)
            ]
        else:
            self.transformer_blocks = [
                BasicTransformerBlock(inner, num_heads, dim_head, context_dim,
                                      use_ma=use_ma, use_ra=use_ra, layer_name=layer_name)
                for _ in range(depth)
            ]
        self.proj_out = nn.Linear(inner, in_channels)

    def __call__(self, x, context=None, xattn=None):  # x: [N,H,W,C]
        N, H, W, C = x.shape
        residual = x
        h = self.norm(x).reshape(N, H * W, C)
        h = self.proj_in(h)
        for blk in self.transformer_blocks:
            h = blk(h, context=context, xattn=xattn)
        h = self.proj_out(h).reshape(N, H, W, C)
        return h + residual

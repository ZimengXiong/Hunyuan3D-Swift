"""HunYuanDiTPlain + MoE — the Hunyuan3D-2 / 2.1 (v2-1) denoiser.

U-Net-style DiT: a time token is prepended to the latent sequence; each of the
`depth` blocks does norm1->self-attn->add, norm2->cross-attn(cond)->add,
norm3->(MoE or MLP)->add; second-half blocks take a LIFO skip connection from
first-half blocks. No adaLN modulation, no positional embedding, no attn pooling.

Ported from reference/.../denoisers/{hunyuandit,moe_layers}.py, cross-checked
against cbun's mlx_hunyuandit.py. Attribute names mirror the torch modules so the
checkpoint loads 1:1. v2-1 config: hidden 2048, depth 21, heads 16 (head_dim 128),
ctx 1024, RMS qk_norm, MoE in the last 6 layers (8 experts, top-2, + shared).

Notable layout quirk (replicated exactly): self/cross attention concatenates
q,k,v then reshapes via view(1, -1, heads, head_dim*K) before splitting — a fixed
permutation the weights were trained against. Do not "fix" it.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from ..layers import LayerNorm, gelu_erf, sdpa


class RMSNormW(nn.Module):
    """torch nn.RMSNorm: param named `weight`, fp32-internal, eps 1e-6."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        dt = x.dtype
        xf = x.astype(mx.float32)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * self.weight.astype(mx.float32)).astype(dt)


def _timesteps(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    """torch Timesteps: [sin, cos], no time_factor (t already in [0,1])."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    emb = t[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # mlp = Sequential(Linear, GELU, Linear); keep index gap for the GELU (no params)
        self.mlp = [nn.Linear(hidden_size, freq_size, bias=True), None,
                    nn.Linear(freq_size, hidden_size, bias=True)]

    def __call__(self, t):
        tf = _timesteps(t, self.hidden_size).astype(self.mlp[0].weight.dtype)
        x = self.mlp[2](gelu_erf(self.mlp[0](tf)))
        return x[:, None, :]  # [B, 1, hidden]


def _heads_qkv(qkv: mx.array, heads: int, head_dim: int, nsplit: int):
    """Replicate torch's concat->view(1,-1,heads,head_dim*nsplit)->split exactly."""
    parts = mx.split(mx.reshape(qkv, (1, -1, heads, head_dim * nsplit)), nsplit, axis=-1)
    return parts  # each [1, B*N, heads, head_dim]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNormW(self.head_dim) if qk_norm else (lambda x: x)
        self.k_norm = RMSNormW(self.head_dim) if qk_norm else (lambda x: x)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x):
        B, N, _ = x.shape
        qkv = mx.concatenate([self.to_q(x), self.to_k(x), self.to_v(x)], axis=-1)
        q, k, v = _heads_qkv(qkv, self.heads, self.head_dim, 3)
        q = self.q_norm(mx.reshape(q, (B, N, self.heads, self.head_dim)))
        k = self.k_norm(mx.reshape(k, (B, N, self.heads, self.head_dim)))
        v = mx.reshape(v, (B, N, self.heads, self.head_dim))
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        out = sdpa(q, k, v, scale=self.head_dim ** -0.5).transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, qdim: int, kdim: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = qdim // heads
        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.q_norm = RMSNormW(self.head_dim) if qk_norm else (lambda x: x)
        self.k_norm = RMSNormW(self.head_dim) if qk_norm else (lambda x: x)
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

    def __call__(self, x, y):
        B, s1, _ = x.shape
        s2 = y.shape[1]
        q = self.to_q(x)
        kv = mx.concatenate([self.to_k(y), self.to_v(y)], axis=-1)
        k, v = _heads_qkv(kv, self.heads, self.head_dim, 2)
        q = self.q_norm(mx.reshape(q, (B, s1, self.heads, self.head_dim)))
        k = self.k_norm(mx.reshape(k, (B, s2, self.heads, self.head_dim)))
        v = mx.reshape(v, (B, s2, self.heads, self.head_dim))
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        out = sdpa(q, k, v, scale=self.head_dim ** -0.5).transpose(0, 2, 1, 3).reshape(B, s1, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x):
        return self.fc2(gelu_erf(self.fc1(x)))


class _Proj(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner)


class _FeedForward(nn.Module):
    """diffusers FeedForward(activation='gelu'): net = [GELU(proj), Dropout, Linear].

    Keys read net.0.proj.{weight,bias} / net.2.{weight,bias}.
    """

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.net = [_Proj(dim, inner), None, nn.Linear(inner, dim)]

    def __call__(self, x):
        return self.net[2](gelu_erf(self.net[0].proj(x)))


class MoEBlock(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = _Gate(dim, num_experts)
        self.experts = [_FeedForward(dim, dim * 4) for _ in range(num_experts)]
        self.shared_experts = _FeedForward(dim, dim * 4)

    def __call__(self, x):
        orig = x.shape
        flat = mx.reshape(x, (-1, orig[-1]))
        scores = mx.softmax(self.gate(flat), axis=-1)            # [T, E]
        # top-k mask-weighting (dense): correct, matches torch routing for top_k<<E
        kth = self.num_experts - self.top_k
        idx = mx.argpartition(scores, kth=kth, axis=-1)[:, kth:]  # [T, top_k]
        topw = mx.take_along_axis(scores, idx, axis=-1)
        y = mx.zeros_like(flat)
        for e in range(self.num_experts):
            mask = mx.sum(topw * (idx == e).astype(scores.dtype), axis=-1, keepdims=True)
            y = y + self.experts[e](flat) * mask.astype(flat.dtype)
        y = y + self.shared_experts(flat)
        return mx.reshape(y, orig)


class _Gate(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, dim))  # F.linear(x, weight) -> x @ weight.T

    def __call__(self, x):
        return x @ self.weight.T


class HunYuanDiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ctx: int, qkv_bias: bool, qk_norm: bool,
                 skip: bool, use_moe: bool, num_experts: int, top_k: int):
        super().__init__()
        self.skip = skip
        self.use_moe = use_moe
        self.norm1 = LayerNorm(hidden, eps=1e-6, affine=True)
        self.attn1 = SelfAttention(hidden, heads, qkv_bias, qk_norm)
        self.norm2 = LayerNorm(hidden, eps=1e-6, affine=True)
        self.attn2 = CrossAttention(hidden, ctx, heads, qkv_bias, qk_norm)
        self.norm3 = LayerNorm(hidden, eps=1e-6, affine=True)
        if skip:
            self.skip_norm = LayerNorm(hidden, eps=1e-6, affine=True)
            self.skip_linear = nn.Linear(2 * hidden, hidden)
        if use_moe:
            self.moe = MoEBlock(hidden, num_experts, top_k)
        else:
            self.mlp = MLP(hidden)

    def __call__(self, x, cond, skip_value=None):
        if self.skip:
            x = self.skip_norm(self.skip_linear(mx.concatenate([skip_value, x], axis=-1)))
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), cond)
        h = self.norm3(x)
        x = x + (self.moe(h) if self.use_moe else self.mlp(h))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden, eps=1e-6, affine=True)
        self.linear = nn.Linear(hidden, out_channels, bias=True)

    def __call__(self, x):
        return self.linear(self.norm_final(x)[:, 1:])


class HunYuanDiTPlain(nn.Module):
    def __init__(self, in_channels=64, hidden_size=2048, context_dim=1024, depth=21,
                 num_heads=16, qk_norm=True, qkv_bias=False, num_moe_layers=6,
                 num_experts=8, moe_top_k=2, **kwargs):
        super().__init__()
        self.depth = depth
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)
        self.blocks = [
            HunYuanDiTBlock(
                hidden_size, num_heads, context_dim, qkv_bias, qk_norm,
                skip=(layer > depth // 2),
                use_moe=(depth - layer <= num_moe_layers),
                num_experts=num_experts, top_k=moe_top_k,
            )
            for layer in range(depth)
        ]
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def __call__(self, x, t, cond, guidance=None):
        c = self.t_embedder(t)                       # [B,1,hidden]
        x = self.x_embedder(x)                       # [B,L,hidden]
        x = mx.concatenate([c, x], axis=1)           # prepend time token
        skips = []
        for layer, block in enumerate(self.blocks):
            sv = None if layer <= self.depth // 2 else skips.pop()
            x = block(x, cond, skip_value=sv)
            if layer < self.depth // 2:
                skips.append(x)
        return self.final_layer(x)

"""PBR-specific attention pieces for the 2.1 paint model (MLX):
3D-voxel PoseRoPE, material self-attention (MDA), reference attention (RA with per-material V).
Mirrors reference attn_processor.py (RotaryEmbedding, SelfAttnProcessor2_0, RefAttnProcessor2_0).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention


# ---------------- 3D PoseRoPE ----------------

def get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2] / dim))
    freqs = mx.outer(pos.astype(mx.float32), freqs)          # [P, dim/2]
    cos = mx.repeat(mx.cos(freqs), 2, axis=1)                # repeat_interleave(2)
    sin = mx.repeat(mx.sin(freqs), 2, axis=1)
    return cos, sin


def get_3d_rotary_pos_embed(position, embed_dim, voxel_resolution, theta=10000):
    """position [..., 3] int voxel indices -> (cos, sin) each [..., embed_dim]."""
    dim_xy = embed_dim // 8 * 3
    dim_z = embed_dim // 8 * 2
    grid = mx.arange(voxel_resolution, dtype=mx.float32)
    xy_cos, xy_sin = get_1d_rotary_pos_embed(dim_xy, grid, theta)
    z_cos, z_sin = get_1d_rotary_pos_embed(dim_z, grid, theta)
    flat = position.reshape(-1, position.shape[-1]).astype(mx.int32)
    cos = mx.concatenate([xy_cos[flat[:, 0]], xy_cos[flat[:, 1]], z_cos[flat[:, 2]]], axis=-1)
    sin = mx.concatenate([xy_sin[flat[:, 0]], xy_sin[flat[:, 1]], z_sin[flat[:, 2]]], axis=-1)
    cos = cos.reshape(*position.shape[:-1], embed_dim)
    sin = sin.reshape(*position.shape[:-1], embed_dim)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """x [B, heads, L, head_dim]; cos/sin [B, L, head_dim]."""
    cos = cos[:, None]
    sin = sin[:, None]
    x2 = x.reshape(*x.shape[:-1], -1, 2)
    real = x2[..., 0]
    imag = x2[..., 1]
    rot = mx.stack([-imag, real], axis=-1).reshape(x.shape)
    return x * cos + rot * sin


# ---------------- material / reference attention + PBR block ----------------

from .attention import Attention, FeedForward  # noqa: E402
from .layers2d import GroupNorm2d  # noqa: E402


def _sdpa(q, k, v, heads, scale):
    """q,k,v: [B, L, inner] -> attention -> [B, L, heads*headdim_v]."""
    B, Lq, _ = q.shape
    Lk = k.shape[1]
    hd = q.shape[-1] // heads
    hdv = v.shape[-1] // heads
    qh = q.reshape(B, Lq, heads, hd).transpose(0, 2, 1, 3)
    kh = k.reshape(B, Lk, heads, hd).transpose(0, 2, 1, 3)
    vh = v.reshape(B, Lk, heads, hdv).transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=scale)
    return o.transpose(0, 2, 1, 3).reshape(B, Lq, heads * hdv), hdv


class MaterialProc(nn.Module):
    """Per-material (mr) linears attached as attn.processor, matching reference key names."""

    def __init__(self, dim, inner, kinds):
        super().__init__()
        if "q" in kinds:
            self.to_q_mr = nn.Linear(dim, inner, bias=False)
            self.to_k_mr = nn.Linear(dim, inner, bias=False)
        if "v" in kinds:
            self.to_v_mr = nn.Linear(dim, inner, bias=False)
        if "out" in kinds:
            self.to_out_mr = [nn.Linear(inner, dim, bias=True)]


class PBRTransformerBlock(nn.Module):
    """2.1 PBR block: base self/cross attn + material self-attn (MDA), reference attn (RA, per-material V),
    multiview attn (MA, 3D PoseRoPE), DINO cross-attn. Base parts live under `.transformer` (key match)."""

    def __init__(self, dim, heads, dim_head, context_dim, use_ma=False, use_ra=False,
                 use_mda=False, use_dino=False, layer_name=None):
        super().__init__()
        from .attention import BasicTransformerBlock
        self.transformer = BasicTransformerBlock(dim, heads, dim_head, context_dim)  # norm1/attn1/norm2/attn2/norm3/ff
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.layer_name = layer_name
        self.use_ma, self.use_ra, self.use_mda, self.use_dino = use_ma, use_ra, use_mda, use_dino
        inner = heads * dim_head
        if use_mda:
            self.transformer.attn1.processor = MaterialProc(dim, inner, ["q", "v", "out"])
        if use_ma:
            self.attn_multiview = Attention(dim, heads, dim_head, context_dim=None, bias=False)
        if use_ra:
            self.attn_refview = Attention(dim, heads, dim_head, context_dim=None, bias=False)
            self.attn_refview.processor = MaterialProc(dim, inner, ["v", "out"])
        if use_dino:
            self.attn_dino = Attention(dim, heads, dim_head, context_dim=context_dim, bias=False)

    def _self_attn_material(self, norm_h, n, n_pbr):
        """MDA: norm_h [(b n_pbr n), l, c]; per-material self-attn -> same shape."""
        a = self.transformer.attn1
        proc = a.processor
        Bt, l, c = norm_h.shape
        b = Bt // (n_pbr * n)
        x = norm_h.reshape(b, n_pbr, n, l, c)
        outs = []
        for mi in range(n_pbr):
            hs = x[:, mi].reshape(b * n, l, c)
            if mi == 0:  # albedo -> base weights
                q, k, v = a.to_q(hs), a.to_k(hs), a.to_v(hs)
                o, _ = _sdpa(q, k, v, self.heads, self.scale)
                o = a.to_out[0](o)
            else:        # mr -> material weights
                q, k, v = proc.to_q_mr(hs), proc.to_k_mr(hs), proc.to_v_mr(hs)
                o, _ = _sdpa(q, k, v, self.heads, self.scale)
                o = proc.to_out_mr[0](o)
            outs.append(o.reshape(b, 1, n, l, c))
        return mx.concatenate(outs, axis=1).reshape(Bt, l, c)

    def _ref_attn(self, ref_norm, cond, n_pbr):
        """RA: query ref_norm [b, (n l), c] (albedo); key/value from cond [b, (nref l), c];
        per-material V -> per-material out -> stack [b, n_pbr, (n l), c]."""
        a = self.attn_refview
        proc = a.processor
        b = ref_norm.shape[0]
        q = a.to_q(ref_norm)
        k = a.to_k(cond)
        v_alb = a.to_v(cond)
        v_mr = proc.to_v_mr(cond)
        v = mx.concatenate([v_alb, v_mr], axis=-1)
        o, _ = _sdpa(q, k, v, self.heads, self.scale)   # [b, Lq, heads*(2*dim_head)]
        # split heads then per-material head_dim
        Lq = o.shape[1]
        oh = o.reshape(b, Lq, self.heads, 2 * self.dim_head)
        alb = oh[..., : self.dim_head].reshape(b, Lq, self.heads * self.dim_head)
        mrh = oh[..., self.dim_head:].reshape(b, Lq, self.heads * self.dim_head)
        out_alb = a.to_out[0](alb)
        out_mr = proc.to_out_mr[0](mrh)
        return mx.stack([out_alb, out_mr], axis=1)      # [b, n_pbr, Lq, c]

    def _mv_attn(self, mv, cos, sin):
        """MA: mv [(b n_pbr), (n l), c]; q/k/v + 3D RoPE on q,k."""
        a = self.attn_multiview
        B, L, _ = mv.shape
        q = a.to_q(mv).reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = a.to_k(mv).reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = a.to_v(mv).reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        if cos is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.dim_head)
        return a.to_out[0](o)

    def __call__(self, hidden, context, xattn):
        t = self.transformer
        mode = xattn.get("mode", "")
        n = xattn.get("num_in_batch", 1)
        n_pbr = xattn.get("n_pbr", 1)
        ced = xattn.get("condition_embed_dict")
        norm_h = t.norm1(hidden)
        # MDA (material self-attn) or plain self-attn (dual stream)
        if self.use_mda:
            attn_out = self._self_attn_material(norm_h, n, n_pbr)
        else:
            attn_out = t.attn1(norm_h)
        hidden = attn_out + hidden
        Bt, l, c = norm_h.shape
        b = Bt // (n_pbr * n)
        # 'w' write (dual ref pass): n_pbr is 1 here
        if "w" in mode and ced is not None:
            ced[self.layer_name] = norm_h.reshape(b, n * l, c)
        # RA
        if "r" in mode and self.use_ra and ced is not None:
            cond = ced[self.layer_name]                       # [b, (nref l), c]
            ref_norm = norm_h.reshape(b, n_pbr, n * l, c)[:, 0]   # albedo only
            ra = self._ref_attn(ref_norm, cond, n_pbr)        # [b, n_pbr, (n l), c]
            ra = ra.reshape(b, n_pbr, n, l, c).reshape(Bt, l, c)
            hidden = xattn.get("ref_scale", 1.0) * ra + hidden
        # MA (multiview + PoseRoPE)
        if n > 1 and self.use_ma:
            mv = norm_h.reshape(b, n_pbr, n, l, c).reshape(b * n_pbr, n * l, c)
            rbt = xattn.get("rope_by_tokens")
            cos, sin = (rbt.get(mv.shape[1], (None, None)) if rbt else xattn.get("rope", (None, None)))
            o = self._mv_attn(mv, cos, sin)
            o = o.reshape(b, n_pbr, n, l, c).reshape(Bt, l, c)
            hidden = xattn.get("mva_scale", 1.0) * o + hidden
        # text cross-attn
        norm_h2 = t.norm2(hidden)
        hidden = t.attn2(norm_h2, context=context) + hidden
        # DINO cross-attn
        if self.use_dino:
            dino = xattn.get("dino")
            dino_r = mx.broadcast_to(dino[:, None], (dino.shape[0], n_pbr * n, dino.shape[1], dino.shape[2]))
            dino_r = dino_r.reshape(dino.shape[0] * n_pbr * n, dino.shape[1], dino.shape[2])
            a = self.attn_dino
            q = a.to_q(norm_h2); k = a.to_k(dino_r); v = a.to_v(dino_r)
            o, _ = _sdpa(q, k, v, self.heads, self.scale)
            hidden = a.to_out[0](o) + hidden
        # FF
        hidden = t.ff(t.norm3(hidden)) + hidden
        return hidden

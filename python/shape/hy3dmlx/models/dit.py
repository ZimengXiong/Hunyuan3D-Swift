"""Hunyuan3DDiT — FLUX-style double/single stream, no RoPE (pe=None), no MoE.

Ported 1:1 from reference/.../denoisers/hunyuan3ddit.py. Attribute names match
the torch module so `model.*` checkpoint keys load with no remapping.
Config (2mini): in_ch 64, ctx_in 1536, hidden 1024, heads 16, depth 8,
depth_single_blocks 16, mlp_ratio 4, qkv_bias True, guidance_embed False.
"""
import mlx.core as mx
import mlx.nn as nn

from ..layers import LayerNorm, MLPEmbedder, RMSNorm, gelu_tanh, sdpa, timestep_embedding


def _split_qkv(qkv: mx.array, heads: int):
    # [B, L, 3*hidden] -> q,k,v each [B, heads, L, head_dim]  (einops "B L (K H D) -> K B H L D")
    B, L, _ = qkv.shape
    qkv = qkv.reshape(B, L, 3, heads, -1).transpose(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def _merge_heads(x: mx.array) -> mx.array:
    # [B, H, L, D] -> [B, L, H*D]
    B, H, L, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, H * D)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q, k):
        return self.query_norm(q), self.key_norm(k)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec: mx.array):
        out = self.lin(nn.silu(vec))[:, None, :]
        chunks = mx.split(out, self.multiplier, axis=-1)
        mod1 = (chunks[0], chunks[1], chunks[2])  # shift, scale, gate
        mod2 = (chunks[3], chunks[4], chunks[5]) if self.is_double else None
        return mod1, mod2


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.img_norm2 = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.img_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), None,
                        nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]  # idx 0,2 (1=GELU)

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.txt_norm2 = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), None,
                        nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

    def _mlp(self, mlp, x):
        return mlp[2](gelu_tanh(mlp[0](x)))

    def __call__(self, img, txt, vec):
        (img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2) = self.img_mod(vec)
        (txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2) = self.txt_mod(vec)

        img_mod = (1 + img_sc1) * self.img_norm1(img) + img_s1
        img_qkv = self.img_attn.qkv(img_mod)
        img_q, img_k, img_v = _split_qkv(img_qkv, self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        txt_mod = (1 + txt_sc1) * self.txt_norm1(txt) + txt_s1
        txt_qkv = self.txt_attn.qkv(txt_mod)
        txt_q, txt_k, txt_v = _split_qkv(txt_qkv, self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)
        attn = _merge_heads(sdpa(q, k, v))
        txt_len = txt.shape[1]
        txt_attn, img_attn = attn[:, :txt_len], attn[:, txt_len:]

        img = img + img_g1 * self.img_attn.proj(img_attn)
        img = img + img_g2 * self._mlp(self.img_mlp, (1 + img_sc2) * self.img_norm2(img) + img_s2)
        txt = txt + txt_g1 * self.txt_attn.proj(txt_attn)
        txt = txt + txt_g2 * self._mlp(self.txt_mlp, (1 + txt_sc2) * self.txt_norm2(txt) + txt_s2)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.pre_norm = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x, vec):
        (shift, scale, gate), _ = self.modulation(vec)
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        out = self.linear1(x_mod)
        qkv = out[..., : 3 * self.hidden_size]
        mlp = out[..., 3 * self.hidden_size:]
        q, k, v = _split_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k)
        attn = _merge_heads(sdpa(q, k, v))
        out = self.linear2(mx.concatenate([attn, gelu_tanh(mlp)], axis=2))
        return x + gate * out


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # adaLN_modulation = Sequential(SiLU, Linear); keep index 1 = Linear to match keys
        self.adaLN_modulation = [None, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

    def __call__(self, x, vec):
        mod = self.adaLN_modulation[1](nn.silu(vec))
        shift, scale = mx.split(mod, 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class Hunyuan3DDiT(nn.Module):
    def __init__(self, in_channels=64, context_in_dim=1536, hidden_size=1024, mlp_ratio=4.0,
                 num_heads=16, depth=8, depth_single_blocks=16, qkv_bias=True,
                 guidance_embed=False, time_factor=1000.0, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.time_factor = time_factor
        self.guidance_embed = guidance_embed

        self.latent_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.cond_in = nn.Linear(context_in_dim, hidden_size)
        if guidance_embed:
            self.guidance_in = MLPEmbedder(256, hidden_size)

        self.double_blocks = [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
                              for _ in range(depth)]
        self.single_blocks = [SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
                              for _ in range(depth_single_blocks)]
        self.final_layer = LastLayer(hidden_size, 1, in_channels)

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array,
                 guidance: mx.array | None = None) -> mx.array:
        """x: [B, L, in_ch]; t: [B] in [0,1]; cond: [B, Lc, ctx_in]."""
        latent = self.latent_in(x)
        # NOTE: the torch reference calls timestep_embedding(t, 256, self.time_factor)
        # positionally, so time_factor lands in the max_period slot -> effective
        # max_period == time_factor (1000), not the 10000 default. Replicate exactly.
        vec = self.time_in(timestep_embedding(t, 256, max_period=self.time_factor).astype(latent.dtype))
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256, max_period=self.time_factor).astype(latent.dtype))
        cond = self.cond_in(cond)

        for block in self.double_blocks:
            latent, cond = block(latent, cond, vec)

        latent = mx.concatenate([cond, latent], axis=1)
        for block in self.single_blocks:
            latent = block(latent, vec)
        latent = latent[:, cond.shape[1]:, ...]
        return self.final_layer(latent, vec)

"""ShapeVAE decoder + dense grid query (vecset / CLAY lineage).

Decode path only: post_kl -> transformer(16x self-attn) -> geo_decoder
(Fourier-embed query pts -> cross-attn -> SDF logit). Encoder/pre_kl skipped
(training-only, needs torch_cluster.fps). Attribute names mirror
reference/.../autoencoders/{model,attention_blocks}.py for 1:1 key loading.

Critical parity points (VERIFICATION.md):
  * latents are divided by scale_factor at decode ENTRY (done in pipeline._export).
  * geo_decoder.ln_post uses eps=1e-5 (torch LayerNorm default); all else 1e-6.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ..layers import FourierEmbedder, LayerNorm, gelu_erf, sdpa


def _near_surface_mask(grid: np.ndarray, alpha: float) -> np.ndarray:
    """Cells bracketing the iso-surface: sign changes vs a 6-neighbor, or |value| < 0.95.
    NaN cells (inactive from a previous octree level) are excluded. Returns bool mask.
    """
    g = grid + alpha
    valid = ~np.isnan(g)
    s = np.sign(np.where(valid, g, 0.0))
    near = np.zeros(g.shape, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3; lo[axis] = slice(0, -1)
        hi = [slice(None)] * 3; hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        change = (s[lo] != s[hi]) & valid[lo] & valid[hi]
        near[lo] |= change
        near[hi] |= change
    near |= (np.abs(np.where(valid, g, 1e9)) < 0.95)
    near &= valid
    return near


def _self_qkv(qkv: mx.array, heads: int, q_norm, k_norm):
    # qkv [bs, n, 3*width] -> per-head [bs,n,heads,3*hd], split q/k/v, norm, -> [bs,heads,n,hd]
    bs, n, _ = qkv.shape
    qkv = qkv.reshape(bs, n, heads, -1)
    hd = qkv.shape[-1] // 3
    q, k, v = qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:]
    q, k = q_norm(q), k_norm(k)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    return q, k, v


class MLP(nn.Module):
    def __init__(self, width: int, expand_ratio: int = 4, output_width: int | None = None):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width or width)

    def __call__(self, x):
        return self.c_proj(gelu_erf(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads: int, width: int, qk_norm: bool):
        super().__init__()
        self.heads = heads
        if qk_norm:
            self.q_norm = LayerNorm(width // heads, eps=1e-6, affine=True)
            self.k_norm = LayerNorm(width // heads, eps=1e-6, affine=True)
        else:
            self.q_norm = lambda x: x
            self.k_norm = lambda x: x

    def __call__(self, qkv):
        bs, n, _ = qkv.shape
        q, k, v = _self_qkv(qkv, self.heads, self.q_norm, self.k_norm)
        out = sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(bs, n, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, width: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, width, qk_norm)

    def __call__(self, x):
        return self.c_proj(self.attention(self.c_qkv(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.attn = MultiheadAttention(width, heads, qkv_bias, qk_norm)
        self.ln_1 = LayerNorm(width, eps=1e-6, affine=True)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width, eps=1e-6, affine=True)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.resblocks = [ResidualAttentionBlock(width, heads, qkv_bias, qk_norm)
                          for _ in range(layers)]

    def __call__(self, x):
        for blk in self.resblocks:
            x = blk(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, heads: int, width: int, qk_norm: bool):
        super().__init__()
        self.heads = heads
        if qk_norm:
            self.q_norm = LayerNorm(width // heads, eps=1e-6, affine=True)
            self.k_norm = LayerNorm(width // heads, eps=1e-6, affine=True)
        else:
            self.q_norm = lambda x: x
            self.k_norm = lambda x: x

    def __call__(self, q, kv):
        bs, n_ctx, _ = q.shape
        _, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.reshape(bs, n_ctx, self.heads, -1)
        kv = kv.reshape(bs, n_data, self.heads, -1)
        k, v = kv[..., :attn_ch], kv[..., attn_ch:]
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, qkv_bias: bool, qk_norm: bool, data_width=None):
        super().__init__()
        data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads, width, qk_norm)

    def __call__(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        return self.c_proj(self.attention(x, data))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_expand_ratio: int, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        self.attn = MultiheadCrossAttention(width, heads, qkv_bias, qk_norm)
        self.ln_1 = LayerNorm(width, eps=1e-6, affine=True)
        self.ln_2 = LayerNorm(width, eps=1e-6, affine=True)
        self.ln_3 = LayerNorm(width, eps=1e-6, affine=True)
        self.mlp = MLP(width, expand_ratio=mlp_expand_ratio)

    def __call__(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self, fourier_embedder: FourierEmbedder, out_channels: int, width: int,
                 heads: int, mlp_expand_ratio: int, enable_ln_post: bool, qkv_bias: bool,
                 qk_norm: bool):
        super().__init__()
        self.fourier_embedder = fourier_embedder
        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        self.enable_ln_post = enable_ln_post
        if not enable_ln_post:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width, heads, mlp_expand_ratio, qkv_bias, qk_norm)
        if enable_ln_post:
            self.ln_post = LayerNorm(width, eps=1e-5, affine=True)  # torch default eps — CRITICAL
        self.output_proj = nn.Linear(width, out_channels)

    def __call__(self, queries: mx.array, latents: mx.array) -> mx.array:
        qe = self.query_proj(self.fourier_embedder(queries).astype(latents.dtype))
        x = self.cross_attn_decoder(qe, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        return self.output_proj(x)


class ShapeVAE(nn.Module):
    """Decode-only ShapeVAE. scale_factor stored for the pipeline divide step."""

    def __init__(self, num_latents=512, embed_dim=64, width=1024, heads=16,
                 num_decoder_layers=16, num_freqs=8, include_pi=False, qkv_bias=False,
                 qk_norm=True, scale_factor=1.0188137142395404,
                 geo_decoder_mlp_expand_ratio=4, geo_decoder_ln_post=True, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(width, num_decoder_layers, heads, qkv_bias, qk_norm)
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder, out_channels=1, width=width, heads=heads,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio, enable_ln_post=geo_decoder_ln_post,
            qkv_bias=qkv_bias, qk_norm=qk_norm)

    def decode(self, latents: mx.array) -> mx.array:
        """latents [B, num_latents, embed_dim] (already divided by scale_factor) -> kv [B, N, width]."""
        return self.transformer(self.post_kl(latents))

    @staticmethod
    def _default_chunks(num_latents, num_chunks):
        if num_chunks is not None:
            return num_chunks
        return 8000 if num_latents <= 1024 else 100000

    @staticmethod
    def _bbox(bounds):
        if isinstance(bounds, (int, float)):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        return np.array(bounds[0:3], dtype=np.float32), np.array(bounds[3:6], dtype=np.float32)

    def _decode_points(self, kv: mx.array, pts: np.ndarray, num_chunks: int) -> np.ndarray:
        """Evaluate the geo-decoder SDF at arbitrary query points. pts [P,3] -> [P] (numpy)."""
        kv_dt = kv.dtype
        out = []
        for s in range(0, pts.shape[0], num_chunks):
            q = mx.array(pts[s:s + num_chunks][None], dtype=kv_dt)  # [1, c, 3]
            logits = self.geo_decoder(q, kv)
            mx.eval(logits)
            out.append(np.asarray(logits.astype(mx.float32)).reshape(-1))
        return np.concatenate(out) if out else np.zeros((0,), np.float32)

    def query_grid(self, kv: mx.array, bounds=1.01, octree_resolution=256, num_chunks=None):
        """Dense (R+1)^3 SDF grid. num_chunks tunes only GPU dispatch (output is identical)."""
        num_chunks = self._default_chunks(self.latent_shape[0], num_chunks)
        bbox_min, bbox_max = self._bbox(bounds)
        r = int(octree_resolution)
        ax = [np.linspace(bbox_min[i], bbox_max[i], r + 1, dtype=np.float32) for i in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        xyz = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        grid = self._decode_points(kv, xyz, num_chunks).reshape((r + 1, r + 1, r + 1))
        return grid, bbox_min, bbox_max, (r + 1, r + 1, r + 1)

    def query_grid_octree(self, kv: mx.array, bounds=1.01, octree_resolution=256,
                          num_chunks=None, mc_level=0.0, min_resolution=63):
        """Octree (FlashVDM-style) SDF grid: decode a coarse grid densely, then refine only the
        near-surface band level-by-level (x2). Inactive cells are NaN (masked at marching cubes).
        Returns the same (grid, bbox_min, bbox_max, grid_size) tuple as query_grid.
        Octree bookkeeping is numpy on the small grids; only the neural queries hit MLX.
        """
        from scipy import ndimage
        num_chunks = self._default_chunks(self.latent_shape[0], num_chunks)
        bbox_min, bbox_max = self._bbox(bounds)
        bbox_size = bbox_max - bbox_min

        res_list = []
        r = int(octree_resolution)
        while r >= min_resolution:
            res_list.append(r)
            r //= 2
        res_list = res_list[::-1] or [int(octree_resolution)]  # [coarse, ..., fine]

        # coarse dense decode
        r0 = res_list[0]
        ax = [np.linspace(bbox_min[i], bbox_max[i], r0 + 1, dtype=np.float32) for i in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        xyz = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        grid = self._decode_points(kv, xyz, num_chunks).reshape((r0 + 1,) * 3)

        struct = np.ones((3, 3, 3), dtype=bool)
        for res in res_list[1:]:
            gs = res + 1
            step = bbox_size / res
            near = _near_surface_mask(grid, mc_level)
            expand = 0 if res == res_list[-1] else 1
            for _ in range(expand):
                near = ndimage.binary_dilation(near, struct)
            cidx = np.argwhere(near) * 2  # upsample coarse->fine (x2)
            nxt = np.zeros((gs, gs, gs), dtype=bool)
            cidx = np.clip(cidx, 0, gs - 1)
            nxt[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = True
            for _ in range(2 - expand):
                nxt = ndimage.binary_dilation(nxt, struct)
            nidx = np.argwhere(nxt)  # active fine-grid voxels [P,3]
            pts = nidx.astype(np.float32) * step + bbox_min
            vals = self._decode_points(kv, pts, num_chunks)
            grid = np.full((gs, gs, gs), np.nan, dtype=np.float32)
            grid[nidx[:, 0], nidx[:, 1], nidx[:, 2]] = vals
        return grid, bbox_min, bbox_max, grid.shape

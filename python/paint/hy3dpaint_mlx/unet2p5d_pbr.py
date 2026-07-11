"""MLX UNet2p5DConditionModel — the large 2.1 PBR paint UNet (albedo + metallic-roughness).

Superset of the 2.0 stack: 12-ch conv_in (noise + VAE normal + VAE position), n_pbr=2 doubled
batch, material self-attn (MDA), reference attn with per-material V (RA), multiview attn with
3D-voxel PoseRoPE (MA), DINOv2 cross-attn, learned per-material text tokens. No camera class
embedding (PoseRoPE replaces it). Mirrors reference hunyuanpaintpbr UNet2p5DConditionModel.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .unet import UNet2DConditionModel
from .attention_pbr import get_3d_rotary_pos_embed


def compute_voxel_indices(position, grid_res, voxel_res):
    """position [b,n,H,W,3] in [0,1] -> voxel indices [b, n*grid_res*grid_res, 3] int32.
    Ports compute_discrete_voxel_indice (window-average downsample + quantize)."""
    # Emulate the reference's fp16 arithmetic (it casts position.half() and computes in fp16),
    # quantizing to fp16 after each op so voxel indices match at rounding boundaries.
    f16 = np.float16
    position = position.astype(f16)
    b, n, H, W, c = position.shape
    gh, gw = H // grid_res, W // grid_res
    valid = np.all(position != 1, axis=-1)              # [b,n,H,W]
    pos = np.where(valid[..., None], position, f16(0)).astype(f16)
    pos = pos.reshape(b, n, grid_res, gh, grid_res, gw, c)
    val = valid.reshape(b, n, grid_res, gh, grid_res, gw)
    grid_pos = pos.sum(axis=(3, 5), dtype=f16)          # fp16 sum
    count = val.sum(axis=(3, 5))
    grid_pos = (grid_pos / np.clip(count, 1, None)[..., None].astype(f16)).astype(f16)
    thres = (gh * gw) // 16
    grid_pos[count < thres] = 0
    grid_pos = np.clip(grid_pos, f16(0), f16(1)).astype(f16)
    vox = np.round((grid_pos * f16(voxel_res - 1)).astype(f16)).astype(np.int32)
    return vox.reshape(b, n * grid_res * grid_res, 3)


class ImageProjModel(nn.Module):
    """DINO feature projector: [B,N,1536] -> [B, N*4, 1024]. Mirrors reference ImageProjModel."""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1536, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(self, image_embeds):
        if image_embeds.ndim == 3:
            num_token = image_embeds.shape[1]
            embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        else:
            num_token = 1
            embeds = image_embeds
        tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        tokens = self.norm(tokens)
        b = embeds.shape[0] // num_token
        return tokens.reshape(b, num_token * self.clip_extra_context_tokens, self.cross_attention_dim)


_PBR_ON = {"use_ma": True, "use_ra": True, "use_mda": True, "use_dino": True}
_PBR_OFF = {"use_ma": False, "use_ra": False, "use_mda": False, "use_dino": False}


class UNet2p5DPBRConditionModel(nn.Module):
    def __init__(self, base_cfg: dict, pbr_setting=("albedo", "mr")):
        super().__init__()
        cfg = dict(
            block_out_channels=tuple(base_cfg.get("block_out_channels", (320, 640, 1280, 1280))),
            layers_per_block=base_cfg.get("layers_per_block", 2),
            cross_attention_dim=base_cfg.get("cross_attention_dim", 1024),
            attention_head_dim=tuple(base_cfg.get("attention_head_dim", (5, 10, 20, 20))),
            norm_num_groups=base_cfg.get("norm_num_groups", 32),
        )
        self.n_pbr = len(pbr_setting)
        self.unet = UNet2DConditionModel(**cfg, conv_in_channels=12, pbr=_PBR_ON)
        self.unet_dual = UNet2DConditionModel(**cfg, conv_in_channels=4, pbr=_PBR_OFF)
        self.unet.learned_text_clip_albedo = mx.zeros((77, 1024))
        self.unet.learned_text_clip_mr = mx.zeros((77, 1024))
        self.unet.learned_text_clip_ref = mx.zeros((77, 1024))
        self.unet.image_proj_model_dino = ImageProjModel()

    def _rope_by_tokens(self, position_maps_np, H_lat, n_gen):
        """Precompute (cos,sin) per multiview token-count from pixel position maps."""
        grids = [H_lat, H_lat // 2, H_lat // 4, H_lat // 8]
        vres = [H_lat * 8, H_lat * 4, H_lat * 2, H_lat]
        out = {}
        for g, vr in zip(grids, vres):
            vox = compute_voxel_indices(position_maps_np, g, vr)        # [b, n_gen*g*g, 3]
            vox = np.repeat(vox[:, None], self.n_pbr, axis=1).reshape(vox.shape[0] * self.n_pbr, vox.shape[1], 3)
            cos, sin = get_3d_rotary_pos_embed(mx.array(vox), 64, vr)
            out[n_gen * g * g] = (cos, sin)
        return out

    def prepare(self, ref_lat, dino_hidden_states, position_maps_np, H, N_gen):
        """Compute the diffusion-step-invariant conditioning once: dual-stream reference features,
        DINO projection, and PoseRoPE tables (all constant across steps)."""
        B, N_ref = ref_lat.shape[0], ref_lat.shape[1]
        rl = ref_lat.reshape(B * N_ref, H, ref_lat.shape[3], 4)
        ref_text = mx.broadcast_to(self.unet.learned_text_clip_ref[None], (B * N_ref, 77, 1024))
        ced = {}
        self.unet_dual(rl, mx.zeros((B * N_ref,)), ref_text, class_labels=None,
                       cross_attention_kwargs={"mode": "w", "num_in_batch": N_ref, "n_pbr": 1,
                                               "condition_embed_dict": ced})
        dino_tok = self.unet.image_proj_model_dino(dino_hidden_states)
        rope = self._rope_by_tokens(position_maps_np, H, N_gen)
        mx.eval(list(ced.values()) + [dino_tok])
        return {"condition_embed_dict": ced, "dino": dino_tok, "rope": rope}

    def __call__(self, sample, timestep, normal_lat, position_lat, ref_lat,
                 dino_hidden_states, position_maps_np, mva_scale=1.0, ref_scale=1.0, cond=None):
        # sample: [B, N_pbr, N_gen, H, W, 4]; normal_lat/position_lat: [B, N_gen, H, W, 4] (repeated over n_pbr)
        B, N_pbr, N_gen, H, W, _ = sample.shape
        nrep = mx.broadcast_to(normal_lat[:, None], (B, N_pbr, N_gen, H, W, 4))
        prep = mx.broadcast_to(position_lat[:, None], (B, N_pbr, N_gen, H, W, 4))
        s = mx.concatenate([sample, nrep, prep], axis=-1).reshape(B * N_pbr * N_gen, H, W, 12)

        alb, mr = self.unet.learned_text_clip_albedo, self.unet.learned_text_clip_mr
        ehs = mx.stack([alb, mr], axis=0)[None]                          # [1, N_pbr, 77, 1024]
        ehs = mx.broadcast_to(ehs[:, :, None], (B, N_pbr, N_gen, 77, 1024)).reshape(B * N_pbr * N_gen, 77, 1024)

        if cond is None:
            cond = self.prepare(ref_lat, dino_hidden_states, position_maps_np, H, N_gen)
        ced = cond["condition_embed_dict"] if ref_scale != 0 else None

        out = self.unet(s, timestep, ehs, class_labels=None,
                        cross_attention_kwargs={"mode": "r", "num_in_batch": N_gen, "n_pbr": N_pbr,
                                                "condition_embed_dict": ced, "dino": cond["dino"],
                                                "rope": (None, None), "rope_by_tokens": cond["rope"],
                                                "mva_scale": mva_scale, "ref_scale": ref_scale})
        return out.reshape(B, N_pbr, N_gen, H, W, 4)

"""MLX UNet2p5DConditionModel — the multiview 2.5D paint UNet (small / 2.0 RGB variant).

Wraps two base UNets:
  * `unet`      : 12-ch conv_in (noise + VAE normal + VAE position), MA + RA, camera class-embedding.
  * `unet_dual` : 4-ch conv_in, base only; runs once in mode 'w' to write per-layer reference features
                  (condition_embed_dict) that the main UNet's reference-attention reads in mode 'r'.
Mirrors reference hy3dpaint hunyuanpaint UNet2p5DConditionModel (keys: unet.*, unet_dual.*,
unet.learned_text_clip_{gen,ref}, unet.class_embedding).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .unet import UNet2DConditionModel


class UNet2p5DConditionModel(nn.Module):
    def __init__(self, base_cfg: dict, max_num_ref: int = 5, max_num_gen: int = 44):
        super().__init__()
        cfg = dict(
            in_channels=base_cfg.get("in_channels", 4),
            out_channels=base_cfg.get("out_channels", 4),
            block_out_channels=tuple(base_cfg.get("block_out_channels", (320, 640, 1280, 1280))),
            layers_per_block=base_cfg.get("layers_per_block", 2),
            cross_attention_dim=base_cfg.get("cross_attention_dim", 1024),
            attention_head_dim=tuple(base_cfg.get("attention_head_dim", (5, 10, 20, 20))),
            norm_num_groups=base_cfg.get("norm_num_groups", 32),
        )
        self.max_num_ref = max_num_ref
        self.unet = UNet2DConditionModel(**cfg, conv_in_channels=12,
                                         num_class_embeds=max_num_ref + max_num_gen,
                                         use_ma=True, use_ra=True)
        self.unet_dual = UNet2DConditionModel(**cfg, conv_in_channels=4,
                                              num_class_embeds=None, use_ma=False, use_ra=False)
        self.unet.learned_text_clip_gen = mx.zeros((1, 77, 1024))
        self.unet.learned_text_clip_ref = mx.zeros((1, 77, 1024))

    def compute_condition_embed(self, ref_latents):
        """Run the dual-stream reference UNet once (mode 'w'). Constant across diffusion steps
        (ref_latents + timestep_ref=0 fixed), so the pipeline computes it once and caches it."""
        B, N_ref, H, W, _ = ref_latents.shape
        rl = ref_latents.reshape(B * N_ref, H, W, 4)
        ehs_ref = mx.broadcast_to(self.unet.learned_text_clip_ref, (B * N_ref, 77, 1024))
        ced = {}
        self.unet_dual(rl, mx.zeros((B * N_ref,)), ehs_ref, class_labels=None,
                       cross_attention_kwargs={"mode": "w", "num_in_batch": N_ref,
                                               "condition_embed_dict": ced})
        mx.eval(list(ced.values()))
        return ced

    def __call__(self, sample, timestep, encoder_hidden_states, normal_imgs, position_imgs,
                 ref_latents, camera_info_gen, camera_info_ref=None, mva_scale=1.0, ref_scale=1.0,
                 condition_embed_dict=None):
        # sample/normal/position: [B, N_gen, H, W, 4]; ref_latents: [B, N_ref, H, W, 4]
        B, N_gen, H, W, _ = sample.shape
        s = mx.concatenate([sample, normal_imgs, position_imgs], axis=-1).reshape(B * N_gen, H, W, 12)
        ehs = encoder_hidden_states                                   # [B, 77, 1024]
        Lt, Ct = ehs.shape[-2], ehs.shape[-1]
        ehs_gen = mx.broadcast_to(ehs[:, None], (B, N_gen, Lt, Ct)).reshape(B * N_gen, Lt, Ct)
        cam_gen = (camera_info_gen + self.max_num_ref).reshape(-1)

        ced = condition_embed_dict
        if ced is None and ref_scale != 0:
            ced = self.compute_condition_embed(ref_latents)          # None for uncond (ref_scale=0) -> RA skipped

        out = self.unet(s, timestep, ehs_gen, class_labels=cam_gen,
                        cross_attention_kwargs={"mode": "r", "num_in_batch": N_gen,
                                                "condition_embed_dict": ced,
                                                "mva_scale": mva_scale, "ref_scale": ref_scale})
        return out.reshape(B, N_gen, H, W, 4)

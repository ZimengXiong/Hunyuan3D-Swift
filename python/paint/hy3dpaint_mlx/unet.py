"""MLX SD2.1 UNet2DConditionModel (NHWC), the base backbone the paint 2.5D wrapper extends.

Module tree mirrors diffusers (conv_in, time_embedding, down_blocks[*].{resnets,attentions,
downsamplers}, mid_block.{resnets,attentions}, up_blocks[*].{resnets,attentions,upsamplers},
conv_norm_out, conv_out) so a torch state_dict maps 1:1 (Conv2d NCHW->NHWC at load).
SD2.1 layout: down = 3x CrossAttnDownBlock2D + DownBlock2D; up = UpBlock2D + 3x CrossAttnUpBlock2D.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .layers2d import GroupNorm2d, ResnetBlock2D, Downsample2D, Upsample2D, silu
from .attention import Transformer2DModel, TimestepEmbedding, timestep_embedding


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, n_layers, heads, dim_head, ctx_dim, groups, add_down,
                 use_ma=False, use_ra=False, name="", pbr=None):
        super().__init__()
        self.resnets, self.attentions = [], []
        for i in range(n_layers):
            self.resnets.append(ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch, groups))
            self.attentions.append(Transformer2DModel(out_ch, heads, dim_head, ctx_dim, 1, groups,
                                                      use_ma=use_ma, use_ra=use_ra, layer_name=f"{name}_{i}_0", pbr=pbr))
        self.downsamplers = [Downsample2D(out_ch, out_ch)] if add_down else []

    def __call__(self, x, temb, ctx, xattn=None):
        res = []
        for r, a in zip(self.resnets, self.attentions):
            x = a(r(x, temb), context=ctx, xattn=xattn)
            res.append(x)
        for d in self.downsamplers:
            x = d(x); res.append(x)
        return x, res


class DownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, n_layers, groups, add_down):
        super().__init__()
        self.resnets = [ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_ch, groups) for i in range(n_layers)]
        self.downsamplers = [Downsample2D(out_ch, out_ch)] if add_down else []

    def __call__(self, x, temb, ctx=None, xattn=None):
        res = []
        for r in self.resnets:
            x = r(x, temb); res.append(x)
        for d in self.downsamplers:
            x = d(x); res.append(x)
        return x, res


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, ch, temb_ch, heads, dim_head, ctx_dim, groups, use_ma=False, use_ra=False, name="mid", pbr=None):
        super().__init__()
        self.attentions = [Transformer2DModel(ch, heads, dim_head, ctx_dim, 1, groups,
                                              use_ma=use_ma, use_ra=use_ra, layer_name=f"{name}_0_0", pbr=pbr)]
        self.resnets = [ResnetBlock2D(ch, ch, temb_ch, groups), ResnetBlock2D(ch, ch, temb_ch, groups)]

    def __call__(self, x, temb, ctx, xattn=None):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context=ctx, xattn=xattn)
        x = self.resnets[1](x, temb)
        return x


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, prev_ch, temb_ch, n_layers, heads, dim_head, ctx_dim, groups, add_up,
                 use_ma=False, use_ra=False, name="", pbr=None):
        super().__init__()
        self.resnets, self.attentions = [], []
        for i in range(n_layers):
            skip = in_ch if i == n_layers - 1 else out_ch
            rin = prev_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock2D(rin + skip, out_ch, temb_ch, groups))
            self.attentions.append(Transformer2DModel(out_ch, heads, dim_head, ctx_dim, 1, groups,
                                                      use_ma=use_ma, use_ra=use_ra, layer_name=f"{name}_{i}_0", pbr=pbr))
        self.upsamplers = [Upsample2D(out_ch, out_ch)] if add_up else []

    def __call__(self, x, res, temb, ctx, xattn=None):
        for r, a in zip(self.resnets, self.attentions):
            x = mx.concatenate([x, res.pop()], axis=-1)
            x = a(r(x, temb), context=ctx, xattn=xattn)
        for u in self.upsamplers:
            x = u(x)
        return x


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, prev_ch, temb_ch, n_layers, groups, add_up):
        super().__init__()
        self.resnets = []
        for i in range(n_layers):
            skip = in_ch if i == n_layers - 1 else out_ch
            rin = prev_ch if i == 0 else out_ch
            self.resnets.append(ResnetBlock2D(rin + skip, out_ch, temb_ch, groups))
        self.upsamplers = [Upsample2D(out_ch, out_ch)] if add_up else []

    def __call__(self, x, res, temb, ctx=None, xattn=None):
        for r in self.resnets:
            x = mx.concatenate([x, res.pop()], axis=-1)
            x = r(x, temb)
        for u in self.upsamplers:
            x = u(x)
        return x


class UNet2DConditionModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, block_out_channels=(320, 640, 1280, 1280),
                 layers_per_block=2, cross_attention_dim=1024, attention_head_dim=(5, 10, 20, 20),
                 norm_num_groups=32, conv_in_channels=None, num_class_embeds=None,
                 use_ma=False, use_ra=False, pbr=None):
        super().__init__()
        boc = list(block_out_channels)
        heads = list(attention_head_dim)
        temb_ch = boc[0] * 4
        self.time_embed_in = boc[0]
        self.conv_in = nn.Conv2d(conv_in_channels or in_channels, boc[0], 3, stride=1, padding=1)
        self.time_embedding = TimestepEmbedding(boc[0], temb_ch)
        # camera/class embedding (paint 2.5D wrapper): class_labels -> Embedding added to temb
        self.class_embedding = nn.Embedding(num_class_embeds, temb_ch) if num_class_embeds else None

        # down
        self.down_blocks = []
        out_ch = boc[0]
        for i in range(len(boc)):
            in_ch = out_ch
            out_ch = boc[i]
            last = i == len(boc) - 1
            dim_head = out_ch // heads[i]
            if i < len(boc) - 1:
                self.down_blocks.append(CrossAttnDownBlock2D(
                    in_ch, out_ch, temb_ch, layers_per_block, heads[i], dim_head,
                    cross_attention_dim, norm_num_groups, add_down=not last,
                    use_ma=use_ma, use_ra=use_ra, name=f"down_{i}", pbr=pbr))
            else:
                self.down_blocks.append(DownBlock2D(
                    in_ch, out_ch, temb_ch, layers_per_block, norm_num_groups, add_down=not last))

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            boc[-1], temb_ch, heads[-1], boc[-1] // heads[-1], cross_attention_dim, norm_num_groups,
            use_ma=use_ma, use_ra=use_ra, name="mid", pbr=pbr)

        # up
        self.up_blocks = []
        rev_boc = list(reversed(boc))
        rev_heads = list(reversed(heads))
        prev_ch = rev_boc[0]
        for i in range(len(rev_boc)):
            out_ch = rev_boc[i]
            in_ch = rev_boc[min(i + 1, len(rev_boc) - 1)]
            last = i == len(rev_boc) - 1
            dim_head = out_ch // rev_heads[i]
            if i == 0:
                self.up_blocks.append(UpBlock2D(
                    in_ch, out_ch, prev_ch, temb_ch, layers_per_block + 1, norm_num_groups, add_up=not last))
            else:
                self.up_blocks.append(CrossAttnUpBlock2D(
                    in_ch, out_ch, prev_ch, temb_ch, layers_per_block + 1, rev_heads[i], dim_head,
                    cross_attention_dim, norm_num_groups, add_up=not last,
                    use_ma=use_ma, use_ra=use_ra, name=f"up_{i}", pbr=pbr))
            prev_ch = out_ch

        self.conv_norm_out = GroupNorm2d(norm_num_groups, boc[0], 1e-5)
        self.conv_out = nn.Conv2d(boc[0], out_channels, 3, stride=1, padding=1)

    def __call__(self, sample, timestep, encoder_hidden_states, class_labels=None, cross_attention_kwargs=None):
        # sample: [N,H,W,C]; timestep: scalar or [N]; encoder_hidden_states: [N,L,ctx]
        if not isinstance(timestep, mx.array):
            timestep = mx.array([timestep] * sample.shape[0])
        elif timestep.ndim == 0:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        t_emb = timestep_embedding(timestep, self.time_embed_in, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        temb = self.time_embedding(t_emb)
        if self.class_embedding is not None and class_labels is not None:
            temb = temb + self.class_embedding(class_labels)
        ctx = encoder_hidden_states

        xattn = cross_attention_kwargs
        h = self.conv_in(sample)
        res_samples = [h]
        for blk in self.down_blocks:
            h, res = blk(h, temb, ctx, xattn=xattn)
            res_samples += res
        h = self.mid_block(h, temb, ctx, xattn=xattn)
        for blk in self.up_blocks:
            n = len(blk.resnets)
            skips = res_samples[-n:]
            del res_samples[-n:]
            h = blk(h, skips, temb, ctx, xattn=xattn)
        h = self.conv_out(silu(self.conv_norm_out(h)))
        return h

"""MLX AutoencoderKL (SD2.1 VAE) — used by both paint models to encode control/reference
images and decode the generated views.

Module tree mirrors diffusers AutoencoderKL exactly (encoder/decoder, down_blocks/up_blocks,
resnets/downsamplers/upsamplers, mid_block.attentions, quant_conv/post_quant_conv), so a torch
state_dict maps 1:1 (Conv2d NCHW->NHWC at load). NHWC throughout; fp32-internal norms.

Default config (SD2.1): latent_channels=4, block_out_channels=[128,256,512,512],
layers_per_block=2, norm_num_groups=32, in/out=3, scaling_factor read from config (0.18215).
The scaling_factor is applied by the *pipeline*, not here — encode/decode are raw.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .layers2d import GroupNorm2d, ResnetBlock2D, Downsample2D, Upsample2D, VAEAttnBlock, silu


class _DownEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, groups, eps, add_downsample):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_channels=None, groups=groups, eps=eps)
            for i in range(num_layers)
        ]
        self.downsamplers = [Downsample2D(out_ch, out_ch, padding=0)] if add_downsample else []

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        for d in self.downsamplers:
            x = d(x)
        return x


class _UpDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, groups, eps, add_upsample):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(in_ch if i == 0 else out_ch, out_ch, temb_channels=None, groups=groups, eps=eps)
            for i in range(num_layers)
        ]
        self.upsamplers = [Upsample2D(out_ch, out_ch)] if add_upsample else []

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        for u in self.upsamplers:
            x = u(x)
        return x


class _MidBlock(nn.Module):
    def __init__(self, ch, groups, eps):
        super().__init__()
        self.attentions = [VAEAttnBlock(ch, groups=groups, eps=eps)]
        self.resnets = [
            ResnetBlock2D(ch, ch, temb_channels=None, groups=groups, eps=eps),
            ResnetBlock2D(ch, ch, temb_channels=None, groups=groups, eps=eps),
        ]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, block_out_channels, layers_per_block, latent_channels, groups, eps):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, stride=1, padding=1)
        self.down_blocks = []
        out_ch = block_out_channels[0]
        for i, boc in enumerate(block_out_channels):
            in_ch = out_ch
            out_ch = boc
            is_last = i == len(block_out_channels) - 1
            self.down_blocks.append(
                _DownEncoderBlock(in_ch, out_ch, layers_per_block, groups, eps, add_downsample=not is_last)
            )
        self.mid_block = _MidBlock(block_out_channels[-1], groups, eps)
        self.conv_norm_out = GroupNorm2d(groups, block_out_channels[-1], eps)
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, 3, stride=1, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        x = self.mid_block(x)
        x = silu(self.conv_norm_out(x))
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, block_out_channels, layers_per_block, latent_channels, groups, eps):
        super().__init__()
        rev = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, stride=1, padding=1)
        self.mid_block = _MidBlock(rev[0], groups, eps)
        self.up_blocks = []
        out_ch = rev[0]
        for i, boc in enumerate(rev):
            in_ch = out_ch
            out_ch = boc
            is_last = i == len(rev) - 1
            self.up_blocks.append(
                _UpDecoderBlock(in_ch, out_ch, layers_per_block + 1, groups, eps, add_upsample=not is_last)
            )
        self.conv_norm_out = GroupNorm2d(groups, rev[-1], eps)
        self.conv_out = nn.Conv2d(rev[-1], out_channels, 3, stride=1, padding=1)

    def __call__(self, z):
        z = self.conv_in(z)
        z = self.mid_block(z)
        for b in self.up_blocks:
            z = b(z)
        z = silu(self.conv_norm_out(z))
        return self.conv_out(z)


class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=4,
                 block_out_channels=(128, 256, 512, 512), layers_per_block=2,
                 norm_num_groups=32, scaling_factor=0.18215, eps=1e-6):
        super().__init__()
        boc = list(block_out_channels)
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.encoder = Encoder(in_channels, boc, layers_per_block, latent_channels, norm_num_groups, eps)
        self.decoder = Decoder(out_channels, boc, layers_per_block, latent_channels, norm_num_groups, eps)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, stride=1, padding=0)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1, stride=1, padding=0)

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            in_channels=cfg.get("in_channels", 3),
            out_channels=cfg.get("out_channels", 3),
            latent_channels=cfg.get("latent_channels", 4),
            block_out_channels=cfg.get("block_out_channels", (128, 256, 512, 512)),
            layers_per_block=cfg.get("layers_per_block", 2),
            norm_num_groups=cfg.get("norm_num_groups", 32),
            scaling_factor=cfg.get("scaling_factor", 0.18215),
        )

    def encode_moments(self, x):
        """x: [N,H,W,3] in [-1,1]. Returns moments [N,h,w,2*latent] (mean, logvar concat)."""
        return self.quant_conv(self.encoder(x))

    def encode_mean(self, x):
        return mx.split(self.encode_moments(x), 2, axis=-1)[0]

    def decode(self, z):
        """z: [N,h,w,latent]. Returns [N,H,W,3]."""
        return self.decoder(self.post_quant_conv(z))

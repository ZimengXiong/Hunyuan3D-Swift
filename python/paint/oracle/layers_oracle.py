"""Dump diffusers/torch reference tensors for the 2D primitives (run in .venv-oracle).

Usage:  .venv-oracle/bin/python oracle/layers_oracle.py <out_dir>

For each primitive: fixed seed, random weights + input, run the diffusers/torch module in
fp32, dump input/output (as NHWC .npy) + the torch state_dict (.npz, torch names). The MLX
side (tests/test_layers2d_parity.py) loads these, builds the MLX module, and compares.
"""

import os
import sys

import numpy as np
import torch

torch.use_deterministic_algorithms(False)

try:
    from diffusers.models.resnet import ResnetBlock2D
except Exception:
    from diffusers.models.resnets import ResnetBlock2D
try:
    from diffusers.models.downsampling import Downsample2D
    from diffusers.models.upsampling import Upsample2D
except Exception:
    from diffusers.models.resnet import Downsample2D, Upsample2D


def _nchw_to_nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def dump_case(out_dir, name, module, x_nchw, temb=None):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    module = module.eval().float()
    with torch.no_grad():
        if temb is not None:
            y = module(x_nchw, temb)
        else:
            try:
                y = module(x_nchw)
            except TypeError:
                y = module(x_nchw, None)
    if isinstance(y, tuple):
        y = y[0]
    np.save(os.path.join(d, "input.npy"), _nchw_to_nhwc(x_nchw.detach().numpy()))
    np.save(os.path.join(d, "output.npy"), _nchw_to_nhwc(y.detach().numpy()))
    if temb is not None:
        np.save(os.path.join(d, "temb.npy"), temb.detach().numpy())
    sd = {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}
    np.savez(os.path.join(d, "weights.npz"), **sd)
    print(f"  dumped {name}: in {tuple(x_nchw.shape)} -> out {tuple(y.shape)} ({len(sd)} params)")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    # GroupNorm (torch reference)
    gn = torch.nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6)
    gn.weight.data.normal_(); gn.bias.data.normal_()
    dump_case(out_dir, "groupnorm", gn, torch.randn(2, 64, 16, 16))

    # ResnetBlock2D without temb (VAE-style)
    rb = ResnetBlock2D(in_channels=64, out_channels=128, temb_channels=None, groups=32, eps=1e-6)
    dump_case(out_dir, "resnet_notemb", rb, torch.randn(2, 64, 16, 16))

    # ResnetBlock2D with temb (UNet-style), same in/out to exercise no-shortcut path too
    rb2 = ResnetBlock2D(in_channels=128, out_channels=128, temb_channels=512, groups=32, eps=1e-6)
    dump_case(out_dir, "resnet_temb", rb2, torch.randn(2, 128, 16, 16), temb=torch.randn(2, 512))

    # Downsample2D (use_conv)
    ds = Downsample2D(channels=64, use_conv=True, out_channels=64, padding=0)
    dump_case(out_dir, "downsample", ds, torch.randn(2, 64, 16, 16))

    # Upsample2D (use_conv)
    us = Upsample2D(channels=64, use_conv=True, out_channels=64)
    dump_case(out_dir, "upsample", us, torch.randn(2, 64, 16, 16))

    print(f"OK -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/layers_oracle")

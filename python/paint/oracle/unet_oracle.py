"""Dump a diffusers SD2.1 UNet2DConditionModel forward (random weights) for parity gating."""

import os
import sys

import numpy as np
import torch
from diffusers import UNet2DConditionModel


def nchw2nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=1024, attention_head_dim=(5, 10, 20, 20),
        use_linear_projection=True, norm_num_groups=32,
    ).eval().float()
    x = torch.randn(1, 4, 32, 32)
    t = torch.tensor(10.0)
    ctx = torch.randn(1, 77, 1024)
    with torch.no_grad():
        y = unet(x, t, encoder_hidden_states=ctx).sample
    np.save(os.path.join(out_dir, "input.npy"), nchw2nhwc(x.numpy()))
    np.save(os.path.join(out_dir, "context.npy"), ctx.numpy())
    np.save(os.path.join(out_dir, "timestep.npy"), np.array([10.0], dtype=np.float32))
    np.save(os.path.join(out_dir, "output.npy"), nchw2nhwc(y.numpy()))
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.detach().numpy() for k, v in unet.state_dict().items()})
    print(f"OK {len(unet.state_dict())} params -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])

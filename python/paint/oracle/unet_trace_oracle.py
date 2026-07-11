"""Dump diffusers UNet per-stage intermediates to localize a wiring bug."""

import os
import sys

import numpy as np
import torch
from diffusers import UNet2DConditionModel


def nhwc(a):
    return np.ascontiguousarray(a.detach().numpy().transpose(0, 2, 3, 1))


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

    store = {}
    unet.conv_in.register_forward_hook(lambda m, i, o: store.__setitem__("conv_in", o))
    unet.mid_block.register_forward_hook(lambda m, i, o: store.__setitem__("mid", o))
    for idx, b in enumerate(unet.down_blocks):
        b.register_forward_hook(lambda m, i, o, k=idx: store.__setitem__(f"down{k}", o))
    def pre_hook(m, args, kwargs, k):
        h = kwargs.get("hidden_states", args[0] if args else None)
        res = kwargs.get("res_hidden_states_tuple", args[1] if len(args) > 1 else None)
        store[f"up{k}_in"] = (h, res)

    for idx, b in enumerate(unet.up_blocks):
        b.register_forward_pre_hook(lambda m, a, kw, k=idx: pre_hook(m, a, kw, k), with_kwargs=True)
        b.register_forward_hook(lambda m, i, o, k=idx: store.__setitem__(f"up{k}", o))

    x = torch.randn(1, 4, 32, 32); t = torch.tensor(10.0); ctx = torch.randn(1, 77, 1024)
    with torch.no_grad():
        unet(x, t, encoder_hidden_states=ctx)

    np.save(os.path.join(out_dir, "input.npy"), nhwc(x))
    np.save(os.path.join(out_dir, "context.npy"), ctx.numpy())
    np.save(os.path.join(out_dir, "conv_in.npy"), nhwc(store["conv_in"]))
    for k in range(4):
        np.save(os.path.join(out_dir, f"down{k}.npy"), nhwc(store[f"down{k}"][0]))
    np.save(os.path.join(out_dir, "mid.npy"), nhwc(store["mid"]))
    for k in range(4):
        h_in, res_in = store[f"up{k}_in"][0], store[f"up{k}_in"][1]
        np.save(os.path.join(out_dir, f"up{k}_hin.npy"), nhwc(h_in))
        for j, r in enumerate(res_in):
            np.save(os.path.join(out_dir, f"up{k}_res{j}.npy"), nhwc(r))
        np.save(os.path.join(out_dir, f"up{k}.npy"), nhwc(store[f"up{k}"]))
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.detach().numpy() for k, v in unet.state_dict().items()})
    print("OK ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])

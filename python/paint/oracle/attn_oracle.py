"""Dump diffusers Transformer2DModel + timestep-embedding references (run in .venv-oracle)."""

import os
import sys

import numpy as np
import torch
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


def nchw2nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    # --- Transformer2DModel (SD2.1: use_linear_projection) ---
    t2d = Transformer2DModel(
        num_attention_heads=5, attention_head_dim=64, in_channels=320,
        num_layers=1, cross_attention_dim=1024, norm_num_groups=32,
        use_linear_projection=True,
    ).eval().float()
    x = torch.randn(2, 320, 8, 8)
    ctx = torch.randn(2, 77, 1024)
    with torch.no_grad():
        y = t2d(x, encoder_hidden_states=ctx).sample
    d = os.path.join(out_dir, "transformer2d"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "input.npy"), nchw2nhwc(x.numpy()))
    np.save(os.path.join(d, "context.npy"), ctx.numpy())
    np.save(os.path.join(d, "output.npy"), nchw2nhwc(y.numpy()))
    np.savez(os.path.join(d, "weights.npz"), **{k: v.detach().numpy() for k, v in t2d.state_dict().items()})

    # --- timestep embedding ---
    ts_mod = Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0)
    te = TimestepEmbedding(320, 1280).eval().float()
    timesteps = torch.tensor([10, 500], dtype=torch.float32)
    with torch.no_grad():
        emb = ts_mod(timesteps)
        temb = te(emb)
    d = os.path.join(out_dir, "timestep"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "timesteps.npy"), timesteps.numpy())
    np.save(os.path.join(d, "emb.npy"), emb.numpy())
    np.save(os.path.join(d, "temb.npy"), temb.numpy())
    np.savez(os.path.join(d, "weights.npz"), **{k: v.detach().numpy() for k, v in te.state_dict().items()})

    print("OK ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])

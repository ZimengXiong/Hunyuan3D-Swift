"""Dump diffusers AutoencoderKL encode/decode references (run in .venv-oracle).

Usage: .venv-oracle/bin/python oracle/vae_oracle.py <vae_dir> <out_dir>
"""

import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL


def nchw2nhwc(a):
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


def main(vae_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch.float32).eval()
    torch.manual_seed(0)

    img = torch.randn(1, 3, 256, 256)               # [-? ] random image-space input
    z = torch.randn(1, vae.config.latent_channels, 32, 32)
    with torch.no_grad():
        mean = vae.encode(img).latent_dist.mode()   # deterministic (mean)
        dec = vae.decode(z).sample

    np.save(os.path.join(out_dir, "input.npy"), nchw2nhwc(img.numpy()))
    np.save(os.path.join(out_dir, "encode_mean.npy"), nchw2nhwc(mean.numpy()))
    np.save(os.path.join(out_dir, "z.npy"), nchw2nhwc(z.numpy()))
    np.save(os.path.join(out_dir, "decode.npy"), nchw2nhwc(dec.numpy()))
    print(f"OK scaling_factor={vae.config.scaling_factor} -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

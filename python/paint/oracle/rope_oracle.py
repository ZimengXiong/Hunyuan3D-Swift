"""Dump reference 3D-RoPE (RotaryEmbedding) outputs for parity (run in .venv-oracle)."""

import os
import sys
import importlib.util

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "metadata", "Hunyuan3D-2.1-paintpbr", "unet", "attn_processor.py")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    spec = importlib.util.spec_from_file_location("ap", MOD)
    ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)
    RE = ap.RotaryEmbedding

    torch.manual_seed(0)
    head_dim = 64
    vres = 512
    B, L = 2, 16
    pos = torch.randint(0, vres, (B, L, 3))
    cos, sin = RE.get_3d_rotary_pos_embed(pos, head_dim, vres)
    x = torch.randn(B, 5, L, head_dim)  # [B, heads, L, head_dim]
    out = RE.apply_rotary_emb(x, (cos, sin))

    np.save(os.path.join(out_dir, "pos.npy"), pos.numpy())
    np.save(os.path.join(out_dir, "cos.npy"), cos.numpy())
    np.save(os.path.join(out_dir, "sin.npy"), sin.numpy())
    np.save(os.path.join(out_dir, "x.npy"), x.numpy())
    np.save(os.path.join(out_dir, "out.npy"), out.numpy())
    print("OK head_dim", head_dim, "vres", vres, "->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])

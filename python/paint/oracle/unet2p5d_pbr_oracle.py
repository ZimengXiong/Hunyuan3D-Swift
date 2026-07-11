"""Run the reference 2.1 PBR UNet2p5DConditionModel forward and dump it (run in .venv-oracle)."""

import os
import sys
import types
import importlib.util
import json

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, "metadata", "Hunyuan3D-2.1-paintpbr", "unet")
WUNET = os.path.join(ROOT, "weights", "hunyuan3d-paintpbr-v2-1", "unet")


def load_ref():
    # patched copies with cuda hardcodes removed (run on CPU)
    tmp = os.path.join(ROOT, ".parity_dumps", "_pbr_ref"); os.makedirs(tmp, exist_ok=True)
    for name in ["attn_processor", "modules"]:
        src = open(os.path.join(UD, f"{name}.py")).read()
        src = src.replace('.to("cuda:0")', "").replace('.to("cuda:1")', "").replace("'cuda:0'", "'cpu'").replace("'cuda:1'", "'cpu'")
        open(os.path.join(tmp, f"{name}.py"), "w").write(src)
    open(os.path.join(tmp, "__init__.py"), "w").write("")
    pkg = types.ModuleType("pbr"); pkg.__path__ = [tmp]; sys.modules["pbr"] = pkg
    spec = importlib.util.spec_from_file_location("pbr.attn_processor", os.path.join(tmp, "attn_processor.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["pbr.attn_processor"] = m; spec.loader.exec_module(m)
    spec = importlib.util.spec_from_file_location("pbr.modules", os.path.join(tmp, "modules.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["pbr.modules"] = mod; spec.loader.exec_module(mod)
    cfg = json.load(open(os.path.join(WUNET, "config.json")))
    base = UNet2DConditionModel(**cfg)
    model = mod.UNet2p5DConditionModel(base)
    ci = model.unet.conv_in
    model.unet.conv_in = torch.nn.Conv2d(12, ci.out_channels, ci.kernel_size, ci.stride, ci.padding,
                                         ci.dilation, ci.groups, ci.bias is not None)
    sd = load_file(os.path.join(WUNET, "diffusion_pytorch_model.safetensors"))
    model.load_state_dict(sd, strict=True)
    return model.eval().float()


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = load_ref()
    torch.manual_seed(0)
    B, N_pbr, N_gen, N_ref, Hl = 1, 2, 2, 1, 8
    Hp = Hl * 8
    sample = torch.randn(B, N_pbr, N_gen, 4, Hl, Hl)
    normal = torch.randn(B, N_gen, 4, Hl, Hl)
    position = torch.randn(B, N_gen, 4, Hl, Hl)
    ref_lat = torch.randn(B, N_ref, 4, Hl, Hl)
    dino = torch.randn(B, 64, 1536)
    posmap = torch.rand(B, N_gen, 3, Hp, Hp)
    ehs = torch.stack([model.unet.learned_text_clip_albedo, model.unet.learned_text_clip_mr], 0)[None]  # [1,2,77,1024]
    with torch.no_grad():
        out = model(sample, torch.tensor(10.0), ehs,
                    embeds_normal=normal, embeds_position=position, ref_latents=ref_lat,
                    dino_hidden_states=dino, position_maps=posmap, mva_scale=1.0, ref_scale=1.0)
        out = out[0] if isinstance(out, (tuple, list)) else out
    # out: [(B N_pbr N_gen), 4, Hl, Hl]
    o = out.detach().numpy().reshape(B, N_pbr, N_gen, 4, Hl, Hl).transpose(0, 1, 2, 4, 5, 3)  # NHWC

    def t(a):  # [B,N,4,H,W]->[B,N,H,W,4]
        return np.ascontiguousarray(a.numpy().transpose(0, 1, 3, 4, 2))
    np.save(f"{out_dir}/sample.npy", np.ascontiguousarray(sample.numpy().transpose(0, 1, 2, 4, 5, 3)))
    np.save(f"{out_dir}/normal.npy", t(normal))
    np.save(f"{out_dir}/position.npy", t(position))
    np.save(f"{out_dir}/ref_lat.npy", t(ref_lat))
    np.save(f"{out_dir}/dino.npy", dino.numpy())
    np.save(f"{out_dir}/posmap.npy", np.ascontiguousarray(posmap.numpy().transpose(0, 1, 3, 4, 2)))  # [B,N,Hp,Wp,3]
    np.save(f"{out_dir}/out.npy", o)
    print("OK ->", out_dir, "out", o.shape)


if __name__ == "__main__":
    main(sys.argv[1])

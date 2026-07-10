"""Dump the real 2.1 PBR UNet forward (self.unet with PBR blocks) + its conditioning
(dual-pass condition embeds, DINO tokens, PoseRoPE tables) for the Swift PBR-UNet gate."""
import sys, os, json; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.unet2p5d_pbr import UNet2p5DPBRConditionModel
from hy3dpaint_mlx.convert import load_torch_weights

PBR = "weights/hunyuan3d-paintpbr-v2-1"
model = UNet2p5DPBRConditionModel(json.load(open(f"{PBR}/unet/config.json")))
load_torch_weights(model, mx.load(f"{PBR}/unet/diffusion_pytorch_model.safetensors"))

rng = np.random.RandomState(0)
B, Np, Ng, H, W = 1, 2, 2, 8, 8
def r(*s): return mx.array(rng.randn(*s).astype(np.float32))
sample = r(B, Np, Ng, H, W, 4)
normal_lat = r(B, Ng, H, W, 4); position_lat = r(B, Ng, H, W, 4)
ref_lat = r(B, 1, H, W, 4)
dino_hs = r(B, 5, 1536)
posmap = mx.array(rng.rand(B, Ng, 64, 64, 3).astype(np.float32))
t = mx.array(np.full(B * Np * Ng, 500.0, np.float32))

cond = model.prepare(ref_lat, dino_hs, np.asarray(posmap), H, Ng)
# replicate __call__'s self.unet inputs
nrep = mx.broadcast_to(normal_lat[:, None], (B, Np, Ng, H, W, 4))
prep = mx.broadcast_to(position_lat[:, None], (B, Np, Ng, H, W, 4))
s = mx.concatenate([sample, nrep, prep], axis=-1).reshape(B * Np * Ng, H, W, 12)
alb, mr = model.unet.learned_text_clip_albedo, model.unet.learned_text_clip_mr
ehs = mx.stack([alb, mr], axis=0)[None]
ehs = mx.broadcast_to(ehs[:, :, None], (B, Np, Ng, 77, 1024)).reshape(B * Np * Ng, 77, 1024)
ced = cond["condition_embed_dict"]
out = model.unet(s, t, ehs, class_labels=None,
                 cross_attention_kwargs={"mode": "r", "num_in_batch": Ng, "n_pbr": Np,
                                         "condition_embed_dict": ced, "dino": cond["dino"],
                                         "rope": (None, None), "rope_by_tokens": cond["rope"],
                                         "mva_scale": 1.0, "ref_scale": 1.0})
mx.eval(out)

params = dict(tree_flatten(model.unet.parameters()))
dump = {k: v.astype(mx.float32) for k, v in params.items() if v.dtype != mx.bool_}
dump.update({"_in.s": s, "_in.t": t, "_in.ehs": ehs, "_out": out})
for k, v in ced.items(): dump[f"ced::{k}"] = v
dump["dino"] = cond["dino"]
for tok, (c, sn) in cond["rope"].items():
    dump[f"rope::{tok}::cos"] = c; dump[f"rope::{tok}::sin"] = sn
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/pbr_unet_fixture.safetensors", dump)
print("unet params:", len(params), "| s", s.shape, "ehs", ehs.shape, "out", out.shape,
      "std", round(float(out.std()), 4))
print("ced keys:", len(ced), "| rope toks:", sorted(cond["rope"].keys()), "| dino", cond["dino"].shape)

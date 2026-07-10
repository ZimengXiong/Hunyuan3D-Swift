"""Dump a full PBR diffusion loop (small) + weights for the Swift end-to-end gate."""
import sys, os, json; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.unet2p5d_pbr import UNet2p5DPBRConditionModel
from hy3dpaint_mlx.convert import load_torch_weights
from hy3dpaint_mlx.scheduler import UniPCScheduler

PBR = "weights/hunyuan3d-paintpbr-v2-1"
model = UNet2p5DPBRConditionModel(json.load(open(f"{PBR}/unet/config.json")))
load_torch_weights(model, mx.load(f"{PBR}/unet/diffusion_pytorch_model.safetensors"))

B, Np, Ng, H, W, STEPS = 1, 2, 2, 8, 8, 3
GUID = float(os.environ.get("GUIDANCE", "3.0"))   # PBR pipeline ships 3.0; recorded in the fixture
rng = np.random.RandomState(0)
def r(*s): return mx.array(rng.randn(*s).astype(np.float32))
normal_lat, position_lat = r(B, Ng, H, W, 4), r(B, Ng, H, W, 4)
ref_lat = r(B, 1, H, W, 4)
dino_hs = r(B, 5, 1536)
posmap = mx.array(rng.rand(B, Ng, 64, 64, 3).astype(np.float32))
sched = UniPCScheduler(); sched.set_timesteps(STEPS)
latents0 = r(B, Np, Ng, H, W, 4)

cond = model.prepare(ref_lat, dino_hs, np.asarray(posmap), H, Ng)
uncond = {"condition_embed_dict": None, "dino": mx.zeros_like(cond["dino"]), "rope": cond["rope"]}
latents = latents0
for t in sched.timesteps:
    ti = int(t)
    vc = model(latents, ti, normal_lat, position_lat, ref_lat, dino_hs, np.asarray(posmap), 1.0, 1.0, cond=cond)
    vu = model(latents, ti, normal_lat, position_lat, ref_lat, dino_hs, np.asarray(posmap), 1.0, 0.0, cond=uncond)
    v = vu + GUID * (vc - vu)
    latents = sched.step(v, ti, latents)
mx.eval(latents)

dump = {}
for k, v in tree_flatten(model.unet.parameters()): dump[f"main::{k}"] = v.astype(mx.float32)
for k, v in tree_flatten(model.unet_dual.parameters()): dump[f"dual::{k}"] = v.astype(mx.float32)
from hy3dpaint_mlx.unet2p5d_pbr import compute_voxel_indices
for tok,(c,sn) in cond["rope"].items():
    dump[f"prope::{tok}::cos"]=c; dump[f"prope::{tok}::sin"]=sn
dump["pvox8"]=mx.array(compute_voxel_indices(np.asarray(posmap),8,64).astype(np.int32))
dump.update({"normal_lat": normal_lat, "position_lat": position_lat, "ref_lat": ref_lat,
             "dino_hs": dino_hs, "posmap": posmap, "latents0": latents0, "final": latents,
             "guidance": mx.array(np.array([GUID], np.float32)),
             "unipc_sigmas": mx.array(np.asarray(sched.sigmas, np.float32)),
             "unipc_timesteps": mx.array(np.asarray(sched.timesteps, np.int32))})
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/pbr_e2e_fixture.safetensors", dump)
print("final latents", latents.shape, "std", round(float(latents.std()), 4),
      "| steps", STEPS, "guid", GUID, "| rope toks", sorted(cond["rope"].keys()))

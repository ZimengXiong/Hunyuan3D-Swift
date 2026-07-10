"""Dump a full 2.0 (RGB) diffusion loop (small) + weights for the Swift 2.0 gate."""
import sys, os, json; sys.path.insert(0, ".")
import numpy as np, mlx.core as mx
from mlx.utils import tree_flatten
from hy3dpaint_mlx.unet2p5d import UNet2p5DConditionModel
from hy3dpaint_mlx.convert import load_torch_weights
from hy3dpaint_mlx.scheduler import UniPCScheduler

FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
P20 = "weights/hunyuan3d-paint-v2-0"
model = UNet2p5DConditionModel(json.load(open(f"{P20}/unet/config.json")))
load_torch_weights(model, mx.load(f"{P20}/unet/diffusion_pytorch_model.safetensors"),
                   renames=[("transformer_blocks.0.transformer.", "transformer_blocks.0.")])

B, N, h, STEPS, GUID = 1, 2, 8, 3, 3.0
rng = np.random.RandomState(0)
def r(*s): return mx.array(rng.randn(*s).astype(np.float32))
normal_lat, position_lat = r(B, N, h, h, 4), r(B, N, h, h, 4)
ref_lat = r(B, 1, h, h, 4)
latents0 = r(B, N, h, h, 4)
cam_gen = mx.array(np.arange(N)[None].astype(np.int32))
gen = model.unet.learned_text_clip_gen; neg = mx.zeros_like(gen)
sched = UniPCScheduler(); sched.set_timesteps(STEPS)
ced = model.compute_condition_embed(ref_lat)
latents = latents0
for t in sched.timesteps:
    ti = int(t)
    vc = model(latents, ti, gen, normal_lat, position_lat, ref_lat, cam_gen, None, 1.0, 1.0, condition_embed_dict=ced)
    vu = model(latents, ti, neg, normal_lat, position_lat, mx.zeros_like(ref_lat), cam_gen, None, 1.0, 0.0, condition_embed_dict=None)
    v = vu + GUID * (vc - vu)
    latents = sched.step(v, ti, latents)
mx.eval(latents)

dump = {}
for k, v in tree_flatten(model.unet.parameters()): dump[f"main::{k}"] = v.astype(mx.float32)
for k, v in tree_flatten(model.unet_dual.parameters()): dump[f"dual::{k}"] = v.astype(mx.float32)
dump.update({"normal_lat": normal_lat, "position_lat": position_lat, "ref_lat": ref_lat,
             "latents0": latents0, "final": latents})
mx.save_safetensors(f"{FIX}/p20_e2e_fixture.safetensors", dump)
print("2.0 e2e: final", latents.shape, "std", round(float(latents.std()), 4))

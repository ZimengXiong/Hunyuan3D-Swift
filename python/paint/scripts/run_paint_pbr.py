"""End-to-end MLX PBR paint (large / 2.1): mesh + image -> albedo + metallic-roughness textured GLB.

Renders normal/position control maps, VAE-encodes them + reference, DINOv2-giant on reference,
runs the 2.5D PBR multiview diffusion (DDIM, 2-way CFG, n_pbr=2), decodes albedo+mr views,
bakes both into UV textures, exports a PBR GLB (baseColor + metallicRoughness).
"""

import os
import sys
import json
import time

import numpy as np
import trimesh
import xatlas
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx
from mlx.utils import tree_unflatten
from hy3dpaint_mlx.vae import AutoencoderKL
from hy3dpaint_mlx.unet2p5d_pbr import UNet2p5DPBRConditionModel
from hy3dpaint_mlx.dinov2 import Dinov2Model
from hy3dpaint_mlx.scheduler import DDIMScheduler, UniPCScheduler
from hy3dpaint_mlx.mesh_render import MeshRender
from hy3dpaint_mlx.raster import cr_raster
from hy3dpaint_mlx.convert import load_torch_weights

PBR = "weights/hunyuan3d-paintpbr-v2-1"
VAEW = "weights/hunyuan3d-paint-v2-0/vae/diffusion_pytorch_model.safetensors"  # same SD2.1 VAE
R = int(os.environ.get("PAINT_RES", "512"))      # native-res generation (crisp); set 384 for a faster/softer run
SCHED = os.environ.get("PAINT_SCHED", "unipc")   # UniPC (15 steps) — reference solver, ~25% fewer steps than DDIM
STEPS = int(os.environ.get("PAINT_STEPS", "15" if SCHED == "unipc" else "20"))
TEX = int(os.environ.get("PAINT_TEX", "4096"))   # 4096 atlas + RealESRGAN x4 views → crisp 2K-class texture
GUID = 3.0
AZIMS = [0, 90, 180, 270, 0, 180]
ELEVS = [0, 0, 0, 0, 90, -90]
VW = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
# Supply your own mesh + image via PAINT_MESH / PAINT_IMG (a shape-port output, or the
# Hunyuan3D-2.1 case_1 asset from the reference checkout — see README).
MESH = os.environ.get("PAINT_MESH", "reference/Hunyuan3D-2.1/hy3dpaint/assets/case_1/mesh.glb")
IMG = os.environ.get("PAINT_IMG", "reference/Hunyuan3D-2.1/hy3dpaint/assets/case_1/image.png")
SF = 0.18215
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep_rgb(pil, size):
    pil = pil.resize((size, size))
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255)); bg.paste(pil, mask=pil.getchannel("A")); pil = bg
    return np.asarray(pil.convert("RGB"), np.float32) / 255.0


def main():
    t0 = time.time()
    print(f"[2.1 PBR] res={R} steps={STEPS} guidance={GUID}")
    vae = AutoencoderKL.from_config(json.load(open(f"{PBR}/vae/config.json")))
    load_torch_weights(vae, mx.load(VAEW), renames=[(".to_out.0.", ".to_out.")])
    unet = UNet2p5DPBRConditionModel(json.load(open(f"{PBR}/unet/config.json")))
    load_torch_weights(unet, mx.load(f"{PBR}/unet/diffusion_pytorch_model.safetensors"))
    dino = Dinov2Model()
    dino.update(tree_unflatten([(k, (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v))
                                for k, v in mx.load("weights/dinov2-giant/model.safetensors").items()]))
    mx.eval(dino.parameters())
    print(f"models loaded ({time.time()-t0:.0f}s)")

    mesh = trimesh.load(MESH, force="mesh")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    V = np.asarray(mesh.vertices)[vmapping]; F = indices.astype(np.int64)
    rend = MeshRender(); rend.load_mesh(V, F); rend.set_uv(uvs, F)
    ctrl = [rend.render_control(e, a, R) for e, a in zip(ELEVS, AZIMS)]
    normals = [c[0] for c in ctrl]; positions = [c[1] for c in ctrl]
    ref_img = prep_rgb(Image.open(IMG), R)

    def enc(imgs):
        return vae.encode_mean(mx.array(np.stack(imgs)) * 2 - 1) * SF
    normal_lat = enc(normals)[None]            # [1, N, h, w, 4]
    position_lat = enc(positions)[None]
    ref_lat = enc([ref_img])[None]             # [1, 1, h, w, 4]

    # DINO on reference (518, ImageNet norm)
    di = prep_rgb(Image.open(IMG), 518)
    di = (di - IMAGENET_MEAN) / IMAGENET_STD
    dino_hs = dino(mx.array(di[None]))         # [1, 1370, 1536]
    # position maps (pixel) for PoseRoPE voxel indices
    posmap = np.stack(positions)[None]         # [1, N, R, R, 3]

    N = len(AZIMS); h = R // 8
    sched = (UniPCScheduler() if SCHED == "unipc" else DDIMScheduler()); sched.set_timesteps(STEPS)
    mx.random.seed(0)
    latents = mx.random.normal((1, 2, N, h, h, 4)) * sched.init_noise_sigma()
    zero_dino = mx.zeros_like(dino_hs)
    print(f"controls + dino ready ({time.time()-t0:.0f}s)")

    cond = unet.prepare(ref_lat, dino_hs, posmap, h, N)               # dual pass + DINO + RoPE: once, not per step
    uncond = {"condition_embed_dict": None, "dino": mx.zeros_like(cond["dino"]), "rope": cond["rope"]}
    for i, t in enumerate(sched.timesteps):
        ti = int(t)
        v_c = unet(latents, ti, normal_lat, position_lat, ref_lat, dino_hs, posmap, mva_scale=1.0, ref_scale=1.0, cond=cond)
        v_u = unet(latents, ti, normal_lat, position_lat, ref_lat, dino_hs, posmap, mva_scale=1.0, ref_scale=0.0, cond=uncond)
        v = v_u + GUID * (v_c - v_u)
        latents = sched.step(v, ti, latents)
        mx.eval(latents)
        print(f"  step {i+1}/{STEPS} {time.time()-t0:.0f}s", flush=True)

    # decode albedo (material 0) and mr (material 1)
    alb = np.clip((np.asarray(vae.decode(latents[0, 0] / SF)) + 1) / 2, 0, 1)   # [N,H,W,3]
    mr = np.clip((np.asarray(vae.decode(latents[0, 1] / SF)) + 1) / 2, 0, 1)
    Image.fromarray((np.concatenate([alb[k] for k in range(N)], 1) * 255).astype(np.uint8)).save("outputs/pbr_albedo_views.png")
    Image.fromarray((np.concatenate([mr[k] for k in range(N)], 1) * 255).astype(np.uint8)).save("outputs/pbr_mr_views.png")
    print(f"views decoded ({time.time()-t0:.0f}s)")

    alb = [alb[k] for k in range(N)]; mr = [mr[k] for k in range(N)]
    if os.environ.get("PAINT_SR", "1") == "1":     # RealESRGAN x4 super-res before baking (sharper textures)
        from hy3dpaint_mlx.realesrgan import load_rrdbnet, upscale
        srm = load_rrdbnet()
        alb = [np.asarray(upscale(srm, a, tile=256)) for a in alb]
        mr = [np.asarray(upscale(srm, m, tile=256)) for m in mr]
        print(f"super-res x4 ({time.time()-t0:.0f}s, views -> {alb[0].shape[0]}px)")

    (tex_a, tex_m), cov = rend.bake_multi([alb, mr], ELEVS, AZIMS, texture_size=TEX, weights=VW)
    tex_a = rend.inpaint(tex_a, cov); tex_m = rend.inpaint(tex_m, cov)
    Image.fromarray((tex_a * 255).astype(np.uint8)).save("outputs/pbr_albedo_texture.png")
    Image.fromarray((tex_m * 255).astype(np.uint8)).save("outputs/pbr_mr_texture.png")

    # PBR GLB: baseColor=albedo; metallicRoughness from mr (G=roughness, B=metallic)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray((tex_a * 255).astype(np.uint8)),
        metallicRoughnessTexture=Image.fromarray((tex_m * 255).astype(np.uint8)),
        metallicFactor=1.0, roughnessFactor=1.0)
    uv_exp = rend.vtx_uv.copy(); uv_exp[:, 1] = 1.0 - uv_exp[:, 1]   # v-up (OpenGL) UVs for common viewers
    # export the ORIGINAL geometry (V), not the renderer's internally-normalized vtx_pos whose
    # Y<->Z axis swap inverts face winding (would show inside-out / see-through in viewers)
    out = trimesh.Trimesh(vertices=V, faces=F, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv_exp, material=mat))
    out.export("outputs/textured_mesh_pbr.glb")
    print(f"DONE {time.time()-t0:.0f}s -> outputs/textured_mesh_pbr.glb (+ pbr_albedo/mr textures)")


if __name__ == "__main__":
    main()

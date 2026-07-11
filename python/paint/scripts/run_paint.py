"""End-to-end MLX paint (small / 2.0 RGB): mesh + image -> generated multiview textures.

Renders normal/position control maps (cr rasterizer), VAE-encodes them + the reference image,
runs the 2.5D multiview diffusion (DDIM, 2-way CFG), decodes the 6 views, saves a grid.
"""

import os
import sys
import json
import time

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx
from hy3dpaint_mlx.vae import AutoencoderKL
from hy3dpaint_mlx.unet2p5d import UNet2p5DConditionModel
from hy3dpaint_mlx.scheduler import DDIMScheduler, UniPCScheduler
from hy3dpaint_mlx.mesh_render import MeshRender
from hy3dpaint_mlx.raster import cr_raster
from hy3dpaint_mlx.convert import load_torch_weights

W = "weights/hunyuan3d-paint-v2-0"
R = int(os.environ.get("PAINT_RES", "512"))      # native-res generation (crisp); set 384 for a faster/softer run
SCHED = os.environ.get("PAINT_SCHED", "unipc")   # UniPC reference solver
STEPS = int(os.environ.get("PAINT_STEPS", "15"))
GUID = 2.0
AZIMS = [0, 90, 180, 270, 0, 180]
ELEVS = [0, 0, 0, 0, 90, -90]
# Supply your own mesh + image via PAINT_MESH / PAINT_IMG (a shape-port output, or the
# Hunyuan3D-2.1 case_1 asset from the reference checkout — see README).
MESH = os.environ.get("PAINT_MESH", "reference/Hunyuan3D-2.1/hy3dpaint/assets/case_1/mesh.glb")
IMG = os.environ.get("PAINT_IMG", "reference/Hunyuan3D-2.1/hy3dpaint/assets/case_1/image.png")
SF = 0.18215


def prep_img(pil, size):
    pil = pil.resize((size, size))
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255)); bg.paste(pil, mask=pil.getchannel("A")); pil = bg
    return np.asarray(pil.convert("RGB"), np.float32) / 255.0  # [H,W,3] in [0,1]


def main():
    t0 = time.time()
    print(f"res={R} steps={STEPS} guidance={GUID}")
    vae = AutoencoderKL.from_config(json.load(open(f"{W}/vae/config.json")))
    load_torch_weights(vae, mx.load(f"{W}/vae/diffusion_pytorch_model.safetensors"), renames=[(".to_out.0.", ".to_out.")])
    unet = UNet2p5DConditionModel(json.load(open(f"{W}/unet/config.json")))
    load_torch_weights(unet, mx.load(f"{W}/unet/diffusion_pytorch_model.safetensors"),
                       renames=[("transformer_blocks.0.transformer.", "transformer_blocks.0.")])
    print(f"models loaded ({time.time()-t0:.0f}s)")

    # mesh + UV unwrap (xatlas)
    import xatlas
    mesh = trimesh.load(MESH, force="mesh")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    V = np.asarray(mesh.vertices)[vmapping]
    F = indices.astype(np.int64)
    print(f"uv-unwrapped: {V.shape[0]} verts, {F.shape[0]} faces")
    rend = MeshRender()
    rend.load_mesh(V, F)
    rend.set_uv(uvs, F)
    ctrl = [rend.render_control(e, a, R) for e, a in zip(ELEVS, AZIMS)]
    normals = [c[0] for c in ctrl]; positions = [c[1] for c in ctrl]
    ref_img = prep_img(Image.open(IMG), R)
    Image.fromarray((np.concatenate(normals, 1) * 255).astype(np.uint8)).save("outputs/paint_normals.png")

    def enc(imgs):  # list of [H,W,3] in [0,1] -> [N,h,w,4] latents
        x = mx.array(np.stack(imgs)) * 2 - 1
        return vae.encode_mean(x) * SF

    normal_lat = enc(normals)              # [6,h,w,4]
    position_lat = enc(positions)
    ref_lat = enc([ref_img])               # [1,h,w,4]
    print(f"controls encoded ({time.time()-t0:.0f}s)")

    N = len(AZIMS)
    h = R // 8
    cam_gen = mx.array(np.arange(N)[None, :].astype(np.int32))
    cam_ref = mx.array(np.array([[0]], np.int32))
    gen_text = unet.unet.learned_text_clip_gen          # [1,77,1024]
    neg_text = mx.zeros_like(gen_text)
    zero_ref = mx.zeros_like(ref_lat)[None]

    sched = UniPCScheduler() if SCHED == "unipc" else DDIMScheduler()
    sched.set_timesteps(STEPS)
    mx.random.seed(0)
    latents = mx.random.normal((1, N, h, h, 4)) * sched.init_noise_sigma()

    nlat = normal_lat[None]; plat = position_lat[None]; rlat = ref_lat[None]
    ced = unet.compute_condition_embed(rlat)   # dual-stream reference pass: once, not per step
    for i, t in enumerate(sched.timesteps):
        ti = int(t)
        v_c = unet(latents, ti, gen_text, nlat, plat, rlat, cam_gen, cam_ref, mva_scale=1.0, ref_scale=1.0, condition_embed_dict=ced)
        v_u = unet(latents, ti, neg_text, nlat, plat, zero_ref, cam_gen, cam_ref, mva_scale=1.0, ref_scale=0.0, condition_embed_dict=None)
        v = v_u + GUID * (v_c - v_u)
        latents = sched.step(v, ti, latents)
        mx.eval(latents)
        print(f"  step {i+1}/{STEPS} (t={ti}) {time.time()-t0:.0f}s", flush=True)

    dec = vae.decode(latents[0] / SF)      # [N,H,W,3]
    imgs = np.clip((np.asarray(dec) + 1) / 2, 0, 1)
    grid = np.concatenate([imgs[k] for k in range(N)], axis=1)
    Image.fromarray((grid * 255).astype(np.uint8)).save("outputs/paint_views.png")
    print(f"views done {time.time()-t0:.0f}s")

    # ---- bake the 6 views into a UV texture, inpaint, export GLB ----
    TEX = int(os.environ.get("PAINT_TEX", "2048"))
    view_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
    views = [imgs[k] for k in range(N)]
    if os.environ.get("PAINT_SR", "1") == "1":     # RealESRGAN x4 super-res before baking
        from hy3dpaint_mlx.realesrgan import load_rrdbnet, upscale
        srm = load_rrdbnet()
        views = [np.asarray(upscale(srm, v, tile=256)) for v in views]
        print(f"super-res x4 ({time.time()-t0:.0f}s, views -> {views[0].shape[0]}px)")
    texture, covered = rend.bake(views, ELEVS, AZIMS, texture_size=TEX, weights=view_weights)
    texture = rend.inpaint(texture, covered)
    Image.fromarray((texture * 255).astype(np.uint8)).save("outputs/paint_texture.png")

    tex_img = Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8))
    uv_exp = rend.vtx_uv.copy(); uv_exp[:, 1] = 1.0 - uv_exp[:, 1]   # v-up (OpenGL) UVs for common viewers
    # export the ORIGINAL geometry (V), not the renderer's internally-normalized vtx_pos whose
    # Y<->Z axis swap inverts face winding (would show inside-out / see-through in viewers)
    out = trimesh.Trimesh(vertices=V, faces=F, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv_exp, image=tex_img))
    out.export("outputs/textured_mesh.glb")
    print(f"DONE {time.time()-t0:.0f}s -> outputs/textured_mesh.glb (+ paint_texture.png)")

    # textured preview render (rasterize + sample texture) for a couple views
    previews = []
    for e, a in [(0, 0), (0, 120), (0, 240)]:
        pc = rend._clip(e, a)
        fi, ba = cr_raster.rasterize(pc, F, 512)
        uvm = cr_raster.interpolate(rend.vtx_uv, fi, ba, F)
        tnp = np.asarray(texture, np.float32)
        ui = np.clip((uvm[..., 0] * (TEX - 1)).astype(int), 0, TEX - 1)
        vi = np.clip(((uvm[..., 1]) * (TEX - 1)).astype(int), 0, TEX - 1)
        col = tnp[vi, ui]
        col[fi == 0] = 1.0
        previews.append((col * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(previews, 1)).save("outputs/textured_preview.png")
    print("preview -> outputs/textured_preview.png")


if __name__ == "__main__":
    main()

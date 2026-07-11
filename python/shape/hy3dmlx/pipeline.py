"""End-to-end image -> mesh pipeline, fully MLX-native (no torch in the path).

rembg/preprocess (numpy) -> DINOv2 (MLX) -> FlowMatch denoise loop (MLX) ->
ShapeVAE decode + dense grid query (MLX) -> the single grid->numpy hop ->
skimage marching cubes -> trimesh export.
"""
import time

import numpy as np
import mlx.core as mx
import trimesh
from skimage import measure

from .convert import load_models
from .preprocess import dino_transform, load_image
from .sampler import consistency_sigmas, denoise, flow_match_sigmas


class Hunyuan3DShapePipeline:
    def __init__(self, dino, dit, vae, cfg, dtype=mx.float32):
        self.dino = dino
        self.dit = dit
        self.vae = vae
        self.cfg = cfg
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype=mx.float32, quantize=None, verbose=True):
        """model_dir: the folder with config.yaml + model.fp16.safetensors
        (e.g. weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini).
        quantize: None | 8 | 4 — quantize the DiT+DINO block linears (VAE stays fp16)."""
        dino, dit, vae, cfg = load_models(model_dir, dtype=dtype, quantize=quantize, verbose=verbose)
        return cls(dino, dit, vae, cfg, dtype)

    def encode_image(self, image_path: str, border_ratio=0.15):
        img = load_image(image_path, size=512, border_ratio=border_ratio)  # [1,3,512,512] in [-1,1]
        pixel = dino_transform(img[0]).astype(self.dtype)                   # NHWC [1,518,518,3]
        cond = self.dino(pixel)                                            # [1,1370,1536]
        uncond = self.dino.unconditional_embedding(1, dtype=self.dtype)
        cond_cat = mx.concatenate([cond, uncond], axis=0)                  # [2,1370,1536] (cond first)
        mx.eval(cond_cat)
        return cond_cat

    def generate(self, image_path: str, num_inference_steps=30, guidance_scale=5.0,
                 octree_resolution=256, num_chunks=None, box_v=1.01, mc_level=0.0,
                 seed=0, border_ratio=0.15, octree_decode=False, compile_dit=False,
                 verbose=True):
        t0 = time.time()
        cond_cat = self.encode_image(image_path, border_ratio)
        if verbose:
            print(f"[dino] cond {cond_cat.shape} in {time.time() - t0:.1f}s")

        # Distilled (guidance-embedded) models skip CFG: single forward with a guidance token.
        guidance_embed = bool(getattr(self.dit, "guidance_embed", False))
        sched = self.cfg.get("scheduler", {}).get("target", "")
        if "Consistency" in sched:  # turbo / PCM-distilled
            pcm = self.cfg["scheduler"].get("params", {}).get("pcm_timesteps", 100)
            sigmas = consistency_sigmas(num_inference_steps, pcm_timesteps=pcm)
        else:
            sigmas = flow_match_sigmas(num_inference_steps)

        if guidance_embed:
            cond1 = cond_cat[:1]  # conditional embedding only (no uncond branch)
            gvec = mx.full((1,), guidance_scale, dtype=self.dtype)

            def velocity_fn(x, t):
                tt = mx.broadcast_to(t.reshape(1), (x.shape[0],))
                gg = mx.broadcast_to(gvec, (x.shape[0],))
                return self.dit(x, tt, cond1, guidance=gg)
        else:
            def velocity_fn(x, t):
                tt = mx.broadcast_to(t.reshape(1), (x.shape[0],))
                return self.dit(x, tt, cond_cat)
        if compile_dit:
            velocity_fn = mx.compile(velocity_fn)

        t1 = time.time()
        latents = denoise(
            velocity_fn, (1, *self.vae.latent_shape),
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            cfg=not guidance_embed, sigmas=sigmas, seed=seed, dtype=self.dtype,
            progress=(lambda i, n: print(f"\r[denoise] {i}/{n}", end="")) if verbose else None,
        )
        mx.eval(latents)
        if verbose:
            mode = "guidance-embed" if guidance_embed else "CFG"
            print(f"\n[denoise] {num_inference_steps} steps ({mode}) in {time.time() - t1:.1f}s")

        t2 = time.time()
        latents = latents / self.vae.scale_factor          # scale_factor divide at decode entry
        kv = self.vae.decode(latents)
        mx.eval(kv)
        if octree_decode:
            grid, bbox_min, bbox_max, grid_size = self.vae.query_grid_octree(
                kv, bounds=box_v, octree_resolution=octree_resolution,
                num_chunks=num_chunks, mc_level=mc_level)
        else:
            grid, bbox_min, bbox_max, grid_size = self.vae.query_grid(
                kv, bounds=box_v, octree_resolution=octree_resolution, num_chunks=num_chunks)
        if verbose:
            active = np.isfinite(grid).mean()
            print(f"[vae] grid {grid.shape} (range {np.nanmin(grid):.3f}..{np.nanmax(grid):.3f}, "
                  f"active {active:.1%}) in {time.time() - t2:.1f}s")

        mesh = self._grid_to_mesh(grid, bbox_min, bbox_max, grid_size, mc_level)
        if verbose:
            print(f"[mesh] {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
                  f"total {time.time() - t0:.1f}s")
        return mesh

    @staticmethod
    def _grid_to_mesh(grid, bbox_min, bbox_max, grid_size, mc_level):
        # Octree decode leaves far-from-surface cells as NaN; marching cubes treats NaN
        # edges as non-crossing, so the iso-surface is extracted only in the valid band
        # with no spurious band-boundary geometry (matches the torch reference).
        verts, faces, _, _ = measure.marching_cubes(grid.astype(np.float32), mc_level,
                                                    method="lewiner")
        grid_size = np.array(grid_size)
        bbox_size = bbox_max - bbox_min
        verts = verts / grid_size * bbox_size + bbox_min
        faces = faces[:, ::-1]  # reverse winding (matches export_to_trimesh)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.update_faces(mesh.nondegenerate_faces())  # drop sliver faces at NaN band edges
        mesh.remove_unreferenced_vertices()
        return mesh


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini",
                    help="model dir with config.yaml + model.fp16.safetensors")
    ap.add_argument("--out", default="demo.glb")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--octree", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                    help="4 or 8 bit DiT+DINO; 0 = off (VAE always fp16)")
    ap.add_argument("--octree-decode", action="store_true",
                    help="FlashVDM-style octree decode (query only near-surface; faster)")
    ap.add_argument("--compile-dit", action="store_true",
                    help="mx.compile the DiT step (~1.1x; perturbs the fp16 sample slightly)")
    args = ap.parse_args()

    dtype = {"float32": mx.float32, "float16": mx.float16}[args.dtype]
    pipe = Hunyuan3DShapePipeline.from_pretrained(args.weights, dtype=dtype,
                                                  quantize=args.quantize or None)
    mesh = pipe.generate(args.image, num_inference_steps=args.steps,
                         guidance_scale=args.guidance, octree_resolution=args.octree,
                         seed=args.seed, octree_decode=args.octree_decode,
                         compile_dit=args.compile_dit)
    mesh.export(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

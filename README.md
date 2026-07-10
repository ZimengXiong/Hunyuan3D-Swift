# Hunyuan3D-Swift

Native MLX-Swift port of the Tencent Hunyuan3D image-to-3D pipelines for Apple silicon:
`Hy3DMLX` (shape generation — DiT flow-matching + ShapeVAE + DINOv2 conditioning + marching
cubes) and `HunyuanPaintMLX` (texture painting — SD2.1 multiview diffusion, RGB and PBR, with
xatlas UV unwrap, rasterize/bake, and RealESRGAN super-resolution), plus a single `hy3d` CLI
that chains them (image → textured GLB) entirely on-device. Both libraries are verified against
the known-good Python MLX ports by threshold-gated parity fixtures (see `parity/`).

## Build

```bash
swift build                      # debug
swift build -c release          # release (hy3d CLI)
```

## Run

Weights: download the model folders (e.g. the `zimengxiong/hunyuan3d-mlx-shape-{small,large}`
and `zimengxiong/hunyuan3d-mlx-paint-{small,large}` bundles on Hugging Face, or convert the
official Tencent checkpoints) and point the flags at them.

```bash
# image -> shape mesh
swift run -c release hy3d shape photo.png -o out.glb --weights /path/to/shape-small

# mesh + image -> textured mesh (RGB or PBR)
swift run -c release hy3d paint mesh.glb photo.png -o textured.glb --weights /path/to/paint-weights --model rgb

# image -> textured mesh (chained shape + paint)
swift run -c release hy3d generate photo.png -o out.glb \
    --shape-weights /path/to/shape-small --paint-weights /path/to/paint-weights

# parity print-panels (need fixtures; see parity/README.md)
swift run -c release hy3d parity-shape --fixtures ./fixtures --weights /path/to/shape-small
swift run -c release hy3d parity-paint --fixtures ./fixtures
```

`hy3d --help` lists every flag.

## Test

```bash
swift test                       # green with zero fixtures (everything skips)
HY3D_FIXTURES=/path/to/fixtures swift test   # threshold-gated parity vs Python MLX
```

Fixture regeneration is documented in [`parity/README.md`](parity/README.md).

## Parity (measured)

Full 2×2 lineup (shape `2mini` + `2.0-turbo`, paint `v2-0` RGB + `paintpbr-v2-1` PBR) vs the
known-good Python MLX ports, fixtures regenerated end-to-end. Values from `hy3d parity-shape`
/ `hy3d parity-paint`; every XCTest gate asserts the threshold column.

| Gate | Threshold | Measured |
|---|---|---|
| Shape DiT forward (mini) | cos ≥ 0.99999 | cos 1.0000000, maxabs 0 |
| Shape DiT forward (2.0-turbo, guidance-embed) | cos ≥ 0.99999 | cos 1.0000000, maxabs 0 |
| Shape DINOv2 | cos ≥ 0.9999 | cos 0.9999999, maxabs 0 |
| Shape VAE / geo-decoder grid | cos ≥ 0.9999 | cos 0.9999999, maxabs 0 |
| Sigmas (flow-match + consistency) | maxabs ≤ 1e-6 | 0 / 0 (exact) |
| Shape e2e mesh, mini (256 octree) | Chamfer/bbox ≤ 0.01 | 0.00260 (148 852 verts) |
| Shape e2e mesh, 2.0-turbo (256 octree) | Chamfer/bbox ≤ 0.01 | 0.00257 (150 430 verts) |
| Paint VAE encode / decode | maxabs ≤ 1e-6 | 0 / 0 (bit-exact) |
| DDIM trajectory | maxabs ≤ 1e-6 | 0 (bit-exact) |
| UniPC trajectory | maxabs ≤ 1e-5 | 3.6e-7 |
| SD2.1 UNet forward | maxabs ≤ 1e-4 | 3.2e-6 |
| PBR UNet forward (MDA+RA+MA+DINO+RoPE) | cos ≥ 0.9999 | cos 1.0000000, maxabs 8.2e-5 |
| RealESRGAN ×4 | maxabs ≤ 1e-6 | 0 (bit-exact) |
| Rasterizer face-id / bary | 100% / ≤ 2e-4 | 100% / 0 |
| Control maps (normal / position) | PSNR ≥ 80 dB | 167.3 dB / ∞ (bit-exact) |
| Bake (coverage / texture) | 100% / PSNR ≥ 100 dB | 100% / 151.3 dB |
| Inpaint (EDT nearest-fill + Navier-Stokes) | bit-exact | maxabs 0 (indices, fill, NS) |
| PoseRoPE fp16 voxel indices (4 levels, 512²) | exact | maxdiff 0, tables maxabs 0 |
| Paint RGB e2e (3-step, guidance 2.0) | cos ≥ 0.999 | cos 0.9999998 |
| Paint PBR e2e (3-step, guidance 3.0, self-computed RoPE) | cos ≥ 0.999 | cos 1.0000000 |

Notes: the inpaint implementation is an exact port of scipy's Euclidean feature transform
(including tie-breaking, verified index-exact on a real bake mask with ~10 k ties) and of
OpenCV's `INPAINT_NS` fast-marching estimator, so the whole fill is bit-identical to Python.
PoseRoPE voxel indices replicate numpy's fp16 accumulation exactly, so the PBR e2e gate runs
on Swift-computed RoPE tables (no injection).

## License

MIT (see `LICENSE`). Vendored xatlas and the model weights carry their own licenses — see
`THIRD_PARTY_LICENSES.md`.

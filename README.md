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

## License

MIT (see `LICENSE`). Vendored xatlas and the model weights carry their own licenses — see
`THIRD_PARTY_LICENSES.md`.

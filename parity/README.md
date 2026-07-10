# Parity fixtures

The Swift port is verified against the known-good Python MLX ports by fixture files:
the Python side dumps inputs + expected outputs (deterministic seeds) as `.safetensors`;
the Swift side (`Tests/`, and the `hy3d parity-shape` / `hy3d parity-paint` print-panels)
replays them and gates on the DESIGN thresholds.

Fixtures are **not** checked in (see `.gitignore`) — regenerate them with the scripts here.

## Which venv runs what

The dumpers import the Python MLX ports, so each runs from its own repo root with `uv run`:

| Scripts | Run from (repo root) | Needs |
|---|---|---|
| `dump_dit_fixture.py`, `dump_swift_fixtures.py`, `python_from_fixture.py` | `Hunyuan3D-Shape-MLX` | `hy3dmlx` package + shape checkpoints under `weights/` |
| `dump_vae_fixture.py`, `dump_sched_fixture.py`, `dump_unet_base_fixture.py`, `dump_resrgan_fixture.py`, `dump_dino_fixture.py`, `dump_raster_fixture.py`, `dump_render_fixture.py`, `dump_bake_fixture.py`, `dump_pbr_unet_fixture.py`, `dump_pbr_e2e_fixture.py`, `dump_p20_e2e.py` | `Hunyuan-3D-Paint-MLX` | `hy3dpaint_mlx` package + paint weights under `weights/` |

Invocation pattern (shape example):

```bash
cd <Hunyuan3D-Shape-MLX>
PYTHONPATH=. FIXTURES_OUT=/path/to/fixtures uv run python /path/to/Hunyuan3D-Swift/parity/dump_dit_fixture.py
```

## Environment knobs

| Env | Meaning | Default |
|---|---|---|
| `FIXTURES_OUT` | Directory where fixtures are written | `./fixtures` (relative to the CWD) |
| `SHAPE_VARIANT` | Shape model slot: `mini` (shape-small) or `turbo` (shape-large) | `mini` |
| `SHAPE_MODEL` | Shape checkpoint dir override | per-variant default under `weights/` |
| `DEMO_IMAGE` | Conditioning image for the shape run fixture | `reference/Hunyuan3D-2.1/assets/demo.png` |
| `PARITY_MESH` | Mesh for the render/bake fixtures (any small manifold mesh) | required (no default) |

The Swift consumers resolve the same directory via `HY3D_FIXTURES` (tests and CLI) or
`--fixtures <dir>` (CLI), default `./fixtures`. The e2e mesh tests additionally resolve
checkpoints via `HY3D_SHAPE_SMALL` / `HY3D_SHAPE_LARGE` (else `<fixtures>/shape-{small,large}/`).

## What each script dumps

Shape fixtures are namespaced `shape_*` so shape and paint fixtures can share one directory
(both families have a "vae" and a "dino" fixture).

| Script | Fixture file(s) | Gate(s) |
|---|---|---|
| `dump_dit_fixture.py` | `shape_dit_fixture.safetensors` (mini) / `shape_dit_fixture_turbo.safetensors` (keys `x,t,cond,v[,guidance]`) | DiT forward cos ≥ 0.99999 |
| `dump_swift_fixtures.py` | `shape_vae_fixture[_turbo]` (`lat,q,sdf`) · `shape_dino_fixture` (`pixels,out`) · `shape_sigmas_fixture` (`flowmatch,consistency`) · `shape_run_fixture[_turbo]` (`cond,noise,sigmas,guidance`) | VAE grid cos ≥ 0.9999 · DINOv2 cos ≥ 0.9999 · sigmas maxabs ≤ 1e-6 · (input to e2e) |
| `python_from_fixture.py` | `shape_mesh_python_{mini,turbo}.safetensors` (`V,F`) + `.glb` for viewers | e2e mesh Chamfer ≤ 0.01 · bbox |
| `dump_vae_fixture.py` | `vae_weights` + `vae_fixture` (`z,img,ximg,mean`) | paint VAE enc/dec maxabs ≤ 1e-6 |
| `dump_sched_fixture.py` | `sched_fixture` (DDIM + UniPC tables/trajectories) | DDIM maxabs ≤ 1e-6 · UniPC maxabs ≤ 1e-5 |
| `dump_unet_base_fixture.py` | `unet_base_weights` + `unet_base_fixture` | SD2.1 UNet maxabs ≤ 1e-4 |
| `dump_pbr_unet_fixture.py` | `pbr_unet_fixture` (weights + ced/rope/dino + fwd) | PBR UNet cos ≥ 0.9999 |
| `dump_resrgan_fixture.py` | `resrgan_weights` + `resrgan_fixture` | RealESRGAN maxabs ≤ 1e-6 |
| `dump_dino_fixture.py` | `dino_weights` + `dino_fixture` (`px,out`) | DINOv2-giant cos ≥ 0.9999 |
| `dump_raster_fixture.py` | `raster_fixture` | face-id 100% · bary ≤ 2e-4 |
| `dump_render_fixture.py` | `render_fixture` (exact xatlas V/F/uv + control maps) | control maps PSNR ≥ 80 dB |
| `dump_bake_fixture.py` | `bake_fixture` (+ Python inpaint reference) | bake PSNR ≥ 100 dB |
| `dump_p20_e2e.py` | `p20_e2e_fixture` (weights + 3-step loop, guidance 3.0) | paint RGB e2e cos ≥ 0.999 |
| `dump_pbr_e2e_fixture.py` | `pbr_e2e_fixture` (weights + 3-step loop, guidance 3.0) | paint PBR e2e cos ≥ 0.999 |

Not yet scripted (tests exist and skip until the files appear):

- `inpaint_fixture.safetensors` (`texture,covered,filled`) — Python EDT fill + smoothing
  reference for the inpaint gate.
- `image_fixture.safetensors` (`ref512`) + `input.png` — optional `prepRGB` diagnostic in the
  `hy3d parity-paint` panel (not a DESIGN gate).

Note: the two e2e loop fixtures bake `guidance = 3.0` into their expected trajectories (the
value the dumpers used). The *pipeline* defaults are per-model (RGB 2.0 / PBR 3.0); the parity
replay intentionally matches the fixture, not the pipeline default.

## Full regeneration, 2×2 model lineup

Model slots: **shape-small** = `hunyuan3d-dit-v2-mini`, **shape-large** = `hunyuan3d-dit-v2-0-turbo`,
**paint-small** = `hunyuan3d-paint-v2-0` (RGB), **paint-large** = `hunyuan3d-paintpbr-v2-1` (PBR).
Run everything serially — one MLX job on the machine at a time.

```bash
FIX=/path/to/Hunyuan3D-Swift/fixtures            # or wherever HY3D_FIXTURES will point
D=/path/to/Hunyuan3D-Swift/parity

# ---- shape (from the Hunyuan3D-Shape-MLX repo root) ----
cd <Hunyuan3D-Shape-MLX>
PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/dump_dit_fixture.py
PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/dump_swift_fixtures.py
PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/python_from_fixture.py 256
SHAPE_VARIANT=turbo PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/dump_dit_fixture.py
SHAPE_VARIANT=turbo PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/dump_swift_fixtures.py
SHAPE_VARIANT=turbo PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/python_from_fixture.py 256

# ---- paint (from the Hunyuan-3D-Paint-MLX repo root) ----
cd <Hunyuan-3D-Paint-MLX>
MESH=<some small manifold mesh.glb>              # reference runs used the Hunyuan3D-2.1 case_1 asset
for s in dump_vae_fixture dump_sched_fixture dump_unet_base_fixture dump_resrgan_fixture \
         dump_dino_fixture dump_raster_fixture dump_p20_e2e dump_pbr_unet_fixture \
         dump_pbr_e2e_fixture; do
  PYTHONPATH=. FIXTURES_OUT=$FIX uv run python $D/$s.py
done
PYTHONPATH=. FIXTURES_OUT=$FIX PARITY_MESH=$MESH uv run python $D/dump_render_fixture.py
PYTHONPATH=. FIXTURES_OUT=$FIX PARITY_MESH=$MESH uv run python $D/dump_bake_fixture.py

# ---- verify on the Swift side ----
cd /path/to/Hunyuan3D-Swift
HY3D_FIXTURES=$FIX swift test
swift run -c release hy3d parity-shape --fixtures $FIX --weights <mini ckpt dir> --weights-turbo <turbo ckpt dir>
swift run -c release hy3d parity-paint --fixtures $FIX
```

Sizes: the e2e/weight-carrying paint fixtures are large (the PBR e2e fixture embeds the full
UNet, ~7.8 GB); expect ~28 GB for the complete set.

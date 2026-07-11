# Hunyuan3D-Paint — MLX

> **Reference implementation.** This is the known-good Python MLX paint port that the native
> [Hunyuan3D-Swift](../../README.md) package is parity-tested against. It sits in the middle of
> the parity chain: Swift ↔ this MLX port ↔ the torch/CUDA reference (via the `cr_cpu` oracle).
> Run everything below from `python/paint` via `uv run`.

Fully **MLX-native** Apple-Silicon texture/paint generation for Hunyuan3D, built
**parity-first**: every stage is gated against a PyTorch/CUDA-equivalent oracle so the
output matches the reference CUDA pipeline — not just "looks close."

Sibling of the shape port [`python/shape`](../shape). Paint consumes
only a **mesh + the input image** (never shape latents), so any shape mesh is textureable.

## Targets (two models)

| Role | Checkpoint | Output | Base |
|---|---|---|---|
| **small** | `hunyuan3d-paint-v2-0` | RGB texture | SD2.1 UNet + 2 multiview-attn paths |
| **large** | `hunyuan3d-paintpbr-v2-1` | PBR (albedo + metallic-roughness) | SD2.1 UNet + 4 attn paths + DINO + PoseRoPE |

Both share ~90% of the stack (VAE, CLIP, base UNet, multiview/reference attention,
scheduler, **renderer/rasterizer/baker**), so the small model is the parity-establishing
step and the large model layers material/DINO/PoseRoPE on top.

## Why a fresh build (the parity diagnosis)

The earlier `ZimengXiong/Hunyuan3D-MLX` repo textures on MLX but **does not reach CUDA
parity**. Root cause is structural, not "library immaturity": the Hunyuan CUDA reference
rasterizer is **`custom_rasterizer`** (a scanline rasterizer: pixel-center `+0.5`,
`(W-1)/(H-1)` screen map, 18-bit depth-token min-reduce, 3 perspective-correct
barycentrics, **no antialiasing**). The old repo renders with **`mtldiffrast`** — a faithful
**nvdiffrast** (different barycentric layout, edge-function fill, float z, and silhouette
**antialiasing on**). nvdiffrast ≠ custom_rasterizer, so it can never bit-match CUDA, and the
AA smooths silhouettes the model was never trained on. See
[`research/`](research/) for the full analysis.

**The fix:** a `custom_rasterizer`-faithful Apple-Silicon backend. `custom_rasterizer`'s core
CPU functions are pure C++ (raw pointers, no CUDA, no torch), so we compile them as a
**bit-exact oracle** (`oracle/cr_cpu`), then build a fresh numpy→MLX/Metal rasterizer gated
bit-exact against it. AA off on the control path.

## Layout

```
hy3dpaint_mlx/          # the MLX-native package (WIP)
  raster/cr_raster.py   # custom_rasterizer-faithful rasterizer (numpy now; Metal later)
oracle/cr_cpu/          # torch-free pybind11 build of custom_rasterizer's CPU twin (parity oracle)
tests/                  # per-stage parity gates
research/               # model-family map, paint architecture, interop, port plan (see 01–04)
metadata/               # downloaded model configs for all paint variants
reference/Hunyuan3D-MLX          # prior repo (reference only — not reused wholesale)
reference/Hunyuan3D-2.1          # torch reference (the CUDA pipeline; git-ignored checkout)
```

`reference/` is a git-ignored checkout you clone yourself when regenerating oracle dumps; the
`PAINT_MESH` / `PAINT_IMG` defaults point into it (`reference/Hunyuan3D-2.1/hy3dpaint/assets/case_1/`),
and any shape-port mesh works too — override both with the env vars below.

## Status — both models work end-to-end, every stage parity-gated

| Stage | Parity vs CUDA/diffusers/reference |
|---|---|
| Rasterizer (cr-faithful) | **bit-exact** (face-id 100%, bary ~1.8e-7) |
| 2D primitives (GroupNorm/Resnet/Up·Downsample/VAEAttn) | cosine 0.9999999 |
| Attention / Transformer2D / timestep | cosine 1.0 |
| AutoencoderKL VAE | encode cos 1.0, decode 111 dB |
| SD2.1 base UNet | cosine 1.0 |
| Renderer control maps (normal/position) | cosine 1.000000, 88–159 dB |
| **Small 2.5D multiview UNet (2.0 RGB)** | **cosine 1.0000000** (real weights) |
| **Large 2.5D PBR UNet (2.1, MDA+RA+MA+DINO+RoPE)** | **neural-exact cos 1.0000000**; cos 0.99996 incl. fp16 voxel-quant |
| 3D PoseRoPE | cosine 1.0 |
| DINOv2-giant | cosine 1.0 |

Run (no PyTorch in the inference path):
```bash
uv sync --group dev
bash oracle/cr_cpu/build.sh                      # CUDA-equivalent parity oracle (no torch/CUDA)
uv run pytest tests/ -q                           # all parity gates
uv run python scripts/run_paint.py                # small (2.0 RGB) -> outputs/textured_mesh.glb
uv run python scripts/run_paint_pbr.py            # large (2.1 PBR) -> outputs/textured_mesh_pbr.glb (albedo+MR)
```
Env knobs: `PAINT_RES`, `PAINT_STEPS`, `PAINT_TEX`, `PAINT_MESH`, `PAINT_IMG`.

### Torch oracle venv (for the parity tests)

Most tests in [`tests/`](tests) gate MLX stages against a **PyTorch oracle** and are
`skipif`-gated on a separate `.venv-oracle` interpreter (and, for the render/bake tests, a
mesh at `PAINT_MESH`). Without it they skip cleanly; to actually run them, stand up the
oracle venv once (it is git-ignored, torch-only, and never on the inference path):

```bash
uv venv .venv-oracle --python 3.12
VIRTUAL_ENV=.venv-oracle uv pip install torch diffusers transformers safetensors trimesh opencv-python
# oracle scripts also import the torch reference from reference/Hunyuan3D-2.1 (clone it there)
```

The oracle interpreter is discovered at `.venv-oracle/bin/python`. `bash oracle/cr_cpu/build.sh`
compiles the torch-free `custom_rasterizer` CPU twin (`libcr_cpu_oracle.dylib`) — also git-ignored;
rebuild it locally.

**Remaining polish** (not parity-critical): Metal-compute rasterizer (speed; numpy works now),
exact UniPC/EulerAncestral scheduler (DDIM zero-SNR used now), `meshVerticeInpaint` C++ for exact
texture-hole parity (scipy nearest-fill now), and matching torch's fp16 voxel-quantization.

Parity methodology mirrors the shape port: per-stage torch oracle, fp32→fp16, cosine/PSNR/IoU gates.

# Hunyuan3D Shape → MLX Port — Execution Plan

**Goal:** the best-possible **fully MLX-native** Apple-Silicon image→mesh shape generator.
**No PyTorch in the inference path** — torch exists only as a dev-time parity oracle.
Optimize for speed + low memory. Target: the 0.6B `mini` (architecture generalizes to full 2.1).

**Scope:** shape only (`hy3dshape` / image→mesh). NOT texture/paint (`hy3dpaint`). The shape
pipeline is **pure PyTorch, zero custom CUDA/C++** — that is what makes a clean native port possible.

This document is the *what* and the *acceptance bar*. Sequencing/pacing is the team's call.

---

## 1. Target model — verified facts (from `tencent/Hunyuan3D-2mini` config.yaml)

| Sub-model | Architecture | Key dims | Params | Notes |
|---|---|---|---|---|
| **DiT** `Hunyuan3DDiT` | FLUX-style: 8 double-stream + 16 single-stream blocks, modulation/adaLN, RMS QK-norm | hidden 1024, heads 16, in_ch 64, ctx_in 1536, mlp×4 | ~0.56B | **No MoE. No RoPE** (`pe=None`, hunyuan3ddit.py:391). `qkv_bias=true`, `guidance_embed=false` |
| **Conditioner** DINOv2-giant | ViT, **SwiGLU FFN** | hidden 1536, 40 layers, 24 heads, patch 14, img 518 → 1370 tokens | ~1.1B | Bigger than the DiT. Runs once per generation |
| **VAE** `ShapeVAE` | vecset (3DShape2VecSet/CLAY lineage) | num_latents **512**, embed_dim 64, width 1024, 16 decoder layers, heads 16, num_freqs 8 | ~0.21B | `qk_norm=true`, `scale_factor=1.0188137142395404` (read from config at runtime, don't hardcode) |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | num_train_timesteps 1000 | — | sigmas = `linspace(0,1,steps)` + shift warp |

Mini ships `model.fp16.safetensors`. fp16 footprint ≈ 3.8 GB; 4-bit DiT/DINO brings it well under
2 GB. The same module code, loaded with full-2.1 configs (`HunYuanDiTPlain` + MoE, larger VAE),
extends the port upward later.

---

## 2. Dataflow & device placement

```
image ─rembg(CPU/onnx)─► preprocess(cv2/PIL) ─► DINOv2-giant (MLX) ──┐  once
                                                                     ▼
              ┌─────────── MLX denoise loop (N steps × CFG) ───────────┐
   noise ───► │  Hunyuan3DDiT velocity  →  CFG combine  →  Euler step  │  HOT PATH #1
              └────────────────────────────────────────────────────────┘
                                                                     ▼
        ShapeVAE decoder (MLX): 512 latents → dense/FlashVDM SDF grid query   HOT PATH #2
                                                                     ▼
        marching cubes (skimage, CPU/numpy)  ─►  trimesh/pymeshlab  ─►  GLB/PLY
```

- **MLX (everything compute-heavy):** DINOv2-giant, DiT, sampler, VAE decoder + grid query.
- **Irreducible CPU/numpy (keep as-is):** rembg, image preprocess, marching cubes, mesh post/export.
- **The one unavoidable boundary:** `grid_logits` (MLX) → numpy → skimage marching cubes.
  Everything before it stays on-device in unified memory — no host copies inside the loops.

---

## 3. Prior-art reuse map — `reference/hunyuan3d-omni-mlx` (cbun)

| Component | cbun file | How to use it |
|---|---|---|
| VAE decoder + grid query | `models/autoencoders/mlx_shape_vae.py` | **Reuse** — set mini config; fix key prefix; add `scale_factor` divide; `ln_post` eps=1e-5 |
| Euler loop / CFG / device bridges | `pipelines/pipeline_generation_sit_omni.py` | **Reuse** loop skeleton + bridge helpers; re-derive sigma schedule against FlowMatchEuler (not cbun's SiT transport) |
| Device/dtype runtime | `runtime.py` | **Reuse ~as-is** (incl. bf16→fp16 guard) |
| DINOv2 | `models/conditioners/mlx_dinov2.py` | **Adapt** — extend to **SwiGLU FFN + giant dims**; fix Resize antialias for parity |
| DiT | `models/denoisers/mlx_hunyuandit.py` | **Don't reuse** — it's `HunYuanDiTPlain`+MoE. Build the FLUX-style `Hunyuan3DDiT` from the **mlx-examples FLUX** template (drop RoPE & MoE) |
| Loader | (cbun loads per-submodule dirs) | **Rewrite** — mini is one `config.yaml` + one `.safetensors` split by `model.`/`vae.`/`conditioner.` prefix |

Torch reference source: `reference/Hunyuan3D-2.1/hy3dshape/`. DiT template: mlx-examples FLUX.

---

## 4. Proposed package layout

```
src/hy3dmlx/
  layers.py            # shared: linear / layernorm(fp32) / rmsnorm / sdpa / gelu / swiglu / quantized-linear
  models/dit.py        # Hunyuan3DDiT — FLUX double/single stream, no RoPE/MoE
  models/shape_vae.py  # decoder path + Vanilla & FlashVDM volume decoders
  models/dinov2.py     # DINOv2-giant + SwiGLU
  sampler.py           # FlowMatchEuler in MLX
  pipeline.py          # orchestration, the single MLX↔numpy boundary, generate()
  convert.py           # torch safetensors → MLX safetensors (+ optional quantization)
reference/             # Hunyuan3D-2.1 (torch oracle) + hunyuan3d-omni-mlx (cbun)
tests/                 # per-stage parity tests vs torch oracle
```

---

## 5. Build items (each ships with its acceptance gate)

**Weights & conversion** — `convert.py`
- Download `tencent/Hunyuan3D-2mini`. Load safetensors → split by prefix → `mx.array(...).astype(fp16)`.
- `Linear` copies 1:1 (`[out,in]` matches MLX). **Only** transpose DINO patch `Conv2d` NCHW→NHWC.
- Emit MLX-native safetensors; support a `--quantize {none,8,4}` flag (`mx.quantize`, group_size 64)
  for the DiT and DINO large-linears; keep norms, embedders, and the VAE in fp16.
- *Gate:* converted (and 4-bit) weights load into the MLX modules with no missing/extra keys.

**Shared layers** — `layers.py`
- fp32-internal LayerNorm/RMSNorm, `mlx.fast.scaled_dot_product_attention`, GELU(erf), SwiGLU,
  and a quantized `Linear` wrapper so any module can run dense or quantized.
- *Gate:* unit parity vs torch for each primitive on random inputs.

**DiT** — `models/dit.py` (the core)
- `latent_in` / `cond_in` / `time_in`; 8× `DoubleStreamBlock` (separate img/cond streams + joint
  attention over concat); 16× `SingleStreamBlock`; `LastLayer` (adaLN). Modulation + RMS QK-norm.
- Omit RoPE (`pe=None`) and MoE (mini has neither). fp32 norms; SDPA fused.
- *Gate:* per-block activation parity vs torch oracle on a fixed input; full-forward max-abs within
  fp16 tolerance; quantized forward within an agreed looser tolerance.

**Sampler** — `sampler.py`
- Reimplement `FlowMatchEulerDiscreteScheduler` (shift-warped sigma schedule + `x += dt·v`); CFG
  `split(2)`. Skip `transport/` + torchdiffeq. One `mx.eval(latents)` per step; nothing else evals
  inside the loop (keep the whole step graph fused).
- *Gate:* with a fixed DINO embedding + seed, MLX latent trajectory matches torch oracle.

**VAE decoder + grid query** — `models/shape_vae.py`
- Decoder path only: `post_kl → transformer(16×) → geo_decoder` (Fourier-embed query pts → cross-attn
  → SDF logit) over the dense `(R+1)³` grid, chunked with `mx.eval` per chunk.
- **Divide latents by `scale_factor` before `post_kl`.** **`geo_decoder.ln_post` eps=1e-5** (others 1e-6).
- Skip the encoder/`pre_kl`/Gaussian (training-only; needs `torch_cluster.fps`).
- Also implement **FlashVDM** octree decode (top-k sparse cross-attention via `mx.take_along_axis` +
  `mx.argpartition`; 3×3×3 dilate via `mlx.nn.Conv3d`; edge-pad via slicing) for high-res speed.
- *Gate:* MLX grid logits match torch oracle (fp32 < 1e-4); Vanilla and FlashVDM paths agree on the
  mesh within Chamfer tolerance; surface visually matches.

**Conditioner** — `models/dinov2.py`
- DINOv2-giant in MLX: patch Conv2d (NHWC), CLS + interpolated pos-embed, 40× (LayerNorm → MHSA →
  LayerScale → LayerNorm → **SwiGLU** → LayerScale), final LN. CFG null branch = zeros.
- Match reference preprocessing (Resize **antialias=True** + CenterCrop + ImageNet normalize).
- *Gate:* `last_hidden_state` cosine-sim > 0.999 vs torch DINOv2-giant; meshes match the torch path.

**Pipeline integration** — `pipeline.py`
- Reuse cbun `runtime.py`. Bespoke loader: parse `config.yaml`, split one safetensors by prefix,
  instantiate the three MLX sub-models (dense or quantized).
- `generate(image) → mesh`: rembg → preprocess → DINO(MLX) → MLX sample → MLX VAE decode → the single
  numpy hop → skimage MC → trimesh export. rembg/preprocess/postprocess unchanged. Use pymeshlab
  `FaceReducer` (the `MeshSimplifier` binary is x86/CUDA — absent on arm64).
- *Gate:* `image → demo.glb` end-to-end with zero torch imports in the inference path.

**Validation & benchmark** — `tests/`
- End-to-end mesh parity vs torch oracle (Chamfer distance); profile both hot paths; record
  wall-clock + peak memory across fp16 / 4-bit / step-count / octree-resolution sweeps.
- *Gate:* parity within agreed Chamfer threshold; benchmark table committed.

---

## 6. Parity strategy (non-negotiable)

Marching cubes thresholds the iso-surface at level 0, so small fp16 drift moves vertices.
- Stand up the torch oracle early; dump per-stage tensors (DINO out, per-step latents, VAE grid).
- **Validate every stage in fp32 first**, then drop to fp16, then quantized — gating at each step.
- Run all norms in fp32 internally.
- Don't chase bit-exactness (MLX RNG ≠ torch RNG; reduction order differs): gate on per-stage tensor
  tolerance and end-to-end Chamfer distance.

---

## 7. Performance & memory — every lever, pulled

- **Stay fully native:** the conditioner, DiT, sampler, and VAE all run in MLX, so there are **no
  torch↔MLX bridges inside any loop** — the only host hop is the single grid→numpy handoff to skimage.
- **Quantize** the DiT and DINO large-linears (4-bit) — cuts weight RAM/bandwidth to <2 GB total and
  speeds the memory-bound matmuls; norms/embedders/VAE stay fp16.
- **FlashVDM octree decode** instead of full-dense at high `octree_resolution` — fewer neural queries
  and a smaller marching-cubes grid.
- **Distilled checkpoints:** `mini-turbo` / `mini-fast` cut the step count (the dominant cost is N×CFG
  DiT forwards) with the same module code.
- **Tune** `octree_resolution`, `num_inference_steps`, and chunk size; `mx.compile` the DiT block and
  the geo-decoder query; one `mx.eval` per outer iteration to bound graphs.
- Marching cubes is a CPU floor — keep it off the critical path by overlapping where possible and
  pruning the grid via FlashVDM.

---

## 8. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| `scale_factor` omitted before `post_kl` | Wrong mesh scale, silent | explicit step; assert vs config |
| `ln_post` eps 1e-5 vs 1e-6 | Logit drift at iso-surface | pass eps per-call |
| bf16 unsupported on MPS (oracle runs) | Oracle crash/incorrect | coerce bf16→fp16/fp32 in oracle |
| Wrong sigma schedule (cbun linspace vs FlowMatchEuler) | Mesh drifts | re-derive from `schedulers.py`, gate vs oracle |
| DINO SwiGLU/giant dims or antialias missed | Garbage conditioning, no crash | cosine-sim gate vs torch |
| Attention head-split layout mismatch | Subtly wrong outputs | single-layer activation diff |
| Lazy-eval graph blowup | OOM / stalls | one `mx.eval` per step & per grid chunk |
| FlashVDM top-k routing correctness | Holes/artifacts at high res | gate FlashVDM vs Vanilla on the same latent |
| Quantization accuracy loss | Surface degradation | per-stage tolerance gate; keep norms/VAE fp16 |
| `MeshSimplifier` binary absent on arm64 | Export fails | pymeshlab `FaceReducer` |

---

## 9. Dependencies (logical, not a timeline)

```
convert + layers ─► DiT ───┐
                  ├► sampler ┤
                  └► VAE ────┼─► pipeline ─► validation/benchmark
        DINOv2-giant ───────┘
```
`convert`/`layers` first (they unblock everything). DiT, sampler, VAE, and DINOv2 are independent
once the torch oracle exists — parallelize freely. `pipeline` joins them; validation closes the loop.

---

## 10. Environment (uv)

```bash
cd python/shape                   # this reference implementation, from the repo root
uv init --python 3.12 .            # MLX wheels: 3.11/3.12, NOT 3.14
uv add mlx numpy safetensors huggingface-hub pyyaml \
       trimesh pymeshlab scikit-image opencv-python rembg pillow einops
uv add --optional oracle torch torchvision transformers   # torch = parity oracle ONLY
uv run hf download tencent/Hunyuan3D-2mini --local-dir weights/Hunyuan3D-2mini
uv run python -m hy3dmlx.convert weights/Hunyuan3D-2mini --quantize 4
```

**References:** torch source `reference/Hunyuan3D-2.1/hy3dshape/`; MLX prior art
`reference/hunyuan3d-omni-mlx/hy3dshape/`; DiT template = mlx-examples FLUX.

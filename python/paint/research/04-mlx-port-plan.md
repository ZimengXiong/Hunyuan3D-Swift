# 04 — MLX Port Plan: Hunyuan3D-2.1 PBR Paint Pipeline

A "what + acceptance bar" execution plan for a **fully MLX-native** Hunyuan3D-2.1 PBR paint pipeline with **per-stage parity against a PyTorch oracle**. Modeled on the shape-port `PLAN.md`: every build item carries an explicit acceptance gate, and parity is enforced stage-by-stage so errors never compound.

---

## 0. Corrections & status (supersedes stale parts below — read first)

This doc was drafted 2.1-only and before the prior-repo discovery. Corrected scope/decisions:

1. **Two targets, not one.** Build **small `hunyuan3d-paint-v2-0` (RGB)** first (SD2.1 UNet + 2 attn paths: multiview + reference), then **large `hunyuan3d-paintpbr-v2-1` (PBR)** layering material/DINO/PoseRoPE/12-ch input on top. ~90% shared infra; the renderer is shared. The §2–§7 deep-dive below is the large (PBR) model; the small model is a strict subset.

2. **Rasterizer decision = `custom_rasterizer`-faithful, NOT mtldiffrast.** §3.1 below ranks mtldiffrast #1 — **rejected.** Diagnosis: the prior repo's CUDA-parity gap is structural — mtldiffrast is a faithful *nvdiffrast* (different barycentric layout, edge-function fill, float z, silhouette **AA on**), but the Hunyuan CUDA reference is `custom_rasterizer` (scanline, pixel-center `+0.5`, 18-bit depth-token min-reduce, 3 perspective-correct barycentrics, **no AA**). nvdiffrast can't bit-match custom_rasterizer. The real fix is a custom_rasterizer-faithful backend, AA off on the control path. mtldiffrast may stay as an optional fast/approx mode only.

3. **DONE & verified:** a **torch-free CUDA-equivalent oracle** (`oracle/cr_cpu`, the reference CPU twin over a plain C ABI via ctypes) + a **numpy custom_rasterizer-faithful rasterizer** (`hy3dpaint_mlx/raster/cr_raster.py`), gated **bit-exact** (`tests/test_raster_parity.py`: face-id 100%, barycentric maxabs ~1.8e-7 on random + axis-aligned scenes). Parity landmine found: build the oracle with **`-ffp-contract=off`** so clang FMA doesn't flip exact-edge pixels (numpy/Metal must match: no fused multiply-add in the signed-area math).

4. **NEXT:** a Metal-compute rasterizer implementing the same algorithm (gated vs the oracle, for speed), then the renderer (normal/position control maps, `back_sample` bake, UV inpaint), then the shared MLX neural core.

---

## 1. Goal, Scope, Target

**Goal.** Port the Hunyuan3D-**2.1** PBR texture (paint) pipeline to a torch-free MLX inference package (`hy3dpaint_mlx`), achieving numerical parity with a PyTorch oracle built from the reference `hy3dpaint` code, gated stage-by-stage with cosine / maxabs (neural) and IoU / PSNR / Chamfer (geometry) metrics.

**Target model (fixed).** `tencent/Hunyuan3D-2.1`, subfolder `hunyuan3d-paintpbr-v2-1`. This is the **first open-weights PBR paint** release. It emits **albedo + metallic-roughness (`mr`)** materials (`pbr_settings=["albedo","mr"]`, `num_view=6`). Base diffusion backbone = stock SD2.1-base zero-terminal-SNR v-prediction UNet wrapped as the custom `UNet2p5DConditionModel`.

**In scope.**
- The neural core: SD2.1 `UNet2DConditionModel` + the 2.5D multiview/PBR wrapper, `AutoencoderKL`, DINOv2-giant conditioner, learned material tokens, UniPC scheduler, 3-way CFG.
- The geometry/host stack required to actually produce a textured mesh: rasterizer (normal/position control maps + UV-space rasterization), back-projection baker, UV inpaint, remesh + UV unwrap, GLB packing.
- Per-stage torch oracle harness + parity gates.

**Out of scope (explicitly).**
- Training / fine-tuning. Inference only. The training-time conv_in zero-init, loss split, and `set_learned_parameters` logic are reference-only context.
- The 2.0 RGB paint model, 2.0-turbo, delight model. (2.0 configs in the corpus are reference for shared backbone topology only.)
- The shape DiT / ShapeVAE (lives in the separate shape MLX port; paint consumes only `mesh + input image`, never shape latents — confirmed: paint `__call__(mesh_path, image_path, output_mesh_path)`).
- ControlNet conditioning path (exists at `model.py:105-110` but `control_net_config` defaults None; `init_control_from` empty in the released yaml). The canonical model uses **channel-concat** conditioning.
- The `grid_neighbor.cpp` octree builder (dead code for the paint render/bake path).
- Real-ESRGAN super-res is **optional** for first-pass parity (stub with bicubic/Lanczos x4); the actual `RealESRGAN_x4plus` weights are needed only for final visual parity.

**Definition of done.** End-to-end: `(mesh.glb, input.png) → textured .glb` runs on Apple Silicon with no torch in the runtime path, and every parity-gated stage passes its acceptance bar against the oracle.

---

## 2. Verified Architecture Facts

All values verified against on-disk `unet/config.json`, `vae/config.json`, `text_encoder/config.json`, `image_encoder/config.json`, `scheduler/scheduler_config.json` for `hunyuan3d-paintpbr-v2-1`, plus the reference source (`modules.py`, `model.py`, `attn_processor.py`, `pipeline.py`).

### 2.1 Sub-models

| Sub-model | Class | Key dims | Params (approx) | Role |
|---|---|---|---|---|
| Base UNet | `UNet2DConditionModel` (SD2.1-base) | in/out=4, cross_attention_dim=1024, block_out=[320,640,1280,1280], heads=[5,10,20,20], per-head dim 64, sample_size=64, layers_per_block=2, norm_num_groups=32, act=silu, use_linear_projection=true | part of ~2B paint stack | v-prediction backbone |
| 2.5D wrapper | `UNet2p5DConditionModel` | conv_in expanded 4→12; n_pbr=2; pbr_token_channels=77 | wrapper + trained adds | multiview + PBR + reference + DINO injection |
| Reference UNet | `unet_dual` = `deepcopy(unet)` (frozen) | identical topology, no extra-attn flags | mirrors base | dual-stream reference KV writer |
| VAE | `AutoencoderKL` | latent_channels=4, block_out=[128,256,512,512], in/out=3, sample_size=768, layers_per_block=2, norm_num_groups=32, **scaling_factor not in config → diffusers default 0.18215** | ~standard SD VAE | encode normal/position/reference; decode views |
| Text encoder | `CLIPTextModel` (from `stabilityai/stable-diffusion-2`) | hidden 1024, intermediate 4096, 23 layers, 16 heads, max_pos 77, vocab 49408 | ~340M | **bypassed at inference** (learned tokens used); prompt "high quality" inert |
| Image encoder | `CLIPVisionModelWithProjection` (CLIP-H/14) | hidden 1280, intermediate 5120, 32 layers, 16 heads, patch 14, image 224, projection 1024 | ~630M | **listed in model_index but unused** in forward; verify dormant |
| DINOv2 | `facebook/dinov2-giant` (stock HF) | hidden 1536, 24 heads, 40 layers, patch 14, image 518 → 1370 tokens, SwiGLU FFN, LN eps 1e-6 | ~1.1B | semantic conditioner on first reference image only |
| ImageProjModel (dino) | Linear 1536→4×1024 + LayerNorm | 1536→4 tokens × 1024 | trivial | projects DINO features to 4 cross-attn tokens |

> Note: the digest's "DINOv2 dino-large hidden 1024" reading is contradicted by `textureGenPipeline.py:45 dino_ckpt_path="facebook/dinov2-giant"` and `modules.py:38-52 AutoModel.from_pretrained("facebook/dinov2-giant")`. **DINOv2-giant (1536-dim) is authoritative.**

### 2.2 The five attention mechanisms (per `Basic2p5DTransformerBlock`)

| # | Name | Processor | Reshape semantics | Notes |
|---|---|---|---|---|
| 0 | MDA material-aware self-attn | `SelfAttnProcessor2_0` (=attn1) | `(b n_pbr n) l c` per-view per-material | albedo→base q/k/v/out; mr→cloned `to_*_mr`; concat |
| 1 | RA reference attn | `RefAttnProcessor2_0` (`attn_refview`) | query = **albedo only** `[:,0,...]`; cross to `b (n_ref l) c` | shared Q/K, per-material V (`to_v`/`to_v_mr`); scaled by `ref_scale` |
| 2 | MA multiview attn | `PoseRoPEAttnProcessor2_0` (`attn_multiview`) | `(b n_pbr) (n l) c` — all 6 views concat on seq | 3D RoPE from voxelized position maps; scaled by `mva_scale`; only if num_in_batch>1 |
| 3 | Text cross-attn | (=attn2) | encoder_hidden_states = learned 77×1024 tokens | NOT real CLIP text |
| 4 | DINO cross-attn | zero-init `attn_dino` | dino tokens repeated over n_pbr·n | parallel to attn2 |

Zero-init residual strategy: MA/RA/DINO copy base weights then **zero their `to_out`** so the wrapper starts as identity.

### 2.3 Scheduler & CFG

- Inference scheduler: `UniPCMultistepScheduler.from_config(orig_config, timestep_spacing="trailing")`, **15 steps**, guidance_scale=3.0. (On-disk `scheduler_config.json` is `DDIMScheduler` v_prediction / scaled_linear / trailing / rescale_betas_zero_snr=true — used as the base config UniPC is built from.)
- **3-way CFG** (uncond / ref / full): `prompt_embeds = cat([neg, pos, pos])`, latents repeated ×3. `noise = uncond + g·vs·(ref-uncond) + g·vs·(full-ref)`.
- `ref_scale = [0.0, 1.0, 1.0]` (uncond branch zeroes reference attention); `dino = cat([zeros, zeros, dino])` (only full branch sees DINO).
- `view_scale_tensor` from `cam_mapping(azim)`, but `camera_azims` is never passed in production → all-ones. Keep `cam_mapping` for fidelity; can hardcode 1.0.

### 2.4 Canonical tensor shapes (6 views, 512px, B=1, N_pbr=2, N_ref=1)

- Reference → ref_latents `[1,1,4,64,64]`.
- Normal & position maps each VAE-encoded → `[1,6,4,64,64]`, repeated over n_pbr.
- Sample `[1,2,6,4,64,64]`; after channel concat `[1,2,6,12,64,64]`; flattened `[12,12,64,64]` into conv_in.
- encoder_hidden_states (learned tokens) → `[1,2,77,1024]` → flattened `[12,77,1024]`.
- At inference CFG tripling: latents `[36,4,64,64]` → `[36,12,64,64]` after concat.
- UNet out `[12,4,64,64]` → split albedo/mr each `[1,1,6,4,64,64]` → VAE decode → 512×512.

### 2.5 Renderer / baker facts

- Rasterizer outputs `findices [H,W] int32` (1-based, 0=bg) + `barycentric [H,W,3] f32` (perspective-correct).
- Z-buffer token: `token = (int)(depth*262144)*2147483647 + (idx+1)`, min wins. Screen map `sx=(x/w*0.5+0.5)*(W-1)+0.5`, `sz=z/w*0.49999+0.5`. **CPU path `rasterize_image_cpu` is bit-identical to the CUDA kernel.**
- MeshRender defaults: camera_distance=1.45, camera_type='orth' ortho_scale=1.2, texture_size=1024 (paint config render_size=2048, texture_size=4096), bake_mode='back_sample', shader_type='face', bake_angle_thres=75°.
- Mesh normalization: flip X/Y, swap Y/Z, UV V-flip, auto-center, scale_factor=1.15; remesh target 40000 faces (trimesh quadric decimation). These directly affect baked-texture coordinates.

---

## 3. Dataflow + Device Placement

```
input image (PIL) ─┐
                   ├─ host: rembg/composite-white, resize 512        [numpy/PIL/opencv]
mesh (.glb/.obj) ──┘
        │
        ├─ remesh → 40k faces (trimesh + pymeshlab)                   [HOST/CPU]
        ├─ UV unwrap (xatlas)                                         [HOST/CPU]
        ├─ MeshRender.set_mesh (axis flip/swap, scale 1.15)           [HOST/numpy]
        │
        ├─ render_normal_multiview / render_position_multiview ───────[RASTERIZER ★ crux]
        │      ↓ (6 normal PIL + 6 position PIL control maps)
        │
   ════ MLX NEURAL CORE ════════════════════════════════════════════════════
        ├─ VAE.encode(reference, normal, position) → latents          [MLX]
        ├─ DINOv2-giant(first ref image) → [1,1370,1536]              [MLX] (one-shot, cacheable)
        ├─ learned_text_clip_{albedo,mr,ref} (stored params)          [MLX]
        ├─ position maps → voxel RoPE indices (precompute)            [HOST/numpy → MLX const]
        ├─ unet_dual once @ t=0, mode="w" → condition_embed_dict      [MLX] (cached over 15 steps)
        ├─ UniPC loop ×15: main UNet (mode="r") + 3-way CFG           [MLX]
        └─ VAE.decode → {albedo:[6], mr:[6]} 512×512 PIL              [MLX]
   ═════════════════════════════════════════════════════════════════════════
        │
        ├─ super-res x4 each view (Real-ESRGAN | bicubic stub)        [MLX or torch-MPS stopgap]
        ├─ resize to render_size 2048                                  [HOST/opencv]
        ├─ bake_from_multiview (back_sample) albedo + mr ────────────[RASTERIZER ★ + numpy gather/scatter]
        ├─ uv_inpaint (meshVerticeInpaint C++ + cv2.inpaint)          [HOST/CPU, recompile as-is]
        └─ save OBJ+MTL+JPG → create_glb_with_pbr_materials (pygltflib) [HOST/CPU]
```

**What runs in MLX:** VAE encode/decode, DINOv2-giant, ImageProjModel, the entire UNet2p5D + base UNet + dual-stream UNet, all attention, the UniPC step math, CFG combine.

**Irreducible CPU/numpy/host:** remesh (trimesh/pymeshlab), UV unwrap (xatlas), camera algebra, voxel-index precompute (fed as MLX constants), back_sample gather/scatter + cos-weight blend (numpy or MLX, parity-trivial in fp32), Canny edge / boundary dilation (opencv), UV inpaint (C++ pybind11 + cv2), GLB packing (pygltflib).

### 3.1 The RASTERIZER / BAKER — the crux

The rasterizer is the only genuinely CUDA-bound piece and the single highest-risk parity surface. It feeds the diffusion conditioning (normal/position maps the UNet was trained on) **and** the baker (face-id + barycentric + depth + visibility). A one-pixel shift in sample convention misaligns conditioning and shifts baked texels.

**Recommended approach (ranked):**

1. **Primary — native Metal `mtldiffrast` (already in-repo).** This working directory's project ships `libraries/mtldiffrast` (nvdiffrast-API-compatible Metal differentiable rasterizer: `rasterize`/`interpolate`/`texture`/`antialias`) and `libraries/mtlbvh` (cubvh-compatible BVH), already wired into `mesh_render.py` with backend auto-select (`'mtl'` on MPS/CPU, `'cr'` only on CUDA). Use this as the production backend. **Two documented parity fixes must be carried:** (a) depth convention — mtldiffrast expects Metal `z∈[0,w]` vs pipeline OpenGL-style `z∈[-w,w]`, remap `pos_mtl[...,2] = pos_mtl[...,2]*0.5+0.5`; (b) winding/culling — mtldiffrast culls opposite winding vs custom_rasterizer, flip face index or disable culling.

2. **Fallback for guaranteed bit-parity — compile the reference `rasterize_image_cpu` as a standalone pybind11 module.** The CUDA kernel has a bit-identical scalar CPU twin (`rasterizer.cpp:3-123`). Drop the `.cu` from setup, swap `CUDAExtension`→`CppExtension` (or a hand `c++ -O3 -shared` build like `compile_mesh_painter.sh`). Runs fine on M-series CPU, ~hours of work, exact parity. Use this as the **parity oracle for the rasterizer in CI** (there is no CUDA reference available on Mac) and as a correctness fallback if mtldiffrast quality is insufficient.

3. **Independent correctness check — Open3D `RaycastingScene.cast_rays`** returns `primitive_ids` (face id) + `primitive_uvs` (barycentric) + `t_hit` (depth) on arm64. Use as a second oracle in CI.

**Avoid for parity:** nvdiffrast (CUDA-only), official `custom_rasterizer` compiled as-is (CUDA, fails on Mac), pytorch3d optimized kernel (no MPS C++), OpenGL/EGL (deprecated/absent on macOS). These differ in sampling conventions and will not bit-match.

**Baker + inpaint:** back_sample gather/scatter, cos^6 weighting, `Σtex·cos/Σcos` blend are elementwise → parity automatic in fp32 (MLX or numpy). `meshVerticeInpaint` is pure CPU pybind11 — **recompile unchanged** for exact parity. `cv2.inpaint` is CPU and identical.

---

## 4. Reuse Map from the Shape MLX Port

Source: the sibling shape port (`python/shape`). Classification per artifact.

| Artifact (shape port) | Action | Detail |
|---|---|---|
| `hy3dmlx/layers.py` — LayerNorm, RMSNorm, sdpa, gelu_tanh, gelu_erf, MLPEmbedder, timestep_embedding, FourierEmbedder | **Reuse as-is** | fp32-internal norm convention + `mx.fast.scaled_dot_product_attention` wrapper are exactly what SD2.1 UNet needs; SD time embed = same sinusoidal+MLP shape |
| `hy3dmlx/models/dinov2.py` — full DINOv2-giant module (`DinoImageEncoder`/`Dinov2Model`/`Dinov2Layer`/`Dinov2SelfAttention`/`Dinov2SwiGLUFFN`/`LayerScale`/`Dinov2Embeddings`) | **Reuse as-is (module body)** | Paint conditioner IS facebook/dinov2-giant, identical giant config (1536/24/40/SwiGLU/patch14/1370). Forward math + key layout unchanged |
| `_assign` strict loader (`convert.py:31-41`) | **Reuse as-is** | flatten params → key→shape map, raise on missing/mismatch, `load_weights(strict=True)`. Domain-agnostic safety net |
| `cos()`/maxabs/std comparison harness (`mlx_compare.py:15-17`) | **Reuse as-is** | flattened float64 cosine + maxabs + std |
| ModelScope downloaders (`dl_modelscope.py`, `dl_any.py`) | **Reuse as-is (re-point)** | change repo/file list to paint UNet/VAE + facebook/dinov2-giant |
| DINOv2 **weight source** | **Adapt** | shape reads `conditioner.main_image_encoder.model.*`; paint loads stock HF `facebook/dinov2-giant` with top-level keys (`embeddings.*`/`encoder.layer.N.*`/`layernorm.*`, no prefix). Patch Conv2d NCHW→NHWC still applies. The `.model` wrapper already matches HF's missing-prefix layout |
| DINO call wrapper | **Adapt** | paint feeds multi-view: `dino(cond_imgs[:,:1])` then `rearrange('(b n) l c -> b (n l) c')`. Add thin batching wrapper; core module unchanged |
| `convert.py` prefix-split pattern | **Adapt** | reuse the one-safe_open + split-by-prefix + strict `_assign` pattern, but with paint prefixes (`unet.`, `vae.`, `text_encoder.`) + separate HF dino load. Confirm whether paint ships one safetensors vs diffusers subfolders |
| `sampler.py` (FlowMatchEuler loop skeleton + CFG) | **Adapt** | reuse loop skeleton + `mx.eval`-per-step + progress cb; re-derive UniPC multistep math (NOT FlowMatch, NOT simple DDIM) |
| `pipeline.py` orchestration shape (from_pretrained → encode → loop → decode) | **Adapt** | good template; paint adds render/bake/UV host steps |
| Oracle methodology (`.venv-oracle`, `oracle_compare.py`/`mlx_compare.py`, per-stage `.npy` dump, `oracle_dit_trace.py`/`mlx_dit_trace.py` per-block trace) | **Adapt (carry wholesale)** | single most valuable thing to port. Write `paint_oracle_compare.py` + `paint_mlx_compare.py` + a per-UNet-block trace |
| SD2.1 UNet 2D stack (GroupNorm, ResnetBlock2D, Conv2d up/down/mid, Transformer2DModel cross-attn blocks, Downsample2D/Upsample2D) | **Build new** | NONE exists in shape port — no GroupNorm, no ResnetBlock2D, no 2D conv blocks. This is the bulk of the work |
| `UNet2p5DConditionModel` wrapper (5 attentions, n_pbr routing, 3D RoPE, dual-stream, DINO injection) | **Build new** | custom multiview/PBR logic; no diffusers equivalent |
| `AutoencoderKL` (SD VAE encoder+decoder Conv2d/Resnet/attn, GroupNorm, diagonal-Gaussian, 0.18215 scaling) | **Build new** | shape ShapeVAE is a vecset/cross-attn decoder, structurally unrelated |
| CLIP text path | **Build new IF needed** | likely unneeded at inference (learned tokens used). Confirm and probably skip |
| UniPC / DDPM scheduler math | **Build new** | re-derive from diffusers UniPCMultistep |
| Renderer / rasterizer / baker / inpaint | **Build new (mostly in-repo already)** | see §3.1; mtldiffrast + mtlbvh already present |

**Parity landmines pre-empted from shape lessons:** (1) GroupNorm + all norms compute fp32 then cast back (torch GroupNorm is fp32-internal). (2) Watch SD's GELU variant per block (GEGLU vs plain). (3) VAE scaling_factor 0.18215 is the analogue of the shape `scale_factor` decoy — read from config, apply at documented boundary, assert. (4) Conv2d NCHW→NHWC transpose for every conv, not just patch embed. (5) Use `_assign` everywhere so SD's many conv shapes fail loud.

---

## 5. Build Items (each with acceptance gate)

Ordered roughly by dependency. Each gate is checked against the torch oracle unless noted.

### B1 — Weights + convert
**What.** Download `hunyuan3d-paintpbr-v2-1` (unet/vae/text_encoder + scheduler/tokenizer configs) and `facebook/dinov2-giant` via ModelScope/HF downloaders. Write `convert.py`: one safe_open, prefix-split into `unet.`/`vae.`/(`text_encoder.`), strip/route DINO from HF top-level keys, transpose every Conv2d NCHW→NHWC, cast dtype, `_assign` strict-load per module.
**Gate.** `_assign` loads every module with zero missing keys and zero shape mismatches. Round-trip: re-export a converted tensor and assert maxabs==0 vs the safetensors source. Confirm checkpoint actually ships non-zero `learned_text_clip_mr`, `to_v_mr`, `attn_multiview.*` (they init to zero/clone but are trained).

### B2 — Shared 2D layers
**What.** Build new in `hy3dmlx/layers2d.py`: fp32-internal `GroupNorm` (mirror LayerNorm cast convention, norm_num_groups=32), `Conv2d` (NHWC), `ResnetBlock2D` (GroupNorm→SiLU→Conv2d, time-embed add, skip conv), `Downsample2D`/`Upsample2D`, cross/self attention via `Transformer2DModel` block reusing `layers.sdpa`, GEGLU/GELU feed-forward.
**Gate.** Per-layer: feed an oracle input `.npy`, compare output cosine > 0.9999 and maxabs < 1e-3 (fp32). GroupNorm matches torch `F.group_norm` to maxabs < 1e-5 on random input.

### B3 — VAE (`AutoencoderKL`)
**What.** Build encoder + decoder Conv2d/Resnet/attention stack, GroupNorm, diagonal-Gaussian, 0.18215 scaling (read from config). Encode: `(x-0.5)*2 → encode → .sample()*scaling_factor`. Decode: `latents/scaling_factor → decode`.
**Gate.** VAE encode (use `.mode()`/mean for determinism, see §6): latents cosine > 0.9999 vs oracle. VAE decode of oracle final latents: image PSNR > 40 dB. Assert scaling_factor == config value.

### B4 — DINOv2-giant (reuse + adapt)
**What.** Reuse `dinov2.py` module body; new HF-key loader; multi-view batching wrapper.
**Gate.** `last_hidden_state` cosine == 1.000000 vs oracle (shape port achieved this). Token count == 1370 (518/14 → 37×37 + cls). ImageProjModel output cosine > 0.9999.

### B5 — UniPC scheduler
**What.** Build new: `set_timesteps` with trailing spacing (timesteps from rounded linspace, last lands near t=0), UniPC multistep update (Adams-Bashforth-like with order ramp), `init_noise_sigma`, `scale_model_input` (identity for UniPC, call anyway), v_prediction handling. Read `solver_order`/`predict_x0`/`thresholding`/`solver_type` from the checkpoint `scheduler_config.json`.
**Gate.** Given an identical `noise_pred` sequence and seed, MLX `scheduler.step` output matches oracle latents at each of 15 steps to maxabs < 1e-4.

### B6 — Conditioners
**What.** Load `learned_text_clip_{albedo,mr,ref}` (each `[77,1024]`) as stored params → encoder_hidden_states. Precompute voxel RoPE indices in numpy (`calc_multires_voxel_idxs`, grid_resolutions=[H,H/2,H/4,H/8]=[64,32,16,8], voxel_resolutions=[H·8…H]=[512,256,128,64]) and feed as MLX constants. ImageProjModel (Linear 1536→4·1024 + LayerNorm). Confirm CLIP text encoder is unused (probably skip B-CLIP entirely).
**Gate.** Learned tokens load bit-exact. Voxel indices match oracle `calc_multires_voxel_idxs` element-wise (integer equality). DINO proj tokens cosine > 0.9999.

### B7 — 2.5D multiview UNet (`UNet2p5DConditionModel`) + base UNet
**What.** Build base SD2.1 UNet from B2 layers. Build `Basic2p5DTransformerBlock` as an MLX module taking explicit `condition_embed_dict` / `dino_tokens` / `voxel_rope` tensors (no in-place hooks — thread state explicitly). Implement the **two-pass design**: `unet_dual` runs once at t=0 mode "w" writing `condition_embed_dict`; main UNet mode "r" reads it. conv_in = 12 channels (do NOT zero channels 4:12 — released ckpt carries trained weights). Five attention paths with exact einops (MDA `(b n_pbr n) l c`; RA albedo-only query → `b (n_ref l) c`; MA `(b n_pbr) (n l) c`; DINO repeat). Material weights as explicit per-material Linear layers (flatten processor-held `to_*_mr`). 3D RoPE: `get_3d_rotary_pos_embed` dim split xy=3/8 each, z=2/8, head_dim 64, interleaved real/imag `reshape(...,-1,2)`.
**Gate.** **Per-block trace** (mirror `oracle_dit_trace.py`): dump `norm_hidden_states` and each attn output for one block, compare cosine > 0.9999 / maxabs < 1e-3 before trusting end-to-end. Full UNet single-step `noise_pred` (fed oracle latents/cond) cosine > 0.9999. Confirm n_pbr vs n_view ordering with shape asserts at every reshape.

### B8 — PBR dual-branch
**What.** Implement the n_pbr=2 doubled-batch routing: albedo→base attn weights, mr→cloned material weights in MDA + RA. `SelfAttnProcessor` loops materials and concatenates; `RefAttnProcessor` shares Q/K, concatenates per-material V then splits by head_dim. Output split into v_pred_albedo / v_pred_mr.
**Gate.** Albedo and mr `noise_pred` slices each cosine > 0.9999 vs oracle. Verify mr branch is genuinely distinct (not accidentally aliasing albedo weights).

### B9 — Pipeline integration
**What.** `HunyuanPaintPipeline` MLX: build conditioning (ref/normal/position VAE encode, learned tokens, DINO, voxel RoPE), assemble 3-way CFG (ref_latents×3 with ref_scale=[0,1,1], dino=[0,0,dino], normal/position×3), UniPC loop with 3-way CFG combine, mirror the `latents[:, :num_channels_latents]` no-op slice, decode, split {albedo, mr}.
**Gate.** Per-step `noise_pred` (all 15 steps, one mesh) cosine > 0.999 vs oracle dump. Final decoded albedo/mr images: cosine of decoded images > 0.99 (exact pixels impossible — RNG differs across frameworks; see §6).

### B10 — Rasterizer / renderer
**What.** Wire mtldiffrast as the MPS/CPU backend in MeshRender (depth remap + winding fix). Port camera algebra verbatim. Produce normal (use_abs_coor) + position multiview maps and UV-space rasterization (`extract_textiles` → tex_position/tex_normal/tex_grid/texture_indices).
**Gate.** Rasterizer findices: **mask IoU > 0.99** vs the CPU-compiled reference (B-fallback). Barycentric maxabs < 1e-3 on covered pixels. Normal/position control maps PSNR > 35 dB vs oracle (these condition the UNet — misalignment poisons everything downstream).

### B11 — Baker + inpaint
**What.** Port back_sample (project tex_position through w2c·proj, depth-occlusion test `|v_z-sampled_z|<3e-3`, bilinear RGB sample, scatter into tex_grid), cos^6 weighting + `Σtex·cos/Σcos` blend, boundary dilation (opencv). Recompile `meshVerticeInpaint` C++ unchanged; cv2.inpaint NS for residual holes.
**Gate.** Baked albedo + mr UV textures: **PSNR > 35 dB** vs oracle baked texture (JPEG-lossy → compare decoded pixels with tolerance, or use PNG for testing). Inpaint output exact (C++ recompiled). Coverage mask IoU > 0.99.

### B12 — Super-res (optional)
**What.** First pass: bicubic/Lanczos x4 stub to validate the rest. Final: reimplement RRDBNet (23 RRDB, x4) in MLX + convert `RealESRGAN_x4plus.pth`, OR run torch-MPS as stopgap (drop `half=True`).
**Gate.** Stub: pipeline runs end-to-end (no parity gate). Final RRDBNet: enhanced view PSNR > 40 dB vs torch Real-ESRGAN.

### B13 — GLB packing + glue
**What.** Reuse `convert_utils.create_glb_with_pbr_materials` (pygltflib, pure CPU) — **avoid Blender path entirely**. Standardize all callers on pygltflib. metallic-roughness packing AO=R=255, roughness=G, metallic=B. save OBJ+MTL+JPG via `save_obj_mesh`.
**Gate.** Output GLB loads in a glTF validator; baseColor + metallicRoughness textures present and channel-correct; mesh decoded vertices/UVs match the remeshed input.

---

## 6. Parity Strategy

**Per-stage oracle, isolated.** Stand up `.venv-oracle` (torch, diffusers, transformers, safetensors) invoked via `VIRTUAL_ENV=.venv-oracle uv run --no-project python paint_oracle_compare.py`. The oracle builds torch modules from `reference/.../hy3dpaint`, loads the SAME checkpoint, runs each stage in fp32 (`.float()` upcast), dumps per-stage `.npy`: `dino_out`, `dino_proj`, `ref_latents`, `normal_latents`, `position_latents`, `voxel_idxs`, `learned_tokens`, `condition_embed_dict[layer]`, `noise_pred_step{0..14}`, `decoded_albedo`, `decoded_mr`, plus rasterizer `findices`/`barycentric`/`normal_map`/`position_map` and baked textures. `paint_mlx_compare.py` feeds each MLX stage the **oracle's** intermediate tensors so errors never compound (UNet gets oracle cond, VAE-decode gets oracle latents, baker gets oracle views).

**fp32 → fp16.** Develop and gate in fp32 first (matches oracle `.float()`). Only after fp32 parity passes, switch the runtime to fp16 and re-check that fp16 drift stays within a looser bar (cosine > 0.999). Norms always compute fp32-internal regardless.

**Tolerance gates (neural).**
- Cosine (flattened float64) > 0.9999 and maxabs < 1e-3 per intermediate stage in fp32.
- Per-block UNet trace before end-to-end (the tool that localized the shape port's time-embedding bug).
- End-to-end decoded images: cosine > 0.99 (NOT MSE) — exact pixels are impossible because the reference uses global-seeded `latent_dist.sample()` (not the passed generator) and MLX cannot reproduce torch RNG. **Accept latent-distribution-level parity, not bit-exact pixels.** For deterministic encode, use VAE `.mode()`/mean instead of `.sample()` during parity runs.

**The rasterizer parity question — bit-exact vs approximate, and how to gate.**
There is **no CUDA reference on Mac**, so "bit-exact vs the production CUDA kernel" is gated against the **CPU-compiled reference** (B2-fallback), which is bit-identical to the CUDA kernel by construction.
- **Bit-exact tier (CI oracle):** the compiled `rasterize_image_cpu` — used to validate any rasterizer choice. Use as the ground truth.
- **Approximate tier (production via mtldiffrast):** gate with geometry metrics, not bit-equality:
  - **Mask IoU > 0.99** on findices > 0 (coverage agreement).
  - **Texture PSNR > 35 dB** on baked albedo/mr vs oracle bake (catches pixel-shift seams).
  - **Barycentric maxabs < 1e-3** on covered pixels.
  - Optionally **Chamfer distance** between the rasterized visible-point cloud and the reference, if seam alignment is suspect.
- Decision rule: if mtldiffrast fails the IoU/PSNR bars (e.g. residual winding/depth misalignment degrades UNet conditioning), fall back to the compiled CPU rasterizer for exact parity and treat mtldiffrast as a speed optimization to be re-qualified later. The depth-remap + winding-flip fixes already in `mesh_render.py` are the first thing to verify against the IoU gate.

**Baker/inpaint parity.** Elementwise ops (cos^6, blend) are fp32-automatic. `meshVerticeInpaint` (recompiled C++) and cv2.inpaint give exact parity for free.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Rasterizer mtldiffrast misaligns normal/position maps → poisons UNet conditioning before any neural bug is visible | Med | High | Gate B10 control maps (PSNR>35dB) FIRST against compiled-CPU rasterizer; carry depth-remap + winding fixes; CPU rasterizer fallback for exact parity |
| R2 | UniPC step math wrong (multistep order ramp, trailing spacing) — silent trajectory drift | Med | High | Build B5 in isolation; gate step-by-step against oracle latents given identical noise_pred; read solver config from checkpoint |
| R3 | 3-way CFG collapsed incorrectly (it is NOT 2-way; branches differ in conditioning not just text) | Med | High | Replicate ref_scale=[0,1,1], dino=[0,0,dino], normal/position×3; do not merge into single UNet call; gate per-branch chunk |
| R4 | n_pbr vs n_view einops ordering off-by-one → silent corruption | Med | High | Shape-assert every reshape; per-block trace; explicit reshape helpers (no einops in MLX) |
| R5 | 3D RoPE interleaving / dim-split mismatch → cross-view geometry breaks | Med | Med | Precompute voxel idxs in numpy, gate integer-equal; match xy=3/8,z=2/8 split + reshape(...,-1,2); compare MA output per-block |
| R6 | conv_in 12-channel weights wrongly zeroed (training init zeroed 4:12; inference must keep them) | Low | High | Load released 12ch weight as-is; assert non-zero in channels 4:12; `_assign` strict |
| R7 | negative_prompt_embeds wrongly set to zeros (reference uses SAME learned tokens) | Med | Med | Mirror `prompt_embeds==negative_prompt_embeds`; gate uncond branch noise_pred |
| R8 | VAE scaling_factor assumed 0.18215 but checkpoint differs | Low | Med | Read from vae/config (absent → 0.18215 default); assert at boundary; B3 gate |
| R9 | Dual-stream reference UNet not cached → runs every step / wrong KV | Low | Med | Thread `condition_embed_dict` explicitly; run unet_dual once at t=0; cache like reference |
| R10 | Material `to_*_mr` weights ship as zero/clone (untrained) → port is dead conditioning | Low | Med | Verify state_dict keys non-zero in B1 |
| R11 | CLIPVisionModelWithProjection / CLIP text wrongly ported as live conditioning (likely dormant) | Med | Low | Confirm unused in forward before porting; skip if dead |
| R12 | RNG divergence makes pixel-exact parity impossible; team over-tightens gates | High | Low | Gate on cosine/PSNR/IoU, NOT MSE; use VAE .mode() for deterministic parity runs |
| R13 | mtldiffrast quality "not as good as CUDA" (repo's own caveat) | Med | Med | CPU-rasterizer fallback; Open3D second oracle; texture PSNR gate catches degradation |
| R14 | pymeshlab / xatlas / trimesh decimation backend nondeterminism → texture coords drift | Low | Med | Pin versions; confirm trimesh decimation backend; remesh target 40k fixed |
| R15 | DINOv2 dim ambiguity (digest says large/1024, source says giant/1536) | Low | Med | Authoritative = giant/1536; shape-assert pos_embed token count 1370 |

---

## 8. Open Questions to Resolve Before Coding

1. **Checkpoint layout.** Does `hunyuan3d-paintpbr-v2-1` ship the UNet2p5D weights as one safetensors with `unet.`/`vae.` prefixes, or as diffusers-style subfolders? Decides whether convert.py uses single-file prefix-split or a multi-folder loader. (Metadata shows separate `unet/`, `vae/`, `text_encoder/` subfolders + per-subtree `model_index.json` → likely multi-folder; confirm the `.bin` vs `.safetensors` and exact key prefixes.)
2. **CLIP text encoder live?** Confirm `pipeline.py` never runs the SD2.1 CLIP text encoder at inference (learned tokens replace it; "high quality" inert). If confirmed, skip porting CLIP text entirely.
3. **CLIP vision dormant?** `model_index.json` lists `CLIPVisionModelWithProjection` — confirm it is genuinely unused at inference (no IP-adapter weights loaded) vs loaded-but-dormant, to avoid porting dead conditioning.
4. **DINOv2-giant byte-identical to shape's bundled conditioner?** If HF `facebook/dinov2-giant` matches the bundled shape weights (same SwiGLU inner dim, same pos_embed), the existing `dinov2.py` loads with only a prefix change. Verify by shape-asserting both.
5. **DINO token count.** Confirm 1370 (518/14 → 37×37 + cls/registers) by running `AutoImageProcessor`; drives `L_dino` and the ImageProjModel input.
6. **Working resolution.** Does `config.resolution` (custom_view_size) differ from cfg `view_size=512` at the call site? The 2.1 multiview cfg `view_size` default is 320 while config passes `resolution`. This sets H,W and therefore `grid_resolutions=[H,H/2,H/4,H/8]` and latent spatial dims — must be pinned before B6/B7.
7. **UniPC config.** Exact `solver_order` / `predict_x0` / `thresholding` / `solver_type` from the checkpoint `scheduler_config.json` (UniPC built `from_config` of the DDIM base) — needed for B5 step math.
8. **VAE scaling_factor.** Absent from vae/config.json → diffusers default 0.18215; confirm `model.py` reads `vae.config.scaling_factor` and that the default applies.
9. **Rasterizer parity bar.** Is bit-exact parity with the (CPU-equivalent-of-)CUDA reference required, or is mask-IoU/texture-PSNR equivalence acceptable? Decides whether mtldiffrast is production or whether the compiled CPU rasterizer is mandatory.
10. **Production render config.** Confirm always `shader_type='face'` + `bake_mode='back_sample'` (skip vertex-normal / linear / mip-map paths) and that `use_depth_prior`/`occlusion_truncation` is never exercised (always 0). Confirm `meshVerticeInpaint` always `method='smooth'`.
11. **Real-ESRGAN bar.** Must super-res be bit-exact for acceptance, or is a bicubic stub acceptable for the initial milestone?
12. **Normal texture output.** `save_obj_mesh` supports `_normal.jpg` but `textureGenPipeline` only sets diffuse + mr — confirm no normal texture is produced in the 2.1 path.

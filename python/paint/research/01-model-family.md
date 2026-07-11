# Hunyuan3D Model Family — Taxonomy

> Scope: a precise taxonomy of Tencent's Hunyuan3D model family, intended to orient an MLX port of the **paint / texture** stage. Naming conventions, repo ids, dims, and the PBR-vs-RGB and open-vs-API splits are drawn entirely from the research corpus. Where the corpus marks a fact as uncertain or conflicting, that is called out inline.
>
> Confidence note (from corpus `webFamily`): high confidence on repo ids, release dates, the PBR-vs-RGB split, open-vs-API split, and the 2.0/2.1 model-zoo parameter counts. Lower confidence on a few exact per-submodel parameter numbers and on the 2.5/3.x ~10B figures (which come from secondary reporting, not primary spec sheets). Several details lean on GitHub model tables + search snippets where rendered HF model-card fetches failed.

## Naming conventions (applies throughout)

- **Hugging Face org:** `tencent/...`
- **ModelScope org:** `Tencent-Hunyuan/...`
- **GitHub org:** `Tencent-Hunyuan/...`
- **License (open releases):** Tencent Hunyuan Community / Non-Commercial license. HF tag is `tencent-hunyuan-community`; the underlying agreement is the "Tencent Hunyuan Non-Commercial License Agreement," restricted in the EU/UK/South Korea. 2.1 ships its own variant ("Tencent Hunyuan 3D 2.1 Community/Non-Commercial," with `LICENSE` + `Notice.txt`).
- **Do not conflate:** `tencent/HunyuanImage-3.0` is a separate **80B text-to-IMAGE** model, NOT a 3D model.

---

## Version comparison table

| Version | HF repo | ModelScope repo | First release | Shape (DiT) params | Submodels shipped | Shape? | Paint? | PBR? | License | Open-weights vs API |
|---|---|---|---|---|---|---|---|---|---|---|
| **1.0** | `tencent/Hunyuan3D-1` | `Tencent-Hunyuan/Hunyuan3D-1` | Nov 2024 | `lite`, `std` (std ~3× lite; not published) + `svrm` | multiview RGB diffusion (lite/std) + sparse-view reconstruction (SVRM) | Yes | No standalone paint net (texture = baked from multiview RGB) | RGB-only | tencent-hunyuan-community | **Open** |
| **2.0** | `tencent/Hunyuan3D-2` | `Tencent-Hunyuan/Hunyuan3D-2` | 2025-01-21 | DiT-v2-0 1.1B (+Fast/Turbo, both 1.1B) | shape DiT + ShapeVAE; Paint-v2-0 1.3B (+Turbo 1.3B); Delight-v2-0 1.3B | Yes | Yes (Paint-v2-0) | RGB-only | tencent-hunyuan-community | **Open** |
| **2mini** | `tencent/Hunyuan3D-2mini` | `Tencent-Hunyuan/Hunyuan3D-2mini` (mirror likely; not directly verified) | 2025-03-18 | DiT-v2-mini 0.6B (+Fast/Turbo, both 0.6B) | shape DiT only; reuses 2.0 Paint | Yes | No (reuses 2.0 Paint) | RGB-only | tencent-hunyuan-community | **Open** |
| **2mv** | `tencent/Hunyuan3D-2mv` | `Tencent-Hunyuan/Hunyuan3D-2mv` (mirror likely; not directly verified) | 2025-03-18 | DiT-v2-mv 1.1B (+Fast/Turbo, both 1.1B) | multiview-controlled shape DiT only; reuses 2.0 Paint | Yes | No (reuses 2.0 Paint) | RGB-only | tencent-hunyuan-community | **Open** |
| **2.1** | `tencent/Hunyuan3D-2.1` | `Tencent-Hunyuan/Hunyuan3D-2.1` | 2025-06-13/14 | Shape-v2-1 ~3.0–3.3B (see note) | shape DiT + ShapeVAE; **Paint-PBR-v2-1 ~1.3–2B** (RomanTex + MaterialMVP) | Yes | Yes (Paint-PBR-v2-1) | **PBR (albedo + metallic-roughness)** | Tencent Hunyuan 3D 2.1 Community/Non-Commercial | **Open** (full weights AND training code) |
| **2.5** | none | none | 2025-04-23 (report Jun 2025) | ~10B total (secondary reporting) | upgraded ~10B shape DiT + multiview PBR texture model (per-submodel split not public) | Yes | Yes | PBR (multiview) | Commercial (Tencent Cloud API) | **API only** |
| **3.0** | none | none | Sept 2025 | ~10B hierarchical 3D-DiT, up to 1536³ res (secondary reporting) | hierarchical 3D-DiT + PBR material maps (split not published) | Yes | Yes | PBR | Commercial (Tencent Cloud API) | **API only** |
| **3.1 (intl/Pro)** | none | none | ~Nov 2025 | 3.x generation | PBR material maps | Yes | Yes | PBR | Commercial (Tencent Cloud API + 3rd parties) | **API only** |
| **Omni** | `tencent/Hunyuan3D-Omni` | (HF; MS mirror unconfirmed) | 2025-09-25 | ~3.3B (2.1 base: DiT + VAE decoder) | 2.1-based shape DiT + unified control encoder (point cloud / voxel / bbox / skeleton); uses 2.1 PBR paint | Yes | Yes (via 2.1 paint) | PBR (via 2.1) | tencent community / non-commercial | **Open** |
| **Part** | `tencent/Hunyuan3D-Part` | (HF; MS mirror unconfirmed) | ~Aug 2025 | part segmentation + part-based generation | part-level seg + part generation | Yes (parts) | n/a | n/a | tencent community / non-commercial | **Open** |

**Summary splits**

- **PBR vs RGB:** PBR = 2.1, 2.5, 3.0/3.1, Omni (via 2.1 paint). RGB-only = 1.0, 2.0, 2mini, 2mv.
- **Open-weights vs API:** Open = 1.0, 2.0, 2mini, 2mv, 2.1, Omni, Part. API-only = 2.5, 3.0, 3.1.

---

## Per-version detail

### 1. Hunyuan3D-1.0 (Nov 2024)

- **Repos:** HF `tencent/Hunyuan3D-1` · ModelScope `Tencent-Hunyuan/Hunyuan3D-1` · GitHub `Tencent-Hunyuan/Hunyuan3D-1` · paper arXiv 2411.02293.
- **Architecture:** two-stage. Stage 1 = multi-view RGB diffusion (~4s); Stage 2 = feed-forward sparse-view reconstruction (SVRM, ~7s). The multiview model generates 6 fixed-azimuth RGB views (relative azimuth +0/+60/+120/+180/+240/+300), with adaptive classifier-free guidance and a lightweight super-resolution module. Text-to-3D uses the upstream Hunyuan-DiT text-to-image model.
- **Submodels:** `lite` (multiview diffusion), `std` (multiview, ~3× lite params), `svrm` (sparse-view reconstruction).
- **Params:** not officially broken out (only "std ~3× lite"; per-variant counts never published — corpus open question).
- **Texture:** RGB-only, baked from multiview RGB (DUSt3R/Dust3R alignment; `--do_texture_mapping`, optional `--do_bake`). The baking module is CC-BY-NC-SA 4.0. **There is no standalone "Paint-v1" diffusion network** — texturing is bake-only.
- **License / availability:** Tencent Hunyuan Non-Commercial; **open weights**.

### 2. Hunyuan3D-2.0 family (Jan–Apr 2025)

- **Repos:** HF `tencent/Hunyuan3D-2` · ModelScope `Tencent-Hunyuan/Hunyuan3D-2` · GitHub `Tencent-Hunyuan/Hunyuan3D-2` · report arXiv 2501.12202.
- **Pipeline:** shape (Hunyuan3D-DiT, flow-based diffusion transformer) + texture (Hunyuan3D-Paint) + Delight (de-lighting). **Shape and Paint are separate downloads / subfolders.** Stages are fully decoupled — paint consumes only a mesh + the input image, never shape latents.
- **Model zoo (GitHub table):**
  - `Hunyuan3D-DiT-v2-0` shape DiT — 1.1B — 2025-01-21
  - `Hunyuan3D-DiT-v2-0-Fast` — 1.1B; `Hunyuan3D-DiT-v2-0-Turbo` — 1.1B
  - `Hunyuan3D-Paint-v2-0` RGB texture — 1.3B — 2025-01-21; `Hunyuan3D-Paint-v2-0-Turbo` — 1.3B (released 2025-04-01)
  - `Hunyuan3D-Delight-v2-0` delight — 1.3B
- **Shape VAE:** a 3D ShapeVAE (vector-set latent autoencoder) is bundled in the DiT package.

#### Paint-v2-0 (subfolder `hunyuan3d-paint-v2-0`)

- **Output:** single RGB texture (4K after bake). No PBR.
- **Backbone:** stock SD2.1-base UNet wrapped at runtime by a custom `UNet2p5DConditionModel`. Verified UNet config: `_class_name UNet2DConditionModel`, `in_channels=4`, `out_channels=4`, `cross_attention_dim=1024`, `attention_head_dim=[5,10,20,20]`, `block_out_channels=[320,640,1280,1280]`, `sample_size=64`, `use_linear_projection=true`. Pipeline `_class_name StableDiffusionPipeline`.
- **Scheduler:** `DDIMScheduler`, `prediction_type=v_prediction`, `rescale_betas_zero_snr=true`, `timestep_spacing=trailing`, betas 0.00085→0.012 scaled_linear (zero-terminal-SNR v-model). Inference ~28 steps with CFG.
- **Conditioners:** `text_encoder` (CLIPTextModel, hidden 1024, 23 layers, from `stabilityai/stable-diffusion-2`) + tokenizer + `feature_extractor` (CLIPImageProcessor, size 224). **No `image_encoder` folder** — the reference image is fed through the dual-stream reference UNet (VAE latent of reference), not a CLIP/IP-adapter image tower.
- **Wrapper attention (from `modules.py`):** multiview attention (`use_ma`), reference attention (`use_ra`, scaled by `ref_scale`), dual-stream reference branch (`unet_dual = copy.deepcopy(unet)`), learnable camera/view embedding (`use_camera_embedding`, `class_embedding = nn.Embedding(max_num_ref_image + max_num_gen_image, ...)`; `max_num_ref_image=5`, `max_num_gen_image=44`). Each `BasicTransformerBlock` is replaced by a `Basic2p5DTransformerBlock`. Multi-task attention: `Z_MVA = Z_SA + λ_ref·RefAttn + λ_mv·MultiviewAttn`.
- **Delighting:** a separate external model (Hunyuan3D-Delight-v2-0, InstructPix2Pix-style i2i) removes lighting/shadow from the input image before synthesis.
- **License / availability:** tencent-hunyuan-community; **open weights**.

#### Paint-v2-0-Turbo (subfolder `hunyuan3d-paint-v2-0-turbo`)

- Same architecture/output (RGB) as v2-0, but step-distilled (consistency/step distillation via DDIMSolver). Cited at ~10 steps (vs ~28) and ~60% faster — **step count and speed are from secondary write-ups; distillation mechanism is confirmed in code.**
- **Conditioner difference:** turbo **adds** `image_encoder = CLIPVisionModelWithProjection` (verified: hidden 1280, 32 layers, patch 14, image_size 224, projection_dim 1024 — CLIP-H/14, traced to `sudo-ai/zero123plus-v1.1/vision_encoder`). This is the IP-Adapter-style CLIP-vision reference conditioner. Released together with the "multiview texture generation pipeline" (`fast_texture_gen_multiview.py`).

#### Delight-v2-0 (subfolder `hunyuan3d-delight-v2-0`)

- Pipeline `StableDiffusionInstructPix2PixPipeline` (diffusers 0.30.1); **stock** diffusers `UNet2DConditionModel` (no 2.5D wrapper). UNet `in_channels=8` (4 noisy + 4 conditioning-image latents), `out_channels=4`, `cross_attention_dim=1024`, `sample_size=96`, `upcast_attention=true`. VAE has explicit `scaling_factor=0.18215`, `force_upcast=true`. Scheduler: DDIM v_prediction, scaled_linear, **leading** spacing, `rescale_betas_zero_snr=false`, `set_alpha_to_one=false`. (Distinct scheduler config from the paint models.)

#### 2a. Hunyuan3D-2mini (shape-only)

- HF `tencent/Hunyuan3D-2mini`. Released 2025-03-18. Subfolder `hunyuan3d-dit-v2-mini`. Submodels: `Hunyuan3D-DiT-v2-mini` 0.6B (smaller/faster than the 1.1B base), `-mini-Fast`, `-mini-Turbo` (all 0.6B). **Shape-only**; ships no paint model; reuses the 2.0 RGB Paint. RGB-only. Open.

#### 2b. Hunyuan3D-2mv (multiview shape, shape-only)

- HF `tencent/Hunyuan3D-2mv`. Released 2025-03-18. Finetuned from Hunyuan3D-2 for **multiview-controlled SHAPE generation** (front/back/left/right → mesh). Subfolders `hunyuan3d-dit-v2-mv`, `hunyuan3d-dit-v2-mv-turbo`. Submodels: `Hunyuan3D-DiT-v2-mv` 1.1B, `-mv-Fast`, `-mv-Turbo` (all 1.1B). **It is a shape model, not a texture model.** RGB-only via the reused 2.0 Paint. Open.
  - **Disambiguation:** any reference to a "2mv multiview texture model" most likely means the multiview texture *pipeline* (multiple reference views fed to Paint-v2-0-Turbo via its CLIP-vision encoder), not a distinct paint checkpoint.

### 3. Hunyuan3D-2.1 (Jun 2025) — first OPEN PBR release  ← MLX paint-port target

- **Repos:** HF `tencent/Hunyuan3D-2.1` · ModelScope `Tencent-Hunyuan/Hunyuan3D-2.1` · GitHub `Tencent-Hunyuan/Hunyuan3D-2.1` · paper arXiv 2506.15442.
- **Release:** 2025-06-13/14. **First fully open-source Hunyuan3D — full weights AND training code.**
- **Model zoo (GitHub 2.1 table):**
  - `Hunyuan3D-Shape-v2-1` (= `Hunyuan3D-DiT-v2-1`), image-to-shape — **3.0–3.3B** (table 3.3B; DiT cited 3.0B) — 2025-06-14
  - `Hunyuan3D-Paint-v2-1` PBR texture — **~1.3–2B** (GitHub 2.1 table 2B; DiT model-zoo 1.3B) — 2025-06-14
  - **Param discrepancies (corpus open questions):** Paint listed as 2B (2.1 table) vs 1.3B (DiT zoo) — the 2B figure likely reflects the full PBR stack (RomanTex + MaterialMVP, i.e. the added metallic-roughness branch). Shape shows 3.0B (DiT only) vs 3.3B (Shape package incl. VAE). Treat both as ranges until confirmed against the actual checkpoint.
- **Submodels:** shape DiT + shape VAE (in the Shape package) + the PBR Paint model. PBR pipeline is powered by **RomanTex** (3D-aware rotary multi-attention for texture synthesis) and **MaterialMVP** (illumination-invariant multi-view PBR diffusion).
- **Shape vs Paint:** separate folders (`hy3dshape` / `hy3dpaint`), separate downloads.

#### Paint-PBR-v2-1 (subfolder `hunyuan3d-paintpbr-v2-1`) — the port target

- **Output:** PBR materials. `num_view=6`, `pbr_settings=["albedo","mr"]` — albedo + a combined metallic-roughness (MR) map (n_pbr=2). Normal/position maps are geometry **conditions**, not generated outputs.
- **Pipeline:** `HunyuanPaintPipeline` (diffusers 0.24.0).
- **Backbone:** same stock SD2.1-base UNet topology as v2-0 (verified `unet/config.json` identical: in/out 4ch, `cross_attention_dim=1024`, `block_out_channels=[320,640,1280,1280]`, `attention_head_dim=[5,10,20,20]`, `sample_size=64`, per-head dim 64), wrapped by a custom `UNet2p5DConditionModel`. The wrapper **expands `conv_in` from 4 → 12 channels** = 4 noisy gen latent + 4 VAE-encoded normal-map latent + 4 VAE-encoded position-map latent (channel-concat conditioning; `noise_in_channels=12`).
- **VAE:** `AutoencoderKL`, `latent_channels=4`, `block_out_channels=[128,256,512,512]`, sample_size 768, 8× downsample. `scaling_factor` is **not present** in `vae/config.json` → diffusers default **0.18215** is used (`model.py` reads `vae.config.scaling_factor`). Confirm against the downloaded config.
- **Scheduler:** on-disk `scheduler_config.json` is `DDIMScheduler`, `v_prediction`, scaled_linear (0.00085→0.012), `timestep_spacing=trailing`, `rescale_betas_zero_snr=true`, `set_alpha_to_one=true`, `clip_sample=false`. **At inference the pipeline swaps to `UniPCMultistepScheduler.from_config(..., timestep_spacing="trailing")`, 15 steps, `guidance_scale=3.0`.** Training uses DDPM.
- **Conditioner stack:** `text_encoder` (CLIPTextModel hidden 1024 / 23 layers) + `image_encoder` (`CLIPVisionModelWithProjection`, CLIP-H/14: hidden 1280, 32 layers, patch 14, projection 1024) + `feature_extractor` + a **DINOv2-giant** encoder inside the UNet (`dino_ckpt_path="facebook/dinov2-giant"`; hidden 1536, 24 heads, 40 layers, patch 14, image 518 → 1370 tokens).
  - **Conditioner-config caveat (corpus conflict):** the `meta.key_config_digest` describes the in-UNet DINO as "dino-large: hidden 1024, 24 layers, patch 14, image 518." This contradicts the verified shape-port finding and `textureGenPipeline.py`/`modules.py` (`facebook/dinov2-giant`, hidden 1536, 40 layers). The corpus treats **dinov2-giant** as authoritative for paint. Verify against the actual checkpoint before porting. Also note: `model_index.json` lists `CLIPVisionModelWithProjection`, but the corpus flags it as likely **loaded-but-dormant** at inference (image conditioning is carried by the dual-stream reference path + DINO); confirm before porting that path.
- **What's new over 2.0 (per arXiv 2506.15442 + code):**
  1. **Spatial-aligned multi-attention / parallel dual-branch UNet** — albedo and MR are processed in a doubled batch (`n_pbr=2`) by a shared UNet; albedo features are propagated to the MR branch to keep materials spatially aligned. Five attention paths per block: material-aware self-attn (MDA), reference attn (RA, dual-stream), multiview attn (MA), text cross-attn (using learned per-material 77×1024 tokens, not CLIP text), and DINO cross-attn.
  2. **3D-aware rotary position embeddings (PoseRoPE)** — `get_1d/3d_rotary_pos_embed`, multi-resolution coordinate encodings keyed by voxelized position maps, applied in multiview attention (xy/z embed-dim split 3/8, 3/8, 2/8).
  3. **Illumination-invariant training** for light-/shadow-free albedo — replaces the explicit external Delight model of 2.0 (delighting is baked into albedo generation).
- **View selection:** candidate bake cameras azim `[0,90,180,270,0,180]`, elev `[0,0,0,0,90,-90]`, weights `[1,0.1,0.5,0.1,0.05,0.05]`; `max_num_view=6`, `resolution=512`, render_size 2048, texture_size 4096.
- **VRAM / runtime env:** ~10 GB shape, ~21 GB texture, ~29 GB combined; reference is A100, Python 3.10, PyTorch 2.5.1 + cu124.
- **License / availability:** Tencent Hunyuan 3D 2.1 Community/Non-Commercial; **open weights + training code**. PBR: yes (albedo + metallic-roughness).

### 4. Hunyuan3D-2.5 (Apr/Jun 2025) — API-only

- Announced ~2025-04-23; technical report 2025-06-23 (arXiv 2506.16504).
- **Specs (secondary reporting):** params scaled ~1B → **10B**; **1024 geometric resolution**; 4K textures; bump mapping; first multi-view input generation of PBR models; optimized skeletal skinning. Per-submodel split not public.
- **Availability:** **API-only**, no open weights. Tencent Cloud Hunyuan 3D API (~20 free gens/day). Open-source request still pending (Hunyuan3D-2.1 GitHub issue #111). The `tencent/Hunyuan3D-2` card was updated to mention 2.5, but those weights are not 2.5.
- PBR: yes (multiview). Open weights: **no**.

### 5. Hunyuan3D-3.0 / 3.1 (Sept 2025 →) — API-first

- **3.0:** unveiled Sept 2025 (Tencent Global Digital Ecosystem Summit). Hierarchical 3D-DiT "sculpting" (coarse → progressive refinement). **~10B params**; resolution up to **1536³** (~3.6B voxels); production-ready PBR material maps. (All from secondary reporting.)
- **3.1:** international/global "Pro" tier, launched ~Nov 25–26 2025; via Tencent Cloud API + 3rd parties (Replicate `tencent/hunyuan-3d-3.1`, Layer, Scenario).
- **Availability:** API-first/commercial; **no open weights** on HF/ModelScope for 3.0/3.1.
- PBR: yes. Open weights: **no**.

### 6. 2.1-based open extensions

- **Hunyuan3D-Omni** — HF `tencent/Hunyuan3D-Omni` · GitHub `Tencent-Hunyuan/Hunyuan3D-Omni` · arXiv 2509.21245. Released 2025-09-25. **~3.3B** params; inherits the Hunyuan3D-2.1 structure (DiT + VAE decoder). A unified control encoder adds 4 modalities: point cloud, voxel, bounding box, skeleton (pose). Uses the 2.1 PBR paint for texturing. License = tencent community/non-commercial. **Open weights.**
- **Hunyuan3D-Part** — HF `tencent/Hunyuan3D-Part` · GitHub `Tencent-Hunyuan/Hunyuan3D-Part`. ~Aug 2025. Part-level segmentation + part-based 3D generation. **Open weights.**
- **PolyGen** (polygon/topology generation, mentioned with 2.5) — open-source status pending per the 2.1 issue tracker.

---

## Interop / compatibility (relevant to a paint-only port)

- **Shape and paint are fully decoupled.** The paint stage consumes **only a mesh (file path or trimesh object) + the original input image** — never shape-internal latents, VAE features, or any shape-side tensor. Verified in `hy3dpaint/textureGenPipeline.py` (`__call__(mesh_path, image_path, output_mesh_path, ...)`).
- **Any mesh can be textured by any paint model.** A 2mini- or 2mv-generated mesh (or a hand-crafted mesh) can be fed to the 2.1 PBR paint pipeline; nothing in the handoff is version-locked. There is no version handshake or compatibility check in code. Cross-version pairings are mechanically supported but not officially benchmarked.
- **Shape and paint do NOT share DINOv2-giant weights.** The shape DiT bundles its own (fine-tuned) `conditioner.*` DINO weights inside its checkpoint (`conditioner.load_state_dict(ckpt['conditioner'])`); the paint pipeline independently loads stock `facebook/dinov2-giant`. Two separate instances, two weight sources.
- **Mesh handoff:** shape returns a trimesh (savable to .glb/.obj). 2.1 paint reads a mesh from disk, remeshes (`remesh_mesh` → `white_mesh_remesh.obj`, quadric decimation to ~40k faces) when `use_remesh=True`, UV-unwraps via xatlas (`mesh_uv_wrap`), runs multiview PBR diffusion + RealESRGAN x4 super-res, bakes, inpaints, and writes OBJ+MTL+JPG (optionally GLB).

---

## What this means for an MLX paint port

**Target version: Hunyuan3D-2.1 Paint-PBR (`tencent/Hunyuan3D-2.1`, subfolder `hunyuan3d-paintpbr-v2-1`).**

Rationale, drawn from the corpus:

1. **It is the only open-weights PBR paint model.** 2.5 / 3.0 / 3.1 are API-only with no downloadable weights, so they cannot be ported at all. 1.0 has no paint diffusion net (bake-only). 2.0 / 2mini / 2mv produce **RGB-only** textures. If the goal is a relightable, production-grade material output (albedo + metallic-roughness), **2.1 is the only viable open target.** Omni and Part are open but *reuse* the 2.1 paint stack, so porting 2.1 paint covers them too.

2. **The backbone is a standard SD2.1-base UNet**, which maximizes reuse from the existing shape MLX port and from standard diffusers building blocks: in/out 4ch, `cross_attention_dim=1024`, `block_out_channels=[320,640,1280,1280]`, `attention_head_dim=[5,10,20,20]`, per-head dim 64, v-prediction. The DINOv2-giant conditioner is **identical to the one the shape MLX port already implements** (hidden 1536, 24 heads, 40 layers, SwiGLU, patch 14, 1370 tokens) — only the weight source changes (paint loads stock `facebook/dinov2-giant`; shape loads bundled `conditioner.main_image_encoder.model.*` keys). The shared-primitives convention, the strict weight loader, and the dump-and-compare parity oracle methodology all transfer.

3. **Decoupled architecture keeps the port self-contained.** Because paint takes only mesh + image (no shape latents), a paint-only MLX port is a complete, runnable product: pair it with any mesh source (including the existing shape MLX port or hand-crafted meshes).

4. **The hard, paint-specific parts are well-scoped:** the 12-channel `conv_in` (channel-concat of noisy + normal + position VAE latents), the five-path `Basic2p5DTransformerBlock` (MDA / RA / MA / text / DINO), the dual-stream reference UNet run once at t=0, 3D RoPE from voxelized position maps, per-material weight routing for albedo vs MR, learned 77×1024 material tokens (CLIP text is bypassed), the 3-way CFG (uncond / ref / full with `ref_scale=[0,1,1]`), and the `UniPCMultistepScheduler` (trailing spacing, 15 steps, guidance 3.0). The renderer/rasterizer is non-neural and already addressed in this project by the in-repo Metal `mtldiffrast` + `mtlbvh` backends; RealESRGAN x4 super-res can be stubbed with bicubic for first-pass parity.

**Caveats to resolve before/while porting 2.1 (corpus-flagged):**

- The in-UNet DINO is **dinov2-giant (hidden 1536, 40 layers)** per the code, not the "dino-large 1024" mentioned in one config digest — verify against the downloaded checkpoint.
- VAE `scaling_factor` is not in `vae/config.json` (defaults to 0.18215); read it from config rather than hardcoding, and assert it.
- `CLIPVisionModelWithProjection` is listed in `model_index.json` but appears **loaded-but-dormant** at inference — confirm before porting that conditioning path (avoid porting dead code).
- The Paint param figure (1.3B vs 2B) and Shape figure (3.0B vs 3.3B) are unresolved in the corpus; not load-bearing for the port, but note them when sizing memory.
- Exact `UniPCMultistepScheduler` internals (solver_order, predict_x0, solver_type) come from the checkpoint's `scheduler_config.json` and must be read from the downloaded weights for exact step() parity.

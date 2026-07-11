# Hunyuan3D-2.1 PBR Paint Pipeline Architecture

This document describes the full Hunyuan3D-2.1 PBR texture-painting pipeline end to end: how a mesh plus an input image become a textured, relightable mesh. It draws exclusively from the research corpus assembled for the MLX port. All `file:line` references point at the reference implementation under `reference/Hunyuan3D-2.1/hy3dpaint/` (paths abbreviated to the source filename, e.g. `modules.py`, `pipeline.py`, `MeshRender.py`).

The paint stage is fully decoupled from shape generation: it consumes **only a mesh (path or in-memory trimesh) + the original input image** — no shape-internal latents — so any mesh (generated or hand-crafted) can be textured by this pipeline.

---

## 1. Pipeline Overview

```
                          INPUT
        ┌───────────────────────────────────────────┐
        │  mesh (.glb/.obj)        reference image    │
        │  from shape stage        (RGBA, post-rembg) │
        └───────────────┬───────────────────┬─────────┘
                        │                   │
                        ▼                   │
        ┌───────────────────────────┐       │
        │ MESH PRE-PROCESS           │       │
        │  remesh -> 40k faces       │       │
        │   (simplify_mesh_utils)    │       │
        │  UV unwrap (xatlas)        │       │
        │   (uvwrap_utils)           │       │
        │  load_mesh -> MeshRender   │       │
        │   normalize: flip XY,      │       │
        │   swap YZ, scale 1.15,     │       │
        │   flip UV-v                │       │
        └───────────────┬───────────┘       │
                        │                   │
                        ▼                   │
        ┌───────────────────────────┐       │
        │ GEOMETRY CONTROL RENDER    │       │
        │  custom_rasterizer +       │       │
        │  MeshRender per camera:    │       │
        │   normal maps (abs coords) │       │
        │   position maps (CCM)      │       │
        │  6 selected views          │       │
        └───────────────┬───────────┘       │
                        │ normal[6] + position[6]    │ ref image (512, white bg)
                        ▼                   ▼
        ┌─────────────────────────────────────────────┐
        │ MULTIVIEW 2.5D PBR DIFFUSION                  │
        │  HunyuanPaintPipeline                         │
        │   UNet2p5DConditionModel (SD2.1 base)         │
        │   conv_in 12ch = noise(4)+normal(4)+pos(4)    │
        │   5 attentions: MDA / RA / MA(3D-RoPE) /      │
        │                 text(learned) / DINO          │
        │   conditioners: DINOv2-giant, CLIP text/vis   │
        │   UniPC trailing, 15 steps, CFG 3.0 (3-way)   │
        │   batch = B(1) x n_pbr(2) x N_view(6)         │
        └───────────────┬───────────────────────────────┘
                        │ {albedo:[6 PIL], mr:[6 PIL]}  (512x512 each)
                        ▼
        ┌───────────────────────────┐
        │ SUPER-RESOLUTION           │
        │  RealESRGAN x4 per view    │
        │  (image_super_utils)       │
        │  then resize -> 2048       │
        └───────────────┬───────────┘
                        │ albedo[6], mr[6] @ 2048
                        ▼
        ┌───────────────────────────┐
        │ BACK-PROJECTION BAKE       │
        │  MeshRender.back_project   │
        │   per view: cos-weighted   │
        │   depth-tested UV scatter  │
        │  fast_bake_texture blend   │
        │  separately for albedo,mr  │
        └───────────────┬───────────┘
                        │ UV textures + coverage masks
                        ▼
        ┌───────────────────────────┐
        │ INPAINT + EXPORT           │
        │  meshVerticeInpaint (C++)  │
        │  + cv2 Navier-Stokes       │
        │  set_texture / _mr         │
        │  save_mesh -> OBJ+MTL+JPG  │
        │  -> GLB (pygltflib/Blender)│
        └───────────────┬───────────┘
                        ▼
                  TEXTURED MESH
              (albedo + metallic-roughness PBR)
```

Three entry points converge on the same handoff (`Hunyuan3DPaintPipeline.__call__(mesh_path, image_path, output_mesh_path, ...)`):

- `demo.py` (CLI) — exports shape to `.glb`, calls paint with `save_glb=True` (Blender GLB writer).
- `model_worker.py` (API) — passes the in-memory PIL image, `save_glb=False`, then `quick_convert_with_obj2gltf` (pygltflib). On paint exception falls back to the untextured initial `.glb` (`model_worker.py:208-212`).
- `gradio_app.py` (UI) — passes the in-memory PIL image, `save_glb=False`, pygltflib conversion.

The same post-rembg input image is reused for both shape and paint; background removal happens once, before shape generation.

---

## 2. The Multiview 2.5D PBR UNet (`UNet2p5DConditionModel`)

### 2.1 Base SD2.1 UNet (frozen backbone)

A stock diffusers `UNet2DConditionModel` from `stabilityai/stable-diffusion-2-1` (`cfgs/hunyuan-paint-pbr.yaml:12`). Verified topology (identical across paint-v2-0 and paintpbr-v2-1 `unet/config.json`):

| Field | Value |
|---|---|
| `in_channels` / `out_channels` | 4 / 4 (overridden to 12 in; see 2.2) |
| `cross_attention_dim` | 1024 |
| `block_out_channels` | `[320, 640, 1280, 1280]` |
| `attention_head_dim` (num heads/block) | `[5, 10, 20, 20]` |
| per-head dim | 64 (320 / 5 = 64) |
| `sample_size` | 64 |
| `layers_per_block` | 2 |
| `norm_num_groups` | 32 |
| `act_fn` | silu |
| `use_linear_projection` | true |
| prediction | `v_prediction` |

Block layout: down = 3× `CrossAttnDownBlock2D` + 1× `DownBlock2D`; mid = 1× `CrossAttn`; up = 1× `UpBlock2D` + 3× `CrossAttnUpBlock2D`. Inference scheduler is `EulerAncestralDiscreteScheduler` with `timestep_spacing="trailing"` (training uses DDPM) — see `model.py:82-99`. (In production `multiview_utils.py` substitutes `UniPCMultistepScheduler.from_config(..., timestep_spacing="trailing")`; see §3.)

### 2.2 The 2.5D wrapping and the 12-channel `conv_in`

`UNet2p5DConditionModel.__init__` (`modules.py:774`) stores `self.unet` and enables all feature flags: `use_ma = use_ra = use_mda = use_dino = use_position_rope = use_learned_text_clip = use_dual_stream = True` (`modules.py:785-791`), with `pbr_setting=["albedo","mr"]` and `pbr_token_channels=77` (`modules.py:792-793`).

**conv_in expansion to 12 channels** (`modules.py:818-827`, applied in `from_pretrained`): `unet.conv_in` is replaced with `Conv2d(12, out_channels, ...)`. The `noise_in_channels=12` setting comes from `cfgs/hunyuan-paint-pbr.yaml:9`. Breakdown:

```
12 = 4 (noisy generation latent)
   + 4 (VAE-encoded normal-map latent)
   + 4 (VAE-encoded position-map latent)
```

At training (`train.py:247-263`) the new conv_in weight is zero-initialized, the original 4-channel weights copied into channels `[0:4]`, extra 8 channels start at 0, bias copied. **At inference the released checkpoint already ships the full trained 12-channel conv_in — channels 4:12 carry trained conditioning and must NOT be zeroed.**

**Conditioning is channel-concat, not ControlNet.** In `forward` (`modules.py:965-970`): `sample = [noisy latent]`; if `embeds_normal` present, append it (unsqueeze pbr dim, repeat over `N_pbr`); if `embeds_position` present, append; `torch.cat(dim=-3)` (channel). Normal/position maps are first VAE-encoded to 4-ch latents by `HunyuanPaint.encode_images` (`model.py:351-356`; encoded at `pipeline.py:257,265`). A ControlNet path exists (`model.py:105-110`, `add_controlnet`, conditioning_scale 0.75) but `control_net_config` defaults `None` and the released yaml `init_control_from` is empty (`yaml:51`), so the canonical model uses channel-concat only.

### 2.3 The five attention mechanisms (`Basic2p5DTransformerBlock`)

`init_attention(self.unet, ...)` recursively swaps each `BasicTransformerBlock` -> `Basic2p5DTransformerBlock` (`modules.py:277`) in the down/mid/up attention layers, naming them e.g. `"down_0_0_0"`, `"mid_0_0"`, `"up_1_0_0"` (`modules.py:859-913`). Forward order per block (`modules.py:472-707`):

**(0) MDA — Material-aware self-attention** (`attn1`, `SelfAttnProcessor2_0`). Input reshaped `(b n_pbr n) l c -> b n_pbr n l c`. The processor splits along `n_pbr`: albedo runs through the base `attn1.to_q/k/v/out`; `mr` runs through processor-resident `to_q_mr/to_k_mr/to_v_mr/to_out_mr`; then concatenates (`attn_processor.py:714-755`). Per-view, per-material self-attention (no cross-view mixing). Residual add (`modules.py:556-576`).

**(1) RA — Reference attention** (`attn_refview`, `RefAttnProcessor2_0`). Two modes:
- The dual-stream reference UNet runs first in mode **"w"** (`modules.py:1051-1064`): in each block it WRITES `condition_embed_dict[layer_name] = norm_hidden_states` rearranged `(b n) l c -> b (n l) c` (`modules.py:581-584`).
- The main UNet runs in mode **"r"**: it uses **only albedo features as query** (`ref_norm_hidden_states = [...][:,0,...]`, "Only using albedo features for reference attention", `modules.py:589-592`) and cross-attends to the stored reference `condition_embed`. RA shares Q/K but uses per-material V (`to_v` / `to_v_mr`) and per-material out, scaled by `ref_scale` (`modules.py:586-609`, `attn_processor.py:777-839`).

**(2) MA — Multiview attention** (`attn_multiview`, `PoseRoPEAttnProcessor2_0`), only if `num_in_batch > 1`. `norm_hidden_states` rearranged `(b n_pbr n) l c -> (b n_pbr) (n l) c` so **all N=6 views' tokens are concatenated along sequence and attend to each other within each (batch, material) group** — this is the cross-view mixing (`modules.py:612-615`). Uses 3D rotary position embedding keyed by per-pixel voxel indices computed from the position maps (`compute_discrete_voxel_indice` / `calc_multires_voxel_idxs`, `modules.py:204-274`; RoPE in `attn_processor.py:553-635`). `get_3d_rotary_pos_embed` splits embed dims xy = 3/8 each, z = 2/8. Scaled by `mva_scale`, residual add (`modules.py:621-633`).

**(3) Text cross-attention** (`attn2`): `norm2 -> attn2` with `encoder_hidden_states` = the learned per-material 77×1024 tokens (NOT real CLIP text), residual add (`modules.py:640-663`).

**(4) DINO cross-attention** (`attn_dino`, zero-init): a parallel cross-attention of the SAME `norm_hidden_states` to the projected DINO tokens (repeated over `n_pbr * num_in_batch`), residual add (`modules.py:665-676`).

**(5) Feed-forward** (`modules.py:678-705`).

Summary of who attends to whom: **MA** is the only path where the 6 views attend to each other (token-concat with 3D-RoPE position bias from voxelized position maps). **RA** injects reference-image identity. **DINO** injects semantic features. **Text cross-attn** carries the per-material identity token.

### 2.4 PBR dual-branch (albedo + metallic-roughness)

A **single shared UNet with a doubled batch** (`n_pbr=2`). Sample shape `[B, N_pbr, N_gen, C, H, W]` is flattened to `(b n_pbr n) c h w` before the base UNet (`modules.py:959,972`). Albedo uses the base `attn1`/`attn_refview` weights; `mr` uses material-specific cloned weights registered by `register_pbr_modules` (`attn_processor.py:308-363`): `to_q_mr, to_k_mr, to_v_mr, to_out_mr` in MDA, and `to_v_mr, to_out_mr` in RA. Per-material identity also enters via learned tokens `learned_text_clip_albedo` / `learned_text_clip_mr`, each `nn.Parameter(zeros(77,1024))` (`modules.py:845-849`).

`pbr_settings=["albedo","mr"]` drives the number of learned tokens, the number of material weight clones, the `n_pbr` batch factor, the loss split (`albedo_loss` vs `mr_loss`, `model.py:407-446`), and the output split. There are **no separate decoder heads** — albedo and mr are simply different slots in the `n_pbr` batch dimension, recovered by split at the end (`model.py:407-413`). Reference attention deliberately uses albedo-only query but produces both-material V (a single shared reference identity broadcast to both materials).

The `mr` map is metallic-roughness packed together (note: on export, metallic and roughness are written as separate grayscale JPGs; see §7).

### 2.5 Conditioning paths (with dims)

- **Learned material "text" tokens** (`use_learned_text_clip`) — replace CLIP text completely. `shading_embeds` = stack of `learned_text_clip_{albedo,mr}` -> `[B, N_pbr, 77, 1024]`, passed as `encoder_hidden_states` to `attn2` (`model.py:331-337`; `pipeline.py:267-275`). The prompt string `"high quality"` is passed but IGNORED (`pipeline.py:267-286`). `learned_text_clip_ref` (zeros 77×1024) is the `encoder_hidden_states` for the reference dual-stream pass (`modules.py:1029`). `pbr_token_channels=77` matches CLIP's 77-token context.

- **DINOv2-giant** (`use_dino`) — `Dino_v2("facebook/dinov2-giant")` frozen (`modules.py:38-86`, `model.py:118-120`). Forward gives `last_hidden_state [B, N*num_patches, 1536]` (giant hidden = 1536). Projected by `ImageProjModel` (`image_proj_model_dino`, `modules.py:853-857`): `clip_embeddings_dim=1536 -> 4 context tokens of dim 1024` (`modules.py:710-755`). Feeds the dedicated zero-init `attn_dino` (`modules.py:1000-1007, 665-676`). DINO is computed only on the FIRST reference image `cond_imgs[:, :1]` (`model.py:340`).

- **CLIP text encoder** — present in the SD2.1 pipeline (`CLIPTextModel`, hidden 1024 / 23 layers / 16 heads, from `stabilityai/stable-diffusion-2`) and used by `encode_prompt`, but **bypassed in normal operation** because learned tokens are used.

- **CLIP image encoder** (`CLIPVisionModelWithProjection`, hidden 1280 / 32 layers / patch 14 / image 224 / projection_dim 1024 — CLIP-H/14, listed in `model_index.json`) — the released paint forward does NOT wire an IP-Adapter/ImageProjModel-from-CLIP path. Image conditioning is carried by (a) the reference latent through the dual-stream RA and (b) DINO. `ip_adapter` args in `denoise()` are accepted but unused by default.

- **Reference image** — VAE-encoded to `ref_latents [B, N_ref, 4, H/8, W/8]` (`pipeline.py:231-233`), run through `unet_dual` in mode "w" at timestep 0 to populate `condition_embed_dict` (`modules.py:1011-1065`).

### 2.6 Learned vs frozen parameters

`set_learned_parameters` (`pipeline.py:121-143`): `freezed_names=["attn1","unet_dual"]`, `added_learned_names=["albedo","mr","dino"]`. A param is FROZEN iff its name contains `"attn1"` or `"unet_dual"` AND contains none of albedo/mr/dino.

- **Frozen** = base self-attn (`attn1.to_q/k/v/out` for albedo) + the entire dual-stream reference UNet.
- **Trained** = everything else: `conv_in` (12ch), `attn_multiview` (MA), `attn_refview` (RA, incl. `to_v_mr`/`to_out_mr`), `attn_dino`, material weight clones `to_*_mr`, `learned_text_clip_*`, `image_proj_model_dino`, and all base `attn2`/conv/resnet params not named `attn1`/`unet_dual`. (Note: albedo MDA weights live in `attn1` and are frozen, but mr MDA weights `to_q_mr` etc. contain "mr" and are trained.)

**Zero-init residual strategy** (`_initialize_attn_weights`, `modules.py:405-464`): MA/RA/DINO copy base weights then zero their `to_out`, so the wrapper starts as an identity transform; mr material weights copy from albedo; learned tokens init to zeros.

### 2.7 Exact tensor shapes (num_view=6, view_size=512, B=1, N_pbr=2, N_ref=1)

| Tensor | Shape |
|---|---|
| Reference image -> `ref_latents` | `[1,1,4,64,64]` (512/8) |
| Normal map (VAE-encoded), repeated over n_pbr | `[1,6,4,64,64]` -> `[1,2,6,4,64,64]` |
| Position map (VAE-encoded), repeated over n_pbr | `[1,6,4,64,64]` -> `[1,2,6,4,64,64]` |
| Noisy gen latents `sample` | `[1,2,6,4,64,64]` |
| After channel-concat (noise+normal+position) | `[1,2,6,12,64,64]` |
| Flattened into conv_in | `[(1*2*6)=12, 12, 64, 64]` |
| `encoder_hidden_states` (learned tokens), per pass | `[1,2,77,1024]` -> flattened over views `[12,77,1024]` |
| DINO `last_hidden_state` (512px ref) | `[1, ~1370, 1536]` -> ImageProjModel -> `[1, 4, 1024]` |
| MDA self-attn at down_0 (64×64, L=4096, C=320) | `[12, 4096, 320]` |
| MA cross-view reshape at down_0 | `[(b n_pbr)=2, (n l)=6*4096=24576, 320]` |
| UNet output | `[12, 4, 64, 64]` -> `[1,2,6,4,64,64]` -> split into `v_pred_albedo`/`v_pred_mr` each `[1,1,6,4,64,64]` |

Voxel/position RoPE indices: `grid_resolutions=[H,H/2,H/4,H/8]=[64,32,16,8]`, `voxel_resolutions=[H*8,H*4,H*2,H]=[512,256,128,64]`, keyed by flattened token count (`modules.py:991-995`); RoPE head_dim = 64.

At inference with `guidance_scale>1` the batch is tripled (uncond/ref/full): `prompt_embeds [3,2,77,1024]`; latents prepared as `batch_size*num_in_batch*n_pbr = 1*6*2 = 12`, tripled to 36 -> `[36,4,64,64]` entering, `[36,12,64,64]` after concat (`pipeline.py:578,595-606,635-643`).

---

## 3. Diffusion Sampling

### 3.1 Scheduler and steps

`UniPCMultistepScheduler.from_config(orig_config, timestep_spacing="trailing")` (`multiview_utils.py:49`). `num_inference_steps = 15` (the UniPC entry of `infer_steps_dict`, `multiview_utils.py:108-115`). `guidance_scale = 3.0` (`multiview_utils.py:118`). `torch_dtype = float16` (`multiview_utils.py:46`).

The on-disk `scheduler/scheduler_config.json` (DDIM, the checkpoint default) declares `prediction_type=v_prediction`, `beta_schedule=scaled_linear` (0.00085 -> 0.012), `num_train_timesteps=1000`, `timestep_spacing=trailing`, `rescale_betas_zero_snr=true`, `set_alpha_to_one=true`, `clip_sample=false`. This is the zero-terminal-SNR v-model setup; UniPC inherits its config from this.

Timesteps via `retrieve_timesteps(scheduler, 15, ...)` (`pipeline.py:590-592`). `scale_model_input` is effectively identity for UniPC but is called anyway.

### 3.2 Latent packing

`n_pbr = len(pbr_setting) = 2`; `num_channels_latents = unet.config.in_channels` (= 12); `prepare_latents(batch_size * num_in_batch * n_pbr, ...)` (`pipeline.py:594-606`). The standard SD `prepare_latents` creates `randn` of shape `(N, num_channels_latents, h//vae_scale, w//vae_scale)` seeded by the passed generator, scaled by `scheduler.init_noise_sigma`.

Latents are reshuffled throughout as `(b n_pbr n) c h w <-> b n_pbr n c h w` with `n = num_in_batch (=6)`, `n_pbr = 2` (`pipeline.py:635-644, 658`).

### 3.3 The denoise loop (`pipeline.py:347-737`)

- `kwargs["cache"] = {}` opened (`pipeline.py:501`) — caches `condition_embed_dict`, DINO projection, and voxel indices across all 15 steps, so the reference UNet and DINO projection run **once**.
- `prompt_embeds = cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])` -> 3 CFG branches stacked on batch (`pipeline.py:576-578`). **Note: `negative_prompt_embeds == prompt_embeds` — both are the learned tokens, NOT zeros** (`pipeline.py:267-276`, commented-out `zeros_like`).

Per step `t` (`pipeline.py:630-714`):
1. `latents` rearrange `(b n_pbr n) c h w -> b n_pbr n c h w`.
2. `latent_model_input = latents.repeat(3,1,1,1,1,1)` (the 3 CFG branches) -> rearrange to `(b n_pbr n) c h w`.
3. `scheduler.scale_model_input(latent_model_input, t)`, rearrange back to 6D.
4. `noise_pred = self.unet(latent_model_input[b,n_pbr,n,c,h,w], t, encoder_hidden_states=prompt_embeds, **kwargs)`. `kwargs` carries `num_in_batch, cache, embeds_normal, embeds_position, position_maps, ref_latents, ref_scale, dino_hidden_states, mva_scale/ref_scale`.
5. **CFG (3-way)** (`pipeline.py:660-688`): `noise_pred.chunk(3) -> uncond, ref, full`. `cam_mapping(azim)`: `[0,90)->azim/90+1`; `[90,330)->2.0`; else `-azim/90+5`. `view_scale_tensor` = mapping per view, repeated over n_pbr, shape `[N,1,1,1]`. Final:
   ```
   noise_pred  = uncond + g*view_scale*(ref  - uncond)
   noise_pred += g*view_scale*(full - ref)
   ```
   (Collapses to `uncond + g*vs*(full - uncond)`.) **In production `camera_azims` is never passed, so azim=0 for all views, `cam_mapping(0)=1.0`, and `view_scale_tensor` is all-ones.**
6. Optional `rescale_noise_cfg` if `guidance_rescale>0` (default 0, skipped).
7. `latents = scheduler.step(noise_pred, t, latents[:, :num_channels_latents, :, :])[0]`. The `[:, :num_channels_latents]` slice is a no-op on the channel dim (4 latent channels < 12).

Decode: `image = vae.decode(latents / vae.config.scaling_factor)[0]` (`pipeline.py:717`); `run_safety_checker` is a no-op (checker None); `image_processor.postprocess` -> PIL.

### 3.4 Control injection per step (`UNet2p5DConditionModel.forward`, `modules.py:921-1102`)

- Channel-concat `[sample, embeds_normal (unsqueeze+repeat over n_pbr), embeds_position]` along channel dim -3 -> 12 channels (`modules.py:965-970`).
- `encoder_hidden_states` (learned tokens `[B,n_pbr,77,1024]`) repeated over `N_gen` views -> `(b n_pbr n) l c` (`modules.py:974-975`).
- **Position RoPE**: `calc_multires_voxel_idxs(position_maps, grid_resolutions=[H,H/2,H/4,H/8], voxel_resolutions=[H*8,H*4,H*2,H])` cached (`modules.py:986-998`). `position_maps` here is the PIXEL position-map tensor (not the VAE embed) — position maps are used twice: pixel tensor for RoPE, and VAE-encoded for channel concat.
- **DINO**: `image_proj_model_dino(dino_hidden_states)` cached as `dino_hidden_states_proj` (`modules.py:1000-1009`).
- **Reference (dual stream)**: `unet_dual(noisy_ref_latents, timestep_ref=0, ..., cross_attention_kwargs={mode:"w", num_in_batch:N_ref, condition_embed_dict})` runs once to WRITE per-layer reference features (`modules.py:1011-1065`).
- **Main UNet**: `unet(sample, t, encoder_hidden_states_gen, cross_attention_kwargs={mode:"r", num_in_batch:N_gen, dino_hidden_states(proj), condition_embed_dict, mva_scale, ref_scale, position_voxel_indices})` (`modules.py:1072-1101`).

### 3.5 VAE encode/decode conventions

`encode_images` (`pipeline.py:149-170`): rearrange `(b n) c h w`, scale to `[-1,1]` via `(x-0.5)*2`, `vae.encode(...).latent_dist.sample() * vae.config.scaling_factor`, rearrange back to `[b,n,c,h,w]`. Decode: `latents / scaling_factor` (`pipeline.py:717`). The VAE (`AutoencoderKL`): `latent_channels=4`, `block_out_channels=[128,256,512,512]`, in/out=3, `sample_size=768`, `layers_per_block=2`, `norm_num_groups=32`. `scaling_factor` is NOT in `vae/config.json` -> diffusers default **0.18215** (read at runtime via `vae.config.scaling_factor`; do not hardcode — read from config).

### 3.6 CFG conditioning gating

- `ref_latents` repeated ×3 on batch with `ref_scale = [0.0, 1.0, 1.0]` (uncond branch zeroes reference attention) (`pipeline.py:300-305`).
- `dino_hidden_states -> cat([zeros, zeros, dino])` so only the "full" branch sees real DINO (`pipeline.py:307-313`).
- `embeds_normal`, `embeds_position`, `position_maps` each repeated ×3 (`pipeline.py:314-327`).

This means the three CFG branches differ in CONDITIONING, not just text — they cannot be collapsed into a single UNet call unless `ref_latents`/`dino`/`normal`/`position` are also concatenated ×3 as the code does.

### 3.7 Output split

`.images` is a flat list of length `n_pbr * num_view = 12`, split: `mvd_image = {"albedo": images[:num_view], "mr": images[num_view:]}` (`multiview_utils.py:122-124`).

Seeds: `seed_everything(0)` sets random/np/torch seeds; `generator = torch.Generator(device).manual_seed(0)` passed to the pipeline drives `prepare_latents` noise (`multiview_utils.py:60-64, 88`).

---

## 4. Conditioners (exact roles)

| Conditioner | Model | Output dims | Role | Path into UNet |
|---|---|---|---|---|
| **DINOv2-giant** | `facebook/dinov2-giant` (frozen, fp16) | `last_hidden_state [B, N*~1370, 1536]` -> ImageProjModel -> `[B, 4, 1024]` | Semantic features of the **first reference image only** | Dedicated zero-init `attn_dino` cross-attention; gated `[0,0,dino]` across CFG branches |
| **CLIP text encoder** | `CLIPTextModel` (SD2-2, hidden 1024 / 23 layers / 16 heads) | `[*, 77, 1024]` | Present but **bypassed** — `use_learned_text_clip=True` replaces it with learned per-material tokens | (unused at inference; `attn2` consumes learned tokens instead) |
| **CLIP vision** | `CLIPVisionModelWithProjection` (CLIP-H/14, hidden 1280 / 32 layers / projection 1024) | n/a at inference | Listed in `model_index.json` but **not wired** into the released forward | none active (IP-adapter path dormant) |
| **Reference image latent** | VAE-encoded ref | `ref_latents [B, N_ref, 4, 64, 64]` | Identity/appearance via dual-stream RA | `unet_dual` mode "w" at t=0 -> `condition_embed_dict` -> RA mode "r" |
| **Learned material tokens** | `nn.Parameter(zeros(77,1024))` ×3 (albedo/mr/ref) | `[B, N_pbr, 77, 1024]` | Per-material identity (replaces CLIP text) | `attn2` cross-attention (`encoder_hidden_states`) |

`ImageProjModel` (`modules.py:710-755`): `Linear 1536 -> 4*1024` + `LayerNorm`, producing 4 context tokens.

Shape vs paint do **not** share DINOv2-giant weights: the shape DiT bundles fine-tuned `conditioner.*` weights inside its own checkpoint; paint independently loads the stock `facebook/dinov2-giant`. (The bundled shape conditioner and stock HF giant share the same architecture — hidden 1536, 24 heads, 40 layers, SwiGLU FFN, patch 14, image 518 -> 37×37=1369 patches + 1 CLS = 1370 tokens — only the weight source differs.)

---

## 5. Differentiable Renderer + Custom CUDA Rasterizer

The directory is named "DifferentiableRenderer" but the rasterizer is **not** gradient-producing: it returns a face-index map + perspective-correct barycentric weights only (`raster_antialias` is a no-op, `grad_db` unused). The only true CUDA code is `rasterizer_gpu.cu`, which has a bit-identical scalar CPU twin (`rasterize_image_cpu`).

### 5.1 The rasterizer (`custom_rasterizer`)

Python entry `custom_rasterizer/render.py:19 rasterize(pos, tri, resolution, ...)` calls `rasterize_image(pos[0], tri, clamp_depth, W, H, 1e-6, use_depth_prior=0)`. Inputs: `pos` = clip-space verts `[1,V,4]` (homogeneous x,y,z,w), `tri` = int32 faces `[F,3]`. Outputs:
- `findices [H,W]` int32 — 1-based face id, 0 = background.
- `barycentric [H,W,3]` float32 — perspective-correct, sums to 1 inside a face, 0 outside.

Dispatch (`rasterizer.cpp:125`): if `V.get_device()==-1` -> CPU path `rasterize_image_cpu` (`rasterizer.cpp:94`), else GPU (`rasterizer_gpu.cu:100`). Both algorithmically identical.

**Pass 1 — per-face rasterize** (`rasterizer_gpu.cu:83` / `rasterizer.cpp:81`): for each face, fetch its 3 clip-space verts (stride 4 floats), convert to screen+depth:
```
sx = (x/w*0.5 + 0.5)*(W-1) + 0.5
sy = (0.5 + 0.5*y/w)*(H-1) + 0.5
sz = z/w*0.49999 + 0.5
```
(`rasterizer.cpp:87-89`). Loop pixels, sample center `(px+0.5, py+0.5)`, compute screen-space barycentric via signed areas; keep pixel if all 3 in `[0,1]`. Interpolated depth `depth = Σ baryc_i*sz_i`. **Z-buffer is an INT64 token**: `z_quantize = int(depth*(2<<17)); token = (INT64)z_quantize*MAXINT + (idx+1)` where `MAXINT=2147483647`, `2<<17=262144` (18-bit depth) (`rasterizer.h:11`, `rasterizer_gpu.cu:30-34`). The **min token wins** (`atomicMin` GPU / `std::min` CPU) — depth in the high bits gives a nearest-depth z-test, ties broken by smaller face id. `use_depth_prior`/`occlusion_truncation` path is unused (called with 0).

**Pass 2 — per-pixel barycentric** (`rasterizer_gpu.cu:40` / `rasterizer.cpp:41`): decode `f = zbuffer[pix] % MAXINT`; if `f == MAXINT-1` -> background (findices=0, baryc=0). Else `findices = f`, recompute SCREEN baryc, then perspective-correct: `baryc_i /= w_i; renormalize by 1/Σ` (`rasterizer.cpp:67-73`). Degenerate area==0 -> baryc `(-1,-1,-1)` rejected.

**interpolate** (`render.py:27`): given `col [1,V,C]`, `findices`, `barycentric`, `tri`: `f = findices-1 + (findices==0)`, gather `vcol = col[0, tri[f]]` -> `[H,W,3,C]`, `result = Σ baryc·vcol` -> `[H,W,C]`. In `MeshRender.raster_rasterize` the output is packed as `rast_out = cat([barycentric, findices[...,None]], -1)` shape `[1,H,W,4]`.

### 5.2 MeshRender setup and render modes

`set_mesh` (`MeshRender.py:665`): loads vtx/uv as torch, **flips axes** `vtx_pos[:, [0,1]] = -vtx_pos[:, [0,1]]`, swaps y,z (`703-704`), **flips UV v** `vtx_uv[:,1] = 1 - vtx_uv[:,1]` (`706`), auto-centers+scales mesh by `scale_factor/(2·max‖v-center‖)` with `scale_factor=1.15` (`709-717`).

`extract_textiles` (`MeshRender.py:923`): rasterizes the mesh **in UV space** (UVs scaled to clip coords `vtx_uv*2-1`) at texture resolution, interpolates world `vtx_pos` and per-vertex normals into a UV-space `tex_position`, `tex_normal`, builds `tex_grid` (UV pixel coords of each valid texel) and `texture_indices` (HxW -> texel-seq map). This is the key structure for `back_sample`.

Cameras (`camera_utils.py`): `get_mv_matrix(elev,azim,dist)` builds a look-at world->cam matrix (`elev=-elev; azim+=90; up=[0,0,1]`), `get_orthographic_projection_matrix` (default `ortho_scale=1.2`, production `camera_type='orth'`) or perspective (fovy 49.13). `transform_pos` does `posw @ M.T`.

Render modes (`_unified_render_pipeline`, `MeshRender.py:448`):
- **ALPHA** -> `rast_out[...,-1:] > 0`.
- **NORMAL** -> face or vertex normals; face mode (`shader_type='face'`, default) reads `tri_ids=rast_out[...,3]`, looks up precomputed `face_normals` per pixel (`439-444`); normals in camera space (default) or abs/world coords (`use_abs_coor`); masked with bg; optionally `(n+1)*0.5`. **The control normal maps use `use_abs_coor=True`** (world/canonical normals).
- **POSITION** -> interpolates `tex_position = 0.5 - vtx_pos/scale_factor` across the raster (`478-482`) — the canonical coordinate map (CCM).
- **UV_POS** -> `uv_feature_map(vtx_pos*0.5+0.5)` rasterizes vertex positions into UV space.
- Depth is produced inside `back_project` by interpolating camera-space z (not a standalone mode).

### 5.3 Back-projection bake (`bake_mode='back_sample'`, `MeshRender.py:1113`, branch at 1248)

For one view image: compute mv/proj, camera-space verts, per-face normals; rasterize at image resolution -> `visible_mask`, per-pixel normal, uv, camera-space depth (`1159-1180`). Compute `cos_image = cosine(lookat=[0,0,-1], normal)`, zero where `cos < cos(75°)` (`bake_angle_thres`, `1192-1193`). Depth -> Canny sketch edge map (`render_sketch_from_depth`), dilate unreliable boundary by `bake_unreliable_kernel_size` conv (`1196-1211`).

`back_sample` core (`1248-1311`): project the precomputed `tex_position` texels through `w2c·img_proj` to NDC `v_proj` (`1254`); keep texels inside `[-1,1]²` (`inner_mask`); map to image pixels `img_x,img_y` (`1257-1264`); gather rendered depth `sampled_z`, mask `sampled_m`, weight `sampled_w` (`1265-1269`); **keep a texel iff** `|v_z - sampled_z| < 3e-3` AND `sampled_m*sampled_w > 0` AND inside frustum (`1270-1276`); bilinearly sample RGB at the float pixel coord (`1283-1295`); scatter sampled RGB into `texture` at `tex_grid` UV positions, scatter weight into `cos_map`, sketch into `boundary_map` (`1304-1307`).

`bake_texture` (`1317`): loops views, `cos_map = weight·cos_map^exp` (exp=6), then `fast_bake_texture` (`1352`) does weighted blend `texture_merge = Σ tex·cos / clamp(Σcos,1e-8)` (`1369-1378`), with an early-skip if a view adds <1% new coverage (`1370-1372`). Alternative `'linear'`/`'mip-map'` bakes use scatter-add grid-put with bilinear splatting + mipmap hole-filling (`linear_grid_put_2d` / `mipmap_linear_grid_put_2d`, `146-263`).

Albedo and mr are baked **separately** (`textureGenPipeline.py:170-177`), then masks -> uint8.

### 5.4 C++ inpaint (`uv_inpaint`, `MeshRender.py:1380`)

First `meshVerticeInpaint` (C++ vertex-graph fill), then `cv2.inpaint(..., INPAINT_NS)` (Navier-Stokes) on remaining holes. `meshVerticeInpaint` is pure CPU pybind11, compiled by a one-line `c++ -O3 -shared` command (`compile_mesh_painter.sh:1`), no CUDA. The vertex graph G is built from `pos_idx` edges (directed ring per face); UV->pixel via `round(u*(W-1))`, `round((1-v)*(H-1))` (`mesh_inpaint_processor.cpp:55-58,80-87`). The default 'smooth' method is iterative distance-weighted (`1/dist²`) averaging from colored neighbors over uncolored vertices (`mesh_inpaint_processor.cpp:121-168,283-304`).

### 5.5 MeshRender defaults

`camera_distance=1.45`, `camera_type='orth'` (`ortho_scale=1.2`), `default_resolution=1024`, `texture_size=1024` (config raises `render_size=2048`, `texture_size=4096`), `bake_mode='back_sample'`, `raster_mode='cr'`, `shader_type='face'`, `bake_angle_thres=75°`, `device='cuda'` (`MeshRender.py:324-368`).

`grid_neighbor.cpp` (`build_hierarchy`/`build_hierarchy_with_feat`, octree-grid neighbor builder) is dead weight for the paint render/bake path — only `rasterize_image` + `interpolate` are called.

### 5.6 View selection

Candidate views (`textureGenPipeline.py:57-68`): `azims=[0,90,180,270,0,180]`, `elevs=[0,0,0,0,90,-90]`, `weights=[1,0.1,0.5,0.1,0.05,0.05]`, plus 12 tilt views at elev ±20 every 30° (weight 0.01). `ViewProcessor.bake_view_selection` (`pipeline_utils.py:40-109`): first 6 always selected, then greedy area-coverage up to `max_selected_view_num`, threshold new area > 0.01 (`pipeline_utils.py:71-105`).

---

## 6. Super-Resolution (RealESRGAN)

Applied to every generated albedo and mr view BEFORE baking (`textureGenPipeline.py:156-162`): deepcopy the albedo and mr lists, then run `super_model(img)` per view, then resize each enhanced view to `render_size` (2048) (`textureGenPipeline.py:165-169`).

`imageSuperNet` (`image_super_utils.py:19-41`): wraps `realesrgan.RealESRGANer` with a `basicsr` `RRDBNet` (`num_in_ch=3, out=3, feat=64, block=23, grow=32, scale=4`), `scale=4`, `model_path=config.realesrgan_ckpt_path` (`ckpt/RealESRGAN_x4plus.pth`), `tile=0`, `half=True`. `__call__` takes a PIL image -> `np.array` -> `upsampler.enhance` -> PIL. This is RealESRGAN x4plus.

`torchvision_fix.apply_fix` monkey-patches `sys.modules['torchvision.transforms.functional_tensor']` (removed in newer torchvision) with a mock exposing `rgb_to_grayscale`/`resize`, because `basicsr`/RealESRGAN import the dropped module. Called at the top of every entry point before importing basicsr/realesrgan.

Super-resolution is a post-process upscale on the generated views, **not** part of the diffusion sampling.

---

## 7. Mesh Pre/Post-processing

### 7.1 Remesh

`simplify_mesh_utils.remesh_mesh(mesh_path, remesh_path)` -> `mesh_simplify_trimesh(target_count=40000)` (`simplify_mesh_utils.py:19-37`). Loads via `pymeshlab.MeshSet` (`load_in_a_single_layer` for `.glb`), saves to `.obj` with `save_textures=False` (strips existing texture), reloads with `trimesh force='mesh'`, and if `faces>40000` runs `simplify_quadric_decimation(40000)` then exports. The actual decimation is trimesh's quadric decimation; pymeshlab is used only for robust load + format normalization. Triggered when `use_remesh=True` (default); intermediate is `white_mesh_remesh.obj`.

### 7.2 UV unwrap

`uvwrap_utils.mesh_uv_wrap(mesh)` (`uvwrap_utils.py:19-32`): if a Scene, `dump(concatenate)`; hard cap 5e8 faces; `vmapping, indices, uvs = xatlas.parametrize(vertices, faces)`; reindex `mesh.vertices=vertices[vmapping]`, `mesh.faces=indices`, `mesh.visual.uv=uvs`. Pure xatlas. Called after remesh load, before rendering.

### 7.3 Export

`MeshRender.save_mesh(output_mesh_path, downsample=True)` -> `mesh_utils.save_obj_mesh` writes:
- **OBJ**: `mtllib <name>.mtl`, vertices, `vt` UVs, faces as `v/vt` (`mesh_utils.py:104-115`).
- **Texture maps** via `cv2.imwrite`, default `image_format='.jpg'` (`mesh_utils.py:72-85`): diffuse -> `<base>.jpg` (RGB->BGR via `[...,::-1]`); metallic -> `<base>_metallic.jpg` (RGB2GRAY); roughness -> `<base>_roughness.jpg` (RGB2GRAY); normal -> `<base>_normal.jpg`. `downsample=True` halves each texture via `cv2.resize` before saving (`MeshRender.py:638-651`).
- **MTL**: PBR material with `map_Kd=<diffuse>.jpg`, `map_Pm=<metallic>.jpg`, `map_Pr=<roughness>.jpg` (`mesh_utils.py:152-176`).

Inpaint fills holes (`texture_inpaint`), then `set_texture` / `set_texture_mr` on the renderer before save (`textureGenPipeline.py:180-184`).

**OBJ -> GLB**, two implementations:
- **Blender** (demo.py `save_glb=True`): `convert_obj_to_glb` imports OBJ with `bpy.ops.wm.obj_import`, applies SMOOTH shading, exports via `bpy.ops.export_scene.gltf` (`mesh_utils.py:260-285`). Heavy bpy dependency.
- **pygltflib** (worker + gradio via `quick_convert_with_obj2gltf` -> `convert_utils.create_glb_with_pbr_materials`): `trimesh.load(obj)` -> temp `.glb` -> `pygltflib.GLTF2().load` -> `combine_metallic_roughness` merges metallic (B channel) + roughness (G channel) + AO=255 (R) into one RGB PNG per the glTF spec (`convert_utils.py:9-39`) -> embeds albedo + metallicRoughness as base64 data URIs (labeled `image/png` even though source is `.jpg`) -> builds `PbrMetallicRoughness` material (`metallicFactor=1`, `roughnessFactor=1`) -> assigns material 0 to all primitives -> `gltf.save` (`convert_utils.py:42-138`).

---

## Notes / parity landmines (carried forward to the MLX port)

- conv_in channels 4:12 carry trained conditioning — do not zero at inference (`modules.py:818-827`).
- `negative_prompt_embeds == prompt_embeds` (learned tokens), NOT zeros (`pipeline.py:267-276`).
- VAE `scaling_factor` read from config (0.18215 default), not hardcoded.
- 3-way CFG (uncond/ref/full) with `ref_scale=[0,1,1]` and `dino=[0,0,dino]` gating — branches differ in conditioning, not just text.
- `view_scale_tensor` is all-ones in production (`camera_azims` never passed) but `cam_mapping` should be preserved for fidelity.
- Dual-stream reference UNet runs once at `timestep_ref=0` (cacheable across 15 steps).
- DINO computed only on the single first reference image (cacheable).
- Rasterizer parity hinges on fp32 math, pixel-center `+0.5`, `(W-1)/(H-1)` scaling, the 18-bit depth quantize, and the INT64 token min-reduce; the CPU twin `rasterize_image_cpu` is bit-identical to the CUDA kernel and is the recommended parity baseline.
- Renderer mesh normalization (axis flip/swap, scale 1.15, UV-v flip) and the 40k-face remesh target directly affect baked-texture coordinates and must match exactly.

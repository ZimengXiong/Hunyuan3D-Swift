# Hunyuan3D Paint Models — Metadata Manifest

Downloaded from the **ModelScope** mirror (huggingface.co is region-blocked).
Endpoint pattern used:
`https://www.modelscope.cn/api/v1/models/{ORG}/{REPO}/repo?Revision=master&FilePath={PATH}`

Weights (`.bin`, large `.safetensors`, `.ckpt`) and tokenizer `vocab.json`/`merges.txt`
were intentionally **skipped** — only config/text metadata was fetched.

All 36 JSON files validated with `json.load` (none are HTML error pages); `.py`/`.yaml`/`.md`
files were scanned for HTML markers and are genuine source. Every downloaded byte-count
matches the size reported by the repo file listing.

---

## 1. Hunyuan3D-2.1 paint-PBR (primary)
Source repo: **Tencent-Hunyuan/Hunyuan3D-2.1**, subtree `hunyuan3d-paintpbr-v2-1/`.
Local root: `metadata/Hunyuan3D-2.1-paintpbr/`

| Local file | Description |
|---|---|
| `model_index.json` | Diffusers pipeline index — `HunyuanPaintPipeline`, components: UNet `UNet2p5DConditionModel` (custom `modules`), VAE `AutoencoderKL`, `CLIPTextModel`, `CLIPVisionModelWithProjection` image encoder, `CLIPTokenizer`, `DDIMScheduler`. |
| `README.md` | Model card (inherited SD2.1-base zero-terminal-SNR / openrail++ card). |
| `unet/config.json` | UNet2DConditionModel base config (in/out=4, cross-attn 1024, blocks 320/640/1280/1280). |
| `unet/model.py` | `HunyuanPaint` LightningModule — training/inference wrapper; `num_view=6`, `pbr_settings=["albedo","mr"]`, v-prediction, VAE-scaled latents. |
| `unet/modules.py` | Core arch — `UNet2p5DConditionModel`, `Basic2p5DTransformerBlock`, `Dino_v2`, `ImageProjModel`, voxel-grid multiview masks. Multiview + reference + material-aware + DINO attention. |
| `unet/attn_processor.py` | Custom attention processors for the 2.5D multiview / PBR / reference attention. |
| `vae/config.json` | AutoencoderKL config (latent_channels=4, blocks 128/256/512/512, sample_size=768). No explicit `scaling_factor` key (falls back to diffusers default 0.18215). |
| `text_encoder/config.json` | CLIPTextModel (SD2 ViT-H text tower): hidden 1024, 23 layers, 16 heads, vocab 49408. |
| `image_encoder/config.json` | CLIPVisionModelWithProjection: hidden 1280, 32 layers, patch 14, image 224, projection_dim 1024. |
| `scheduler/scheduler_config.json` | DDIMScheduler — v_prediction, scaled_linear betas 0.00085→0.012, trailing spacing, zero-SNR rescale=true. |
| `feature_extractor/preprocessor_config.json` | CLIPFeatureExtractor — 224 crop, CLIP mean/std normalize. |
| `tokenizer/tokenizer_config.json` | CLIPTokenizer config (model_max_length 77). |
| `tokenizer/special_tokens_map.json` | CLIP special tokens (`<|startoftext|>` / `<|endoftext|>`). |

### Cross-reference (shape side) — `metadata/Hunyuan3D-2.1-paintpbr/_repo_root/`
| Local file | Description |
|---|---|
| `configuration.json` | Repo-level: `{framework: pytorch, task: image-to-3d, allow_remote: true}`. |
| `README.md` | Top-level Hunyuan3D-2.1 repo README. |
| `hunyuan3d-dit-v2-1.config.yaml` | Shape DiT (`HunYuanDiTPlain`): in_channels 64, hidden 2048, depth 21, 16 heads, MoE (6 layers, 8 experts, top-2), context_dim 1024; ShapeVAE + DINOv2 (dino-large, image 518) conditioner; FlowMatchEulerDiscreteScheduler. |
| `hunyuan3d-vae-v2-1.config.yaml` | Shape `ShapeVAE`: num_latents 4096, embed_dim 64, width 1024, 8 enc / 16 dec layers, scale_factor 1.00395, point_feats 4. |

---

## 2. Hunyuan3D-2.0 paint / delight (probe)
Source repo: **Tencent-Hunyuan/Hunyuan3D-2** (subtrees exist and were adapted — paths confirmed by recursive listing).
Local root: `metadata/Hunyuan3D-2.0-paint/`

Three relevant subtrees found: `hunyuan3d-paint-v2-0/`, `hunyuan3d-paint-v2-0-turbo/`, `hunyuan3d-delight-v2-0/`.

| Local file | Description |
|---|---|
| **hunyuan3d-paint-v2-0/** | |
| `model_index.json` | `StableDiffusionPipeline`; custom `UNet2p5DConditionModel`; **no** image_encoder component (text-conditioned multiview paint). |
| `unet/config.json` | UNet2DConditionModel base (in/out=4, cross-attn 1024, blocks 320/640/1280/1280, sample_size 64). |
| `unet/modules.py` | 2.0 multiview UNet wrapper (`UNet2p5DConditionModel`, no PBR/DINO branches — smaller than 2.1). |
| `vae/config.json` | AutoencoderKL (latent 4, blocks 128/256/512/512, sample_size 768; no explicit scaling_factor). |
| `text_encoder/config.json` | CLIPTextModel (SD2): hidden 1024, 23 layers. |
| `scheduler/scheduler_config.json` | DDIMScheduler — v_prediction, zero-SNR rescale=true, trailing spacing. |
| `feature_extractor/preprocessor_config.json` | CLIPFeatureExtractor (224). |
| `tokenizer/tokenizer_config.json`, `tokenizer/special_tokens_map.json` | CLIPTokenizer config / special tokens. |
| **hunyuan3d-paint-v2-0-turbo/** | |
| `model_index.json` | `StableDiffusionPipeline`; `UNet2p5DConditionModel`; **adds** `CLIPVisionModelWithProjection` image_encoder. |
| `unet/config.json` | Identical base UNet config to v2-0 (in/out 4, cross-attn 1024, sample_size 64). |
| `unet/modules.py` | Turbo 2.5D UNet wrapper (larger than base 2.0; image-conditioned). |
| `image_encoder/config.json` | CLIPVisionModelWithProjection (zero123plus-v1.1 vision tower): hidden 1280, 32 layers, projection_dim 1024. |
| `image_encoder/preprocessor_config.json` | CLIPImageProcessor for the image encoder. |
| `vae/config.json` | AutoencoderKL (same as base). |
| `text_encoder/config.json` | CLIPTextModel (SD2). |
| `scheduler/scheduler_config.json` | DDIMScheduler (v_prediction, zero-SNR). |
| `feature_extractor/preprocessor_config.json`, `tokenizer/*`, `README.md` | CLIP feature extractor, tokenizer config/special tokens, model card. |
| **hunyuan3d-delight-v2-0/** | |
| `model_index.json` | `StableDiffusionInstructPix2PixPipeline` (diffusers 0.30.1) — image-to-image delighting; stock diffusers `UNet2DConditionModel`. |
| `unet/config.json` | UNet2DConditionModel — **in_channels=8** (InstructPix2Pix concat of input-image latents), out 4, cross-attn 1024, **sample_size 96**, upcast_attention true. |
| `vae/config.json` | AutoencoderKL — **explicit `scaling_factor: 0.18215`**, force_upcast true, latent 4. |
| `text_encoder/config.json` | CLIPTextModel — hidden 1024, 23 hidden layers, 16 heads (same SD2 text tower as paint; differs only by `torch_dtype` float16 / transformers 4.45). |
| `scheduler/scheduler_config.json` | DDIMScheduler — v_prediction, leading spacing, **zero-SNR rescale=false**, set_alpha_to_one=false. |
| `feature_extractor/preprocessor_config.json`, `tokenizer/*` | CLIP feature extractor, tokenizer config/special tokens. |

---

## Key config digest

### Hunyuan3D-2.1 paint-PBR (`hunyuan3d-paintpbr-v2-1`)

**Pipeline:** `HunyuanPaintPipeline` (diffusers 0.24.0). Multiview PBR texture generation,
`num_view=6`, `pbr_settings=["albedo","mr"]` (albedo + metallic-roughness). Custom UNet wrapper
`UNet2p5DConditionModel` adds multiview attention (use_ma), reference attention (use_ra,
dual-stream `unet_dual`), material-aware attention (use_mda) and DINOv2 feature integration (use_dino).

**UNet** (`unet/config.json`, base `UNet2DConditionModel`):
- `in_channels` = 4, `out_channels` = 4
- `cross_attention_dim` = 1024
- `block_out_channels` = [320, 640, 1280, 1280]
- `attention_head_dim` = [5, 10, 20, 20]
- `sample_size` = 64
- `layers_per_block` = 2, `norm_num_groups` = 32, `act_fn` = silu, `use_linear_projection` = true
- down: 3×CrossAttnDownBlock2D + DownBlock2D ; up: UpBlock2D + 3×CrossAttnUpBlock2D

**VAE** (`vae/config.json`, `AutoencoderKL`):
- `latent_channels` = 4
- `block_out_channels` = [128, 256, 512, 512]
- `in_channels`/`out_channels` = 3, `sample_size` = 768, `layers_per_block` = 2, `norm_num_groups` = 32
- `scaling_factor` = **not in file** → diffusers default **0.18215** (model.py reads `vae.config.scaling_factor`)

**Text encoder** (`text_encoder/config.json`, `CLIPTextModel`, from `stabilityai/stable-diffusion-2`):
- `hidden_size` = 1024, `intermediate_size` = 4096
- `num_hidden_layers` = 23, `num_attention_heads` = 16
- `projection_dim` = 512, `max_position_embeddings` = 77, `vocab_size` = 49408, model_type `clip_text_model`

**Image encoder** (`image_encoder/config.json`, `CLIPVisionModelWithProjection`):
- `hidden_size` = 1280, `intermediate_size` = 5120
- `num_hidden_layers` = 32, `num_attention_heads` = 16
- `patch_size` = 14, `image_size` = 224, `projection_dim` = 1024, model_type `clip_vision_model`
- Plus a separate **DINOv2** encoder used inside the UNet (dino-large: hidden 1024, 24 layers, patch 14, image 518) per `modules.py` + shape DiT yaml conditioner.

**Scheduler** (`scheduler/scheduler_config.json`, `DDIMScheduler`):
- `prediction_type` = v_prediction
- `beta_schedule` = scaled_linear, `beta_start` 0.00085, `beta_end` 0.012, `num_train_timesteps` 1000
- `timestep_spacing` = trailing, `rescale_betas_zero_snr` = true, `set_alpha_to_one` = true, `clip_sample` = false

---

### Hunyuan3D-2.0 paint (`hunyuan3d-paint-v2-0`)

- **Pipeline:** `StableDiffusionPipeline`; custom `UNet2p5DConditionModel`; **no image encoder** (text-conditioned multiview).
- **UNet:** in/out = 4, cross_attention_dim = 1024, block_out_channels [320,640,1280,1280], attention_head_dim [5,10,20,20], sample_size = 64.
- **VAE:** AutoencoderKL, latent_channels = 4, block_out_channels [128,256,512,512], sample_size 768; no explicit scaling_factor (→ 0.18215).
- **Text encoder:** CLIPTextModel, hidden 1024, 23 layers, 16 heads.
- **Scheduler:** DDIMScheduler, v_prediction, scaled_linear, trailing spacing, zero-SNR rescale = true.

### Hunyuan3D-2.0 paint turbo (`hunyuan3d-paint-v2-0-turbo`)

- Same as v2-0 **plus** a `CLIPVisionModelWithProjection` image encoder (hidden 1280, 32 layers, projection_dim 1024 — zero123plus-v1.1 vision tower). UNet base config identical (in/out 4, cross-attn 1024, sample_size 64); larger `modules.py` (image-conditioned 2.5D wrapper).

### Hunyuan3D-2.0 delight (`hunyuan3d-delight-v2-0`)

- **Pipeline:** `StableDiffusionInstructPix2PixPipeline` (diffusers 0.30.1) — image-space delighting; **stock** diffusers `UNet2DConditionModel` (no 2.5D wrapper).
- **UNet:** `in_channels` = **8** (InstructPix2Pix: noisy latents 4 + conditioning-image latents 4), `out_channels` = 4, cross_attention_dim = 1024, block_out_channels [320,640,1280,1280], attention_head_dim [5,10,20,20], `sample_size` = **96**, upcast_attention = true, transformer_layers_per_block = 1.
- **VAE:** AutoencoderKL with explicit `scaling_factor` = **0.18215**, force_upcast = true, latent_channels = 4, block_out_channels [128,256,512,512], sample_size 768.
- **Text encoder:** CLIPTextModel, hidden 1024, 23 hidden layers, 16 heads (same SD2 tower as paint; float16, transformers 4.45).
- **Scheduler:** DDIMScheduler, v_prediction, scaled_linear, **leading** spacing, `rescale_betas_zero_snr` = **false**, set_alpha_to_one = false.

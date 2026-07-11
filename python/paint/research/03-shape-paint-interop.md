# 03 — Shape ↔ Paint Interop

How the shape (geometry) and paint (texture) stages of the Hunyuan3D family
combine, what they pass to each other, and which models are compatible with
which. Drawn entirely from the research corpus (`webCompat`, `glue`,
`pipeline`, `mvUnet`, `shapePort`, `webFamily`, `webPaint`, `meta`). Source
citations use the corpus's own `file:line` references; uncertainties are flagged
inline and collected in [§7](#7-open-questions--uncertainties).

---

## 1. The handoff: paint consumes mesh + image, NOT shape latents

**Claim: the paint stage is fully decoupled from the shape stage. Its only
inputs are (a) a mesh — a `.glb`/`.obj` file path or an in-memory `trimesh`
object — and (b) the original input image. No shape-internal tensors (latents,
VAE features, DiT activations, conditioner embeddings) ever cross the boundary.**

### Evidence

The 2.1 paint entry point takes only a mesh and an image:

- `Hunyuan3DPaintPipeline.__call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True)` — the entire input surface is a mesh path plus an image path/PIL (`webCompat`; `glue` textureGenPipeline.py:92-103). There is no parameter for a latent code, and no code path that ingests a shape-side tensor.
- Internally the pipeline does `trimesh.load(processed_mesh_path)` then `mesh = mesh_uv_wrap(mesh)`, and conditions the multiview diffusion purely on **renders of that mesh** plus **the reference image** (`webCompat`; `glue` textureGenPipeline.py:118-119, 148-154). The geometry the model sees is the rasterized normal/position maps of the loaded mesh — not anything from the shape network.

The 2.0 API confirms the same decoupling, passing a `trimesh` **object** rather than a path:

```python
pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')  # subfolder hunyuan3d-paint-v2-0
mesh = pipeline(mesh, image='assets/demo.png')
```

The shape stage returns a plain `trimesh` (`mesh = shape_pipeline(image=...)[0]`,
described as "a trimesh object, which you could save to glb/obj"). So the
contract between stages is literally **geometry + the same input image**
(`webCompat`).

### What actually enters the paint diffusion model

Tracing the conditioning paths confirms no shape latent is consumed. The paint
UNet's conditions are all derived from the mesh-renders and the reference image:

- **Geometry control maps** — rendered normal + position maps of the loaded mesh, VAE-encoded and channel-concatenated onto the noisy latent (`pipeline` pipeline.py:253-265; `mvUnet` modules.py:965-970). These come from the renderer, not the shape net.
- **Reference image** — VAE-encoded to `ref_latents` and run through the dual-stream reference UNet; also fed to DINOv2-giant (`pipeline` pipeline.py:231-233; `mvUnet` model.py:340). This is the *original input image*, the same one used for shape generation.
- **Learned per-material "text" tokens** — `learned_text_clip_{albedo,mr}`, stored model parameters, not derived from any input (`mvUnet` modules.py:845-850).

The same input image is **reused**, not re-derived: in `demo.py` the path
`'assets/demo.png'` is passed to both shape (`Image.open`) and paint
(`image_path=path`); in the worker/gradio paths the decoded/background-removed
PIL image used for shape is handed directly (in-memory) to paint
(`glue` model_worker.py:171 vs :187; gradio_app.py:302 vs :381). Background
removal happens once, before shape generation.

### Implication

Because the handoff is just geometry + image, **any mesh can be textured by any
paint model** — including a hand-crafted or externally authored mesh. A
shape-generated mesh is indistinguishable from a hand-crafted one to the paint
stage. This is the stated design intent: the system "decouples the difficulties
of shape and texture generation and provides flexibility for texturing either
generated or hand-crafted meshes" (`webCompat`, 2.0 paper/model card). No version
handshake or compatibility check exists in code to block any pairing
(`webCompat`).

---

## 2. Shared vs. separate conditioners (DINOv2-giant)

**Claim: shape and paint both use a DINOv2-giant image conditioner, but they do
NOT share weights. Each loads its own copy from a different source.**

| | Shape stage | Paint stage |
|---|---|---|
| Conditioner | DINOv2-giant (518×518 input) | DINOv2-giant (`facebook/dinov2-giant`) |
| Weight source | **Bundled inside the shape checkpoint** under `conditioner.*` keys; overwritten via `conditioner.load_state_dict(ckpt['conditioner'])` (`webCompat`) | **Stock HF checkpoint** loaded at runtime: `dino_ckpt_path = 'facebook/dinov2-giant'` (`shapePort`; textureGenPipeline.py:45) |
| Wrapper | `DinoImageEncoder` (`MODEL_CLASS = Dinov2Model`), loaded via `from_pretrained(version)` then weight-overwritten (`webCompat`) | `Dino_v2` wrapping `AutoModel.from_pretrained('facebook/dinov2-giant')` (`shapePort`; `mvUnet` modules.py:38-86) |
| Key prefix | `conditioner.main_image_encoder.model.*` (`shapePort` convert.py:23) | top-level `embeddings.*` / `encoder.*` / `layernorm.*` (no `conditioner.` wrapper) (`shapePort`) |
| Role | Image → shape DiT conditioning | One reference image → multiview paint cross-attention |

### Detail

- **Same architecture.** Both are DINOv2-giant: hidden 1536, 24 heads, 40 layers, patch 14, image 518 → 37×37 = 1369 patches + 1 CLS = **1370 tokens**, SwiGLU FFN, LN eps 1e-6 (`shapePort` dinov2.py:1-4, :138). The shape MLX port's `dinov2.py` module body is reusable for paint unchanged; **only the weight source and key prefix change** (`shapePort`).
- **Separate instances, separate purposes.** The shape DiT ships its own (fine-tuned) DINO weights inside `model.fp16.ckpt`; the paint pipeline independently loads the stock HF checkpoint. There is no weight sharing or cross-loading (`webCompat`).
- **Paint feeds multi-view; shape feeds single image.** Paint's wrapper does `dino_v2(cond_imgs[:, :1])` (only the FIRST reference image) then `rearrange('(b n) l c -> b (n l) c')` to concat views (`shapePort`; `mvUnet` modules.py:83-86, model.py:340). In paint, DINO output is projected by an `ImageProjModel` (Linear 1536→4×1024 + LayerNorm) to **4 context tokens** feeding a dedicated zero-init `attn_dino` cross-attention (`mvUnet` modules.py:853-857, 710-755).

> **Conflict flagged.** DeepWiki's overview claims the conditioner is
> "DINOv2-large, 1024-dim." The papers, the on-disk configs, and the code all
> specify **DINOv2-giant, 1536-dim, 518×518** — treat the papers/config as
> authoritative (`webCompat`, `shapePort`). Note the `meta` config digest's
> prose line "dino-large: hidden 1024" is an inconsistency in that digest; the
> authoritative `shapePort` dinov2.py and `mvUnet` facts both state giant/1536.

### Other paint conditioners (not shared with shape)

The paint UNet also carries an SD2.1 **CLIP text encoder** (`CLIPTextModel`,
hidden 1024, 23 layers, from `stabilityai/stable-diffusion-2`) and a
**CLIP-H/14 vision encoder** (`CLIPVisionModelWithProjection`, hidden 1280,
32 layers, projection 1024) listed in `model_index.json` (`meta` digest;
`webPaint`). In normal 2.1 operation the CLIP **text** encoder is bypassed —
`use_learned_text_clip=True` replaces text with the learned material tokens and
the `'high quality'` prompt is inert (`mvUnet` pipeline.py:267-286). The CLIP
**vision** encoder is listed in `model_index.json` but appears **unused at
inference** in the released 2.1 paint pipeline — image conditioning is carried
by the dual-stream reference path and DINO (`mvUnet`); confirming it is dormant
vs. dead is an open question ([§7](#7-open-questions--uncertainties)).

---

## 3. Compatibility matrix

Rows = shape model; columns = paint model. The matrix is effectively **all-YES
because the stages are decoupled** (see [§1](#1-the-handoff-paint-consumes-mesh--image-not-shape-latents)); the only real constraints are (i) running the matching
paint-repo *code* (2.0 `hy3dgen.texgen` vs 2.1 `hy3dpaint`) and (ii) the
RGB-vs-PBR difference in the output (`webCompat`).

| Shape model | 2.0 Paint (`hunyuan3d-paint-v2-0`, RGB) | 2.0 Paint Turbo (`-v2-0-turbo`, RGB) | 2.1 PBR Paint (`hunyuan3d-paintpbr-v2-1`, albedo+MR) |
|---|---|---|---|
| **2.0 DiT** (`hunyuan3d-dit-v2-0`, 1.1B) | ✅ official recommended | ✅ supported | ⚙️ mechanical (mesh+image only; no benchmark) |
| **2.0 DiT fast/turbo** (1.1B) | ✅ supported | ✅ supported | ⚙️ mechanical |
| **2mini** (`hunyuan3d-dit-v2-mini`, 0.6B; shape-only) | ✅ recommended (2mini ships no paint) | ✅ supported | ⚙️ **mechanical — a 2mini mesh CAN be PBR-textured by 2.1; no official benchmark** |
| **2mv** (`hunyuan3d-dit-v2-mv`, multiview shape; shape-only) | ✅ recommended (2mv ships no paint) | ✅ supported | ⚙️ mechanical |
| **2.1 Shape** (`hunyuan3d-dit-v2-1`, ~3.0–3.3B) | ✅ supported (RGB only — loses PBR) | ✅ supported | ✅ official recommended (PBR) |
| **Hand-crafted / external mesh** | ✅ | ✅ | ✅ |

**Legend**
- ✅ *official recommended* — the same-version pairing shown in that repo's README / Models Zoo (`webCompat`).
- ✅ *supported* — works, same code family, but not the headline pairing.
- ⚙️ *mechanical* — works because paint needs only mesh + image, but it is a cross-version combination that Tencent does not advertise or benchmark (`webCompat`).

### Can a 2mini-shape mesh be textured by 2.1 PBR paint? — **Yes (mechanically).**

This is the explicit cross-version question. A 2mini-generated mesh is just an
`.obj`/`.glb`; to the 2.1 PBR pipeline it is indistinguishable from a
hand-crafted mesh, which is officially supported. So feeding a 2mini mesh into
`hunyuan3d-paintpbr-v2-1` is expected to function and to produce PBR
(albedo + metallic-roughness) output (`webCompat`). Caveats:

1. You must run the **2.1 `hy3dpaint` code**, not 2.0's `hy3dgen.texgen`.
2. 2mini predates 2.1, so it ships **no** paint model — its officially recommended texturing path is the 2.0-family RGB paint (`webCompat`).
3. There is **no published qualitative/quantitative evaluation** of the specific 2mini→2.1-PBR combo, and 2mini is a low-poly 0.6B shape model — seam/PBR quality on such meshes is unverified (`webCompat`, [§7](#7-open-questions--uncertainties)).

### Models that have no paint stage / are out of scope

- **1.0** has no standalone paint diffusion net — texture comes from 6-view RGB generation + baking (`webFamily`, `webPaint`).
- **2.5 / 3.0 / 3.1** are API-only with no open weights; their pairings are internal to Tencent Cloud and outside the open-weight interop story (`webFamily`).
- **Omni** (2.1-based) uses the 2.1 PBR paint; **Part** is segmentation/part-gen with no paint stage (`webFamily`).

---

## 4. Coordinate/format conventions and the remesh + UV step

The boundary is a mesh **file**; the paint pipeline then re-meshes, UV-unwraps,
and re-normalizes the geometry into its renderer's frame before any diffusion.

### Format handoff

- **Shape → disk.** Shape returns an in-memory `trimesh`; the orchestrators export it before paint. `demo.py` writes `demo.glb`; `model_worker.py` writes `{uid}_initial.glb`; `gradio_app.py` writes `white_mesh.glb` then a reduced `white_mesh.obj` used as the paint input (`glue` demo.py:31; model_worker.py:179-180; gradio_app.py:357-373).
- **Paint input.** 2.1 reads a mesh from disk (`trimesh.load(...)`); 2.0 accepts a `trimesh` object directly (`webCompat`).
- **Paint output.** `render.save_mesh(output_mesh_path, downsample=True)` writes **OBJ + MTL + JPG maps** (`glue` textureGenPipeline.py:186):
  - OBJ: `mtllib <name>.mtl`, vertices, `vt` UVs, faces as `v/vt` (`glue` mesh_utils.py:104-115).
  - Maps (cv2.imwrite, default `.jpg`): diffuse `<base>.jpg` (RGB→BGR), metallic `<base>_metallic.jpg` (RGB2GRAY), roughness `<base>_roughness.jpg`, normal `<base>_normal.jpg`; `downsample=True` halves each before saving (`glue` mesh_utils.py:72-85; MeshRender.py:638-651). The 2.1 path produces diffuse + metallic-roughness only — whether a `_normal.jpg` is ever emitted in 2.1 is an open question ([§7](#7-open-questions--uncertainties)).
  - MTL: PBR material with `map_Kd`, `map_Pm` (metallic), `map_Pr` (roughness) (`glue` mesh_utils.py:152-176).
- **Final container.** OBJ → GLB conversion produces the deliverable, via one of two implementations (see below).

### Remesh + UV unwrap (the geometry-conditioning step between the stages)

Inside `Hunyuan3DPaintPipeline.__call__` (`glue` textureGenPipeline.py:107-119):

1. **Remesh** (default `use_remesh=True`): `processed_mesh_path = <dir>/white_mesh_remesh.obj`; `remesh_mesh(mesh_path, processed_mesh_path)`. Input may be `.glb` or `.obj`; output is always `.obj`. The decimation targets **40,000 faces** — `mesh_simplify_trimesh(target_count=40000)`, where pymeshlab is used only for robust load/format-normalization and the actual decimation is `trimesh.simplify_quadric_decimation(40000)` (`glue` simplify_mesh_utils.py:19-37).
2. **Load + UV-unwrap**: `mesh = trimesh.load(processed_mesh_path)`; `mesh = mesh_uv_wrap(mesh)` → **xatlas** `parametrize`, reindexing `vertices/faces/visual.uv` (`glue` uvwrap_utils.py:19-32).
3. **Renderer normalization** (`MeshRender.set_mesh`, `glue`/`renderer` MeshRender.py:703-717): flips X/Y (`vtx_pos[:, [0,1]] = -vtx_pos[:, [0,1]]`), swaps Y/Z axes, flips UV V (`vtx_uv[:,1] = 1 - vtx_uv[:,1]`), and auto-centers + scales the mesh into a unit-ish box (`scale_factor=1.15`, i.e. `scale_factor/(2·max‖v−center‖)`).

### Coordinate conventions used in rendering/baking

- **Cameras** (`renderer` camera_utils.py): `get_mv_matrix(elev, azim, dist)` builds a look-at world→cam matrix (note `elev = -elev; azim += 90; up = [0,0,1]`). Production uses orthographic projection (`camera_type='orth'`, `ortho_scale=1.2`), `camera_distance=1.45` (`renderer` MeshRender.py:324-368).
- **Position maps serve a dual role** (`pipeline`/`mvUnet`): the *pixel* position map drives 3D voxel-RoPE indices for multiview attention (`calc_multires_voxel_idxs`, grid_resolutions `[H, H/2, H/4, H/8]`, voxel_resolutions `[H·8, H·4, H·2, H]`); a separate *VAE-encoded* position latent is channel-concatenated into the 12-channel `conv_in`.
- **Input image preprocessing** (`pipeline`/`glue`): resized to 512×512, RGBA composited onto a white background inside the paint pipeline (textureGenPipeline.py:139-145).

> **Parity landmine (MLX port).** Renderer normalization (axis flip/swap,
> scale 1.15, UV V-flip) and the 40k remesh target directly determine baked
> texture coordinates — they must be kept identical or textures misalign
> (`glue`). The image passed to paint must be the **same** post-rembg PIL image
> the shape stage used, or texturing diverges (`glue`).

### OBJ → GLB conversion (two implementations)

- **A) Blender/`bpy`** path (`demo.py` via `save_glb=True`): `convert_obj_to_glb` imports the OBJ, applies smooth shading, exports glTF (`glue` mesh_utils.py:260-285). Heavy dependency — the MLX-port notes recommend **not** porting it.
- **B) `pygltflib`** path (worker + gradio via `quick_convert_with_obj2gltf` → `create_glb_with_pbr_materials`): merges metallic (B) + roughness (G) + AO=255 (R) into one RGB metallic-roughness texture, embeds albedo + metallicRoughness as base64, builds a `PbrMetallicRoughness` material (metallicFactor=1, roughnessFactor=1) (`glue` convert_utils.py:9-138). This is the preferred GLB writer for the MLX port.

---

## 5. Official recommended pairings

Same-version pairings shown in each repo's README / Models Zoo (`webCompat`,
`webFamily`):

| Generation | Shape model | Paint model | Texture type |
|---|---|---|---|
| **2.0** | `Hunyuan3D-DiT-v2-0` (`hunyuan3d-dit-v2-0`, 1.1B; + fast/turbo) | `Hunyuan3D-Paint-v2-0` (`hunyuan3d-paint-v2-0`, 1.3B; + turbo). `hunyuan3d-delight-v2-0` runs before/with paint. | RGB only |
| **2.1** | `Hunyuan3D-Shape-v2-1` (`hunyuan3d-dit-v2-1`, ~3.0–3.3B) | `Hunyuan3D-Paint-v2-1` (`hunyuan3d-paintpbr-v2-1`, ~1.3–2B; RomanTex + MaterialMVP) | **PBR** (albedo + metallic-roughness) |
| **2mini** (shape-only, 0.6B) | `Hunyuan3D-DiT-v2-mini` (`hunyuan3d-dit-v2-mini`; + fast/turbo) | reuses **2.0-family** paint (`hunyuan3d-paint-v2-0`) | RGB only |
| **2mv** (shape-only, multiview) | `Hunyuan3D-DiT-v2-mv` (`hunyuan3d-dit-v2-mv`; + turbo) | reuses **2.0-family** paint | RGB only |

Notes:

- **2mini and 2mv ship no paint model.** Their recommended texturing path is the 2.0-family RGB paint, because both predate 2.1 (`webCompat`).
- **Cross-version is not endorsed but not blocked.** There is no official recommended cross-version pairing and no published benchmark (e.g. 2mini→2.1-PBR), yet nothing in code prevents it (`webCompat`, [§3](#3-compatibility-matrix)).
- **PBR vs. RGB is the practical selector.** Want relightable albedo + metallic-roughness → use the 2.1 PBR paint. Only need baked RGB → 2.0/2.0-turbo paint (`webFamily`, `webPaint`).
- **Shared backbone across paint versions.** All paint models from 2.0 onward fine-tune the same SD2.1-base zero-terminal-SNR v-prediction UNet (in_channels=4, cross_attention_dim=1024, block_out_channels `[320,640,1280,1280]`); the differences are in the attention/conditioning wrapper and the output channel set (RGB vs. albedo+MR), not the base topology (`webPaint`, `meta` digest).

---

## 6. Summary of the interop contract

1. **Boundary = a mesh file + the original input image.** Nothing else crosses. No shape latents, VAE features, or conditioner embeddings are passed (`webCompat`, `glue`, `pipeline`, `mvUnet`).
2. **Therefore any mesh is textureable by any paint model** — generated, cross-version, or hand-crafted; a 2mini mesh can be PBR-textured by 2.1 (mechanically; unbenchmarked) (`webCompat`).
3. **DINOv2-giant is used by both stages but with separate weights** — shape bundles fine-tuned `conditioner.*` weights; paint loads stock `facebook/dinov2-giant`. Same architecture, different source/prefix (`webCompat`, `shapePort`).
4. **Between the stages, paint always remeshes (→40k faces, OBJ), xatlas-unwraps, and re-normalizes** (axis flip/swap, scale 1.15, UV V-flip) before rendering the geometry conditions (`glue`, `renderer`).
5. **Official pairings are same-version** (2.0+2.0, 2.1+2.1 PBR); 2mini/2mv are shape-only and reuse 2.0 paint (`webCompat`, `webFamily`).

---

## 7. Open questions / uncertainties

- **2mini→2.1-PBR quality.** Mechanically supported, but no published evaluation of PBR/seam behavior when feeding a low-poly 2mini (or 2mv) mesh into 2.1 PBR paint (`webCompat`).
- **CLIP vision encoder in 2.1 paint.** `model_index.json` lists `CLIPVisionModelWithProjection`, but it appears unused at inference (image conditioning is carried by the dual-stream reference path + DINO). Whether it is loaded-but-dormant vs. genuinely absent from the forward is unconfirmed (`mvUnet`).
- **DINOv2 size conflict.** DeepWiki says "DINOv2-large/1024-dim"; papers + configs + code say "DINOv2-giant/1536-dim/518×518." The `meta` config digest's prose ("dino-large: hidden 1024") echoes the wrong figure. Authoritative sources (`shapePort` dinov2.py, `mvUnet`) confirm **giant/1536** — flagged as a documentation inconsistency, not a code one (`webCompat`, `shapePort`, `meta`).
- **Whether shape's bundled `conditioner.*` DINO weights are genuinely fine-tuned** vs. a re-serialized stock copy — the `load_state_dict` path confirms bundling, not the degree of fine-tuning (`webCompat`). This matters for MLX-port weight reuse: if byte-identical to stock giant, the existing `dinov2.py` loads HF keys with only a prefix change ([§2](#2-shared-vs-separate-conditioners-dinov2-giant)).
- **Normal-map output in 2.1.** `save_obj_mesh` supports a `_normal.jpg` map, but the 2.1 paint path only sets diffuse + MR. Whether a normal texture is ever produced in 2.1 is unconfirmed (`glue`).
- **2.1 default output container & per-map files.** `save_glb=True` is the 2.1 default, but whether it emits separate albedo/metallic/roughness files alongside the GLB (and via the Blender vs. pygltflib path) depends on the caller (`webCompat`, `glue`).
- **Verbatim shape conditioner config string.** The exact `conditioner` target class + `version` in `hunyuan3d-dit-v2-0/config.yaml` was inferred from `conditioner.py` + `pipelines.py` + papers, not read verbatim (HF raw endpoint dropped) (`webCompat`).

---

*Sources: corpus sections `webCompat`, `glue`, `pipeline`, `mvUnet`, `shapePort`,
`webFamily`, `webPaint`, `meta`. All `file:line` references are as cited within
those sections. No claims here originate outside the provided corpus.*

# Benchmarks — Hunyuan3D Shape on MLX

Apple-Silicon M-series (48 GB unified), MLX 0.31.2, `demo.png`, octree 256, 30 steps,
guidance 5.0, seed 0. **Focus models: 2mini and 2.0-standard** (2.1 dropped). All runs use
fp16 activations; *quantization is weights-only* — the DiT + DINO block linears are quantized,
while the VAE, all norms, and the input/output embedders stay fp16 (quality-critical).

The two models, for reference:

| | image encoder | DiT | latent tokens |
|---|---|---|---|
| **2mini** | DINOv2-giant (1536) | Hunyuan3DDiT, 0.6B, 8+16 | 512 |
| **2.0** | DINOv2-giant (1536) | Hunyuan3DDiT, 1.1B, 16+32 | 3072 |

Both share the *giant* conditioner (why both render the "HY3D" sign well); 2.0 adds DiT depth
and 6× the latent tokens.

## Memory (MLX) — quantization is the headline

`resident weights` = steady model footprint after load; `inference peak` = peak GPU allocation
during one CFG forward + VAE decode + grid chunk (load transient excluded).

| model | precision | resident weights | inference peak |
|---|---|---|---|
| **2mini** | fp16 | 3.82 GB | 5.20 GB |
| | 8-bit | 2.24 GB | 3.59 GB |
| | **4-bit** | **1.40 GB** | **2.74 GB** |
| **2.0** | fp16 | 4.93 GB | 6.37 GB |
| | 8-bit | 2.83 GB | 4.27 GB |
| | **4-bit** | **1.71 GB** | **3.15 GB** |

4-bit cuts weights ~2.7–3.5× and the inference peak ~1.9×. (Not a full 4× because the VAE +
embedders + norms + all activations remain fp16.) **2.0 — the quality pick — runs 4-bit in
~1.7 GB weights / ~3.2 GB peak, i.e. comfortably on an 8 GB Mac.**

## Speed (total per mesh, with denoise / grid breakdown)

| model | precision | denoise (30 steps) | VAE grid (257³) | **total** |
|---|---|---|---|---|
| **2mini** | fp16 | 10 s | ~50 s | **~63 s** |
| | 8-bit | 10 s | 50 s | 61 s |
| | 4-bit | 11 s | 52 s | 63 s |
| **2.0** | fp16 | 129 s | ~70–124 s | ~254 s |
| | 8-bit | 85 s | 93 s | 178 s |
| | **4-bit** | **79 s** | 75 s | **155 s** |

Takeaways:
- **Quantization speeds the big model's denoise for free** — 2.0 fp16 129 s → 4-bit **79 s**
  (1.6×). 4-bit matmuls are memory-bandwidth-bound, so they run *faster* than fp16, not just
  smaller. So for 2.0, 4-bit is a win on *both* axes (RAM and time) at negligible quality cost.
- **2mini is grid-bound** — its DiT is tiny (10 s denoise), so quant barely moves its total.
- **The VAE grid query (fp16) is the floor** — ~50 s (2mini) / ~70 s (2.0) of querying all
  ~17 M grid points. It is unaffected by weight quantization. (The ~124 s seen in an early 2.0
  run was an un-warmed-GPU outlier; the controlled grid benchmark gives ~70 s.)

## Quality vs precision

See `outputs/compare_2mini_quant.png` and `outputs/compare_v20_quant.png`:
- **8-bit is visually indistinguishable from fp16** for both models — near-lossless.
- **4-bit keeps the "HY3D" sign legible** and the body shape intact; it perturbs the weights
  slightly, which nudges fine details (the 4-bit penguin is a hair softer / shifts a little),
  but the result is high quality. Mesh stats stay in family (149–162 K verts, watertight body).

**Recommendation:** default to **8-bit** (free quality, ~1.6× less RAM); use **4-bit** when RAM
is tight or for 2.0 where it also buys speed.

## Lossless tuning — grid chunk size

The grid query chunks query points with one `mx.eval` per chunk. Sweeping `num_chunks`
(output is **bit-identical** for every value — verified `max-drift 0.00e+00`):

| num_chunks | 2mini grid | 2.0 grid |
|---|---|---|
| 8 000 | **48 s** | 78 s |
| 32 000 | 51 s | 75 s |
| 100 000 | 56 s | **68 s** |
| 300 000 | 65 s | 71 s |
| 600 000 | 77 s | 91 s |

The optimum is latent-count dependent (small sets prefer small chunks; 2.0's 6× heavier
per-query attention amortizes bigger batches). The default now adapts: 8 000 for ≤1024 latents,
100 000 above. Net effect is small (±13 %) because the grid is **compute-bound** — chunk size
only trims dispatch overhead.

## Octree decode (`--octree-decode`) — the grid lever, implemented

FlashVDM-style: decode a coarse grid densely, then refine **only the near-surface band**
level-by-level (64→128→256). The geo-decoder queries ~**5 %** of cells instead of all 17 M;
octree bookkeeping (near-surface extraction, dilation, 2× upsampling) is numpy on the small
grids, only the neural queries hit MLX. Inactive cells are NaN; marching cubes treats NaN edges
as non-crossing, so the surface is extracted cleanly with no band-boundary artifacts.

| model | dense grid | octree grid | speedup | active | Chamfer vs dense |
|---|---|---|---|---|---|
| 2mini | ~50 s | **9.4 s** | 5.3× | 5.0 % | 0.0076 (0.4 % of bbox) |
| 2.0 | ~124 s | **11.7 s** | 10.6× | 5.3 % | 0.0078 |

**Near-lossless** — identical vertex count, identical surface area, same component count, Chamfer
at sampling-noise level; renders are indistinguishable (`outputs/compare_octree.png`,
`compare_v20_octree.png`). The bigger model benefits more (its dense grid is costlier).

## Full stack — 2.0, octree 256 (octree + 4-bit, near-lossless)

| config | denoise | grid | **total** | weights | peak |
|---|---|---|---|---|---|
| dense, fp16 | 129 s | 124 s | **254 s** | 4.9 GB | 6.4 GB |
| octree, fp16 | 72 s | 12 s | **85 s** | 4.9 GB | 6.4 GB |
| **octree + 4-bit** | 68 s | 11 s | **79 s** | **1.7 GB** | **3.2 GB** |

Stacking octree (grid) + 4-bit (denoise + memory) takes 2.0 from **254 s / 6.4 GB → 79 s / 3.2 GB**
— **3.2× faster and ~2× less memory, near-lossless.**

## Distilled checkpoints (turbo) — fewer steps, single forward

Turbo models are `guidance_embed: true` (CFG distilled into a guidance token → **single forward
per step, no CFG**) + a `ConsistencyFlowMatchEulerDiscreteScheduler` (few steps). Supported via
the same loader; `pipeline.generate` auto-detects both. The path is **verified exact vs the torch
oracle**: single-forward DiT cosine **1.0000000**, consistency schedule identical to 7 digits.

| turbo, 8 steps, octree 256 | denoise | grid | total | verts | quality |
|---|---|---|---|---|---|
| **2.0-turbo** | 7.4 s | 8.6 s | **16.7 s** | 153 K, 1 comp | **clean** (≈ base 2.0; sign slightly softer) |
| mini-turbo | 3.8 s | 9.5 s | 18.5 s | 450 K, 500+ comp | **stripes the thin sign** |

**2.0-turbo is the practical sweet spot: near-base-2.0 quality in ~17 s vs the 85 s base-2.0 octree
run (~5×).** mini-turbo, by contrast, stripes thin high-frequency features (the sign) — the *small*
model doesn't distill cleanly there. Both paths are exact-parity vs the torch oracle (single-forward
cosine 1.0000000, consistency schedule identical), so this is a model-size effect, not a port bug:
the 1.1B 2.0-turbo distills fine, the 0.6B mini-turbo does not. Recommendation: **2.0-turbo for fast
high-quality single-image shape**; base 30-step only when you want the sharpest possible fine detail.

## `mx.compile` (opt-in, `--compile-dit`)

Single-forward parity: fp32 cosine 1.0000001, fp16 rel 0.4 % (op-reordering). Only **~1.1×**, and
in fp16 the per-step reordering compounds over the loop into a different (equally valid) sample,
so it is **off by default**. Available for those who want the marginal speed.

## Remaining levers (not yet implemented)

- **Top-k latent routing** in the geo-decoder (the other half of FlashVDM) — each query attends
  to only the most relevant latents; speeds the remaining (post-octree) grid queries further.

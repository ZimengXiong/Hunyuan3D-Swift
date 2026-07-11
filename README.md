# hunyuan3d-swift
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/534826fa-0a79-45f0-a5af-8c69a49e1fe9" />

swift and python mlx ports of the hunyuan3d shape and paint pipelines.

the repo has two parts:

- swift package at the root
- python-mlx ports under `python/`

the swift code is checked against python fixtures. the python ports are checked against the original pytorch code. this port was done by a team of agents (one spawned per pytorch vertical) with access to both an CUDA as well as a MLX compatiable GPU to ensure parity. total run time ~10hrs across a dozen agents.

## why i'm excited
| run | config | wall time | peak memory |
|---|---|---:|---:|
| `hy3d shape` (small) | 30-step cfg, octree 256 | 20.9 s | ~5.6 gb |
| `hy3d shape` (large) | 8-step turbo, octree 256 | 22.3 s | ~7.3 gb |
| `hy3d paint` (rgb) | 512 render, 15 steps, +sr | 231 s | ~38 gb |
| `hy3d paint` (pbr) | 512 render, 15 steps, 4096 atlas, +sr | 344 s | ~39 gb |
| `hy3d generate` (small / rgb) | chained | 240 s | ~25 gb |
| `hy3d generate` (large / pbr) | chained | 360 s | ~33 gb |

that's hunyuan3D-shape running in FIVE POINT SIX gigabytes of RAM. with a Q8 or Q4, we are easily in mobile territory. now, what would you do with a image to 3d model running on your phone (iPhone 15 onwards)? not sure, but that's really really cool, the first time in history it has been possible (AFAIK). here's a demo:

## package

main products:

- Hy3DMLX for shape generation
- HunyuanPaintMLX for texture generation
- `hy3d` for the command line

main commands:

```bash
swift build -c release
swift run -c release hy3d --help
```

## weights

download model weights into local folders:

```bash
hf download zimengxiong/hunyuan3d-mlx-shape-small --local-dir weights/shape-small
hf download zimengxiong/hunyuan3d-mlx-paint-large --local-dir weights/paint-large
```

four supported slots:

- shape small: `hunyuan3d-dit-v2-mini`
- shape large: `hunyuan3d-dit-v2-0-turbo`
- paint small: `hunyuan3d-paint-v2-0`
- paint large: `hunyuan3d-paintpbr-v2-1`

## run

shape plus paint:

```bash
swift run -c release hy3d generate photo.png -o out.glb \
  --shape-weights weights/shape-small \
  --paint-weights weights/paint-large
```

shape only:

```bash
swift run -c release hy3d shape photo.png -o mesh.glb \
  --weights weights/shape-small
```

paint an existing mesh:

```bash
swift run -c release hy3d paint mesh.glb photo.png -o textured.glb \
  --weights weights/paint-large --model pbr
```

## license

swift source is mit. dependencies, model weights, and algorithm ports keep their own licenses.
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

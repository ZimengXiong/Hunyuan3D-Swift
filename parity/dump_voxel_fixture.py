"""Dump the PoseRoPE voxel-index fixture: a realistic 512x512 position map (exact-1.0
background + near-1 fp16 edge cases) and the reference fp16 voxel indices for all four
RoPE levels used by the PBR pipeline (grid_res [64,32,16,8] / voxel_res [512,256,128,64]).

Keys: pos [1,2,512,512,3] f32; vox64/vox32/vox16/vox8 [1, 2*g*g, 3] i32.

Run from the Hunyuan-3D-Paint-MLX repo root:
    PYTHONPATH=. FIXTURES_OUT=/path/to/fixtures uv run python .../dump_voxel_fixture.py
"""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from hy3dpaint_mlx.unet2p5d_pbr import compute_voxel_indices

FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
rng = np.random.RandomState(11)
B, N, H = 1, 2, 512
pos = rng.rand(B, N, H, H, 3).astype(np.float32)
# background: random axis-aligned rectangles of exact 1.0 (reference posmaps use 1.0 as bg)
for _ in range(40):
    b, n = rng.randint(B), rng.randint(N)
    r0, c0 = rng.randint(0, H - 8, 2)
    rh, cw = rng.randint(4, 96, 2)
    pos[b, n, r0:r0 + rh, c0:c0 + cw] = 1.0
# near-1 texels that round to fp16 1.0 (exercise the fp16 `!= 1` validity edge)
edge = rng.rand(B, N, H, H) < 0.01
pos[edge] = rng.uniform(0.9994, 1.0, (int(edge.sum()), 3)).astype(np.float32)

dump = {"pos": mx.array(pos)}
for g, vr in [(64, 512), (32, 256), (16, 128), (8, 64)]:
    vox = compute_voxel_indices(pos, g, vr)
    dump[f"vox{g}"] = mx.array(vox.astype(np.int32))
    print(f"grid {g} voxel {vr}: vox {vox.shape} range {vox.min()}..{vox.max()}")
mx.save_safetensors(f"{FIX}/voxel_fixture.safetensors", dump)
print("voxel fixture written")

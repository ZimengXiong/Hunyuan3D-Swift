"""Dump bake_multi + inpaint for the Swift bake gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np, mlx.core as mx, trimesh, xatlas
from hy3dpaint_mlx.mesh_render import MeshRender

MESH = os.environ.get("PARITY_MESH") or sys.exit(
    "set PARITY_MESH=/path/to/mesh.glb (the reference runs used the Hunyuan3D-2.1 hy3dpaint case_1 asset)")
AZIMS = [0, 90, 180, 270, 0, 180]; ELEVS = [0, 0, 0, 0, 90, -90]; VW = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
mesh = trimesh.load(MESH, force="mesh")
vm, idx, uv = xatlas.parametrize(mesh.vertices, mesh.faces)
V = np.asarray(mesh.vertices)[vm].astype(np.float32); F = idx.astype(np.int64)
R = MeshRender(); R.load_mesh(V, F); R.set_uv(uv, F)
rng = np.random.RandomState(2)
VRES, T = 256, 256
views = [rng.rand(VRES, VRES, 3).astype(np.float32) for _ in range(6)]
texs, covered = R.bake_multi([views], ELEVS, AZIMS, texture_size=T, weights=VW)
inp = R.inpaint(texs[0], covered)
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/bake_fixture.safetensors", {
    "V": mx.array(V.reshape(-1)), "F": mx.array(F.reshape(-1).astype(np.int32)),
    "uv": mx.array(uv.reshape(-1).astype(np.float32)),
    "views": mx.array(np.stack(views)), "tex": mx.array(texs[0]), "covered": mx.array(covered.astype(np.int32))})
print("tex", texs[0].shape, "covered", int(covered.sum()), "/", T*T)

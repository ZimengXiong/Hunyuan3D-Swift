"""Dump MeshRender control maps + uv_rasterize for the Swift renderer gate.
Feeds Swift the EXACT xatlas V/F (isolates renderer parity from xatlas version)."""
import sys, os; sys.path.insert(0, ".")
import numpy as np, mlx.core as mx, trimesh, xatlas
from hy3dpaint_mlx.mesh_render import MeshRender

MESH = os.environ.get("PARITY_MESH") or sys.exit(
    "set PARITY_MESH=/path/to/mesh.glb (the reference runs used the Hunyuan3D-2.1 hy3dpaint case_1 asset)")
mesh = trimesh.load(MESH, force="mesh")
vm, idx, uv = xatlas.parametrize(mesh.vertices, mesh.faces)
V = np.asarray(mesh.vertices)[vm].astype(np.float32); F = idx.astype(np.int64)
R = MeshRender(); R.load_mesh(V, F); R.set_uv(uv, F)
normal, position = R.render_control(0, 30, 256)
tex_pos, tex_nrm, mask = R.uv_rasterize(512)
FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/render_fixture.safetensors", {
    "V": mx.array(V.reshape(-1)), "F": mx.array(F.reshape(-1).astype(np.int32)),
    "uv": mx.array(uv.reshape(-1).astype(np.float32)),
    "normal": mx.array(np.asarray(normal, np.float32)), "position": mx.array(np.asarray(position, np.float32)),
    "tex_pos": mx.array(np.asarray(tex_pos, np.float32)), "mask": mx.array(mask.astype(np.int32))})
print("V", V.shape, "F", F.shape, "| normal", normal.shape, "tex_pos covered", int(mask.sum()))

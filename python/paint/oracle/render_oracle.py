"""Run the REFERENCE MeshRender (raster_mode='cr', backed by our bit-exact cr CPU oracle) and
dump its normal/position control maps, to gate our MLX renderer against it. Run in .venv-oracle.

The reference custom_rasterizer is replaced by a thin shim over oracle/cr_cpu (== CUDA algorithm),
and cv2 / meshVerticeInpaint are stubbed (unused by render_normal/render_position).
"""

import os
import sys
import types
import importlib.util

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DR = os.path.join(ROOT, "reference", "Hunyuan3D-MLX", "hy3dgen", "texgen", "differentiable_renderer")


def _mvn(vertex_count, faces, face_normals, **kw):  # numpy-2.x-safe standard mean vertex normals
    f = np.asarray(faces); fn = np.asarray(face_normals)
    vn = np.zeros((vertex_count, 3), np.float64)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)


trimesh.geometry.mean_vertex_normals = _mvn
sys.path.insert(0, os.path.join(ROOT, "oracle", "cr_cpu"))
import cr_cpu  # ctypes loader for the CUDA-equivalent rasterizer


# --- custom_rasterizer shim (reference API) over the cr oracle ---
def _rasterize(pos, tri, resolution):
    p = pos[0] if pos.dim() == 3 else pos
    V = np.ascontiguousarray(p.detach().cpu().numpy(), np.float32)
    F = np.ascontiguousarray(tri.detach().cpu().numpy(), np.int32)
    fi, ba = cr_cpu.rasterize_image(V, F, int(resolution[1]), int(resolution[0]))
    return torch.from_numpy(fi.copy()), torch.from_numpy(ba.copy())


def _interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    return torch.sum(result, axis=-2).view(1, *result.shape[:-2], result.shape[-1])


cr = types.ModuleType("custom_rasterizer")
cr.rasterize = _rasterize
cr.interpolate = _interpolate
sys.modules["custom_rasterizer"] = cr
sys.modules["cv2"] = types.ModuleType("cv2")

# load reference mesh_render as a package 'dr' with stubbed siblings
pkg = types.ModuleType("dr"); pkg.__path__ = [DR]; sys.modules["dr"] = pkg
for name in ["camera_utils", "mesh_utils"]:
    spec = importlib.util.spec_from_file_location(f"dr.{name}", os.path.join(DR, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[f"dr.{name}"] = m; spec.loader.exec_module(m)
mp = types.ModuleType("dr.mesh_processor"); mp.meshVerticeInpaint = lambda *a, **k: (a[0], a[1])
sys.modules["dr.mesh_processor"] = mp
spec = importlib.util.spec_from_file_location("dr.mesh_render", os.path.join(DR, "mesh_render.py"))
mr = importlib.util.module_from_spec(spec); sys.modules["dr.mesh_render"] = mr; spec.loader.exec_module(mr)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.load(os.environ.get("PAINT_MESH",
        os.path.join(ROOT, "reference", "Hunyuan3D-2.1", "hy3dpaint", "assets", "case_1", "mesh.glb")), force="mesh")
    R = int(os.environ.get("PAINT_RES", "384"))
    rend = mr.MeshRender(camera_distance=1.45, default_resolution=R, texture_size=(1024, 1024),
                         use_antialias=False, raster_mode="cr", device="cpu")
    rend.load_mesh(mesh)
    views = [(0, 0), (0, 90), (90, 0)]
    for i, (e, a) in enumerate(views):
        nm = rend.render_normal(e, a, use_abs_coor=True, return_type="np")
        pm = rend.render_position(e, a, return_type="np")
        np.save(os.path.join(out_dir, f"normal_{i}.npy"), np.asarray(nm))
        np.save(os.path.join(out_dir, f"position_{i}.npy"), np.asarray(pm))
    np.save(os.path.join(out_dir, "views.npy"), np.array(views))
    print("OK ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1])

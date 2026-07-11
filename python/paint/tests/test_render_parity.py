"""Parity gate: MLX renderer control maps vs the reference MeshRender (raster_mode='cr',
backed by the bit-exact cr CPU oracle). Validates camera + normal + interpolation logic."""

import os
import sys
import subprocess

import numpy as np
import trimesh
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hy3dpaint_mlx.mesh_render import MeshRender

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_DUMP = os.path.join(ROOT, ".parity_dumps", "render")
_MESH = os.environ.get("PAINT_MESH",
    os.path.join(ROOT, "reference", "Hunyuan3D-2.1", "hy3dpaint", "assets", "case_1", "mesh.glb"))

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_ORACLE_PY) and os.path.exists(_MESH)),
    reason="needs oracle venv + a test mesh",
)


def _psnr(a, b):
    m = float(np.mean((a - b) ** 2)); return 99.0 if m == 0 else 10 * np.log10(1.0 / m)


def _cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@pytest.fixture(scope="module")
def setup():
    env = dict(os.environ, PAINT_MESH=_MESH, PAINT_RES="384")
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "render_oracle.py"), _DUMP],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.fail(f"render oracle failed:\n{r.stdout}\n{r.stderr}")
    mesh = trimesh.load(_MESH, force="mesh")
    rend = MeshRender(); rend.load_mesh(mesh.vertices, mesh.faces)
    return rend, np.load(os.path.join(_DUMP, "views.npy"))


def test_control_maps(setup):
    rend, views = setup
    for i, (e, a) in enumerate(views):
        nref = np.load(os.path.join(_DUMP, f"normal_{i}.npy"))
        pref = np.load(os.path.join(_DUMP, f"position_{i}.npy"))
        nm = rend.render_normal(int(e), int(a), 384)
        pm = rend.render_position(int(e), int(a), 384)
        assert _cos(nm, nref) > 0.9999 and _psnr(nm, nref) > 40, f"normal view {i}"
        assert _cos(pm, pref) > 0.9999 and _psnr(pm, pref) > 40, f"position view {i}"

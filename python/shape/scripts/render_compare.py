"""Render front + 3/4 views of several meshes side by side at matched density.
  uv run python scripts/render_compare.py out.png "label=path.glb" "label=path.glb" ...
"""
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = sys.argv[1]
items = [a.split("=", 1) for a in sys.argv[2:]]
views = [("front", (90, -90)), ("3/4", (20, -60))]

fig = plt.figure(figsize=(4.0 * len(items), 4.0 * len(views)))
for col, (label, path) in enumerate(items):
    m = trimesh.load(path)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    V, F = m.vertices, m.faces
    tris = V[F]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    sh = np.clip(n @ np.array([0.3, 0.5, 0.8]) * 0.5 + 0.5, 0.15, 1.0)
    col_rgb = np.stack([sh * 0.55, sh * 0.68, sh], axis=1)
    for row, (vname, (el, az)) in enumerate(views):
        ax = fig.add_subplot(len(views), len(items), row * len(items) + col + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=col_rgb, edgecolors="none"))
        ax.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax.set_ylim(V[:, 1].min(), V[:, 1].max())
        ax.set_zlim(V[:, 2].min(), V[:, 2].max())
        ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
        ax.view_init(el, az)
        ax.set_axis_off()
        if row == 0:
            ax.set_title(f"{label}\n{len(V)//1000}K verts", fontsize=10)
plt.tight_layout()
plt.savefig(OUT, dpi=95, bbox_inches="tight")
print("wrote", OUT)

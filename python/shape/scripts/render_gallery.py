"""Gallery: input image + front + 3/4 mesh views per row.
  uv run python scripts/render_gallery.py out.png "img.png=mesh.glb" ...
"""
import sys

import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT = sys.argv[1]
rows = [a.split("=", 1) for a in sys.argv[2:]]
views = [("front", (90, -90)), ("3/4", (20, -60))]
fig = plt.figure(figsize=(3.6 * (1 + len(views)), 3.6 * len(rows)))

for r, (img_path, mesh_path) in enumerate(rows):
    ax = fig.add_subplot(len(rows), 1 + len(views), r * (1 + len(views)) + 1)
    ax.imshow(Image.open(img_path).convert("RGBA"))
    ax.set_axis_off()
    if r == 0:
        ax.set_title("input", fontsize=11)
    m = trimesh.load(mesh_path)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    V, F = m.vertices, m.faces
    tris = V[F]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    sh = np.clip(n @ np.array([0.3, 0.5, 0.8]) * 0.5 + 0.5, 0.15, 1.0)
    col = np.stack([sh * 0.55, sh * 0.68, sh], axis=1)
    for c, (vname, (el, az)) in enumerate(views):
        ax = fig.add_subplot(len(rows), 1 + len(views), r * (1 + len(views)) + 2 + c,
                             projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="none"))
        ax.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax.set_ylim(V[:, 1].min(), V[:, 1].max())
        ax.set_zlim(V[:, 2].min(), V[:, 2].max())
        ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
        ax.view_init(el, az)
        ax.set_axis_off()
        if r == 0:
            ax.set_title(f"mesh {vname}  ({len(V)//1000}K)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT, dpi=92, bbox_inches="tight")
print("wrote", OUT)

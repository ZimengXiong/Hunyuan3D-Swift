"""Dump the inpaint parity fixture: a real baked texture + coverage mask (from bake_fixture)
with the Python reference fill.

Keys:
  texture     [T,T,3] f32  baked texture (holes = zeros where uncovered)
  covered     [T,T]   i32  coverage mask (1 = baked texel)
  filled      [T,T,3] f32  full Python reference: clip -> EDT nearest-fill -> uint8 round-trip
                           with cv2.INPAINT_NS on the holes (MeshRender.inpaint, the oracle)
  filled_edt  [T,T,3] f32  EDT-only intermediate (clip -> out[tuple(idx)]), pre-quantize/NS
  edt_rows/edt_cols [T,T] i32  scipy feature-transform indices (tie-break ground truth)

Run from python/paint (this repo), AFTER dump_bake_fixture.py:
    PYTHONPATH=. FIXTURES_OUT=/path/to/fixtures uv run python .../dump_inpaint_fixture.py
"""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from scipy import ndimage
from hy3dpaint_mlx.mesh_render import MeshRender

FIX = os.environ.get("FIXTURES_OUT", "fixtures")
bake = mx.load(f"{FIX}/bake_fixture.safetensors")
texture = np.asarray(bake["tex"], dtype=np.float32)
covered = np.asarray(bake["covered"]).astype(bool)

filled = MeshRender.inpaint(texture, covered)             # the oracle, verbatim

clipped = np.clip(texture, 0, 1).astype(np.float32)
idx = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
filled_edt = clipped[tuple(idx)]

# tie diagnostics: how many hole texels have >1 equidistant nearest covered texel?
zr, zc = np.nonzero(covered)
pts = np.stack([zr, zc], 1)
hr, hc = np.nonzero(~covered)
ties = 0
for r, c in zip(hr.tolist(), hc.tolist()):
    d2 = (pts[:, 0] - r) ** 2 + (pts[:, 1] - c) ** 2
    ties += int((d2 == d2.min()).sum() > 1)

mx.save_safetensors(f"{FIX}/inpaint_fixture.safetensors", {
    "texture": mx.array(texture), "covered": mx.array(covered.astype(np.int32)),
    "filled": mx.array(filled.astype(np.float32)),
    "filled_edt": mx.array(filled_edt),
    "edt_rows": mx.array(idx[0].astype(np.int32)), "edt_cols": mx.array(idx[1].astype(np.int32)),
})
print(f"inpaint fixture: T={texture.shape[0]} holes={int((~covered).sum())} "
      f"({100.0 * (~covered).mean():.1f}%) ties={ties} "
      f"| filled range {filled.min():.4f}..{filled.max():.4f}")

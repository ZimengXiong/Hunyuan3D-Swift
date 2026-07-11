"""Octree near-surface band logic (fast, synthetic SDF, no weights)."""
import numpy as np

from hy3dmlx.models.shape_vae import _near_surface_mask


def test_near_surface_brackets_zero_crossing():
    # 1D-ish sphere SDF on a small grid; the mask must cover the iso-surface region
    N = 32
    x = np.linspace(-1, 1, N).astype(np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    sdf = (0.5 ** 2 - (X ** 2 + Y ** 2 + Z ** 2)).astype(np.float32)  # >0 inside
    mask = _near_surface_mask(sdf, alpha=0.0)
    # every cell with a sign change to a neighbor is flagged
    s = np.sign(sdf)
    crossing = np.zeros_like(mask)
    for ax in range(3):
        d = np.diff(s, axis=ax) != 0
        lo = [slice(None)] * 3; lo[ax] = slice(0, -1)
        hi = [slice(None)] * 3; hi[ax] = slice(1, None)
        crossing[tuple(lo)] |= d
        crossing[tuple(hi)] |= d
    assert np.all(mask[crossing])              # band covers all crossings
    assert mask.sum() < sdf.size               # but is sparse (not the whole grid)
    assert not mask[0, 0, 0]                   # far-outside corner excluded


def test_near_surface_excludes_nan():
    sdf = np.full((8, 8, 8), np.nan, np.float32)
    sdf[3:5, 3:5, 3:5] = np.array([-0.1, 0.1]).repeat(4).reshape(2, 2, 2)
    mask = _near_surface_mask(sdf, alpha=0.0)
    assert not mask[np.isnan(sdf)].any()       # NaN cells never flagged

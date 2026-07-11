"""ctypes loader for the custom_rasterizer CPU-twin oracle (libcr_cpu_oracle.dylib).

Bit-exact CPU equivalent of the reference CUDA rasterizer. Build first:
    bash oracle/cr_cpu/build.sh
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_DIR, "libcr_cpu_oracle.dylib")

_lib = None


def available() -> bool:
    return os.path.exists(_LIB_PATH)


def _load():
    global _lib
    if _lib is None:
        if not available():
            raise RuntimeError(
                "libcr_cpu_oracle.dylib not built — run: bash oracle/cr_cpu/build.sh"
            )
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.cr_rasterize.restype = None
        _lib.cr_rasterize.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # V
            ctypes.POINTER(ctypes.c_int),    # F
            ctypes.c_int,                    # num_vertices
            ctypes.c_int,                    # num_faces
            ctypes.c_int,                    # width
            ctypes.c_int,                    # height
            ctypes.POINTER(ctypes.c_int),    # findices_out
            ctypes.POINTER(ctypes.c_float),  # bary_out
        ]
    return _lib


def rasterize_image(V, F, width, height):
    """V: [N,4] float32 clip-space homogeneous. F: [M,3] int32.

    Returns (findices [H,W] int32, barycentric [H,W,3] float32).
    """
    lib = _load()
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    assert V.ndim == 2 and V.shape[1] == 4
    assert F.ndim == 2 and F.shape[1] == 3
    findices = np.empty((height, width), dtype=np.int32)
    bary = np.empty((height, width, 3), dtype=np.float32)
    lib.cr_rasterize(
        V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        F.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(V.shape[0]),
        ctypes.c_int(F.shape[0]),
        ctypes.c_int(width),
        ctypes.c_int(height),
        findices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        bary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return findices, bary

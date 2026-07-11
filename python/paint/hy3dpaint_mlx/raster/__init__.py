"""custom_rasterizer-faithful rasterization for Apple Silicon (numpy now, Metal later)."""

from .cr_raster import rasterize, interpolate

__all__ = ["rasterize", "interpolate"]

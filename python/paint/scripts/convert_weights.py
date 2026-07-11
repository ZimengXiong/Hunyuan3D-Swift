"""Pre-convert paint torch safetensors to MLX-native safetensors (Conv2d NCHW->NHWC, optional fp16).

Keys are preserved (module-specific renames stay at load time via convert.load_torch_weights), so
this just bakes the layout transpose + dtype for faster loading / distribution.

Usage:
    uv run python scripts/convert_weights.py <in.safetensors> <out.safetensors> [--fp16]
"""

import sys
import mlx.core as mx


def convert(src, dst, fp16=False):
    sd = mx.load(src)
    out = {}
    for k, v in sd.items():
        if v.ndim == 4:                       # conv weight NCHW -> NHWC
            v = v.transpose(0, 2, 3, 1)
        if fp16 and v.dtype == mx.float32:
            v = v.astype(mx.float16)
        out[k] = v
    mx.save_safetensors(dst, out)
    print(f"converted {len(out)} tensors -> {dst} ({'fp16' if fp16 else 'fp32'})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    convert(args[0], args[1], fp16="--fp16" in sys.argv)

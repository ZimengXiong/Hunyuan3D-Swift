"""Torch/diffusers safetensors -> MLX module weight loading.

Shared by the parity tests and the eventual convert scripts. Two universal rules:
  * Conv2d weights are NCHW [out,in,kH,kW] in torch, NHWC [out,kH,kW,in] in MLX -> transpose (0,2,3,1).
  * Linear/Norm weights are identical.
Module trees in this package mirror the diffusers names, so loading is a flat key map plus
optional substring renames (e.g. diffusers' ".to_out.0." -> our ".to_out.").
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_unflatten


def load_torch_weights(module, sd, renames=(), verbose=False):
    """Load a torch state_dict (dict of mx.array or numpy) into an MLX module.

    sd: {torch_key: array}. renames: iterable of (old_substr, new_substr).
    Raises on missing/extra keys (strict), so shape/name mismatches fail loud.
    """
    flat = []
    for k, v in sd.items():
        for a, b in renames:
            k = k.replace(a, b)
        arr = v if isinstance(v, mx.array) else mx.array(v)
        if arr.ndim == 4:  # conv weight NCHW -> NHWC
            arr = arr.transpose(0, 2, 3, 1)
        flat.append((k, arr))
    module.update(tree_unflatten(flat))
    mx.eval(module.parameters())
    if verbose:
        print(f"loaded {len(flat)} tensors")
    return module

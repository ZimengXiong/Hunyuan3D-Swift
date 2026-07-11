"""Load a Hunyuan3D shape checkpoint into MLX modules (2mini / v2-0 / v2-1).

The bundled `model.fp16.safetensors` keys are split by prefix: `conditioner.`
(DINOv2), `model.` (DiT), `vae.` (ShapeVAE). Linear weights copy 1:1; only the
DINO patch Conv2d is transposed NCHW->NHWC. Encoder/pre_kl VAE keys are ignored.

The DiT class (FLUX-style `Hunyuan3DDiT` vs U-Net `HunYuanDiTPlain`+MoE) and the
DINO FFN (SwiGLU vs plain MLP) are selected from the config.
"""
import os

import numpy as np
import yaml
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

from .models.dit import Hunyuan3DDiT
from .models.dit_plain import HunYuanDiTPlain
from .models.shape_vae import ShapeVAE
from .models.dinov2 import DinoImageEncoder

COND_PREFIX = "conditioner.main_image_encoder."


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def _assign(module, items, name: str):
    need = {k: tuple(v.shape) for k, v in tree_flatten(module.parameters())}
    provided = dict(items)
    missing = set(need) - set(provided)
    if missing:
        raise KeyError(f"{name}: {len(missing)} missing keys, e.g. {sorted(missing)[:6]}")
    for k in need:
        if tuple(provided[k].shape) != need[k]:
            raise ValueError(f"{name}.{k}: shape {tuple(provided[k].shape)} != expected {need[k]}")
    module.load_weights([(k, provided[k]) for k in need], strict=True)
    return len(need), len(set(provided) - set(need))


def _build_dit(cfg: dict):
    target = cfg["model"]["target"].rsplit(".", 1)[-1]
    params = cfg["model"]["params"]
    if target == "HunYuanDiTPlain":
        return HunYuanDiTPlain(**params)
    return Hunyuan3DDiT(**params)  # Hunyuan3DDiT


def _build_dino(cfg: dict, dino_items: dict):
    dcfg = cfg["conditioner"]["params"]["main_image_encoder"]["kwargs"]["config"]
    swiglu = bool(dcfg.get("use_swiglu_ffn", True))
    # FFN inner dim straight from the weights (SwiGLU weights_out / plain fc2: [hidden, ffn_dim]).
    wkey = "model.encoder.layer.0.mlp." + ("weights_out" if swiglu else "fc2") + ".weight"
    ffn_dim = dino_items[wkey].shape[1]
    return DinoImageEncoder(
        hidden=dcfg["hidden_size"], heads=dcfg["num_attention_heads"],
        layers=dcfg["num_hidden_layers"], ffn_dim=ffn_dim, swiglu=swiglu,
        patch=dcfg["patch_size"], eps=float(dcfg["layer_norm_eps"]),
    )


# Keep these in fp16 even when quantizing: small, quality-critical input/output
# projections (the latent in/out and conditioning projection) and DINO embeddings.
_QUANT_SKIP_DIT = ("latent_in", "time_in", "cond_in", "guidance_in", "final_layer", "x_embedder")
_QUANT_SKIP_DINO = ("embeddings",)


def _quantize(module, skip, bits: int, group_size: int = 64):
    """Quantize the bulk nn.Linear weights of `module` to `bits`, skipping `skip` paths.
    Norms, Conv2d, and the VAE are left in fp16 (never passed here)."""
    def predicate(path, m):
        return (isinstance(m, nn.Linear)
                and not any(s in path for s in skip)
                and m.weight.shape[1] % group_size == 0)
    nn.quantize(module, group_size=group_size, bits=bits, class_predicate=predicate)


def load_models(model_dir: str, dtype=mx.float32, quantize: int | None = None,
                verbose: bool = True):
    cfg = load_config(model_dir)
    st = os.path.join(model_dir, "model.fp16.safetensors")

    dino_items, dit_items, vae_items = {}, {}, {}
    with safe_open(st, framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            if key.startswith(COND_PREFIX):
                k = key[len(COND_PREFIX):]
                if k.endswith("patch_embeddings.projection.weight") and arr.ndim == 4:
                    arr = np.transpose(arr, (0, 2, 3, 1))
                dino_items[k] = mx.array(arr).astype(dtype)
            elif key.startswith("model."):
                dit_items[key[len("model."):]] = mx.array(arr).astype(dtype)
            elif key.startswith("vae."):
                vae_items[key[len("vae."):]] = mx.array(arr).astype(dtype)

    dit = _build_dit(cfg)
    vae = ShapeVAE(**cfg["vae"]["params"])
    dino = _build_dino(cfg, dino_items)

    n_dino = _assign(dino, dino_items.items(), "dino")
    n_dit = _assign(dit, dit_items.items(), "dit")
    n_vae = _assign(vae, vae_items.items(), "vae")
    if quantize in (4, 8):
        _quantize(dit, _QUANT_SKIP_DIT, quantize)
        _quantize(dino, _QUANT_SKIP_DINO, quantize)  # VAE stays fp16
    if verbose:
        q = f" [{quantize}-bit DiT+DINO]" if quantize in (4, 8) else ""
        print(f"loaded {cfg['model']['target'].rsplit('.', 1)[-1]}{q}: "
              f"dino {n_dino[0]} (+{n_dino[1]}), dit {n_dit[0]} (+{n_dit[1]}), "
              f"vae {n_vae[0]} (+{n_vae[1]})")
    mx.eval(dino.parameters(), dit.parameters(), vae.parameters())
    return dino, dit, vae, cfg

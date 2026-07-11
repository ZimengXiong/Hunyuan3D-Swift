"""Parity gates for the 2.1 PBR pieces: 3D PoseRoPE, DINOv2-giant, and the full PBR UNet."""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_unflatten

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_UNET = os.path.join(ROOT, "weights", "hunyuan3d-paintpbr-v2-1", "unet", "diffusion_pytorch_model.safetensors")
_DINO = os.path.join(ROOT, "weights", "dinov2-giant", "model.safetensors")


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@pytest.mark.skipif(not os.path.exists(_ORACLE_PY), reason="oracle venv missing")
def test_rope():
    d = os.path.join(ROOT, ".parity_dumps", "rope")
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "rope_oracle.py"), d], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from hy3dpaint_mlx.attention_pbr import get_3d_rotary_pos_embed, apply_rotary_emb
    c, s = get_3d_rotary_pos_embed(mx.array(np.load(f"{d}/pos.npy")), 64, 512)
    assert _cos(c, np.load(f"{d}/cos.npy")) > 0.99999
    out = apply_rotary_emb(mx.array(np.load(f"{d}/x.npy")), c, s)
    assert _cos(out, np.load(f"{d}/out.npy")) > 0.99999


@pytest.mark.skipif(not (os.path.exists(_ORACLE_PY) and os.path.exists(_DINO)), reason="needs dino weights")
def test_dino():
    d = os.path.join(ROOT, ".parity_dumps", "dino")
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "dino_oracle.py"), d], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from hy3dpaint_mlx.dinov2 import Dinov2Model
    m = Dinov2Model()
    m.update(tree_unflatten([(k, (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v)) for k, v in mx.load(_DINO).items()]))
    out = np.asarray(m(mx.array(np.load(f"{d}/pix.npy").transpose(0, 2, 3, 1))), np.float32)
    assert _cos(out, np.load(f"{d}/out.npy")) > 0.99999


@pytest.mark.skipif(not (os.path.exists(_ORACLE_PY) and os.path.exists(_UNET)), reason="needs 2.1 unet weights")
def test_pbr_unet():
    import json
    d = os.path.join(ROOT, ".parity_dumps", "u2p5d_pbr")
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "unet2p5d_pbr_oracle.py"), d], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from hy3dpaint_mlx.unet2p5d_pbr import UNet2p5DPBRConditionModel
    from hy3dpaint_mlx.convert import load_torch_weights
    m = UNet2p5DPBRConditionModel(json.load(open(os.path.join(ROOT, "weights/hunyuan3d-paintpbr-v2-1/unet/config.json"))))
    load_torch_weights(m, mx.load(_UNET))
    L = lambda n: np.load(f"{d}/{n}")
    out = np.asarray(m(mx.array(L("sample.npy")), 10.0, mx.array(L("normal.npy")), mx.array(L("position.npy")),
                       mx.array(L("ref_lat.npy")), mx.array(L("dino.npy")), L("posmap.npy")), np.float32)
    # neural path is exact; residual is fp16 voxel-quantization in RoPE preprocessing
    assert _cos(out, L("out.npy")) > 0.9999

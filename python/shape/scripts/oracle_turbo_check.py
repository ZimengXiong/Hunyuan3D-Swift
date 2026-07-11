"""Single-forward parity for the turbo (guidance_embed) DiT: torch oracle vs MLX.
Confirms the guidance branch is correct (rules out a port bug for the striping).
  VIRTUAL_ENV=.venv-oracle uv run --no-project python scripts/oracle_turbo_check.py   # writes /tmp/turbo_*.npy
  PYTHONPATH=. uv run python scripts/oracle_turbo_check.py mlx                          # compares
"""
import sys
import numpy as np

WD = "weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini-turbo"

if len(sys.argv) > 1 and sys.argv[1] == "mlx":
    import mlx.core as mx
    from hy3dmlx.convert import load_models
    _, dit, _, _ = load_models(WD, dtype=mx.float32, verbose=False)
    x = mx.array(np.load("/tmp/turbo_x.npy"))
    cond = mx.array(np.load("/tmp/turbo_cond.npy"))
    t = mx.array(np.load("/tmp/turbo_t.npy"))
    g = mx.array(np.load("/tmp/turbo_g.npy"))
    v = dit(x, t, cond, guidance=g); mx.eval(v)
    v = np.array(v.astype(mx.float32)); vt = np.load("/tmp/turbo_v.npy")
    c = (v.flatten() @ vt.flatten()) / (np.linalg.norm(v) * np.linalg.norm(vt))
    print(f"[turbo DiT] cosine {c:.7f}  maxabs {np.abs(v - vt).max():.5f}  std {v.std():.4f}/{vt.std():.4f}")
else:
    import torch, yaml, safetensors.torch
    sys.path.insert(0, "reference/Hunyuan3D-2.1/hy3dshape")
    from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT
    cfg = yaml.safe_load(open(f"{WD}/config.yaml"))
    sd = safetensors.torch.load_file(f"{WD}/model.fp16.safetensors")
    dit = Hunyuan3DDiT(**cfg["model"]["params"]).eval()
    miss = dit.load_state_dict({k[6:]: v.float() for k, v in sd.items() if k.startswith("model.")},
                               strict=False)
    print("guidance_embed:", dit.guidance_embed, "| missing", len(miss.missing_keys))
    dit.float()
    x = np.random.RandomState(0).randn(1, 512, 64).astype(np.float32)
    cond = np.random.RandomState(1).randn(1, 1370, 1536).astype(np.float32)
    t = np.array([0.3], np.float32); g = np.array([5.0], np.float32)
    with torch.no_grad():
        v = dit(torch.from_numpy(x), torch.from_numpy(t),
                {"main": torch.from_numpy(cond)}, guidance=torch.from_numpy(g))
    for n, a in [("x", x), ("cond", cond), ("t", t), ("g", g), ("v", v.numpy())]:
        np.save(f"/tmp/turbo_{n}.npy", a)
    print(f"[oracle turbo DiT] v std {v.std():.4f}")

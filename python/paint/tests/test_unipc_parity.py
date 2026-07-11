"""Parity gate: MLX UniPCScheduler vs diffusers UniPCMultistepScheduler (paint-2.1 config)."""

import os
import sys
import subprocess

import numpy as np
import mlx.core as mx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hy3dpaint_mlx.scheduler import UniPCScheduler

_ORACLE_PY = os.path.join(ROOT, ".venv-oracle", "bin", "python")
_DUMP = os.path.join(ROOT, ".parity_dumps", "unipc")

pytestmark = pytest.mark.skipif(not os.path.exists(_ORACLE_PY), reason="oracle venv missing")


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@pytest.fixture(scope="module")
def dump():
    os.makedirs(_DUMP, exist_ok=True)
    r = subprocess.run([_ORACLE_PY, os.path.join(ROOT, "oracle", "unipc_oracle.py"), _DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"unipc oracle failed:\n{r.stdout}\n{r.stderr}")
    return _DUMP


def test_unipc(dump):
    x0 = np.load(os.path.join(dump, "x0.npy"))            # (1,4,8,8) NCHW
    timesteps = np.load(os.path.join(dump, "timesteps.npy"))  # (15,)
    vs = np.load(os.path.join(dump, "vs.npy"))            # (15,1,4,8,8)
    ref_traj = np.load(os.path.join(dump, "traj.npy"))    # (15,1,4,8,8)

    sched = UniPCScheduler(
        num_train=1000,
        beta_start=0.00085,
        beta_end=0.012,
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
    )
    sched.set_timesteps(15)

    # timesteps must match diffusers exactly
    assert np.array_equal(np.asarray(sched.timesteps, np.int64), timesteps.astype(np.int64)), (
        f"timestep mismatch: mlx={list(sched.timesteps)} ref={list(timesteps)}"
    )

    xx = mx.array(x0.astype(np.float32))
    cosines = []
    maxabs = []
    for i in range(len(timesteps)):
        v = mx.array(vs[i].astype(np.float32))
        xx = sched.step(v, int(timesteps[i]), xx)
        mx.eval(xx)
        out = np.asarray(xx, np.float32)
        ref = ref_traj[i]
        c = _cos(out, ref)
        m = float(np.abs(out - ref).max())
        cosines.append(c)
        maxabs.append(m)

    worst_cos = min(cosines)
    final_maxabs = maxabs[-1]
    overall_maxabs = max(maxabs)
    print(f"unipc per-step cosine min={worst_cos:.7f} "
          f"final_maxabs={final_maxabs:.3e} overall_maxabs={overall_maxabs:.3e}")
    print("per-step cosines:", [f"{c:.7f}" for c in cosines])

    assert worst_cos > 0.9999, f"worst per-step cosine {worst_cos} <= 0.9999"
    assert final_maxabs < 1e-3, f"final maxabs {final_maxabs} >= 1e-3"

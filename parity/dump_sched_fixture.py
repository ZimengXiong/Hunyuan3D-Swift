"""Dump DDIM + UniPC scheduler tables + a trajectory fixture for the mlx-swift gate."""
import sys, os; sys.path.insert(0, ".")
import numpy as np
import mlx.core as mx
from hy3dpaint_mlx.scheduler import DDIMScheduler, UniPCScheduler

N = 15
d = DDIMScheduler(); d.set_timesteps(N)
u = UniPCScheduler(); u.set_timesteps(N)
rng = np.random.RandomState(0)
shape = (2, 4, 4, 4)
x0 = mx.array(rng.randn(*shape).astype(np.float32))
vs = mx.array(rng.randn(N, *shape).astype(np.float32))

xd = x0
for i, t in enumerate(d.timesteps): xd = d.step(vs[i], int(t), xd)
xu = x0
for i, t in enumerate(u.timesteps): xu = u.step(vs[i], int(t), xu)
mx.eval(xd, xu)

FIX = os.environ.get("FIXTURES_OUT", "fixtures"); os.makedirs(FIX, exist_ok=True)
mx.save_safetensors(f"{FIX}/sched_fixture.safetensors", {
    "ddim_acp": mx.array(np.asarray(d.acp, np.float32)),
    "ddim_timesteps": mx.array(np.asarray(d.timesteps, np.int32)),
    "ddim_ratio": mx.array(np.array([d.ratio], np.int32)),
    "unipc_sigmas": mx.array(np.asarray(u.sigmas, np.float32)),
    "unipc_timesteps": mx.array(np.asarray(u.timesteps, np.int32)),
    "x0": x0, "vs": vs, "ddim_traj": xd, "unipc_traj": xu,
})
print("DDIM ratio", d.ratio, "ts", list(d.timesteps))
print("UniPC ts", list(u.timesteps), "sigmas[:3]", [round(float(s),4) for s in u.sigmas[:3]])
print("ddim_traj std", round(float(xd.std()),4), "unipc_traj std", round(float(xu.std()),4))

"""Dump a diffusers UniPCMultistepScheduler trajectory (2.1 paint config) for parity gating.

Builds the scheduler with the exact paint-2.1 config, set_timesteps(15), creates a fixed
initial latent x0 and a fixed list of 15 model_outputs (v_prediction), then iterates step().
Saves x0, timesteps, the stacked model_outputs, and the stacked trajectory (NCHW) to argv[1].
"""

import os
import sys

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    sched = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
    )
    sched.set_timesteps(15)
    timesteps = sched.timesteps  # torch int64, len 15

    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 8, 8) * sched.init_noise_sigma
    vs = [torch.randn(1, 4, 8, 8) for _ in range(len(timesteps))]

    traj = []
    xx = x0
    for i, t in enumerate(timesteps):
        xx = sched.step(vs[i], t, xx).prev_sample
        traj.append(xx.clone())

    vs_stacked = torch.stack(vs, dim=0).numpy()        # (15, 1, 4, 8, 8) NCHW
    traj_stacked = torch.stack(traj, dim=0).numpy()    # (15, 1, 4, 8, 8) NCHW

    np.save(os.path.join(out_dir, "x0.npy"), x0.numpy())                    # (1,4,8,8) NCHW
    np.save(os.path.join(out_dir, "timesteps.npy"), timesteps.numpy().astype(np.int64))
    np.save(os.path.join(out_dir, "vs.npy"), vs_stacked)
    np.save(os.path.join(out_dir, "traj.npy"), traj_stacked)
    print(f"OK unipc traj steps={len(timesteps)} -> {out_dir}")
    print("timesteps:", timesteps.numpy().tolist())


if __name__ == "__main__":
    main(sys.argv[1])

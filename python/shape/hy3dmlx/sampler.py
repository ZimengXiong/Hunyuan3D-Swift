"""FlowMatchEulerDiscrete sampler in MLX, matching Hunyuan3DDiTFlowMatchingPipeline.

The pipeline builds sigmas = linspace(0, 1, N) (note: starts at 0), applies the
rational shift warp (shift=1.0 -> identity), sets timesteps = sigmas*1000, and
appends 1.0 so sigmas_full has N+1 entries. At step i the model receives the
normalized timestep sigma_i in [0,1]; the Euler update is
    x <- x + (sigma_full[i+1] - sigma_i) * v
(sigma_full[i] == sigma_i for i < N, so this matches the reference sigma_next - sigma.)
CFG: model is run on cat([x, x]) -> (v_cond, v_uncond); v = v_uncond + s*(v_cond - v_uncond).
"""
import numpy as np
import mlx.core as mx


def flow_match_sigmas(num_inference_steps: int, shift: float = 1.0):
    """Returns (sigmas [N], sigmas_full [N+1]) as float32 numpy arrays."""
    sigmas = np.linspace(0.0, 1.0, num_inference_steps).astype(np.float32)
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)  # identity when shift=1
    sigmas_full = np.concatenate([sigmas, np.ones(1, dtype=np.float32)])
    return sigmas, sigmas_full


def consistency_sigmas(num_inference_steps: int, pcm_timesteps: int = 100,
                       num_train_timesteps: int = 1000):
    """ConsistencyFlowMatchEulerDiscreteScheduler grid (turbo/PCM-distilled models).
    Picks a phased-consistency subset of the sigma grid; few steps suffice.
    """
    base = np.linspace(0.0, 1.0, num_train_timesteps).astype(np.float32)
    step_ratio = num_train_timesteps // pcm_timesteps
    euler = (np.arange(1, pcm_timesteps) * step_ratio).round().astype(np.int64) - 1
    euler = np.concatenate([[0], euler])
    pcm = base[euler]                                   # [pcm_timesteps]
    idx = np.floor(np.linspace(0, pcm_timesteps, num=num_inference_steps,
                               endpoint=False)).astype(np.int64)
    sigmas = pcm[idx].astype(np.float32)                # [N]
    sigmas_full = np.concatenate([sigmas, np.ones(1, dtype=np.float32)])
    return sigmas, sigmas_full


def denoise(
    velocity_fn,
    latent_shape,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    cfg: bool = True,
    shift: float = 1.0,
    seed: int = 0,
    dtype=mx.float32,
    sigmas=None,
    progress=None,
):
    """Run the flow-matching Euler loop.

    cfg=True:  velocity_fn(cat([x,x]), t) -> [2B,...]; combined as uncond + s*(cond-uncond).
    cfg=False: velocity_fn(x, t) -> [B,...] used directly (guidance_embed / distilled models;
               velocity_fn applies the guidance embedding internally).
    sigmas: optional (sigmas[N], sigmas_full[N+1]) override (e.g. consistency schedule).
    """
    if sigmas is None:
        sigmas = flow_match_sigmas(num_inference_steps, shift)
    sig, sig_full = sigmas
    n = len(sig)
    latents = mx.random.normal(latent_shape, dtype=mx.float32, key=mx.random.key(seed)).astype(dtype)

    for i in range(n):
        dt = float(sig_full[i + 1] - sig[i])
        if dt == 0.0:
            continue
        t = mx.array(float(sig[i]), dtype=dtype)
        if cfg:
            v = velocity_fn(mx.concatenate([latents, latents], axis=0), t)
            v_cond, v_uncond = mx.split(v, 2, axis=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = velocity_fn(latents, t)
        latents = latents + dt * v
        mx.eval(latents)
        if progress is not None:
            progress(i + 1, n)
    return latents

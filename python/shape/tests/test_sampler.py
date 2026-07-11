"""Sampler schedule numerics (fast, no weights)."""
import numpy as np

from hy3dmlx.sampler import consistency_sigmas, flow_match_sigmas


def test_flow_match_sigmas_linspace_identity():
    sig, full = flow_match_sigmas(30, shift=1.0)
    assert sig.shape == (30,) and full.shape == (31,)
    assert np.allclose(sig, np.linspace(0, 1, 30))   # shift=1 -> identity warp
    assert full[-1] == 1.0 and full[-2] == sig[-1]   # appended 1.0


def test_flow_match_shift_warp_monotonic():
    sig, _ = flow_match_sigmas(20, shift=3.0)
    assert np.all(np.diff(sig) > 0)
    assert abs(sig[0]) < 1e-6 and abs(sig[-1] - 1.0) < 1e-6


def test_consistency_sigmas_matches_reference_grid():
    # reference ConsistencyFlowMatchEulerDiscreteScheduler(pcm_timesteps=100), N=8
    sig, full = consistency_sigmas(8, pcm_timesteps=100, num_train_timesteps=1000)
    ref = [0.0, 0.1191, 0.2492, 0.3694, 0.4995, 0.6196, 0.7497, 0.8699]
    assert np.allclose(sig, ref, atol=1e-3)
    assert full[-1] == 1.0


def test_consistency_first_sigma_zero():
    sig, _ = consistency_sigmas(5, 100)
    assert abs(sig[0]) < 1e-6  # starts from noise at t=0

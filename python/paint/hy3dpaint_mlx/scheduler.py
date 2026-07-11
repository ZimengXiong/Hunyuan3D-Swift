"""DDIM scheduler (v-prediction, scaled-linear betas, zero-terminal-SNR, trailing spacing).

Matches the paint checkpoint scheduler_config (DDIMScheduler, v_prediction, rescale_betas_zero_snr,
timestep_spacing='trailing', steps_offset=1). Deterministic — good for a reproducible run.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def _rescale_zero_terminal_snr(alphas_cumprod):
    abs_ = np.sqrt(alphas_cumprod)
    a0, aT = abs_[0].copy(), abs_[-1].copy()
    abs_ -= aT
    abs_ *= a0 / (a0 - aT)
    return abs_ ** 2


def _rescale_betas_zero_snr(betas):
    """Rescale betas to have zero terminal SNR (diffusers rescale_zero_terminal_snr).

    Operates on betas (float32, matching diffusers torch.float32) and returns rescaled betas.
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T
    alphas_bar_sqrt = alphas_bar_sqrt * (alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T))

    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas.astype(np.float32)


class DDIMScheduler:
    def __init__(self, num_train=1000, beta_start=0.00085, beta_end=0.012):
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train, dtype=np.float64) ** 2
        acp = np.cumprod(1.0 - betas)
        self.acp = _rescale_zero_terminal_snr(acp)
        self.acp[-1] = max(self.acp[-1], 1e-8)
        self.num_train = num_train
        self.final_alpha = 1.0  # set_alpha_to_one

    def set_timesteps(self, n):
        ratio = self.num_train / n
        ts = np.round(np.arange(self.num_train, 0, -ratio)).astype(int) - 1  # trailing
        ts = np.clip(ts, 0, self.num_train - 1)
        self.timesteps = ts
        self.ratio = int(round(ratio))

    def init_noise_sigma(self):
        return 1.0

    def scale_model_input(self, sample, t):
        return sample  # identity for DDIM

    def step(self, v, t, sample):
        """v: model output (v_prediction). sample: x_t. Returns x_{t-1}."""
        prev_t = t - self.ratio
        a_t = float(self.acp[t])
        a_prev = float(self.acp[prev_t]) if prev_t >= 0 else self.final_alpha
        sa_t, soma_t = a_t ** 0.5, (1 - a_t) ** 0.5
        x0 = sa_t * sample - soma_t * v
        eps = sa_t * v + soma_t * sample
        x_prev = (a_prev ** 0.5) * x0 + ((1 - a_prev) ** 0.5) * eps
        return x_prev


class UniPCScheduler:
    """UniPC multistep scheduler (v_prediction, predict_x0, bh2, trailing, zero-terminal-SNR).

    Faithful MLX port of diffusers `UniPCMultistepScheduler` for the paint-2.1 config:
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear',
        prediction_type='v_prediction', rescale_betas_zero_snr=True, timestep_spacing='trailing',
        solver_order=2, predict_x0=True, solver_type='bh2', lower_order_final=True.

    Multistep: keeps model-output history across step() calls; reset in set_timesteps.
    numpy is used for the alpha/sigma tables + the small coefficient linear solves; mx for samples.
    """

    def __init__(
        self,
        num_train=1000,
        beta_start=0.00085,
        beta_end=0.012,
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
        disable_corrector=None,
    ):
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train, dtype=np.float32) ** 2
        betas = _rescale_betas_zero_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # Close to 0 without being 0 so first sigma is not inf (FP16 smallest subnormal).
        alphas_cumprod[-1] = 2 ** -24
        self.alphas_cumprod = alphas_cumprod  # float32, matches diffusers torch.float32

        self.num_train = num_train
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.disable_corrector = disable_corrector if disable_corrector is not None else []

    # ---- setup ----------------------------------------------------------------
    def set_timesteps(self, n):
        # trailing spacing
        step_ratio = self.num_train / n
        timesteps = np.arange(self.num_train, 0, -step_ratio).round().copy().astype(np.int64)
        timesteps -= 1

        # sigmas via interp of training sigmas onto these timesteps; final_sigmas_type='zero'
        train_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigmas = np.interp(timesteps, np.arange(0, len(train_sigmas)), train_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.timesteps = timesteps
        self.sigmas = sigmas  # len n+1
        self.num_inference_steps = len(timesteps)

        # reset multistep history
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self.this_order = None

    def init_noise_sigma(self):
        return 1.0

    def scale_model_input(self, sample, t):
        return sample  # identity (noise-space solver)

    # ---- helpers --------------------------------------------------------------
    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1.0 / ((sigma ** 2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    @staticmethod
    def _log(v):
        """log matching torch IEEE semantics: log(0) -> -inf, no exception."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.log(np.float64(v)))

    def _init_step_index(self, t):
        ts = self.timesteps
        cand = np.nonzero(ts == int(t))[0]
        if len(cand) == 0:
            self._step_index = len(ts) - 1
        elif len(cand) > 1:
            self._step_index = int(cand[1])
        else:
            self._step_index = int(cand[0])

    def _coeff_matrices(self, order, rks_np, hh):
        """Compute R (order x order), b (order,), B_h, h_phi_1 in numpy."""
        h_phi_1 = np.expm1(hh)  # e^hh - 1
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = np.expm1(hh)
        else:
            raise NotImplementedError(self.solver_type)

        R = []
        b = []
        for i in range(1, order + 1):
            R.append(np.power(rks_np, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = np.stack(R, axis=0) if len(R) > 0 else np.zeros((0, 0))
        b = np.stack(b, axis=0) if len(b) > 0 else np.zeros((0,))
        return R, b, B_h, h_phi_1

    # ---- predictor ------------------------------------------------------------
    def multistep_uni_p_bh_update(self, sample, order):
        si = self._step_index
        m0 = self.model_outputs[-1]
        x = sample

        sigma_t = float(self.sigmas[si + 1])
        sigma_s0 = float(self.sigmas[si])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = self._log(alpha_t) - self._log(sigma_t)
        lambda_s0 = self._log(alpha_s0) - self._log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            sidx = si - i
            mi = self.model_outputs[-(i + 1)]
            a_si, s_si = self._sigma_to_alpha_sigma_t(float(self.sigmas[sidx]))
            lambda_si = self._log(a_si) - self._log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) * (1.0 / rk))

        rks.append(1.0)
        rks_np = np.array(rks, dtype=np.float64)

        hh = -h if self.predict_x0 else h
        R, b, B_h, h_phi_1 = self._coeff_matrices(order, rks_np, hh)

        if len(D1s) > 0:
            if order == 2:
                rhos_p = np.array([0.5], dtype=np.float64)
            else:
                rhos_p = np.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = (sigma_t / sigma_s0) * x - (alpha_t * h_phi_1) * m0
            if D1s is not None:
                pred_res = self._einsum_k(rhos_p, D1s)
            else:
                pred_res = 0.0
            x_t = x_t_ - (alpha_t * B_h) * pred_res
        else:
            x_t_ = (alpha_t / alpha_s0) * x - (sigma_t * h_phi_1) * m0
            if D1s is not None:
                pred_res = self._einsum_k(rhos_p, D1s)
            else:
                pred_res = 0.0
            x_t = x_t_ - (sigma_t * B_h) * pred_res
        return x_t

    # ---- corrector ------------------------------------------------------------
    def multistep_uni_c_bh_update(self, this_model_output, last_sample, this_sample, order):
        si = self._step_index
        m0 = self.model_outputs[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t = float(self.sigmas[si])
        sigma_s0 = float(self.sigmas[si - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = self._log(alpha_t) - self._log(sigma_t)
        lambda_s0 = self._log(alpha_s0) - self._log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            sidx = si - (i + 1)
            mi = self.model_outputs[-(i + 1)]
            a_si, s_si = self._sigma_to_alpha_sigma_t(float(self.sigmas[sidx]))
            lambda_si = self._log(a_si) - self._log(s_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) * (1.0 / rk))

        rks.append(1.0)
        rks_np = np.array(rks, dtype=np.float64)

        hh = -h if self.predict_x0 else h
        R, b, B_h, h_phi_1 = self._coeff_matrices(order, rks_np, hh)

        if len(D1s) > 0:
            D1s_list = D1s
        else:
            D1s_list = None

        if order == 1:
            rhos_c = np.array([0.5], dtype=np.float64)
        else:
            rhos_c = np.linalg.solve(R, b)

        if self.predict_x0:
            x_t_ = (sigma_t / sigma_s0) * x - (alpha_t * h_phi_1) * m0
            if D1s_list is not None:
                corr_res = self._einsum_k(rhos_c[:-1], D1s_list)
            else:
                corr_res = 0.0
            D1_t = model_t - m0
            x_t = x_t_ - (alpha_t * B_h) * (corr_res + float(rhos_c[-1]) * D1_t)
        else:
            x_t_ = (alpha_t / alpha_s0) * x - (sigma_t * h_phi_1) * m0
            if D1s_list is not None:
                corr_res = self._einsum_k(rhos_c[:-1], D1s_list)
            else:
                corr_res = 0.0
            D1_t = model_t - m0
            x_t = x_t_ - (sigma_t * B_h) * (corr_res + float(rhos_c[-1]) * D1_t)
        return x_t

    @staticmethod
    def _einsum_k(rhos, D1s_list):
        """sum_k rhos[k] * D1s_list[k], where each D1s_list[k] is an mx.array (per-sample residual)."""
        acc = float(rhos[0]) * D1s_list[0]
        for k in range(1, len(D1s_list)):
            acc = acc + float(rhos[k]) * D1s_list[k]
        return acc

    # ---- model-output conversion ---------------------------------------------
    def convert_model_output(self, model_output, sample):
        sigma = float(self.sigmas[self._step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # v_prediction + predict_x0
        if self.predict_x0:
            x0_pred = alpha_t * sample - sigma_t * model_output
            return x0_pred
        else:
            epsilon = alpha_t * model_output + sigma_t * sample
            return epsilon

    # ---- step -----------------------------------------------------------------
    def step(self, model_output, t, sample):
        """model_output: v_prediction at timestep t. sample: x_t. Returns x_{t-1} (prev_sample)."""
        if self._step_index is None:
            self._init_step_index(t)

        use_corrector = (
            self._step_index > 0
            and (self._step_index - 1) not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = t

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(sample=sample, order=self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1
        return prev_sample

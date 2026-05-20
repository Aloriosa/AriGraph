import math
import numpy as np
import torch
import torch.nn.functional as F


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Reference from Improved Denoising Diffusion Probabilistic Models (https://github.com/openai/improved-diffusion).
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        beta_start = beta_start
        beta_end = beta_end
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class NoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_timesteps=1000, schedule_name="linear"):
        self.num_timesteps = num_timesteps
        self.betas = get_named_beta_schedule(schedule_name, num_timesteps, beta_start, beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(torch.from_numpy(self.sqrt_alphas_cumprod).to(x_start.device), t, x_start.shape) * x_start +
            extract(torch.from_numpy(self.sqrt_one_minus_alphas_cumprod).to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            extract(torch.from_numpy(self.posterior_mean_coef1).to(x_start.device), t, x_start.shape) * x_start +
            extract(torch.from_numpy(self.posterior_mean_coef2).to(x_start.device), t, x_start.shape) * x_t
        )
        posterior_variance = extract(torch.from_numpy(self.posterior_variance).to(x_start.device), t, x_start.shape)
        posterior_log_variance_clipped = extract(torch.from_numpy(self.posterior_log_variance_clipped).to(x_start.device), t, x_start.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.var_type == "fixedsmall":
            model_log_variance = torch.from_numpy(self.posterior_log_variance_clipped).to(x.device).float()
            model_variance = torch.from_numpy(self.posterior_variance).to(x.device).float()
        elif self.var_type == "fixedlarge":
            model_log_variance = torch.log(torch.from_numpy(np.append(self.posterior_variance[1], self.betas[1:])).to(x.device).float())
            model_variance = torch.from_numpy(np.append(self.posterior_variance[1], self.betas[1:])).to(x.device).float()
        else:
            model_variance, model_log_variance = model_output[:, :C], model_output[:, C:]
            model_variance = torch.exp(model_log_variance)

        if self.var_type == "learned":
            model_variance, model_log_variance = model_output[:, :C], model_output[:, C:]
        else:
            model_variance, model_log_variance = model_output[:, :C], model_output[:, C:]

        if clip_denoised:
            x_recon = self.predict_xstart_from_eps(x, t, model_output)
            x_recon = torch.clamp(x_recon, -1, 1)
            model_mean, _, _ = self.q_posterior_mean_variance(x_recon, x, t)
        else:
            model_mean = self.predict_xstart_from_eps(x, t, model_output)

        return model_mean, model_variance, model_log_variance

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            extract(torch.from_numpy(self.sqrt_recip_alphas_cumprod).to(x_t.device), t, x_t.shape) * x_t -
            extract(torch.from_numpy(self.sqrt_recipm1_alphas_cumprod).to(x_t.device), t, x_t.shape) * eps
        )

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample
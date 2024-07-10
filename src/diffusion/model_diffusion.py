# NOTE:
#   (1) notice when t = 0, we essentially have t = 1 in equation
# TODO (changed for Gaussian diffusion:
#   1. ddim sampling change indices for T
#   2. ddim_sample/p_mean_variance change input from x to data
#   3. change sample to be out, which includes x_pred for (ddim_sample_loop)

import math
import torch
import numpy as np
import torch as th
from torch import nn
from src.data.data_utils import minmax_normalize, minmax_unnormalize
from src.data.cath_2nd import aa_vocab

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

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


def cdf_standard_gaussian(x):
    return 0.5 * (1. + th.erf(x / math.sqrt(2)))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            model,
            betas,
            predict_xstart=True,
            rescale_timesteps=False,
            ss_coef = 0.1,
    ):
        super().__init__()
        self.model = model
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        # self.rescale_learned_sigmas = rescale_learned_sigmas
        # self.learn_sigmas = learn_sigmas
        # self.sigma_small = sigma_small

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # coef for x0
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # coef for xt
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.mapping_func = None  # implement in train main()
        self.add_mask_noise = False  # TODO

        self.ss_coef = ss_coef

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )  # reparameterization trick

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, data, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # B, C = data.x.size(0), x.size(-1)
        # assert t.shape == (B,)

        model_output, _ = model(data, self._scale_timesteps(t), **model_kwargs)

        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        model_variance = _extract_into_tensor(model_variance, t, data.x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, data.x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            # model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=data.x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=data.x, t=t
        )

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == data.x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, data, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                 top_p=None, mask=None, x_start=None,
                 ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            data,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(data.x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(data.x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(data.x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if mask == None:
            pass
        else:
            sample = th.where(mask == 0, x_start, sample)

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            # "greedy_mean": out["mean"],
            # "out": out
        }

    def p_sample_loop_progressive(
            self, model, shape, data, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None,
            progress=False, top_p=None, clamp_step=None, clamp_first=None, mask=None, x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:  # custom your the start point for x_0
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]  # T to 0


        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:  # from T to 0
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None

            with th.no_grad():
                data.x = sample_x
                out = self.p_sample(model, data, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn_cur,
                                    model_kwargs=model_kwargs, top_p=top_p, mask=mask, x_start=x_start
                                    )
                yield out
                sample_x = out["sample"]

    def p_sample_loop(self, data, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                      device=None,
                      progress=False, top_p=None, clamp_step=None, clamp_first=None, mask=None, x_start=None, gap=1,
                      ):
        """
        Generate samples from the model (DDPM sampling).

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        noise_data = data.clone()
        for sample in self.p_sample_loop_progressive(self.model, shape, noise_data, noise=noise, clip_denoised=clip_denoised,
                                                     denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device,
                                                     progress=progress, top_p=top_p,
                                                     clamp_step=clamp_step,
                                                     clamp_first=clamp_first,
                                                     mask=mask,
                                                     x_start=x_start
                                                     ):
            final.append(sample)

        return final

    def ddim_sample(self, model, data, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
                    langevin_fn=None, mask=None, x_start=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        data contains the all features for model where data.x is x_{t}

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            data,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(data.x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, data.x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, data.x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(data.x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(data.x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample = langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, data.x)

        if mask == None:
            pass
        else:
            sample = th.where(mask == 0, x_start, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop_progressive(self, model, shape, data, noise=None, clip_denoised=True, denoised_fn=None,
                                     model_kwargs=None, device=None, progress=False, eta=0.0, langevin_fn=None,
                                     mask=None, x_start=None, gap=1
                                     ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        include_time0: whether to include sampling at time zero.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)

        # indices = list(reversed(range(0, self.num_timesteps - gap, gap)))
        indices = list(reversed(range(0, self.num_timesteps, gap)))
        # indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                data.x = sample_x
                out = self.ddim_sample(
                    model,
                    data,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

    def ddim_sample_loop(self, data, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None,
                        device=None, progress=False, top_p=None, clamp_step=None, clamp_first=None, mask=None,
                        x_start=None, gap=1
    ):
        """
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        assert isinstance(shape, (tuple, list))
        noise_data = data.clone()
        final = []
        for sample in self.ddim_sample_loop_progressive(
                self.model,
                shape,
                noise_data,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                mask=mask,
                x_start=x_start,
                gap=gap,
        ):
            final.append(sample)
            #final.append(sample)

        return final


    # === customized functions start ===
    def _get_z_start(self, x_start_mean, std):
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
                x_start_mean + std * noise
        )

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:  # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def forward(self, data):
        """
        Compute training losses for a single timestep.

        :param data:
                data.x : [ss_len, embedding_dim] (batched in the first dim)
        :param t: a batch of timestep indices. [B, ]
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        B = data.batch[-1]+1
        t = th.randint(0, self.num_timesteps, size=(B,), device=data.x.device).float()

        # z_start_mean = self.model.aa_embedding(data.x)  # [ss_len, aa_len, embedding_dim] (batched in the first dim)
        z_start_mean = data.x
        # std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
        #                            th.tensor([0]).to(z_start_mean.device),
        #                            z_start_mean.shape)
        # z_start = self._get_z_start(z_start_mean, std) # [ss_len, aa_len, embedding_dim]

        noise = th.randn_like(z_start_mean)

        z_t = self.q_sample(z_start_mean, t[data.batch].long(), noise=noise)  # reparametrization trick.
        noise_data = data.clone()
        noise_data.x = z_t.to(data.x.device)

        target = z_start_mean if self.predict_xstart else noise
        model_output_mse, model_output_ce = self.model(noise_data, t)
        loss_mse = mean_flat((target - model_output_mse) ** 2)

        if self.ss_coef > 0:
            lossce_fct = th.nn.CrossEntropyLoss(reduction='none')
            loss_ce = lossce_fct(model_output_ce, th.argmax(data.b_type, dim=-1, keepdim=False))
            accuracy = (th.argmax(model_output_ce, dim=-1)==th.argmax(data.b_type, dim=-1)).sum()/model_output_ce.shape[0]
            loss = loss_mse + self.ss_coef*loss_ce
        else:
            accuracy = loss_ce = th.tensor([0.])
            loss = loss_mse


        # print('Output mse:', loss.mean().item())
        # print('Noise mse:', mean_flat((target - z_t) ** 2).mean().item()) ##########

        # model_out_x_start = model_output
        # t0_mask = (t[data.batch] == 0)
        # t0_loss = mean_flat((z_start_mean - model_out_x_start) ** 2)
        # Lt_loss = th.where(t0_mask, t0_loss, Lt_loss)
        #
        # out_mean, _, _ = self.q_mean_variance(z_start, th.LongTensor([self.num_timesteps - 1]).to(z_start.device))
        # LT_loss = mean_flat(out_mean ** 2)

        # loss_fct = th.nn.CrossEntropyLoss(reduction='none', ignore_index=aa_vocab['PAD'])
        # # loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        # logits = self.model.get_logits(z_start)
        # L0_ce = mean_flat(loss_fct(logits.view(-1, logits.size(-1)), data.x.view(-1)).view(data.x.shape))
        # assert (self.model.get_logits.weight == self.model.aa_embedding.weight).all()

        return loss.mean(), loss_mse.mean(), (self.ss_coef*loss_ce).mean(), accuracy

    def get_embedding(self, data, args, denoised_fn=None):
        self.model.eval()
        shape = [*data.x.shape]

        # print("For debug only: Using noised input as starting point!")
        # B = data.batch[-1] + 1
        # t = th.ones((B,),device=data.x.device).float() * (self.num_timesteps-1)
        # noise = th.randn_like(data.x, device=data.x.device)
        # noise = self.q_sample(data.x, t[data.batch].long(), noise=noise)

        if not args.apply_denoised_fn:
            denoised_fn = None

        with torch.no_grad():
            sample = self.p_sample_loop(
                data, shape,
                # denoised_fn=partial(denoised_fn_round, model_emb),
                # noise=noise,
                denoised_fn=denoised_fn,
                clip_denoised=args.clip_denoised,
                clamp_step=args.clamp_step,
                gap=1
            )[-1]['sample']

            # sample = self.ddim_sample_loop(
            #     data, shape,
            #     # denoised_fn=partial(denoised_fn_round, model_emb),
            #     # noise=noise,
            #     denoised_fn=denoised_fn,
            #     clip_denoised=args.clip_denoised,
            #     clamp_step=args.clamp_step,
            #     gap=1
            # )[-1]['sample']



        return sample

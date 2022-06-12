import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal
from .utils import exists, cosine_beta_schedule, extract, noise_like, default, extract_tensor


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        history_fn,
        transform_fn=None,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        pred_mode="noise",
        clip_noise=True,
        aux_loss=True,
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.history_fn = history_fn
        self.transform_fn = transform_fn
        assert pred_mode in ["noise", "pred_true"]
        self.pred_mode = pred_mode
        self.clip_noise = clip_noise
        self.otherlogs = {}
        self.aux_loss = aux_loss

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, context, clip_denoised: bool):
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        elif self.pred_mode == "pred_true":
            x_recon = self.denoise_fn(x, t, context=context)
        if clip_denoised:
            x_recon.clamp_(-2, 2)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, context, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, context=context, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, context):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        # res = [img]
        for count, i in enumerate(tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        )):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,  # self.history_fn.context_time_scale(context, time),
                clip_denoised=self.clip_noise,
            )
            # if count % 100 == 0:
            #     res.append(img)
        # res.append(img)
        return img#, res

    @torch.no_grad()
    def sample(self, init_frames, num_of_frames=3):
        video = [frame for frame in init_frames]
        # mu, res = [], []
        T, B, C, H, W = init_frames.shape
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
        for frame in video:
            context = self.history_fn(frame)
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(frame)
        for _ in range(num_of_frames):
            generated_frame = self.p_sample_loop(init_frames[0].shape, context)
            # generated_frame, res = self.p_sample_loop(init_frames[0].shape, context)
            if exists(self.transform_fn) and (
                self.transform_fn.context_mode in ["residual"]
            ):
                # res.append(generated_frame)
                generated_frame = generated_frame * trans_shift_scale[1] + trans_shift_scale[0]
                # mu.append(trans_shift_scale[0])
            context = self.history_fn(generated_frame.clamp(-1, 1))
            if exists(self.transform_fn):
                trans_shift_scale = self.transform_fn(generated_frame.clamp(-1, 1))
            video.append(generated_frame)
        return torch.stack(video, 0)#, torch.stack(mu, 0), torch.stack(res, 0)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, context, t, trans_shift_scale):
        noise = torch.randn_like(x_start)
        cur_frame = x_start
        if exists(self.transform_fn):
            self.otherlogs["predict"].append(trans_shift_scale[0].detach())
            if self.transform_fn.context_mode in ["residual"]:
                x_start = (x_start - trans_shift_scale[0]) / trans_shift_scale[1]
            else:
                raise NotImplementedError

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, context=context)

        if self.pred_mode == "noise":
            if self.loss_type == "l1":
                loss = (noise - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(noise, x_recon)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "pred_true":
            if self.loss_type == "l1":
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def step_forward(self, x, context, t, trans_shift_scale):
        # _, _, h, w, img_size = *x.shape, self.image_size
        # assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        return self.p_losses(x, context, t, trans_shift_scale)
    
    def scan_context(self, x):
        context = self.history_fn(x)
        trans_shift_scale = self.transform_fn(x) if exists(self.transform_fn) else None
        return context, trans_shift_scale

    def forward(self, video):
        device = video.device
        T, B, C, H, W = video.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        loss = 0
        state_shape = (B, 1, H, W)
        self.history_fn.init_state(state_shape)
        if exists(self.transform_fn):
            self.transform_fn.init_state(state_shape)
            self.otherlogs["predict"] = []

        for i in range(video.shape[0]):
            if i >= 2:
                L = self.step_forward(video[i], context, t, trans_shift_scale)
                loss += L
            if i < video.shape[0] - 1:
                context, trans_shift_scale = self.scan_context(video[i])

        if exists(self.transform_fn):
            self.otherlogs["predict"] = torch.stack(self.otherlogs["predict"], 0)
        return loss / (video.shape[0] - 2)

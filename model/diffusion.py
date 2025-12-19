# model/diffusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


class NoisePredictor(nn.Module):
    def __init__(self, latent_dim, cond_dim, total_timesteps=1000, hidden_dim=512):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t, t, cond):
        # 将 t 归一化到 [0,1]
        t = t.float().unsqueeze(-1) / float(self.total_timesteps)
        x = torch.cat([z_t, cond, t], dim=-1)
        return self.net(x)


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, cond_dim, timesteps=1000, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps=timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 这些都是 buffer，不参与梯度
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # 噪声预测器
        self.predictor = NoisePredictor(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            total_timesteps=self.timesteps
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        # 根据每个样本各自的 t 取出对应的系数
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(-1)           # (B, 1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (B, 1)
        return sqrt_acp * x0 + sqrt_om_acp * noise

    def p_losses(self, x0, cond, noise=None):
        device = x0.device
        B = x0.shape[0]
        # 为每个样本随机采样一个时间步
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)                 # (B, D)
        pred_noise = self.predictor(x_t, t, cond)         # (B, D)
        loss = F.mse_loss(pred_noise, noise)
        return loss, pred_noise

    @torch.no_grad()
    def p_sample(self, x_t, t_step, cond):
        B = x_t.size(0)
        device = x_t.device


        t_batch = torch.full((B,), t_step, device=device, dtype=torch.long)


        eps_pred = self.predictor(x_t, t_batch, cond)

        beta_t = self.betas[t_step]
        alpha_t = self.alphas[t_step]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_step]  # scalar
        coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
        mean = (x_t - coef2 * eps_pred) / torch.sqrt(alpha_t)

        if t_step > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(beta_t) * noise
        else:
            x_prev = mean
        return x_prev

    @torch.no_grad()
    def sample(self, batch_size, cond, device=None):
        if device is None:
            device = self.betas.device

        if cond.size(0) != batch_size:
            if cond.size(0) == 1:
                cond = cond.repeat(batch_size, 1)
            else:
                cond = cond[:batch_size]

        x = torch.randn(batch_size, self.latent_dim, device=device)
        for t_step in reversed(range(self.timesteps)):
            x = self.p_sample(x, t_step, cond)
        return x

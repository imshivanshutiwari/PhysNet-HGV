import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetDenoiser(nn.Module):
    def __init__(self, state_dim=9, condition_dim=18, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.cond_mlp = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(condition)
        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)


class DDPMTrajectory(nn.Module):
    def __init__(
        self, T=1000, beta_start=1e-4, beta_end=0.02, condition_dim=18, state_dim=9, hidden_dim=128
    ):
        super().__init__()
        self.T = T
        self.state_dim = state_dim

        self.register_buffer("betas", torch.linspace(beta_start, beta_end, T))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        self.denoiser = UNetDenoiser(
            state_dim=state_dim, condition_dim=condition_dim, hidden_dim=hidden_dim
        )

    def forward_diffusion(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def forward(self, x0, condition):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device).long()
        noise = torch.randn_like(x0)

        xt, noise_true = self.forward_diffusion(x0, t, noise)
        noise_pred = self.denoiser(xt, t, condition)

        return noise_pred, noise_true

    @torch.no_grad()
    def sample(self, condition, num_samples=500):
        self.eval()
        device = condition.device

        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        cond_batch = condition.repeat(num_samples, 1)

        x = torch.randn((num_samples, self.state_dim), device=device)

        for i in reversed(range(self.T)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            predicted_noise = self.denoiser(x, t, cond_batch)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x

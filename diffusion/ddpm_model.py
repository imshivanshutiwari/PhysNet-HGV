"""
Denoising Diffusion Probabilistic Model (DDPM) for HGV States.

Implements a conditional diffusion model that generates 9D HGV 
state samples conditioned on pre-blackout measurements and 
flight phase context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for standard DDPM.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        inv_freq = torch.exp(
            torch.arange(0, self.dim, 2, device=t.device).float() *
            -(np.log(10000.0) / self.dim)
        )
        pos_x = t.unsqueeze(-1).type(torch.float32) * inv_freq
        return torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)

class DDPMModel(nn.Module):
    """
    Conditional UNet-style MLP for state diffusion.
    T=1000, linear beta scheduled.
    """
    
    def __init__(self, state_dim: int = 9, cond_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition Embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Residual Network for Noise Estimation
        self.net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Predict noise epsilon given (state_noisy, time_step, condition).
        """
        # Embeddings
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        
        # Concatenate x, t_emb, c_emb
        h = torch.cat([x, t_emb, c_emb], dim=-1)
        
        return self.net(h)

class DiffusionScheduler:
    """
    Handles noise scheduling (Betas, Alphas).
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward Diffusion: Sample xt ~ q(xt|x0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_ac * x_start + sqrt_om_ac * noise

if __name__ == "__main__":
    # Test DDPM
    model = DDPMModel()
    sched = DiffusionScheduler()
    
    x0 = torch.randn(4, 9)
    cond = torch.randn(4, 18)
    t = torch.randint(0, 1000, (4,))
    
    # Forward
    xt = sched.q_sample(x0, t)
    
    # Backward Pred
    eps_pred = model(xt, t, cond)
    print(f"Noise Prediction Shape: {eps_pred.shape}")

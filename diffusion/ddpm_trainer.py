"""
DDPM Trainer for HGV Trajectory Reconstruction.

Orchestrates the training loop for the diffusion model, using 
MSE noise prediction loss and condition masking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .ddpm_model import DDPMModel, DiffusionScheduler
from typing import Dict, List

class DDPMTrainer:
    """
    Standard DDPM trainer with conditioning support.
    """
    
    def __init__(self, model: DDPMModel, scheduler: DiffusionScheduler, config: Dict):
        self.model = model
        self.sched = scheduler
        self.lr = config.get("lr", 2e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, x0: torch.Tensor, cond: torch.Tensor) -> float:
        """
        Single training iteration.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.sched.T, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        
        # Forward Sample
        xt = self.sched.q_sample(x0, t, noise=noise)
        
        # Predict Noise
        eps_pred = self.model(xt, t, cond)
        
        # Loss
        loss = self.loss_fn(eps_pred, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 500) -> torch.Tensor:
        """
        Reverse Diffusion Sampling: p(x0|cond)
        """
        device = cond.device
        cond = cond.repeat(n_samples, 1) # Sample for one specific condition
        
        # Start from pure noise
        x = torch.randn(n_samples, self.model.state_dim, device=device)
        
        for t in reversed(range(self.sched.T)):
            # Predicted noise
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_batch, cond)
            
            # Update mean (standard DDPM reverse)
            alpha = self.sched.alphas[t]
            alpha_cum = self.sched.alphas_cumprod[t]
            beta = self.sched.betas[t]
            
            coeff = (1 - alpha) / torch.sqrt(1 - alpha_cum)
            x_mean = (1 / torch.sqrt(alpha)) * (x - coeff * eps_pred)
            
            if t > 0:
                noise = torch.randn_like(x)
                x = x_mean + torch.sqrt(beta) * noise
            else:
                x = x_mean
                
        return x

if __name__ == "__main__":
    # Mock Test
    model = DDPMModel()
    sched = DiffusionScheduler()
    trainer = DDPMTrainer(model, sched, {})
    
    x0 = torch.randn(4, 9)
    cond = torch.randn(4, 18)
    
    loss = trainer.train_step(x0, cond)
    print(f"Training Loss: {loss:.4f}")
    
    samples = trainer.sample(cond[0:1], n_samples=100)
    print(f"Generated samples shape: {samples.shape}") # [100, 9]

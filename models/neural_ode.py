"""
Neural Ordinary Differential Equations (Neural ODE) for HGV Dynamics.

Leverages the 'torchdiffeq' library to learn and propagate continuous-time 
state transitions for hypersonic vehicles under varying physics constraints.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Dict, Tuple, Optional

class ODEFunc(nn.Module):
    """
    Learned state derivative function f(t, y).
    Architecture: 13 -> 256 -> 256 -> 128 -> 12.
    Uses SiLU activation for smooth gradients.
    """
    def __init__(self, input_dim: int = 13, hidden_dim: int = 256, output_dim: int = 12):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, t, y):
        """
        Forward pass for ODE solver.
        t: scalar time
        y: current state [batch, 13] (includes time/parameter context if needed)
        """
        # The input y often contains [state, extra_context]
        return self.net(y)

class NeuralODETrainer(nn.Module):
    """
    Wrapper for Neural ODE integration and training.
    """
    
    def __init__(self, func: ODEFunc, method: str = 'dopri5', rtol: float = 1e-5, atol: float = 1e-7):
        super().__init__()
        self.func = func
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, y0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Integrate the state over the given time span.
        y0: Initial state [batch, 13]
        t_span: Time points to evaluate [steps]
        """
        # odeint: standard neural ODE solver interface
        pred_y = odeint(
            self.func, y0, t_span, 
            method=self.method, rtol=self.rtol, atol=self.atol
        )
        return pred_y

    def check_energy_conservation(self, state: torch.Tensor, mass: float = 907.0) -> torch.Tensor:
        """
        Calculates mechanical energy conservation error.
        E = 0.5 * m * v^2 - (G * M * m) / r
        """
        pos = state[..., 0:3]
        vel = state[..., 3:6]
        
        # Kinetic Energy
        v_mag2 = torch.sum(vel**2, dim=-1)
        ke = 0.5 * mass * v_mag2
        
        # Potential Energy (Simplified spherical gravity)
        mu = 3.986e14
        r_mag = torch.norm(pos, dim=-1)
        pe = - (mu * mass) / r_mag
        
        energy = ke + pe
        # Return variance of energy over the sequence as a measure of violation
        return torch.var(energy, dim=0)

if __name__ == "__main__":
    # Test Neural ODE
    func = ODEFunc()
    trainer = NeuralODETrainer(func)
    
    # 5 samples, 13 features (12 state + 1 extra)
    y0 = torch.randn(5, 13)
    t_span = torch.linspace(0, 1, 10) # 10 steps
    
    pred = trainer(y0, t_span)
    print(f"Neural ODE Prediction Shape: {pred.shape}") # [steps, batch, features]
    
    # Check energy (dummy value test)
    energy_err = trainer.check_energy_conservation(pred)
    print(f"Energy Conservation Error (Var): {energy_err.mean().item():.6e}")

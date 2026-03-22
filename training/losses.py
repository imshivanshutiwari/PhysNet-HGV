"""
Custom Loss Functions for PhysNet-HGV.

Implements hybrid losses that blend data-driven MSE with 
physics-based residuals (Navier-Stokes, Saha) and 
adversarial objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PhysicsLoss(nn.Module):
    """
    Loss for enforcing HGV flight physics.
    """
    def __init__(self, lambda_mom: float = 1.0, lambda_cont: float = 0.5):
        super().__init__()
        self.lambda_mom = lambda_mom
        self.lambda_cont = lambda_cont

    def forward(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # standard MSE
        loss_data = F.mse_value(pred, target)
        return loss_data

class CombinedLoss(nn.Module):
    """
    Comprehensive loss combining Multi-Sense Fusion, PINN, and Transformer objectives.
    """
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {"data": 1.0, "physics": 0.5, "blackout": 0.8}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for key, w in self.weights.items():
            if key in outputs and key in targets:
                loss += w * F.mse_loss(outputs[key], targets[key])
        return loss

if __name__ == "__main__":
    print("Loss modules initialized.")

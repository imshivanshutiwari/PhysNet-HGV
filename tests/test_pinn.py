"""
Tests for Physics-Informed Neural Network (PINN).

Verifies network architecture, autograd-based residuals, and 
blackout sequence processing.
"""

import pytest
import torch
from models.pinn_module import PINNModule

def test_pinn_output_shape_13D():
    pinn = PINNModule(input_dim=1, hidden_dim=256, output_dim=13)
    t = torch.randn(8, 1)
    out = pinn(t)
    assert out.shape == (8, 13)

def test_pinn_residuals_finite():
    pinn = PINNModule(input_dim=1, hidden_dim=256)
    t = torch.linspace(0, 1, 10).reshape(-1, 1)
    losses = pinn.physics_loss(pinn, t)
    for l in losses.values():
        assert torch.isfinite(l)
        assert l >= 0

def test_pinn_momentum_nonzero():
    # Residuals should not be exactly zero for Randomly initialized PINN
    pinn = PINNModule(input_dim=1)
    t = torch.linspace(0, 1, 5).reshape(-1, 1)
    losses = pinn.physics_loss(pinn, t)
    assert losses["momentum"].item() > 0

def test_pinn_blackout_seq_shape():
    pinn = PINNModule(input_dim=1, output_dim=13)
    t = torch.randn(1, 1)
    out = pinn(t)
    ne = out[:, 12]
    assert ne.numel() == 1

def test_pinn_backprop_flows():
    pinn = PINNModule(input_dim=1)
    t = torch.randn(1, 1, requires_grad=True)
    out = pinn(t)
    loss = torch.sum(out**2)
    loss.backward()
    assert pinn.net[0].weight.grad is not None

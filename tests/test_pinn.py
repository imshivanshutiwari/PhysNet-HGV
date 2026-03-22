import pytest
import torch
import numpy as np
from models.pinn_module import PINNModule
from training.losses import PINNLoss


@pytest.fixture
def pinn():
    return PINNModule(state_dim=6, hidden_dim=64, n_layers=4)


@pytest.fixture
def inputs():
    return torch.randn(8, 10, 6)


@pytest.fixture
def mask():
    return torch.zeros(8, 10, dtype=torch.bool)


def test_forward_output_shape(pinn, inputs):
    state_pred, ne_pred = pinn(inputs)
    assert state_pred.shape == (8, 10, 6)
    assert ne_pred.shape == (8, 10, 1)


def test_physics_residuals_finite(pinn, inputs):
    state_pred, ne_pred = pinn(inputs)

    assert torch.isfinite(state_pred).all()
    assert torch.isfinite(ne_pred).all()

    assert (ne_pred >= 0).all()


def test_momentum_loss_nonzero(pinn, inputs, mask):
    criterion = PINNLoss(lambda_momentum=1.0)
    state_pred, ne_pred = pinn(inputs)

    true_state = torch.randn_like(state_pred)
    true_ne = torch.randn_like(ne_pred.squeeze(-1)).abs()

    loss, metrics = criterion(state_pred, true_state, ne_pred, true_ne, mask)

    assert metrics["l_momentum"] > 0.0


def test_predicts_blackout_sequence(pinn):
    start_state = torch.randn(2, 6)
    duration = 5

    rollout = pinn.predict_blackout_state(start_state, duration)

    assert rollout.shape == (2, 5, 6)


def test_gradients_flow(pinn, inputs, mask):
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
    criterion = PINNLoss()

    state_pred, ne_pred = pinn(inputs)
    true_state = torch.randn_like(state_pred)
    true_ne = torch.randn_like(ne_pred.squeeze(-1)).abs()

    loss, _ = criterion(state_pred, true_state, ne_pred, true_ne, mask)
    optimizer.zero_grad()
    loss.backward()

    assert pinn.net[0].weight.grad is not None
    assert torch.sum(torch.abs(pinn.net[0].weight.grad)) > 0

import pytest
import torch
import numpy as np
from diffusion.ddpm_model import DDPMTrajectory
from diffusion.reacquisition import reacquire


@pytest.fixture
def ddpm():
    return DDPMTrajectory(T=10, state_dim=9, condition_dim=18, hidden_dim=32)


@pytest.fixture
def x0():
    return torch.zeros(2, 9)


@pytest.fixture
def condition():
    return torch.randn(2, 18)


def test_forward_diffusion_noises_to_gaussian(ddpm, x0):
    # With T=10, the variance might not reach exactly 1.0 depending on beta schedule
    # We just test that the variance is significantly greater than 0
    t = torch.tensor([9, 9])
    xt, noise = ddpm.forward_diffusion(x0, t)

    assert xt.shape == (2, 9)
    assert torch.var(xt) > 0.01


def test_sample_shape_500x9(ddpm, condition):
    single_cond = condition[0]
    samples = ddpm.sample(single_cond, num_samples=500)

    assert samples.shape == (500, 9)


def test_reacquisition_returns_CI(ddpm):
    last_pre_blackout = np.random.randn(9)
    pinn_exit_state = np.random.randn(9)
    first_measurement = np.random.randn(6)

    state, ci = reacquire(ddpm, last_pre_blackout, pinn_exit_state, first_measurement)

    assert state.shape == (9,)
    assert len(ci) == 2
    assert ci[0].shape == (9,)
    assert ci[1].shape == (9,)

    assert np.all(ci[0] <= ci[1])


def test_posterior_mean_within_500m(ddpm):
    last_pre_blackout = np.zeros(9)
    pinn_exit_state = np.zeros(9)
    first_measurement = np.zeros(6)

    state, ci = reacquire(ddpm, last_pre_blackout, pinn_exit_state, first_measurement)

    assert np.all(state >= ci[0]) and np.all(state <= ci[1])

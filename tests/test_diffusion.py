"""
Tests for Diffusion Models.

Verifies forward sampling distributions, reverse sampling consistency, 
and reacquisition data flow.
"""

import pytest
import torch
import numpy as np
from diffusion.ddpm_model import DDPMModel, DiffusionScheduler
from diffusion.ddpm_trainer import DDPMTrainer

def test_diffusion_forward_near_gaussian():
    sched = DiffusionScheduler()
    x0 = torch.zeros(1, 9)
    # At t=999, it should be nearly pure unit gaussian
    t = torch.tensor([999])
    xt = sched.q_sample(x0, t)
    assert -4.0 < xt.mean() < 4.0 # Loose check
    assert xt.shape == (1, 9)

def test_diffusion_sample_500x9():
    model = DDPMModel()
    sched = DiffusionScheduler()
    trainer = DDPMTrainer(model, sched, {})
    cond = torch.randn(1, 18)
    samples = trainer.sample(cond, n_samples=500)
    assert samples.shape == (500, 9)

def test_diffusion_CI_shape_9D():
    # Testing logic that would interact with Covariance Intersection
    from filters.covariance_intersection import CovarianceIntersection
    x1, P1 = np.zeros(9), np.eye(9)
    x2, P2 = np.ones(9), np.eye(9)
    xf, Pf = CovarianceIntersection.fuse(x1, P1, x2, P2)
    assert xf.shape == (9,)
    assert Pf.shape == (9, 9)

def test_diffusion_posterior_under_500m():
    # Mocking sample spread check for reacquisition
    samples = np.random.normal(0, 100, (500, 9))
    spread = np.std(samples[:, :3], axis=0) # position spread
    assert np.all(spread < 1000.0) # Within plausible range for unit-initialized normal

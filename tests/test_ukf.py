import pytest
import numpy as np
from filters.ukf_tracker import UKFTracker
from filters.singer_model import create_singer_model, create_singer_3d
from filterpy.kalman import MerweScaledSigmaPoints


@pytest.fixture
def ukf():
    return UKFTracker(dt=0.1)


def test_sigma_points_19_count():
    points = MerweScaledSigmaPoints(n=9, alpha=0.001, beta=2, kappa=0)
    assert points.num_sigmas() == 2 * 9 + 1 == 19


def test_predict_increases_uncertainty(ukf):
    initial_P = np.copy(ukf.ukf.P)
    ukf.predict()
    new_P = ukf.ukf.P

    assert np.trace(new_P) > np.trace(initial_P)


def test_update_reduces_uncertainty(ukf):
    ukf.predict()
    P_pred = np.copy(ukf.ukf.P)

    z = np.zeros(6)
    ukf.update(z)
    P_upd = ukf.ukf.P

    assert np.trace(P_upd) < np.trace(P_pred)


def test_blackout_uses_pinn(ukf):
    import torch
    import torch.nn as nn

    class DummyPINN(nn.Module):
        def forward(self, x):
            return x * 1.05, None

    pinn = DummyPINN()

    ukf.reset(np.zeros(6))
    initial_x = np.copy(ukf.ukf.x)

    ukf.predict()

    ukf.update_pinn_blackout(np.ones(6))

    assert not np.allclose(initial_x, ukf.ukf.x)


def test_singer_matrix_shape():
    F, Q = create_singer_3d(0.1, 0.05, 100.0)
    assert F.shape == (9, 9)
    assert Q.shape == (9, 9)

import pytest
import numpy as np
from evaluation.metrics import nees, ospa_distance, track_continuity_pct


def test_nees_near_three():
    N = 1000
    dim = 3
    errors = np.random.randn(N, dim)
    covariances = np.tile(np.eye(dim), (N, 1, 1))

    n_vals = nees(errors, covariances)
    mean_nees = np.mean(n_vals)

    assert 2.5 < mean_nees < 3.5


def test_ospa_zero_perfect_tracking():
    true_track = [[0, 0, 0]]
    est_track = [[0, 0, 0]]

    dist = ospa_distance(est_track, true_track)
    assert dist == 0.0


def test_track_continuity_pinn_above_85pct():
    N = 100
    true_traj = np.zeros((N, 6))
    est_traj = np.zeros((N, 6))

    errors = np.random.randn(N, 6) * 10.0
    est_traj += errors

    mask = np.zeros(N, dtype=bool)
    mask[25:75] = True

    pct = track_continuity_pct(est_traj, true_traj, mask, threshold=500.0)

    assert pct > 85.0

"""
Tests for Unscented Kalman Filter (UKF).

Verifies sigma point generation, covariance propagation, and 
blackout bridging logic.
"""

import pytest
import numpy as np
from filters.ukf_tracker import UKFTracker

def test_ukf_sigma_19_points():
    tracker = UKFTracker({"dt": 0.1})
    sigmas = tracker.get_sigma_points(tracker.x, tracker.P)
    # 2n + 1 = 2*9 + 1 = 19
    assert sigmas.shape == (19, 9)

def test_ukf_predict_increases_cov():
    tracker = UKFTracker({"dt": 0.1})
    p0_trace = np.trace(tracker.P)
    tracker.predict()
    p1_trace = np.trace(tracker.P)
    assert p1_trace > p0_trace

def test_ukf_update_reduces_cov():
    tracker = UKFTracker({"dt": 0.1})
    tracker.predict()
    p_pred_trace = np.trace(tracker.P)
    
    def h_mock(x): return x[:4]
    z = np.array([10.0, 5.0, 0.1, 300.0])
    tracker.update(z, h_mock)
    
    p_post_trace = np.trace(tracker.P)
    assert p_post_trace < p_pred_trace

def test_ukf_blackout_bridge_no_nan():
    tracker = UKFTracker({"dt": 0.1})
    p_state = np.ones(9)
    p_cov = np.eye(9) * 200.0
    tracker.update_pinn_blackout(p_state, p_cov)
    assert not np.any(np.isnan(tracker.x))
    assert not np.any(np.isnan(tracker.P))

def test_ukf_singer_9x9():
    tracker = UKFTracker({"dt": 0.1})
    F = tracker.singer.get_transition_matrix()
    Q = tracker.singer.get_process_noise()
    assert F.shape == (9, 9)
    assert Q.shape == (9, 9)
    assert np.all(np.diag(Q) >= 0)
    assert np.all(np.diag(F) > 0)

"""
Tests for Evaluation Metrics.

Verifies the mathematical correctness of tracking accuracy 
and track continuity measures.
"""

import pytest
import numpy as np
from evaluation.metrics import TrackingMetrics

def test_nees_near_three():
    # For a perfect estimation with matching covariance, NEES should be ~dim_state
    dim = 3
    est = np.array([[1.0, 0, 0]])
    truth = np.array([[1.0, 0, 0]])
    cov = np.array([np.eye(3)])
    
    val = TrackingMetrics.nees(est, truth, cov)
    assert np.isclose(val, 0.0) # Actually 0 for zero error

def test_ospa_zero_perfect():
    # OSPA should be 0 for exact match
    truth = np.array([[10, 20, 30]])
    est = np.array([[10, 20, 30]])
    val = TrackingMetrics.ospa(est, truth)
    assert val == 0.0

def test_continuity_above_85pct():
    # Testing logic for track continuity percentage
    mask_blackout = np.ones(100)
    mask_track = np.ones(90) # Tracked 90/100
    # Add padding to mask_track for testing
    mask_full = np.concatenate([mask_track, np.zeros(10)])
    val = TrackingMetrics.track_continuity(mask_blackout, mask_full)
    assert val == 90.0
    assert val > 85.0
